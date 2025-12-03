import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn

from client import Client


class Server(object):
    def __init__(self, args, train_data_list, global_test_dataset, global_distill_dataset,
                 global_student, temperature: float, mini_batch_size_distillation: int, lamda):
        super(Server, self).__init__()
        self.args = args
        self.device = args.device
        self.global_rounds = args.global_rounds
        self.batch_size = args.batch_size

        # Federated configuration
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)

        # Reliability and rehabilitation hyper-parameters
        self.reliability_threshold = args.reliability_threshold
        self.rehab_threshold = args.rehab_threshold
        self.beta_hybrid = args.beta_hybrid
        self.gamma_momentum = args.gamma_momentum
        self.rehab_window = args.rehab_window
        self.rehab_epsilon = args.rehab_epsilon

        # Distillation-related parameters
        self.temperature = temperature
        self.mini_batch_size_distillation = mini_batch_size_distillation
        self.lamda = lamda

        # Datasets
        self.train_data_list = train_data_list
        self.global_test_dataset = global_test_dataset
        self.global_distill_dataset = global_distill_dataset

        # Global model
        self.global_student = copy.deepcopy(global_student)
        self.global_student.to(self.device)
        self.dict_global_params = self.global_student.state_dict()
        self.optimizer = Adam(self.global_student.parameters(),
                              lr=args.global_learning_rate,
                              weight_decay=args.weight_decay)

        # Clients
        self.clients = []
        self.selected_client_indexes = []
        self.list_dicts_local_params = []

        # Moving reliability scores τ_k (initialised to 1)
        self.tau = np.ones(self.num_clients, dtype=float)
        # Pseudo-loss history for each client
        self.pseudo_loss_history = [[] for _ in range(self.num_clients)]

        for i in range(self.num_clients):
            client = Client(
                args=args,
                data_client=self.train_data_list[i],
                train_samples=len(self.train_data_list[i]),
                model=copy.deepcopy(self.global_student),
            )
            self.clients.append(client)

    def select_client_indexes(self):
        selected_clients = np.random.choice(range(self.num_clients), self.join_clients, replace=False)
        return list(selected_clients)

    def send_models(self):
        for select_id in self.selected_client_indexes:
            self.clients[select_id].download_params(self.dict_global_params)

    def train(self):
        self.test_acc = []

        distill_loader = DataLoader(
            self.global_distill_dataset,
            batch_size=self.mini_batch_size_distillation,
            shuffle=True,
        )

        for t in range(1, self.global_rounds + 1):
            P_t = self.select_client_indexes()
            self.selected_client_indexes = P_t
            print(f'Round {t}: selected clients P_t = {P_t}')

            # Step 1: compute ˜U_k for selected clients
            tilde_U = {}
            U_pu = {}
            U_rep = {}
            for k in P_t:
                tilde_U[k], U_pu[k], U_rep[k] = self.clients[k].compute_hybrid_unreliability(
                    self.global_student, distill_loader
                )
                print(f'Round {t}, Client {k}: ˜U={tilde_U[k]:.4f}, U_pu={U_pu[k]:.4f}, U_rep={U_rep[k]:.4f}')

            # Step 2: split into reliable and unreliable
            R_t = [k for k in P_t if tilde_U[k] < self.reliability_threshold]
            N_t = [k for k in P_t if k not in R_t]
            print(f'Round {t}: R_t={R_t}, N_t={N_t}')

            # Step 3: send global model
            self.send_models()
            self.list_dicts_local_params = []

            # Reliable clients: local training only
            for k in R_t:
                client = self.clients[k]
                client.train()
                rare_classes = client.train_with_contrastive_loss()
                print(f'Round {t}, Client {k} (reliable): rare classes {rare_classes}')
                self.list_dicts_local_params.append(copy.deepcopy(client.model.state_dict()))

            # Unreliable clients: local training + adaptive distillation
            for k in N_t:
                client = self.clients[k]
                client.train()
                rare_classes = client.train_with_contrastive_loss()
                print(f'Round {t}, Client {k} (unreliable): rare classes {rare_classes}')

                _, L_pseudo_k = client.adaptive_distillation(
                    self.global_student,
                    tilde_U[k],
                    distill_loader,
                )

                hist = self.pseudo_loss_history[k]
                hist.append(L_pseudo_k)
                self.pseudo_loss_history[k] = hist

                delta_L = None
                if len(hist) > self.rehab_window:
                    L_old = hist[-(self.rehab_window + 1)]
                    L_new = hist[-1]
                    delta_L = L_old - L_new
                    print(f'Round {t}, Client {k}: L_pseudo={L_new:.4f}, ΔL_pseudo={delta_L:.4f}')
                else:
                    print(f'Round {t}, Client {k}: L_pseudo={L_pseudo_k:.4f} (insufficient history)')

                # Eq. (9): update τ_k
                old_tau = self.tau[k]
                self.tau[k] = self.gamma_momentum * self.tau[k] + (1.0 - self.gamma_momentum) * (1.0 - tilde_U[k])
                print(f'Round {t}, Client {k}: τ updated from {old_tau:.4f} to {self.tau[k]:.4f}')

                # Rehab condition (logging)
                if delta_L is not None and self.tau[k] >= self.rehab_threshold and delta_L >= self.rehab_epsilon:
                    print(f'Round {t}, Client {k}: rehabilitation condition met.')

                self.list_dicts_local_params.append(copy.deepcopy(client.model.state_dict()))

            # Step 4: reliability-aware aggregation over all selected clients
            self.aggregate_parameters_with_reliability(P_t, tilde_U)

            acc, loss = self.evaluate()
            self.test_acc.append(acc)
            print(f'Round {t} - Accuracy: {acc:.4f}, Loss: {loss:.4f}')

    def aggregate_parameters_with_reliability(self, selected_clients, tilde_U):
        """Aggregate local models using reliability-aware weights (1 - ˜U_k)."""
        for name_param in self.dict_global_params:
            weighted_params = []
            total_weight = 0.0
            for k, client_params in zip(selected_clients, self.list_dicts_local_params):
                w_k = max(1.0 - tilde_U[k], 0.0)
                weighted_params.append(client_params[name_param] * w_k)
                total_weight += w_k

            if total_weight > 0:
                value_global_param = sum(weighted_params) / total_weight
            else:
                value_global_param = self.dict_global_params[name_param]

            self.dict_global_params[name_param] = value_global_param

        self.global_student.load_state_dict(self.dict_global_params)

    def evaluate(self):
        self.global_student.eval()
        test_loader = torch.utils.data.DataLoader(
            self.global_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_student(images)

                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    outputs = outputs[1]

                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return accuracy, avg_loss
