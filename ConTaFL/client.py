import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import numpy as np
import copy

class Client(object):
    def __init__(self, args, data_client, train_samples, model):
        super(Client, self).__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.num_classes = args.num_classes
        self.eval_step = args.eval_step
        self.data_client = data_client
        self.train_samples = train_samples
        self.model = model.to(self.device)

        # Optimiser and core hyper-parameters
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.temperature = args.temperature
        self.threshold = args.threshold

        # ConTaFL-specific hyper-parameters
        self.mc_samples = args.mc_samples
        self.beta_hybrid = args.beta_hybrid
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2

    def download_params(self, global_params):
        self.model.load_state_dict(global_params)

    def train(self):
        data_loader = DataLoader(dataset=self.data_client, batch_size=self.batch_size, shuffle=True)
        for step in range(self.local_steps):
            for images, labels, indexs in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                logits = outputs[1]
                loss = self.ce_loss(logits, labels)
                loss.backward()
                self.optimizer.step()

    def train_with_contrastive_loss(self):
        self.model.train()
        data_loader = DataLoader(dataset=self.data_client, batch_size=self.batch_size, shuffle=True)
        all_features = []
        all_labels = []

        for step in range(self.local_steps):
            for images, labels, indexs in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                features, _, _ = self.model(images)
                loss = self.contrastive_loss(features, labels)
                loss.backward()
                self.optimizer.step()
                all_features.append(features)
                all_labels.append(labels)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return self.identify_rare_classes_using_contrastive(all_features, all_labels)

    def contrastive_loss(self, features, labels, margin=1.0):
        features = F.normalize(features, p=2, dim=-1)
        similarity_matrix = torch.matmul(features, features.T)
        labels = labels.float()
        positive_pairs = labels * similarity_matrix
        negative_pairs = (1 - labels) * torch.relu(margin - similarity_matrix)
        return torch.mean(positive_pairs + negative_pairs)

    def identify_rare_classes_using_contrastive(self, features, labels, n_clusters=10):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(features.detach().cpu().numpy())
        cluster_labels = kmeans.labels_
        cluster_sizes = [sum(cluster_labels == i) for i in range(n_clusters)]
        rare_classes = [i for i in range(n_clusters) if cluster_sizes[i] < np.percentile(cluster_sizes, 20)]
        return rare_classes

    def calculate_uncertainty(self, logits):
        probabilities = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-6), dim=-1)
        return entropy

    
    def compute_representation_divergence(self, global_model, data_loader):
        """Representation agreement between client and global features (U_rep)."""
        self.model.eval()
        global_model.eval()

        sims = []
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, tuple):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(self.device)

                feat_client, logits_client, _ = self.model(images)
                feat_global, logits_global, _ = global_model(images)

                feat_client = F.normalize(feat_client, p=2, dim=-1)
                feat_global = F.normalize(feat_global, p=2, dim=-1)

                sim = torch.sum(feat_client * feat_global, dim=-1)
                sims.append(sim)

        if not sims:
            return 0.0

        all_sims = torch.cat(sims, dim=0)
        return all_sims.mean().item()

    def compute_prediction_uncertainty_mc(self, data_loader):
        """MC Dropout-based predictive uncertainty (U_pu)."""
        self.model.train()
        all_uncertainties = []

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, tuple):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(self.device)

                logits_samples = []
                for _ in range(self.mc_samples):
                    _, logits_drop, _ = self.model(images)
                    logits_samples.append(F.softmax(logits_drop, dim=-1))

                stack_probs = torch.stack(logits_samples, dim=0)
                var_probs = stack_probs.var(dim=0)
                unc = var_probs.sum(dim=-1)
                all_uncertainties.append(unc)

        if not all_uncertainties:
            return 0.0

        all_unc = torch.cat(all_uncertainties, dim=0)
        return all_unc.mean().item()

    def compute_hybrid_unreliability(self, global_model, data_loader):
        """Hybrid unreliability ˜U_k combining U_pu and U_rep."""
        U_pu = self.compute_prediction_uncertainty_mc(data_loader)
        U_rep = self.compute_representation_divergence(global_model, data_loader)
        tilde_U = self.beta_hybrid * U_pu + (1.0 - self.beta_hybrid) * (1.0 - U_rep)
        return tilde_U, U_pu, U_rep

    def adaptive_distillation(self, global_model, tilde_U_k, data_loader):
        """Adaptive distillation with negative distillation and pseudo-loss."""
        self.model.train()
        self.optimizer.zero_grad()

        total_kl = 0.0
        total_ce = 0.0
        total_samples = 0

        for batch in data_loader:
            if isinstance(batch, tuple):
                images = batch[0]
            else:
                images = batch
            images = images.to(self.device)

            with torch.no_grad():
                _, teacher_logits, _ = global_model(images)
                pseudo_labels = torch.argmax(teacher_logits, dim=-1)

            _, student_logits, _ = self.model(images)

            teacher_prob = F.softmax(teacher_logits / self.temperature, dim=-1)
            student_log_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
            kl = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean')

            ce = F.cross_entropy(student_logits, pseudo_labels)

            loss = self.lambda1 * (1.0 - tilde_U_k) * kl + self.lambda2 * ce
            loss.backward()

            total_kl += kl.item()
            total_ce += ce.item() * images.size(0)
            total_samples += images.size(0)

        self.optimizer.step()

        L_pseudo = total_ce / total_samples if total_samples > 0 else 0.0
        return total_kl, L_pseudo

    def distillation_loss(self, student_outputs, teacher_outputs):
        student_probabilities = F.softmax(student_outputs, dim=-1)
        teacher_probabilities = F.softmax(teacher_outputs, dim=-1)
        return F.kl_div(student_probabilities.log(), teacher_probabilities, reduction='batchmean')
