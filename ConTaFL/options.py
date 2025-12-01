import argparse
import os


def args_parser():
    """Argument parser for ConTaFL experiments.

    Defaults are chosen to match the training specifications described
    in the ConTaFL paper (20 clients, 100 rounds, 40% participation,
    SGD with lr=0.025, momentum=0.9, weight decay=5e-4, etc.).
    """
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)

    # ---------------- Dataset & paths ----------------
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'mnist'],
                        help='Dataset used in experiments.')
    parser.add_argument('--data_path', type=str, default='../data',
                        help='Root path for datasets.')
    parser.add_argument('--data_path_cifar100', type=str,
                        default=os.path.join(path_dir, '../data/cifar-100-python'))

    # ---------------- Federated settings ----------------
    parser.add_argument('--num_clients', type=int, default=20,
                        help='Total number of clients K.')
    parser.add_argument('--global_rounds', type=int, default=100,
                        help='Number of communication rounds T.')
    parser.add_argument('--join_ratio', type=float, default=0.4,
                        help='Fraction of clients participating per round.')
    parser.add_argument('--local_steps', type=int, default=1,
                        help='Local epochs E per selected client (default E=1).')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Local mini-batch size.')

    # ---------------- Optimisation (local & global) ----------------
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use: "cpu" or "cuda".')
    parser.add_argument('--gpu', default='0',
                        help='Comma separated list of GPU(s) to use.')
    parser.add_argument('--num_classes', type=int, default=10)

    parser.add_argument('--local_learning_rate', type=float, default=0.025,
                        help='Client-side SGD learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Client-side SGD momentum.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Client-side SGD weight decay.')

    parser.add_argument('--global_learning_rate', type=float, default=0.025,
                        help='Global model learning rate for refinement.')
    parser.add_argument('--eval_step', default=78, type=int,
                        help='Number of evaluation steps (kept from original code).')

    # ---------------- Non-IID + noisy label configuration ----------------
    parser.add_argument('--iid', type=int, default=0,
                        help='Whether to sample IID data across clients.')
    parser.add_argument('--non_iid_alpha', type=float, default=0.7,
                        help='Dirichlet concentration parameter for non-IID partitioning.')
    # Beta(α, β) for heterogeneous client noise ratios η_k
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Alpha parameter of Beta(α, β) for label noise.')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Beta parameter of Beta(α, β) for label noise.')
    parser.add_argument('--noise_type', type=str, default='symmetric',
                        help='Noise type passed to noisify_label.')
    parser.add_argument('--noise_rate', type=float, default=0.0,
                        help='(Unused) kept for compatibility; per-client noise is drawn from Beta instead.')

    # ---------------- Distillation & contrastive components ----------------
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Temperature for distillation logits.')
    parser.add_argument('--mini_batch_size_distillation', type=int, default=128,
                        help='Mini-batch size for distillation data loader.')
    parser.add_argument('--mc_samples', type=int, default=4,
                        help='Number of Monte Carlo dropout samples for uncertainty.')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Placeholder for legacy code; not used in core ConTaFL logic.')

    # ---------------- Reliability & rehabilitation hyper-parameters ----------------
    parser.add_argument('--reliability_threshold', type=float, default=0.15,
                        help='λ: threshold on hybrid score ˜U_k to mark reliable clients.')
    parser.add_argument('--rehab_threshold', type=float, default=0.7,
                        help='Λ_re: threshold on τ_k for rehabilitation.')
    parser.add_argument('--beta_hybrid', type=float, default=0.5,
                        help='β: trade-off in ˜U_k = β U_pu + (1-β)(1-U_rep).')
    parser.add_argument('--gamma_momentum', type=float, default=0.9,
                        help='γ: EMA momentum for τ_k.')
    parser.add_argument('--rehab_window', type=int, default=5,
                        help='δ: observation window for ΔL_pseudo.')
    parser.add_argument('--rehab_epsilon', type=float, default=1e-3,
                        help='ε: minimum pseudo-loss improvement for rehabilitation.')

    # Loss weights in the unified objective: L_CND + λ1 L_ALT-ND + λ2 L_SSL
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='λ1: weight for negative distillation loss.')
    parser.add_argument('--lambda2', type=float, default=0.0,
                        help='λ2: weight for self-supervised loss (optional).')

    # Other
    parser.add_argument('--seed', type=int, default=5959,
                        help='Random seed for reproducibility.')

    args = parser.parse_args()
    return args
