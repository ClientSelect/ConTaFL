# ConTaFL: Contrastive Tail-Aware Federated Learning with Reliability-Aware Aggregation

This repository provides an implementation of the ConTaFL framework.

ConTaFL is designed for federated learning with heterogeneous and extremely noisy clients under non-IID data. It combines:
- contrastive representation refinement for rare and tail classes,
- hybrid reliability estimation based on prediction uncertainty and representation divergence,
- adaptive negative distillation for unreliable clients,
- reliability-aware aggregation of client updates.

The implementation follows the experimental protocol described in the paper, including:
- Datasets: CIFAR-10, CIFAR-100, and MNIST;
- Non-IID partitioning via a Dirichlet distribution over client label proportions;
- Heterogeneous client-specific noise ratios drawn from a Beta(α, β) distribution;
- 20 clients, 100 communication rounds, and 40% participation per round by default;
- SGD with learning rate 0.025, momentum 0.9, and weight decay 5×10⁻⁴ for local training.

## 1. Environment

- Python ≥ 3.7  
- PyTorch ≥ 1.7  
- torchvision ≥ 0.8  

Install the main dependencies:

```bash
pip install torch torchvision scikit-learn
```

## 2. Datasets

By default, datasets are downloaded to `./data`:

- CIFAR-10 and CIFAR-100 via `torchvision.datasets.CIFAR10/100`
- MNIST via `torchvision.datasets.MNIST`

Dataset and backbone selection:

- `--dataset cifar10`  → ResNet-20 backbone (`ResNet_cifar_feature`)
- `--dataset cifar100` → ResNet-34 backbone
- `--dataset mnist`    → lightweight CNN backbone

## 3. Federated setup and noise model

1. **Non-IID partitioning.** Each dataset is partitioned across `K` clients using a Dirichlet distribution with concentration parameter `--non_iid_alpha`.

2. **Client-wise label noise.** For each client k, a noise ratio η_k is sampled from a Beta distribution:

   η_k ~ Beta(α, β)

with parameters `--alpha` and `--beta`. A fraction η_k of local labels is randomly relabelled within the client’s local label set (excluding the true label).

3. **Auxiliary unlabeled pool.** An auxiliary unlabeled dataset D_U is used for contrastive refinement and distillation. For CIFAR-10 and CIFAR-100 it is drawn from their training split; for MNIST it is the MNIST training split.

## 4. Running ConTaFL

### CIFAR-10

```bash
python main.py \
  --dataset cifar10 \
  --data_path ./data \
  --num_clients 20 \
  --global_rounds 100 \
  --join_ratio 0.4 \
  --batch_size 64 \
  --local_learning_rate 0.025 \
  --momentum 0.9 \
  --weight_decay 5e-4 \
  --alpha 0.1 \
  --beta 0.1 \
  --non_iid_alpha 0.7
```

### CIFAR-100

```bash
python main.py \
  --dataset cifar100 \
  --data_path ./data \
  --num_clients 20 \
  --global_rounds 100 \
  --join_ratio 0.4 \
  --batch_size 64
```

### MNIST

```bash
python main.py \
  --dataset mnist \
  --data_path ./data \
  --num_clients 20 \
  --global_rounds 100 \
  --join_ratio 0.4 \
  --batch_size 64
```

## 5. Key hyper-parameters

- `--reliability_threshold` (λ): threshold on the hybrid unreliability score ˜U_k to mark clients as reliable.
- `--rehab_threshold` (Λ_re): rehabilitation threshold for the moving reliability score τ_k.
- `--beta_hybrid` (β): trade-off between prediction uncertainty and representation divergence.
- `--gamma_momentum` (γ): EMA momentum for updating τ_k.
- `--lambda1` (λ₁): weight for the negative distillation loss.
- `--lambda2` (λ₂): optional weight for self-supervised loss (placeholder in the current code).

The random seed is controlled by `--seed` and is used for data partitioning, label noise generation, and PyTorch initialisation.

