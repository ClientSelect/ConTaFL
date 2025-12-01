# ConTaFL Example Runs

This directory contains example shell scripts that reproduce the main experimental settings from the ConTaFL paper.

- `run_cifar10.sh`  – CIFAR-10 with 20 clients, 100 rounds, 40% participation, Beta(0.1, 0.1) noise.
- `run_cifar100.sh` – CIFAR-100 with the same federated configuration.
- `run_mnist.sh`    – MNIST with the same federated configuration.

To run an experiment from the repository root:

```bash
cd ConTaFL
bash examples/run_cifar10.sh
```

You can adapt these scripts by changing hyper-parameters such as `--alpha`, `--beta`, `--non_iid_alpha`, or the reliability thresholds.
