#!/bin/bash
# Example run script for MNIST with ConTaFL

python main.py \
  --dataset mnist \
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
  --non_iid_alpha 0.7 \
  --reliability_threshold 0.15 \
  --rehab_threshold 0.7 \
  --beta_hybrid 0.5 \
  --gamma_momentum 0.9 \
  --lambda1 1.0 \
  --lambda2 0.0
