import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datetime import datetime
import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from server import Server
from dataset.get_cifar10 import get_cifar10
from dataset.get_cifar100 import get_cifar100
from dataset.get_mnist import get_mnist
from dataset.utils.dataset import Indices2Dataset
from dataset.utils.noisify import noisify_label
from models.model_feature import ResNet_cifar_feature
from models.model_mnist import MnistCNN
from utils.tools import get_set_gpus
from options import args_parser


def get_train_label(dataset, index_list):
    labels = []
    for idx in index_list:
        labels.append(dataset[idx][1])
    return labels


def label_rate(true_labels, noisy_labels):
    assert len(true_labels) == len(noisy_labels)
    matches = sum(int(t == n) for t, n in zip(true_labels, noisy_labels))
    rate = matches / len(true_labels) if true_labels else 0.0
    print(f'Clean label rate: {rate:.4f}')


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def inject_heterogeneous_noise(args, data_local_training, list_client2indices):
    train_data_list = []
    per_client_labels = []

    alpha = args.alpha
    beta = args.beta
    beta_samples = np.random.beta(alpha, beta, size=args.num_clients)
    noise_rate_list = np.sort(beta_samples)
    args.noise_rate_list = noise_rate_list

    for client_id in range(args.num_clients):
        indices = list_client2indices[client_id]
        clean_labels = get_train_label(data_local_training, indices)
        noisy_labels = copy.deepcopy(clean_labels)

        local_classes = sorted(list(set(clean_labels)))

        eta_k = noise_rate_list[client_id]
        num_noisy = int(len(indices) * eta_k)

        for idx_pos in range(num_noisy):
            true_label = clean_labels[idx_pos]
            candidates = [c for c in local_classes if c != true_label]
            if not candidates:
                continue
            noisy_label = random.choice(candidates)
            noisy_labels[idx_pos] = noisy_label

        label_rate(clean_labels, noisy_labels)

        indices2data = Indices2Dataset(data_local_training)
        indices2data.load(indices, noisy_labels)
        train_data_list.append(indices2data)
        per_client_labels.append(noisy_labels)

    return train_data_list, per_client_labels


def build_backbone(args):
    if args.dataset == 'cifar10':
        model = ResNet_cifar_feature(
            resnet_size=20,
            scaling=1,
            save_activations=False,
            group_norm_num_groups=None,
            freeze_bn=False,
            freeze_bn_affine=False,
            num_classes=args.num_classes,
        )
    elif args.dataset == 'cifar100':
        from torchvision.models import resnet34
        model = resnet34(weights=None, num_classes=args.num_classes)
    elif args.dataset == 'mnist':
        model = MnistCNN(num_classes=args.num_classes)
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    return model


def main(args):
    prev_time = datetime.now()

    if args.gpu:
        gpus = get_set_gpus(args.gpu)
        print(f'Using GPU(s): {gpus}')

    if args.dataset == 'cifar10':
        args.num_classes = 10
        data_local_training, data_global_test, list_client2indices, global_distill_dataset = get_cifar10(args)
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        data_local_training, data_global_test, list_client2indices, global_distill_dataset = get_cifar100(args)
    elif args.dataset == 'mnist':
        args.num_classes = 10
        data_local_training, data_global_test, list_client2indices, global_distill_dataset = get_mnist(args)
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

    train_data_list, label_list = inject_heterogeneous_noise(args, data_local_training, list_client2indices)

    for client_id, labels in enumerate(label_list):
        hist = [0 for _ in range(args.num_classes)]
        for y in labels:
            if 0 <= y < args.num_classes:
                hist[y] += 1
        print(f'Client {client_id}: class histogram (first 10) {hist[:10]}')

    model = build_backbone(args)

    server = Server(
        args=args,
        train_data_list=train_data_list,
        global_test_dataset=data_global_test,
        global_distill_dataset=global_distill_dataset,
        global_student=model,
        temperature=args.temperature,
        mini_batch_size_distillation=args.mini_batch_size_distillation,
        lamda=args.lambda1,
    )

    server.train()

    acc = sorted(server.test_acc)
    if len(acc) > 10:
        acc = acc[int(0.9 * len(acc)):]
    final_acc = sum(acc) / len(acc) if acc else 0.0
    print(f'train finished---> final_acc={final_acc}')

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    print('train time %02d:%02d:%02d' % (h, m, s))


if __name__ == '__main__':
    args = args_parser()
    set_seed(args.seed)
    main(args)
