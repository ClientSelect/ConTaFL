from torchvision import datasets, transforms

from .utils.dataset import classify_label
from .utils.sampling import client_iid_indices, clients_non_iid_indices


def get_mnist(args):
    """Return MNIST federated splits and a global distillation set."""

    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_local_training = datasets.MNIST(args.data_path, train=True, download=True,
                                         transform=transform_train)
    data_global_test = datasets.MNIST(args.data_path, train=False, download=True,
                                      transform=transform_test)
    data_global_distill = datasets.MNIST(args.data_path, train=True, download=True,
                                         transform=transform_train)

    if args.iid:
        list_client2indices = client_iid_indices(data_local_training, args.num_clients)
    else:
        list_label2indices = classify_label(data_local_training, args.num_classes)
        list_client2indices = clients_non_iid_indices(
            list_label2indices=list_label2indices,
            num_classes=args.num_classes,
            num_clients=args.num_clients,
            alpha=args.non_iid_alpha,
            seed=args.seed,
        )

    return data_local_training, data_global_test, list_client2indices, data_global_distill
