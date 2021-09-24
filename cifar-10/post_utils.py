import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import Subset


def get_train_loaders_by_class(dir, batch_size):
    train_transform = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
    ])
    train_dataset = tv.datasets.CIFAR10(
        dir, train=True, transform=train_transform, download=True
    )
    indices_list = [[] for _ in range(10)]
    for i in range(len(train_dataset)):
        label = int(train_dataset[i][1])
        indices_list[label].append(i)
    dataset_list = [Subset(train_dataset, indices) for indices in indices_list]
    train_loader_list = [
        torch.utils.data.DataLoader(

        )
    ]