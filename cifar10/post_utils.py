import copy

import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
from torch.utils.data import Subset
import apex.amp as amp


mu = torch.zeros([3, 1, 1]).cuda()
std = torch.ones([3, 1, 1]).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


def cal_accuracy(outputs, labels):
    _, predictions = torch.max(outputs, 1)
    correct = 0
    total = 0
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct += 1
        total += 1
    return correct / total


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


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
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0
        ) for dataset in dataset_list
    ]
    return train_loader_list


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None, random_start=True):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if random_start:
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def post_train(model, images, model_attack, train_loaders_by_class, args):
    alpha = (10 / 255) / std
    epsilon = (8 / 255) / std
    loss_func = nn.CrossEntropyLoss()
    device = torch.device('cuda')
    model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(lr=0.01,
                                params=model.parameters(),
                                momentum=0.9,
                                nesterov=True)
    images = images.detach()
    with torch.enable_grad():
        original_output = model(images, _eval=True)
        original_class = torch.argmax(original_output).reshape(1)

        # neighbour_delta = attack_pgd(model, images, original_class, epsilon, alpha, attack_iters=20,
        #                              restarts=1, random_start=False).detach()
        neighbour_delta = model_attack.perturb(images, original_class, 'mean', False) - images
        neighbour_images = neighbour_delta + images
        neighbour_output = model(neighbour_images, _eval=True)
        neighbour_class = torch.argmax(neighbour_output).reshape(1)

        if original_class == neighbour_class:
            print('original class == neighbour class')
            return model, original_class, neighbour_class, None, None, neighbour_delta

        loss_list = []
        acc_list = []

        for _ in range(20):
            original_data, original_label = next(iter(train_loaders_by_class[original_class]))
            neighbour_data, neighbour_label = next(iter(train_loaders_by_class[neighbour_class]))

            data = torch.vstack([original_data, neighbour_data]).to(device)
            label = torch.hstack([original_label, neighbour_label]).to(device)

            # generate fgsm adv examplesp
            delta = (torch.rand_like(data) * 2 - 1) * epsilon  # uniform rand from [-eps, eps]
            noise_input = data + delta
            noise_input.requires_grad = True
            noise_output = model(noise_input)
            loss = loss_func(noise_output, label)  # loss to be maximized
            # loss = target_bce_loss_func(noise_output, label, original_class, neighbour_class)  # bce loss to be maximized
            input_grad = torch.autograd.grad(loss, noise_input)[0]
            delta = delta + alpha * torch.sign(input_grad)
            delta.clamp_(-epsilon, epsilon)
            adv_input = data + delta

            adv_output = model(adv_input.detach())

            loss = loss_func(adv_output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            defense_acc = cal_accuracy(adv_output, label)
            loss_list.append(loss)
            acc_list.append(defense_acc)
    return model, original_class, neighbour_class, loss_list, acc_list, neighbour_delta