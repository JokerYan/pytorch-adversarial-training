import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision as tv

from time import time

from visualize import visualize_grad, visualize_cam, visualize_delta
from src.model.madry_model import WideResNet
from src.attack import FastGradientSignUntargeted
from src.utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model

from src.argument import parser, print_args
from post_utils import get_train_loaders_by_class, post_train
from blackbox_dataset import BlackboxDataset


class Trainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack

    def standard_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, False)

    def adversarial_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, True)

    def train(self, model, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger

        opt = torch.optim.SGD(model.parameters(), args.learning_rate,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                                                         milestones=[40000, 60000],
                                                         gamma=0.1)
        _iter = 0

        begin_time = time()

        for epoch in range(1, args.max_epoch + 1):
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                if adv_train:
                    # When training, the adversarial example is created from a random
                    # close point to the original data point. If in evaluation mode,
                    # just start from the original data point.
                    adv_data = self.attack.perturb(data, label, 'mean', True)
                    output = model(adv_data, _eval=False)
                else:
                    output = model(data, _eval=False)

                loss = F.cross_entropy(output, label)

                opt.zero_grad()
                loss.backward()
                opt.step()

                if _iter % args.n_eval_step == 0:
                    t1 = time()

                    if adv_train:
                        with torch.no_grad():
                            stand_output = model(data, _eval=True)
                        pred = torch.max(stand_output, dim=1)[1]

                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    else:

                        adv_data = self.attack.perturb(data, label, 'mean', False)

                        with torch.no_grad():
                            adv_output = model(adv_data, _eval=True)
                        pred = torch.max(adv_output, dim=1)[1]
                        # print(label)
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    t2 = time()

                    logger.info(f'epoch: {epoch}, iter: {_iter}, lr={opt.param_groups[0]["lr"]}, '
                                f'spent {time() - begin_time:.2f} s, tr_loss: {loss.item():.3f}')

                    logger.info(f'standard acc: {std_acc:.3f}%, robustness acc: {adv_acc:.3f}%')

                    # begin_time = time()

                    # if va_loader is not None:
                    #     va_acc, va_adv_acc = self.test(model, va_loader, True)
                    #     va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                    #     logger.info('\n' + '='*30 + ' evaluation ' + '='*30)
                    #     logger.info('test acc: %.3f %%, test adv acc: %.3f %%, spent: %.3f' % (
                    #         va_acc, va_adv_acc, time() - begin_time))
                    #     logger.info('='*28 + ' end of evaluation ' + '='*28 + '\n')

                    begin_time = time()

                if _iter % args.n_store_image_step == 0:
                    tv.utils.save_image(torch.cat([data.cpu(), adv_data.cpu()], dim=0),
                                        os.path.join(args.log_folder, f'images_{_iter}.jpg'),
                                        nrow=16)

                if _iter % args.n_checkpoint_step == 0:
                    file_name = os.path.join(args.model_folder, f'checkpoint_{_iter}.pth')
                    save_model(model, file_name)

                _iter += 1
                # scheduler depends on training interation
                scheduler.step()

            if va_loader is not None:
                t1 = time()
                va_acc, va_adv_acc = self.test(model, va_loader, True, False)
                va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                t2 = time()
                logger.info('\n' + '=' * 20 + f' evaluation at epoch: {epoch} iteration: {_iter} ' \
                            + '=' * 20)
                logger.info(f'test acc: {va_acc:.3f}%, test adv acc: {va_adv_acc:.3f}%, spent: {t2 - t1:.3f} s')
                logger.info('=' * 28 + ' end of evaluation ' + '=' * 28 + '\n')

    def test(self, args, model, train_loader, test_loader, adv_test=False, use_pseudo_label=False):
        # adv_test is False, return adv_acc as -1

        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0
        total_adv_post_acc = 0.0
        total_neighbour_acc = 0.0
        total_natural_acc = 0.0
        total_natural_post_acc = 0.0
        pgd_blackbox_success_list = []

        train_loaders_by_class = get_train_loaders_by_class(args.data_root, 128)

        with torch.no_grad():
            for i, (data, label) in enumerate(test_loader):
                self.logger.info('')
                data, label = tensor2cuda(data), tensor2cuda(label)

                num += data.shape[0]

                if adv_test:
                    with torch.enable_grad():
                        if not args.blackbox:
                            # pseudo_label: use predicted label as target label
                            adv_data = self.attack.perturb(data, pred if use_pseudo_label else label,
                                                           'mean', False)
                        else:
                            adv_data = data  # already attacked

                    # evaluate base model against adv
                    adv_output = model(adv_data, _eval=True)
                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                    self.logger.info('Batch: {}\tbase adv acc: {:.4f}'.format(num, total_adv_acc / num))

                    if args.blackbox:  #
                        if (torch.argmax(adv_output) != label).sum().item():  # attack successful
                            pgd_blackbox_success_list.append(str(i))
                        if i % 1000 == 0:
                            with open('./logs/log_exp_blackbox_index.txt', 'w+') as f:
                                f.write('\n'.join(pgd_blackbox_success_list))

                    # evaluate post model against adv
                    post_model, original_class, neighbour_class, loss_list, acc_list, neighbour_delta = \
                        post_train(model, adv_data, self.attack, train_loader, train_loaders_by_class, self.logger, args)
                    post_output = post_model(adv_data, _eval=True)
                    post_pred = torch.max(post_output, dim=1)[1]
                    post_acc = evaluate(post_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_post_acc += post_acc
                    self.logger.info('Batch: {}\tpost adv acc: {:.4f}'.format(num, total_adv_post_acc / num))

                    total_neighbour_acc += 1 if int(label) == int(original_class) or int(label) == int(neighbour_class) else 0
                    self.logger.info('Batch: {}\tneighbour acc: {:.4f}'.format(num, total_neighbour_acc / num))

                    # evaluate base model against natural
                    output = model(data, _eval=True)
                    pred = torch.max(output, dim=1)[1]
                    natural_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_natural_acc += natural_acc
                    self.logger.info('Batch: {}\tbase natural acc: {:.4f}'.format(num, total_natural_acc / num))

                    # evaluate post model against natural
                    post_model, original_class, neighbour_class, loss_list, acc_list, neighbour_delta = \
                        post_train(model, data, self.attack, train_loader, train_loaders_by_class, self.logger, args)
                    post_normal_output = post_model(data, _eval=True)
                    post_normal_pred = torch.max(post_normal_output, dim=1)[1]
                    post_normal_acc = evaluate(post_normal_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_natural_post_acc += post_normal_acc
                    self.logger.info('Batch: {}\tpost natural acc: {:.4f}'.format(num, total_natural_post_acc / num))
                else:
                    total_adv_acc = -num

        return total_natural_acc / num, total_adv_acc / num


def main(args):
    save_folder = '%s_%s' % (args.dataset, args.affix)

    # log_folder = os.path.join(args.log_root, save_folder)
    log_folder = os.path.join(args.log_root, '')
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    # logger = create_logger(log_folder, args.todo, 'info')
    logger = create_logger(args.log_file, args.todo, 'info')

    print_args(args, logger)

    model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)

    attack = FastGradientSignUntargeted(model,
                                        args.epsilon,
                                        args.alpha,
                                        min_val=0,
                                        max_val=1,
                                        max_iters=args.k,
                                        _type=args.perturbation_type)

    if torch.cuda.is_available():
        model.cuda()

    trainer = Trainer(args, logger, attack)

    if args.todo == 'train':
        transform_train = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
        ])
        tr_dataset = tv.datasets.CIFAR10(args.data_root,
                                         train=True,
                                         transform=transform_train,
                                         download=True)

        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # evaluation during training
        te_dataset = tv.datasets.CIFAR10(args.data_root,
                                         train=False,
                                         transform=tv.transforms.ToTensor(),
                                         download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        trainer.train(model, tr_loader, te_loader, args.adv_train)
    elif args.todo == 'test':
        transform_train = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4, fill=0, padding_mode='constant'),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
        ])
        tr_dataset = tv.datasets.CIFAR10(args.data_root,
                                         train=True,
                                         transform=transform_train,
                                         download=True)

        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        if not args.blackbox:
            te_dataset = tv.datasets.CIFAR10(args.data_root,
                                             train=False,
                                             transform=tv.transforms.ToTensor(),
                                             download=True)
        else:
            te_dataset = BlackboxDataset("../../data/cifar10_adv_fast.pickle")

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint)

        std_acc, adv_acc = trainer.test(args, model, tr_loader, te_loader, adv_test=True, use_pseudo_label=False)

        print(f"std acc: {std_acc * 100:.3f}%, adv_acc: {adv_acc * 100:.3f}%")

    else:
        raise NotImplementedError


if __name__ == '__main__':
    args = parser()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)