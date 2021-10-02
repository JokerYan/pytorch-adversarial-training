import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision as tv

from time import time
from src.model.madry_model import WideResNet
from src.attack import FastGradientSignUntargeted
from src.utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model

from src.argument import parser, print_args
from post_utils import get_train_loaders_by_class, post_train


class Trainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack

    def test(self, args, model, loader, adv_test=False, use_pseudo_label=False):
        # adv_test is False, return adv_acc as -1

        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0

        adv_data_list = []
        label_list = []
        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                output = model(data, _eval=True)

                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')

                total_acc += te_acc
                num += output.shape[0]

                if adv_test:
                    # use predicted label as target label
                    with torch.enable_grad():
                        adv_data = self.attack.perturb(data, pred if use_pseudo_label else label,
                                                       'mean', False)

                    adv_data_list.append(adv_data)
                    label_list.append(label)
                    adv_output = model(adv_data, _eval=True)

                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                else:
                    total_adv_acc = -num
        adv_data_concat = torch.cat(adv_data_list, 0)
        label_concat = torch.cat(label_list, 0)
        print(adv_data_concat.shape)
        print(label_concat.shape)

        return total_acc / num, total_adv_acc / num


def main(args):
    save_folder = '%s_%s' % (args.dataset, args.affix)

    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')

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

    te_dataset = tv.datasets.CIFAR10(args.data_root,
                                     train=False,
                                     transform=tv.transforms.ToTensor(),
                                     download=True)

    te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    checkpoint = torch.load(args.load_checkpoint)
    model.load_state_dict(checkpoint)

    std_acc, adv_acc = trainer.test(args, model, te_loader, adv_test=True, use_pseudo_label=False)

    print(f"std acc: {std_acc * 100:.3f}%, adv_acc: {adv_acc * 100:.3f}%")


if __name__ == '__main__':
    args = parser()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    main(args)