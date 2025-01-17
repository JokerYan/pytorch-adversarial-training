import argparse

def parser():
    parser = argparse.ArgumentParser(description='Video Summarization')
    parser.add_argument('--todo', choices=['train', 'valid', 'test', 'visualize'], default='train',
        help='what behavior want to do: train | valid | test | visualize')
    parser.add_argument('--dataset', default='mnist', help='use what dataset')
    parser.add_argument('--data_root', default='/home/yilin/Data', 
        help='the directory to save the dataset')
    parser.add_argument('--log_root', default='log', 
        help='the directory to save the logs or other imformations (e.g. images)')
    parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
    parser.add_argument('--load_checkpoint', default='./model/default/model.pth')
    parser.add_argument('--affix', default='', help='the affix for the save folder')

    # parameters for generating adversarial examples
    parser.add_argument('--epsilon', '-e', type=float, default=0.3, 
        help='maximum perturbation of adversaries')
    parser.add_argument('--alpha', '-a', type=float, default=0.01, 
        help='movement multiplier per iteration when generating adversarial examples')
    # parser.add_argument('--k', '-k', type=int, default=40,
    #     help='maximum iteration when generating adversarial examples')

    parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size')
    parser.add_argument('--max_epoch', '-m_e', type=int, default=60, 
        help='the maximum numbers of the model see a sample')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')
    parser.add_argument('--n_eval_step', type=int, default=100, 
        help='number of iteration per one evaluation')
    parser.add_argument('--n_checkpoint_step', type=int, default=2000, 
        help='number of iteration to save a checkpoint')
    parser.add_argument('--n_store_image_step', type=int, default=2000, 
        help='number of iteration to save adversaries')
    parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf', 
        help='the type of the perturbation (linf or l2)')
    
    parser.add_argument('--adv_train', action='store_true')

    # post train related
    parser.add_argument('--pt-data', default='ori_neigh', choices=['ori_rand', 'ori_neigh', 'train'], type=str)
    parser.add_argument('--pt-method', default='adv', choices=['adv', 'dir_adv', 'normal', 'pgd'], type=str)
    parser.add_argument('--adv-dir', default='na', choices=['na', 'pos', 'neg', 'both'], type=str)
    parser.add_argument('--neigh-method', default='untargeted', choices=['untargeted', 'targeted'], type=str)
    parser.add_argument('--pt-iter', default=50, type=int)
    parser.add_argument('--pt-lr', default=0.001, type=float)
    parser.add_argument('--att-iter', default=40, type=int)
    parser.add_argument('--att-restart', default=1, type=int)
    parser.set_defaults(blackbox=False, type=bool)
    parser.add_argument('--blackbox', dest='blackbox', action='store_true')
    parser.add_argument('--log-file', default='logs/default.log', type=str)

    args = parser.parse_args()
    setattr(args, 'k', args.att_iter)
    if args.adv_dir != 'na':
        assert args.pt_method == 'dir_adv'
    if args.pt_method == 'dir_adv':
        assert args.adv_dir != 'na'
    return args


def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))