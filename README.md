# Adaptive Modeling Against Adversarial Attacks (Fast FGSM Experiment)

This is the official experiment repo on Fast-FGSM base models for the [paper](https://arxiv.org/abs/2112.12431) "Adaptive Modeling Against Adversarial Attacks".

The main code repo: https://github.com/JokerYan/post_training
The Fast-FGSM Model experiment repo: https://github.com/JokerYan/fast_adversarial
The original Madry base model repo: https://github.com/ylsung/pytorch-adversarial-training

* Please note that the algorithm might be referred as **post training** for easy reference.

## Envrionment Setups
We recommend using Anaconda/Miniconda to setup the envrionment, with the following command:
```bash
conda env create -f pt_env.yml
conda activate post_train
```

## Experiments

### CIFAR-10
#### Base Model
The base model provided by the Madry pytorch implementation author can be found on the readme [here](https://github.com/ylsung/pytorch-adversarial-training/tree/master/cifar-10).

#### Attack algorithm
40-step l infinity PGD without restart, with ϵ = 8/255 and step size α = 3/255

#### Post Training setups
* 50 epochs
* Batch Size 128
* SGD optimizer of 0.001, momentum of 0.9

#### Results
| Model | Robust Accuracy | Natural Accuracy |
| ----- | --------------- | ---------------- |
| Madry | 0.4740 | 0.8729 |
| Madry + Post Train (Fast) | **0.5490** | 0.8750 |
| Madry + Post Train (Fix Adv) | 0.5317 | 0.8577 |

#### How to Run
Please change the directory to `cifar10` before the experiment.
You can refer to the `bash` folder for various examples of bash files that runs the experiments. 
The experiment results will be updated in the respective log file in the `logs` folder.

Here is an example of experiment bash command:
```bash
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
CUDA_VISIBLE_DEVICES=0 python main.py \
  --todo test \
  --data_root ../../data/ \
  --batch_size 1 \
  -e 0.0314 \
  -a 0.00784 \
  -p 'linf' \
  --load_checkpoint 7step_2alpha_0.1lr.pth \
  --pt-data ori_neigh \
  --pt-method adv \
  --adv-dir na \
  --neigh-method untargeted \
  --pt-iter 50 \
  --pt-lr 0.001 \
  --att-iter 20 \
  --att-restart 1 \
  --log-file logs/log_exp01_${TIMESTAMP}.txt
```

### MNIST
#### Base Model
The base model is not provided by the original author of the Madry Pytorch implementation.

#### Attack Algorithm
40-step l infinity PGD without restart, with ϵ = 0.3 and step size α = 0.01, unless otherwise specified

#### Post Training setups
* 50 epochs
* Batch Size 64
* SGD optimizer of 0.001, momentum of 0.9

#### Results
| Model | Robust Accuracy | Natural Accuracy |
| ----- | --------------- | ---------------- |
| Madry | 0.9536 | 0.9903 |
| Madry + Post Train (Fast) | 0.9430 | 0.9913 |
| Madry + Post Train (Fix Adv) | 0.9433 | 0.9919 |
| Madry + Post Train (PGD) | **0.9618** | 0.9874 |

#### How to Run
Please change the directory to `MNIST` before the experiment.
You can refer to the `bash` folder for various examples of bash files that runs the experiments. 
The experiment results will be updated in the respective log file in the `logs` folder.

Here is an example of experiment bash command:
```bash
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
CUDA_VISIBLE_DEVICES=1 python main.py \
  --todo test \
  --data_root ../../data/ \
  --batch_size 1 \
  -e 0.3 \
  -a 0.01 \
  -p 'linf' \
  --load_checkpoint checkpoint/mnist_/checkpoint_56000.pth \
  --pt-data ori_neigh \
  --pt-method adv \
  --adv-dir na \
  --neigh-method untargeted \
  --pt-iter 50 \
  --pt-lr 0.001 \
  --att-iter 40 \
  --att-restart 1 \
  --log-file logs/log_exp01_${TIMESTAMP}.txt
```

### ImageNet
The code for ImageNet is not implemented yet.
  
### Arguments
Some bash arguments are shared between CIFAR-10 and MNIST experiments.

The arguement description and accepted values are listed here:
* pt-data: post training data composition
  - ori_rand: 50% original class + 50% random class
  - ori_neigh: 50% original calss + 50% neighbour class
  - train: random training data
* pt-method: post training method
  - adv: fast adversarial training used in Fast FGSM
  - dir_adv: fixed adversarial training proposed in paper
  - normal: normal training instead of adversarial training
  - pgd: only implemented for Madry MNIST model. use PGD as adversarial training for post training
* adv-dir: direction of fixed adversarial training
  - na: not applicable, used for adv and normal pt-method
  - pos: positive direction, data + fix perturbation
  - neg: negative direction, data - fix perturbation
  - both: default for dir_adv, random mixture of positive and negative direction
* neigh-method: attack method to find the neighbour
  - untargeted: use untargeted attack
  - targeted: use targeted attack and choose the highest confidence class
* pt-iter: post training iteration
* pt-lr: post training learning rate
* att-iter: attack iteration used for attack and post adversarial training
* att-restart: attack restart used for attack and post adversarial training
* log-file: log file stored path
