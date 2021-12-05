TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
CUDA_VISIBLE_DEVICES=3 python main.py \
  --todo test \
  --data_root ../../data/ \
  --batch_size 1 \
  -e 0.3 \
  -a 0.01 \
  -p 'linf' \
  --load_checkpoint checkpoint/mnist_/checkpoint_56000.pth \
  --pt-data ori_neigh \
  --pt-method dir_adv \
  --adv-dir neg \
  --neigh-method untargeted \
  --pt-iter 50 \
  --pt-lr 0.001 \
  --att-iter 40 \
  --att-restart 1 \
  --log-file logs/log_exp03_${TIMESTAMP}.txt
