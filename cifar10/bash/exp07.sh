TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
CUDA_VISIBLE_DEVICES=0 python main.py \
  --todo test \
  --data_root ../../data/ \
  --batch_size 1 \
  -e 0.0314 \
  -a 0.00784 \
  -p 'linf' \
  --load_checkpoint 7step_2alpha_0.1lr.pth \
  --pt-data train \
  --pt-method adv \
  --adv-dir na \
  --neigh-method untargeted \
  --pt-iter 50 \
  --pt-lr 0.001 \
  --att-iter 20 \
  --att-restart 1 \
  --log-file logs/log_exp07_${TIMESTAMP}.txt
