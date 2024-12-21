export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG="INFO"
export NCCL_IB_HCA=mlx5_2,mlx5_3

NNODE=1
NUM_GPUS=8
NUM_CPU=128
JOB_NAME=$(basename $0)_gltc_hd
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
QUOTA=reserved

srun -p video5 \
    --job-name=${JOB_NAME} \
    --ntasks=8 \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --quotatype=${QUOTA} \
    --kill-on-bad-exit=1 \
    python \
    -u src/train/train_or_eval.py \
    --mode 'it' \
    --model_path 'logs/scripts/pt/1b_qformer_mistral/ablation/stage2_16f_gltc_hd.sh_gltc_hd/checkpoint-last' \
    --model configs/model/i1b_qformer_mistral7b_f8_hd.py \
    --tokenizer configs/tokenizer/mistral_tokenizer_q96.yaml \
    --train_data configs/dataset/data_config/post_it_f16.py \
    --output_dir logs/${OUTPUT_DIR} \
    --deepspeed configs/deepspeed/stage2_bf16.json \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20000 \
    --num_train_epochs 1 \
    --min_lr_ratio 0.05 \
    --learning_rate 1e-5 \
    --weight_decay 0.02 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --dataloader_num_workers 8 \
    --logging_steps 1 \
    --log_level 'info' \
    --logging_nan_inf_filter "no" \
    --bf16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --adam_epsilon 1e-5 \
    --ignore_data_skip