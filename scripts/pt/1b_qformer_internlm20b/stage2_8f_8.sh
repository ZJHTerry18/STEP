export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG="INFO"
export NCCL_IB_HCA=mlx5_2,mlx5_3

NNODE=4
NUM_GPUS=8
NUM_CPU=128
JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"

srun -p video5 \
    --job-name=${JOB_NAME} \
    --ntasks=8 \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    python \
    -u src/train/train_or_eval.py \
    --mode 'it' \
    --model configs/model/i1b_qformer_internlm20b_f8_clean.py \
    --tokenizer configs/tokenizer/internlm20b_tokenizer_q96.yaml \
    --train_data configs/dataset/data_config/it_baseline_f8.py \
    --output_dir logs/${OUTPUT_DIR} \
    --deepspeed configs/deepspeed/mystage3.json \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --num_train_epochs 3 \
    --min_lr_ratio 0.05 \
    --learning_rate 3e-5 \
    --weight_decay 0.02 \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --report_to "none" \
    --dataloader_num_workers 2 \
    --logging_steps 1 \
    --log_level 'info' \
    --logging_nan_inf_filter "no" \
    --bf16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --adam_epsilon 1e-5 \
    --ignore_data_skip