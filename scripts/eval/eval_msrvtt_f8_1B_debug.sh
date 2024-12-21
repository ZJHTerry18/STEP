export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG="INFO"
export NCCL_IB_HCA=mlx5_2,mlx5_3

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"

 
srun -p video5 \
    --job-name=${JOB_NAME} \
    --ntasks=1 \
    --gres=gpu:1 \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    python \
    -u src/train/train_or_eval.py \
    --is_eval True \
    --model configs/model/internvideo2-1B_qformer_vicuna7b_f8.py \
    --tokenizer configs/tokenizer/multimodal_llama_tokenizer_q96.yaml \
    --train_data configs/dataset/data_config/debug_f8.py \
    --eval_data configs/dataset/data_config/debug_f8.py \
    --pretrained_path /mnt/petrelfs/lixinhao/lxh_exp/multimodalllm/log/internvideo2_unfreezed_vicuna-7b-fix_qformer-8000-further/checkpoint-last/pytorch_model.bin \
    --output_dir logs/${OUTPUT_DIR} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --max_steps 5000 \
    --min_lr_ratio 0.05 \
    --learning_rate 1.5e-4 \
    --weight_decay 5e-2 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --report_to "none" \
    --dataloader_num_workers 2 \
    --logging_steps 1 \
    --log_level "info" \
    --logging_nan_inf_filter "no" \
    --bf16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --adam_epsilon 1e-5 \
    --ignore_data_skip \
    --bf16_full_eval