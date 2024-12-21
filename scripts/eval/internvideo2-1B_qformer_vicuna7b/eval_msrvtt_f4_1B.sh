export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG="INFO"
export NCCL_IB_HCA=mlx5_2,mlx5_3

NNODE=1
NUM_GPUS=1
NUM_CPU=8
JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"

 
srun -p video5 \
    --job-name=${JOB_NAME} \
    -n${NNODE} \
    --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=1 \
    --quotatype=auto \
    --cpus-per-task=${NUM_CPU} \
    --kill-on-bad-exit=1 \
    python \
    -u src/train/train_or_eval.py \
    --is_eval True \
    --model configs/model/internvideo2-1B_qformer_vicuna7b_f4.py \
    --tokenizer configs/tokenizer/multimodal_llama_tokenizer_q96.yaml \
    --train_data configs/dataset/data_config/cc3m_webvid2m_f4.py \
    --eval_data configs/dataset/data_config/cc3m_webvid2m_f4.py \
    --pretrained_path /mnt/petrelfs/lixinhao/lxh_exp/multimodalllm-lxh/logs/scripts/pt/pt_internvideo2-1B_qformer_vicuna7b_f4_5m.sh_20240428_022050/checkpoint-5000/global_step5000/mp_rank_00_model_states.pt \
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
    --per_device_train_batch_size 128 \
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