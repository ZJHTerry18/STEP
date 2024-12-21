export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG="INFO"
export NCCL_IB_HCA=mlx5_2,mlx5_3


JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"

srun -p video5 \
    --preempt \
    -w SH-IDCA1404-10-140-54-[21,28,18,4] \
    --job-name=${JOB_NAME} \
    --ntasks=32 \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    python -u src/train/train_or_eval.py \
    --model configs/model/umt-L_qformer_vicuna7b_f4.py \
    --pretrained_path /mnt/petrelfs/share_data/lixinhao/pt_umt-L_qformer_vicuna7b_f4_5m_smit_no_prompt_new.sh_20240509_010748/checkpoint-last/pytorch_model.bin \
    --tokenizer configs/tokenizer/multimodal_llama_tokenizer_q96.yaml \
    --train_data configs/dataset/data_config/it_baseline.py \
    --output_dir logs/${OUTPUT_DIR} \
    --deepspeed configs/deepspeed/stage2_bf16.json \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --num_train_epochs 1 \
    --min_lr_ratio 0.05 \
    --learning_rate 1e-4 \
    --weight_decay 0.02 \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --report_to "wandb" \
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