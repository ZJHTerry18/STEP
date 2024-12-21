### Perception-val data num=19140
NUM_GPUS=8
NUM_SINGLE_GPU=2400

# internvideo-1b_mistral
# CKPT=/mnt/petrelfs/share_data/wangchenting/models/internvideo2_videochat2/internvideo2_1b_mistral7b_stage3
# SAVE=results/perception-val/internvideo2_1b_mistral7b_stage3

# # internvideo-1b_mistral hd
# CKPT=/mnt/petrelfs/share_data/wangchenting/models/internvideo2_videochat2/internvideo2_1b_mistral7b_stage4_hd_post_f16
# SAVE=results/perception-val/internvideo2_1b_mistral7b_stage4_hd_post_f16
# MODEL_CFG=configs/model/i1b_qformer_mistral7b_f8_st2_sh.py

# # internvideo-1b_mistral
# python -u demo/eval_star.py \
#     --ckpt_path logs/scripts/pt/1b_qformer_mistral/stage2_8f_sh.sh_bsl/checkpoint-2116 \
#     --save_path results/star/stage2_bsl

# # internvideo-1b_mistral_glfg-qformer
# CKPT=logs/scripts/pt/1b_qformer_mistral/ablation/stage2_8f_glfg_sh.sh_glfg/checkpoint-last
# SAVE=results/perception-val/stage2_glfg

# # internvideo-1b_mistral_gltc-qformer
# CKPT=logs/scripts/pt/1b_qformer_mistral/ablation/stage2_8f_gltc_sh.sh_gltc/checkpoint-last
# SAVE=results/perception-val/stage2_gltc

# # internvideo-1b_mistral_glfgtc-qformer
# CKPT=logs/scripts/pt/1b_qformer_mistral/stage2_8f_glfgtc_sh.sh_glfgtc/checkpoint-last
# SAVE=results/perception-val/stage2_glfgtc

# # internvideo-1b_mistral_gl-qformer_hd
# CKPT=logs/scripts/pt/1b_qformer_mistral/ablation/stage4_16f_gl_hd.sh_gl_hd/checkpoint-last
# SAVE=results/perception-val/stage4_gl_hd

# # internvideo-1b_mistral_glfg-qformer_hd
# CKPT=logs/scripts/pt/1b_qformer_mistral/ablation/stage4_16f_glfg_hd.sh_glfg_hd/checkpoint-last
# SAVE=results/perception-val/stage4_glfg_hd

# internvideo-1b_mistral_gltc-qformer_hd
CKPT=logs/scripts/pt/1b_qformer_mistral/ablation/stage4_16f_gltc_hd.sh_gltc_hd/checkpoint-last
SAVE=results/perception-val/stage4_gltc_hd

# # internvideo-1b_mistral_glfgtc-qformer_hd
# CKPT=logs/scripts/pt/1b_qformer_mistral/stage4_16f_glfgtc_hd.sh_glfgtc_hd/checkpoint-last
# SAVE=results/perception-val/stage4_glfgtc_hd
# MODEL_CFG=configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py

for IDX in $(seq 0 $((NUM_GPUS-1))); do
    START=$((IDX * NUM_SINGLE_GPU))
    END=$(($((IDX + 1)) * NUM_SINGLE_GPU))
    CUDA_VISIBLE_DEVICES=$IDX python -u demo/eval_perception-val.py \
        --ckpt_path $CKPT \
        --save_path $SAVE \
        --num_f 8 \
        --start ${START} --end ${END} &
done

wait

python demo/merge_res.py $SAVE