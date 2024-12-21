### GroundVQA test data num=500
NUM_GPUS=8
NUM_SINGLE_GPU=70

# # internvideo-1b_mistral
# CKPT=/mnt/petrelfs/share_data/wangchenting/models/internvideo2_videochat2/internvideo2_1b_mistral7b_stage3
# SAVE=results/groundvqa/internvideo2_1b_mistral7b_stage3
# MODEL_CFG=configs/model/i1b_qformer_mistral7b_f8_st2_sh.py

# # internvideo-1b_mistral hd
# CKPT=/mnt/petrelfs/share_data/wangchenting/models/internvideo2_videochat2/internvideo2_1b_mistral7b_stage4_hd_post_f16
# SAVE=results/groundvqa/internvideo2_1b_mistral7b_stage4_hd_post_f16
# MODEL_CFG=configs/model/i1b_qformer_mistral7b_f8_st2_sh.py

# # internvideo-1b_mistral
# CKPT=logs/scripts/pt/1b_qformer_mistral/ablation/stage2_8f_gl_sh.sh_gl/checkpoint-last
# SAVE=results/groundvqa/stage2_gl
# MODEL_CFG=configs/model/i1b_qformer_mistral7b_f8_st2_sh.py

# # internvideo-1b_mistral_glfg-qformer
# CKPT=logs/scripts/pt/1b_qformer_mistral/ablation/stage2_8f_glfg_sh.sh_glfg/checkpoint-last
# SAVE=results/groundvqa/stage2_glfg
# MODEL_CFG=configs/model/i1b_qformer_mistral7b_f8_st2_sh.py

# # internvideo-1b_mistral_gltc-qformer
# CKPT=logs/scripts/pt/1b_qformer_mistral/ablation/stage2_8f_gltc_sh.sh_gltc/checkpoint-last
# SAVE=results/groundvqa/stage2_gltc
# MODEL_CFG=configs/model/i1b_qformer_mistral7b_f8_st2_sh.py

# # internvideo-1b_mistral_glfgtc-qformer
# CKPT=logs/scripts/pt/1b_qformer_mistral/stage2_8f_glfgtc_sh.sh_glfgtc/checkpoint-last
# SAVE=results/groundvqa/stage2_glfgtc
# MODEL_CFG=configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py

# # internvideo-1b_mistral_gl-qformer_hd
# CKPT=logs/scripts/pt/1b_qformer_mistral/ablation/stage4_16f_gl_hd.sh_gl_hd/checkpoint-last
# SAVE=results/groundvqa/stage4_gl_hd
# MODEL_CFG=configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py

# # internvideo-1b_mistral_glfg-qformer_hd
# CKPT=logs/scripts/pt/1b_qformer_mistral/ablation/stage4_16f_glfg_hd.sh_glfg_hd/checkpoint-last
# SAVE=results/groundvqa/stage4_glfg_hd
# MODEL_CFG=configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py

# internvideo-1b_mistral_gltc-qformer_hd
CKPT=logs/scripts/pt/1b_qformer_mistral/ablation/stage4_16f_gltc_hd.sh_gltc_hd/checkpoint-last
SAVE=results/groundvqa/stage4_gltc_hd
MODEL_CFG=configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py

# # internvideo-1b_mistral_glfgtc-qformer_hd
# CKPT=logs/scripts/pt/1b_qformer_mistral/stage4_16f_glfgtc_hd.sh_glfgtc_hd/checkpoint-last
# SAVE=results/groundvqa/stage4_glfgtc_hd
# MODEL_CFG=configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py

for IDX in $(seq 0 $((NUM_GPUS-1))); do
    START=$((IDX * NUM_SINGLE_GPU))
    END=$(($((IDX + 1)) * NUM_SINGLE_GPU))
    CUDA_VISIBLE_DEVICES=$IDX python -u demo/eval_groundvqa.py \
        --model_cfg ${MODEL_CFG} \
        --ckpt_path $CKPT \
        --save_path $SAVE \
        --num_f 8 \
        --start ${START} --end ${END} &
done

wait

python demo/merge_res.py $SAVE