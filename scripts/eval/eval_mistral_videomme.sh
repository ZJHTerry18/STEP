# # internvideo-1b_mistral
# CKPT=/mnt/petrelfs/share_data/wangchenting/models/internvideo2_videochat2/internvideo2_1b_mistral7b_stage3
# SAVE=results/videomme/internvideo2_1b_mistral7b_stage3_wsub
# MODEL_CFG=configs/model/i1b_qformer_mistral7b_f8_st2_sh.py

# # internvideo-1b_mistral hd
# CKPT=/mnt/petrelfs/share_data/wangchenting/models/internvideo2_videochat2/internvideo2_1b_mistral7b_stage4_hd_post_f16
# SAVE=results/videomme/internvideo2_1b_mistral7b_stage4_hd_post_f16_wsub
# MODEL_CFG=configs/model/i1b_qformer_mistral7b_f8_st2_sh.py

# # internvideo-1b_mistral
# CKPT=logs/scripts/pt/1b_qformer_mistral/300k/ablation/stage2_8f_gl_sh.sh_gl/checkpoint-last
# SAVE=results/videomme/stage2_gl_wsub
# MODEL_CFG=configs/model/i1b_qformer_mistral7b_f8_st2_sh.py

# # internvideo-1b_mistral_glfg-qformer
# CKPT=logs/scripts/pt/1b_qformer_mistral/300k/ablation/stage2_8f_glfg_sh.sh_glfg/checkpoint-last
# SAVE=results/videomme/stage2_glfg
# MODEL_CFG=configs/model/i1b_qformer_mistral7b_f8_st2_sh.py

# # internvideo-1b_mistral_gltc-qformer
# CKPT=logs/scripts/pt/1b_qformer_mistral/300k/ablation/stage2_8f_gltc_sh.sh_gltc/checkpoint-last
# SAVE=results/videomme/stage2_gltc
# MODEL_CFG=configs/model/i1b_qformer_mistral7b_f8_st2_sh.py

# # internvideo-1b_mistral_glfgtc-qformer
# CKPT=logs/scripts/pt/1b_qformer_mistral/stage2_8f_glfgtc_sh.sh_glfgtc/checkpoint-last
# SAVE=results/videomme/stage2_glfgtc
# MODEL_CFG=configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py

# # internvideo-1b_mistral_gl-qformer hd
# CKPT=logs/scripts/pt/1b_qformer_mistral/ablation/stage4_16f_gl_hd.sh_gl_hd/checkpoint-last
# SAVE=results/videomme/stage4_gl_hd
# MODEL_CFG=configs/model/i1b_qformer_mistral7b_f8_st2_sh.py

# # internvideo-1b_mistral_glfg-qformer hd
# CKPT=logs/scripts/pt/1b_qformer_mistral/ablation/stage4_16f_glfg_hd.sh_glfg_hd/checkpoint-last
# SAVE=results/videomme/stage4_glfg_hd
# MODEL_CFG=configs/model/i1b_qformer_mistral7b_f8_st2_sh.py

# # internvideo-1b_mistral_gltc-qformer hd
# CKPT=logs/scripts/pt/1b_qformer_mistral/ablation/stage4_16f_gltc_hd.sh_gltc_hd/checkpoint-last
# SAVE=results/videomme/stage4_gltc_hd
# MODEL_CFG=configs/model/i1b_qformer_mistral7b_f8_st2_sh.py

# # internvideo-1b_mistral_glfgtc-qformer_hd
# CKPT=logs/scripts/pt/1b_qformer_mistral/stage4_16f_glfgtc_hd.sh_glfgtc_hd/checkpoint-last
# SAVE=results/videomme/stage4_glfgtc_hd
# MODEL_CFG=configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py

# # internvideo-1b_mistral_glfgtc-qformer_hd loss ablation
# CKPT=logs/scripts/pt/1b_qformer_mistral/ablation/stage4_16f_glfgtc_hd_loss.sh_glfgtc_hd_loss/checkpoint-last
# SAVE=results/videomme/stage4_glfgtc_hd_loss
# MODEL_CFG=configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py

# # internvideo-1b_mistral_glfgtc-qformer_hd groupvidtc
# CKPT=logs/scripts/pt/1b_qformer_mistral/tests/stage4_16f_glfgtc-test_hd.sh_glfgtc-test_hd/checkpoint-last
# SAVE=results/videomme/stage4_16f_glfgtc_hd_groupvidtc_wsub
# MODEL_CFG=configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py

python -u demo/eval_videomme.py \
    --model_cfg ${MODEL_CFG} \
    --ckpt_path $CKPT \
    --num_f 16 \
    --sub \
    --save_path $SAVE