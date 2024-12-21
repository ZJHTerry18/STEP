# internvideo-1b_mistral
python src/tools/qformer_vis_st1.py \
    --ckpt_path /mnt/petrelfs/share_data/wangchenting/models/internvideo2_videochat2/internvideo2_1b_mistral7b_stage3 \

# # internvideo-1b_mistral-sft
# python src/tools/qformer_vis_st1.py \
#     --model_cfg configs/model/ablations/i1b_gl-qformer_mistral7b_pt_f8_sh.py \
#     --ckpt_path logs/scripts/pt/1b_qformer_mistral/ablation/stage1_8f_gl_sh.sh_gl/checkpoint-last

# # internvideo-1b_mistral_fg-qformer
# python src/tools/qformer_vis_st1.py \
#     --model_cfg configs/model/ablations/i1b_glfg-qformer_mistral7b_pt_f8_sh.py \
#     --tokenizer_cfg configs/tokenizer/mistral_tokenizer_q96.yaml \
#     --ckpt_path logs/scripts/pt/1b_qformer_mistral/ablation/stage1_8f_glfg_sh.sh_glfg/checkpoint-last

# # internvideo-1b_mistral_fg-tc-qformer
# python src/tools/qformer_vis_st1.py \
#     --model_cfg configs/model/i1b_glfgtc-qformer_mistral7b_pt_f8_sh.py \
#     --tokenizer_cfg configs/tokenizer/mistral_tokenizer_q96.yaml \
#     --ckpt_path logs/scripts/pt/1b_qformer_mistral/stage1_8f_glfgtc_sh.sh_glfgtc/checkpoint-last

# # internvideo-1b_mistral_fg-tc hd -qformer
# python src/tools/qformer_vis_st1.py \
#     --model_cfg configs/model/i1b_glfgtc-qformer_mistral7b_pt_f8_sh.py \
#     --tokenizer_cfg configs/tokenizer/mistral_tokenizer_q96.yaml \
#     --ckpt_path logs/scripts/pt/1b_qformer_mistral/stage1_16f_glfgtc_hd.sh_glfgtc/checkpoint-last