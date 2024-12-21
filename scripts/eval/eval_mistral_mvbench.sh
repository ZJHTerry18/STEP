# # internvideo-1b_mistral
# python -u demo/eval_mvbench.py \
#     --ckpt_path /mnt/petrelfs/share_data/wangchenting/models/internvideo2_videochat2/internvideo2_1b_mistral7b_stage3 \
#     --save_path results/mvbench/internvideo2_1b_mistral7b_stage3 \

# # internvideo-1b_mistral_hd_f16
# python -u demo/eval_mvbench.py \
#     --model_cfg configs/model/i1b_qformer_mistral7b_f8_hd.py \
#     --ckpt_path /mnt/petrelfs/share_data/wangchenting/models/internvideo2_videochat2/internvideo2_1b_mistral7b_stage4_hd_post_f16 \
#     --num_f 16 \
#     --save_path results/mvbench/internvideo2_1b_mistral7b_stage4_hd_post_f16 \

# # internvideo-1b_mistral sft
# python -u demo/eval_mvbench.py \
#     --ckpt_path logs/scripts/pt/1b_qformer_mistral/ablation/stage2_8f_gl_sh.sh_gl/checkpoint-last \
#     --save_path results/mvbench/stage2_gl

# # internvideo-1b_mistral_fg-qformer
# python -u demo/eval_mvbench.py \
#     --model_cfg configs/model/ablations/i1b_gl-qformer_mistral7b_it_f8_sh.py \
#     --tokenizer_cfg configs/tokenizer/mistral_tokenizer_q96.yaml \
#     --ckpt_path logs/scripts/pt/1b_qformer_mistral/ablation/stage2_8f_glfg_sh.sh_glfg/checkpoint-last \
#     --save_path results/mvbench/stage2_glfg

# # internvideo-1b_mistral_glfgtc-qformer
# python -u demo/eval_mvbench.py \
#     --model_cfg configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py \
#     --tokenizer_cfg configs/tokenizer/mistral_tokenizer_q96.yaml \
#     --ckpt_path logs/scripts/pt/1b_qformer_mistral/stage2_8f_glfgtc_sh.sh_glfgtc/checkpoint-last \
#     --save_path results/mvbench/stage2_glfgtc

# # internvideo-1b_mistral_glfgtc-qformer
# python -u demo/eval_mvbench.py \
#     --model_cfg configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py \
#     --tokenizer_cfg configs/tokenizer/mistral_tokenizer_q96.yaml \
#     --ckpt_path logs/scripts/pt/1b_qformer_mistral/stage2_8f_glfgtc_sh.sh_glfgtc/checkpoint-last \
#     --save_path results/mvbench/stage2_glfgtc

# # internvideo-1b_mistral_gl-qformer hd
# python -u demo/eval_mvbench.py \
#     --model_cfg configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py \
#     --tokenizer_cfg configs/tokenizer/mistral_tokenizer_q96.yaml \
#     --ckpt_path logs/scripts/pt/1b_qformer_mistral/ablation/stage4_16f_gl_hd.sh_gl_hd/checkpoint-last \
#     --save_path results/mvbench/stage4_gl_hd

# # internvideo-1b_mistral_glfg-qformer hd
# python -u demo/eval_mvbench.py \
#     --model_cfg configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py \
#     --tokenizer_cfg configs/tokenizer/mistral_tokenizer_q96.yaml \
#     --ckpt_path logs/scripts/pt/1b_qformer_mistral/ablation/stage4_16f_glfg_hd.sh_glfg_hd/checkpoint-last \
#     --save_path results/mvbench/stage4_glfg_hd

# # internvideo-1b_mistral_gltc-qformer hd
# python -u demo/eval_mvbench.py \
#     --model_cfg configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py \
#     --tokenizer_cfg configs/tokenizer/mistral_tokenizer_q96.yaml \
#     --ckpt_path logs/scripts/pt/1b_qformer_mistral/ablation/stage4_16f_gltc_hd.sh_gltc_hd/checkpoint-last \
#     --save_path results/mvbench/stage4_gltc_hd

# # internvideo-1b_mistral_hd_glfgtc-qformer
# python -u demo/eval_mvbench.py \
#     --model_cfg configs/model/tests/i1b_glfgtc-test-qformer_mistral7b_it_f8_sh.py \
#     --tokenizer_cfg configs/tokenizer/mistral_tokenizer_q96.yaml \
#     --ckpt_path logs/scripts/pt/1b_qformer_mistral/stage4_16f_glfgtc_hd.sh_glfgtc_hd/checkpoint-last \
#     --num_f 16 \
#     --save_path results/mvbench/stage4_16f_glfgtc_hd

# # internvideo-1b_mistral_hd_glfgtc-qformer loss ablation
# python -u demo/eval_mvbench.py \
#     --model_cfg configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py \
#     --tokenizer_cfg configs/tokenizer/mistral_tokenizer_q96.yaml \
#     --ckpt_path logs/scripts/pt/1b_qformer_mistral/ablation/stage4_16f_glfgtc_hd_loss.sh_glfgtc_hd_loss/checkpoint-last \
#     --num_f 16 \
#     --save_path results/mvbench/stage4_16f_glfgtc_hd_loss

# internvideo-1b_mistral_hd_glfgtc-qformer ablations
python -u demo/eval_mvbench.py \
    --model_cfg configs/model/i1b_glfgtc-qformer_mistral7b_it_f8_sh.py \
    --tokenizer_cfg configs/tokenizer/mistral_tokenizer_q96.yaml \
    --ckpt_path logs/scripts/pt/1b_qformer_mistral/ablation/stage4_16f_glfgtc_hd_loss.sh_glfgtc_hd_loss/checkpoint-last \
    --num_f 16 \
    --save_path results/mvbench/stage4_16f_glfgtc_hd_loss_group64