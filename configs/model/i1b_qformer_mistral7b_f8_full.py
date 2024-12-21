num_frames = 8

custom_lr_wd_dict=dict(
    lm={'lr':4e-6, 'wd':None},
    vision_encoder={'lr':2e-5, 'wd':None}
)

model = dict(
    model_cls="MultiModalLLM_PT",
    vision_encoder=dict(
        name="internvideo2-1B",
        img_size=224, 
        patch_size=14, 
        d_model=1408,
        origin_num_frames=4,
        encoder_embed_dim=1408,
        num_frames="${num_frames}",
        tubelet_size=1,
        use_checkpoint=True,
        checkpoint_num=48,
        pretrained=None,
        sep_image_video_pos_embed=True,
        x_vis_return_idx=-2,
        x_vis_only=True,
        vit_add_ln=True,
    ),
    bridge=dict(
        name='qformer',
        num_query_token=32,
        extra_num_query_token=64,
        qformer_hidden_dropout_prob=0.1,
        qformer_attention_probs_dropout_prob=0.1,
        qformer_drop_path_rate=0.2
    ),
    llm=dict(
        name='mistral_7b',
        pretrained_llm_path='/mnt/petrelfs/share/videointern/MLLM/Mistral-7B-Instruct-v0.3/',
        use_lora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    ),
    loss=dict(
        use_vision_regression_loss=False
    ),
    pretrained_paths=dict(
        pretrained_vit_qformer_path="/mnt/petrelfs/wangchenting/multimodalllm/logs/scripts/pt/1b_qformer_mistral/stage1.sh_20240613_002655/checkpoint-2000/model.safetensors",
    ),
    use_flash_attention=True,
    freeze_vision_encoder=False,
    freeze_bridge=False,
    freeze_llm=False,
)