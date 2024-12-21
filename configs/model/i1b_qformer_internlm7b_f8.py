num_frames = 8

custom_lr_wd_dict=dict(
    lm={'lr':2e-5, 'wd':None},
    vision_encoder={'lr':2e-5, 'wd':None}
)

model = dict(
    model_cls="MultiModalLLM_PT",
    vision_encoder=dict(
        name="internvideo2-1B",
        img_size=224, 
        patch_size=16, 
        d_model=1408,
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
        name='internlm2_5_7b',
        pretrained_llm_path='/mnt/petrelfs/share/videointern/MLLM/internlm2_5_7b_1m',
        use_lora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    ),
    loss=dict(
        use_vision_regression_loss=False
    ),
    pretrained_paths=dict(
    ),
    use_flash_attention=True,
    freeze_vision_encoder=False,
    freeze_bridge=False,
    freeze_llm=True,
)