num_frames = 8

custom_lr_wd_dict=dict(
    lm={'lr':3e-5, 'wd':None},
    vision_encoder={'lr':5e-5, 'wd':None}
)

model = dict(
    model_cls="MultiModalLLM_PT",
    vision_encoder=dict(
        name="internvideo2-1B",
        img_size=224, 
        patch_size=16, 
        d_model=1408,
        encoder_embed_dim=1408,
        origin_num_frames=4,
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
        name='vicuna1.5_7b',
        llm_hidden_size=4096,
        pretrained_llm_path="/mnt/petrelfs/share_data/videointern/vicuna-7b-v1.5",
        use_llama_gradient_checkpointing=True
    ),
    loss=dict(
        use_vision_regression_loss=False
    ),
    pretrained_paths=dict(
        pretrained_vit_qformer_path="/mnt/petrelfs/share_data/lixinhao/avp_1b_qformer_new_smit_e04.pth",
    ),
    freeze_vision_encoder=False,
    freeze_bridge=False,
    freeze_llm=True,
)