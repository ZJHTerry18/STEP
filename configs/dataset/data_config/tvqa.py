from configs.dataset.data_config.it import available_corpus


# ========================= data ==========================
train_corpus = "caption_coco"
train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation

test_file = dict(
    nextqa_test=dict(
        anno_path='/mnt/petrelfs/share_data/likunchang/tvqa/val_msg0630_noSub.json',
        data_root='p2:s3://tvqa/frames_fps3_hq/',
        media_type='video',
        video_reader_type='img'
    )
)

# ========================= input ==========================
num_frames = 4
num_frames_test = 4
max_txt_l = 64

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}")
)
