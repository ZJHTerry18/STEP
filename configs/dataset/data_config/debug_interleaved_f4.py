from configs.dataset.data_config.available_corpus import available_corpus

# ========================= data ==========================
train_file = available_corpus["internvid2_interleaved_1M_avscap"]
test_file = dict(msrvtt_1k_test=available_corpus["msrvtt_1k_test"])


# ========================= input ==========================
num_frames = 4
num_frames_test = 4



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
