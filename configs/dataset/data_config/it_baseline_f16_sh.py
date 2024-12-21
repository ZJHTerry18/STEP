from configs.dataset.data_config.it import available_corpus


# ========================= data ==========================
train_corpus = "videochat2_stage2_sh"
train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation

test_file = dict()

# ========================= input ==========================
num_frames = 16
num_frames_test = 16
max_txt_l = 1024

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
