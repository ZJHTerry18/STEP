from configs.dataset.data_config.available_corpus import available_corpus

# ========================= data ==========================
train_file = available_corpus["debug"]
test_file = dict(coco_test=available_corpus["coco_test_4shot"])

test_file['coco_test_4shot']['use_role'] = False
test_file['coco_test_4shot']['use_prompt'] = False

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
