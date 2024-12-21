from configs.dataset.data_config.available_corpus import available_corpus

# ========================= data ==========================
train_file = available_corpus["internvid2_interleaved_1M_avscap_smit_clip2"]
test_file = dict(msrvtt_1k_test=available_corpus["msrvtt_1k_test"])

for data in train_file:
    data['use_role'] = False
    data['use_prompt'] = False

test_file['msrvtt_1k_test']['use_role'] = False
test_file['msrvtt_1k_test']['use_prompt'] = False

custom_per_device_train_bs_dict = {
    "interleaved_video":2
}
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
