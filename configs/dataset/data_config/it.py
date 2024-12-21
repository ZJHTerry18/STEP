import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy

anno_root_it = "/mnt/petrelfs/share_data/videointern/annotations/anno_instruction/videochat_new"
# anno_root_it = "/mnt/petrelfs/share_data/likunchang/videochat/new_data"


# ============== pretraining datasets=================
available_corpus = dict(
    # image
    caption_coco = dict(
        anno_path=f"{anno_root_it}/image/caption/coco/train.json", 
        data_root="p2:s3://coco_caption",
    ),
    vqa_tvqa = dict (
        anno_path=f'/mnt/petrelfs/share_data/wangchenting/datasets/tvqa.json',
        data_root="p2:s3://tvqa/frames_fps3_hq/",
        media_type='video',
        video_reader_type='img'
    ),
    caption_coco_100k = dict(
        anno_path=f"{anno_root_it}/image/caption/coco/train_100k.json", 
        data_root="p2:s3://coco_caption",
    ),
    caption_llava = dict(
        anno_path=f"{anno_root_it}/image/caption/llava/train.json", 
        data_root="p2:s3://coco-caption",
    ),
    caption_svit = dict(
        anno_path=f"{anno_root_it}/image/caption/svit/train_20k.json", 
        data_root="p2:s3://svit",
    ),
    caption_minigpt4 = dict(
        anno_path=f"{anno_root_it}/image/caption/minigpt4/train.json", 
        data_root="p2:s3://minigpt4/image",
    ),
    caption_paragraph_captioning = dict(
        anno_path=f"{anno_root_it}/image/caption/paragraph_captioning/train.json", 
        data_root="p2:s3://m3it/image-paragraph-captioning",
    ),
    caption_textcaps = dict(
        anno_path=f"{anno_root_it}/image/caption/textcaps/train.json", 
        data_root="p2:s3://m3it/textcap",
    ),
    caption_som = dict(
        anno_path=f"{anno_root_it}/image/caption/som/train.json", 
        data_root="/mnt/petrelfs/share_data/likunchang/som",
    ),
    classification_imagenet = dict(
        anno_path=f"{anno_root_it}/image/classification/imagenet/train.json", 
        data_root="p2:s3://m3it/imagenet",
    ),
    classification_coco_itm = dict(
        anno_path=f"{anno_root_it}/image/classification/coco_itm/train.json", 
        data_root="p2:s3://m3it/coco-itm",
    ),
    conversation_llava = dict(
        anno_path=f"{anno_root_it}/image/conversation/llava/train.json", 
        data_root="p2:s3://coco_caption",
    ),
    conversation_svit = dict(
        anno_path=f"{anno_root_it}/image/conversation/svit/train_30k.json", 
        data_root="p2:s3://svit",
    ),
    conversation_lvis_instruct4v = dict(
        anno_path=f"{anno_root_it}/image/conversation/lvis_instruct4v/train.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data",
    ),
    conversation_som = dict(
        anno_path=f"{anno_root_it}/image/conversation/som/train.json", 
        data_root="/mnt/petrelfs/share_data/likunchang/som",
    ),
    reasoning_clevr = dict(
        anno_path=f"{anno_root_it}/image/reasoning/clevr/train.json", 
        data_root="p2:s3://m3it/clevr",
    ),
    reasoning_visual_mrc = dict(
        anno_path=f"{anno_root_it}/image/reasoning/visual_mrc/train.json", 
        data_root="p2:s3://m3it/visual-mrc",
    ),
    reasoning_llava = dict(
        anno_path=f"{anno_root_it}/image/reasoning/llava/train.json", 
        data_root="p2:s3://coco-caption",
    ),
    reasoning_svit = dict(
        anno_path=f"{anno_root_it}/image/reasoning/svit/train_50k.json", 
        data_root="p2:s3://svit",
    ),
    reasoning_science_qa = dict(
        anno_path=f"{anno_root_it}/image/reasoning/science_qa/train.json", 
        data_root="p2:s3://m3it/science-qa",
    ),
    vqa_vqav2 = dict(
        anno_path=f"{anno_root_it}/image/vqa/vqav2/train.json", 
        data_root="p2:s3://m3it/vqa-v2",
    ),
    vqa_vqav2_chinese = dict(
        anno_path=f"{anno_root_it}/image/vqa/vqav2_chinese/train.json", 
        data_root="p2:s3://m3it/vqa-v2",
    ),
    vqa_gqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/gqa/train.json", 
        data_root="p2:s3://m3it/gqa",
    ),
    vqa_okvqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/okvqa/train.json", 
        data_root="p2:s3://m3it/okvqa",
    ),
    vqa_okvqa_chinese = dict(
        anno_path=f"{anno_root_it}/image/vqa/okvqa_chinese/train.json", 
        data_root="p2:s3://m3it/okvqa",
    ),
    vqa_a_okvqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/a_okvqa/train.json", 
        data_root="p2:s3://m3it/a-okvqa",
    ),
    vqa_viquae = dict(
        anno_path=f"{anno_root_it}/image/vqa/viquae/train.json", 
        data_root="p2:s3://m3it/viquae",
    ),
    vqa_ocr_vqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/ocr_vqa/train.json", 
        data_root="p2:s3://m3it/ocr-vqa",
    ),
    vqa_ocr_vqa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/ocr_vqa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/OCRVQA/images",
    ),
    vqa_text_vqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/text_vqa/train.json", 
        data_root="p2:s3://m3it/text-vqa",
    ),
    vqa_text_vqa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/text_vqa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/TextVQA/train_images",
    ),
    vqa_st_vqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/st_vqa/train.json", 
        data_root="p2:s3://m3it/st-vqa",
    ),
    vqa_st_vqa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/st_vqa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/ST-VQA",
    ),
    vqa_docvqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/docvqa/train.json", 
        data_root="p2:s3://m3it/docvqa",
    ),
    vqa_docvqa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/docvqa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/DocVQA/train/train/documents",
    ),
    vqa_infovqa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/infovqa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/InfoVQA/infographicVQA_train_v1.0_images",
    ),
    vqa_ai2d = dict(
        anno_path=f"{anno_root_it}/image/vqa/ai2d/train.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ai2diagram/ai2d/images",
    ),
    vqa_chart_qa = dict(
        anno_path=f"{anno_root_it}/image/vqa/chart_qa/train.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/chartqa/ChartQA Dataset/train/png",
    ),
    vqa_chart_qa_gpt = dict(
        anno_path=f"{anno_root_it}/image/vqa/chart_qa/train_gpt.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/ocr_data/ChartQA/train",
    ),
    vqa_dvqa = dict(
        anno_path=f"{anno_root_it}/image/vqa/dvqa/train.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/DVQA/images",
    ),
    vqa_dvqa_80k = dict(
        anno_path=f"{anno_root_it}/image/vqa/dvqa/train_80k.json", 
        data_root="/mnt/petrelfs/share_data/wangwenhai/data/DVQA/images",
    ),
    grounding_coco = dict(
        anno_path=f"{anno_root_it}/image/grounding/coco/train.json", 
        data_root="p2hdd:s3://videollava/llava_image_tune/coco",
    ),
    grounding_vg = dict(
        anno_path=f"{anno_root_it}/image/grounding/vg/train.json", 
        data_root="p2hdd:s3://videollava/llava_image_tune/vg",
    ),
    # video
    caption_textvr = dict(
        anno_path=f"{anno_root_it}/video/caption/textvr/train.json", 
        data_root="p2:s3://TextVR/Video",
        media_type="video"
    ),
    caption_videochat = dict(
        anno_path=f"{anno_root_it}/video/caption/videochat/train.json", 
        data_root="p2:s3://webvid10m",
        media_type="video"
    ),
    caption_videochat_chinese = dict(
        anno_path=f"{anno_root_it}/video/caption/videochat_chinese/train.json", 
        data_root="p2:s3://WebVid10M",
        media_type="video"
    ),
    caption_videochatgpt = dict(
        anno_path=f"{anno_root_it}/video/caption/videochatgpt/train.json", 
        data_root="p2:s3://anet/ANet_320p_fps30",
        media_type="video"
    ),
    caption_webvid = dict(
        anno_path=f"{anno_root_it}/video/caption/webvid/train.json", 
        data_root="p2:s3://WebVid2M",
        media_type="video"
    ),
    caption_webvid_80k = dict(
        anno_path=f"{anno_root_it}/video/caption/webvid/train_80k.json", 
        data_root="p2:s3://WebVid2M",
        media_type="video"
    ),
    caption_youcook2 = dict(
        anno_path=f"{anno_root_it}/video/caption/youcook2/train_debug.json", 
        data_root="p2:s3://youcook2/split_videos",
        media_type="video"
    ),
    caption_youcook2_L20 = dict(
        anno_path=f"{anno_root_it}/video/caption/youcook2/train_L20.json", 
        data_root="p2:s3://youcook2/split_videos",
        media_type="video"
    ),
    caption_smit = dict(
        anno_path=f"{anno_root_it}/video/caption/s_mit/train.json", 
        data_root="p2hdd:s3://S-MiT",
        media_type="video"
    ),
    caption_smit_40k = dict(
        anno_path=f"{anno_root_it}/video/caption/s_mit/train_40k.json", 
        data_root="p2hdd:s3://S-MiT",
        media_type="video"
    ),
    caption_vatex_chinese = dict(
        anno_path=f"{anno_root_it}/video/caption/vatex_chinese/train.json", 
        data_root="p2:s3://k600",
        media_type="video"
    ),
    classification_k710 = dict(
        anno_path=f"{anno_root_it}/video/classification/k710/train.json", 
        # anno_path=f"{anno_root_it}/video/classification/k710/train_60k.json", 
        data_root="",
        media_type="video"
    ),
    classification_ssv2 = dict(
        anno_path=f"{anno_root_it}/video/classification/ssv2/train.json", 
        # anno_path=f"{anno_root_it}/video/classification/ssv2/train_60k.json", 
        data_root="p2:s3://ssv2_video",
        media_type="video"
    ),
    classification_mitv1 = dict(
        anno_path=f"{anno_root_it}/video/classification/mitv1/train.json", 
        data_root="p2hdd:s3://Moments_in_Time_Raw/videos",
        media_type="video"
    ),
    conversation_videochat1 = dict(
        anno_path=f"{anno_root_it}/video/conversation/videochat1/train.json", 
        data_root="p2:s3://WebVid10M",
        media_type="video"
    ),
    conversation_videochat2 = dict(
        anno_path=f"{anno_root_it}/video/conversation/videochat2/train.json", 
        data_root="p2hdd:s3://videointernsegvideos",
        media_type="video"
    ),
    conversation_videochat2_chinese = dict(
        anno_path=f"{anno_root_it}/video/conversation/videochat2_chinese/train.json", 
        data_root="p2hdd:s3://videointernsegvideos",
        media_type="video"
    ),
    vqa_rgbd = dict(
        anno_path='/mnt/petrelfs/share_data/wangchenting/rgbd.json',
        data_root='p2:s3://nturgbd',
        media_type='video',
        video_reader_type='av'
    ),
    conversation_videochatgpt = dict(
        anno_path=f"{anno_root_it}/video/conversation/videochatgpt/train.json", 
        data_root="p2:s3://ANet/ANet_320p_fps30",
        media_type="video"
    ),
    reasoning_next_qa = dict(
        anno_path=f"{anno_root_it}/video/reasoning/next_qa/train.json", 
        data_root="p2:s3://nextqa",
        media_type="video"
    ),
    reasoning_clevrer_qa = dict(
        anno_path=f"{anno_root_it}/video/reasoning/clevrer_qa/train.json", 
        # anno_path=f"{anno_root_it}/video/reasoning/clevrer_qa/train_mc.json", 
        # anno_path=f"{anno_root_it}/video/reasoning/clevrer_qa/train_mc_plus.json", 
        data_root="p2:s3://clevrer/video_train",
        media_type="video"
    ),
    reasoning_clevrer_mc = dict(
        anno_path=f"{anno_root_it}/video/reasoning/clevrer_mc/train.json",
        # anno_path=f"{anno_root_it}/video/reasoning/clevrer_mc/train_43k.json",  
        # anno_path=f"{anno_root_it}/video/reasoning/clevrer_mc/train_43k_debug.json",  
        data_root="p2:s3://clevrer/video_train",
        media_type="video"
    ),
    reasoning_star = dict(
        anno_path=f"{anno_root_it}/video/reasoning/star/train.json", 
        data_root="p2:s3://star/Charades_v1_480",
        media_type="video"
    ),
    vqa_ego_qa = dict(
        anno_path=f"{anno_root_it}/video/vqa/ego_qa/train.json", 
        data_root="p2:s3://egoqa/split_videos",
        # anno_path="/mnt/petrelfs/zhaojiahe/data/ego_feedback/qaego4d_simple/train_mc_it.json",
        # data_root="p2:s3://ego4d/clips",
        media_type="video"
    ),
    vqa_ego_qa_L15 = dict(
        anno_path=f"{anno_root_it}/video/vqa/ego_qa/train_L15.json", 
        data_root="p2:s3://egoqa/split_videos",
        media_type="video"
    ),
    vqa_tgif_frame_qa = dict(
        anno_path=f"{anno_root_it}/video/vqa/tgif_frame_qa/train.json", 
        data_root="p2:s3://tgif",
        media_type="video",
        video_reader_type="gif", # read from gif
    ),
    vqa_tgif_transition_qa = dict(
        anno_path=f"{anno_root_it}/video/vqa/tgif_transition_qa/train.json", 
        data_root="p2:s3://tgif",
        media_type="video",
        video_reader_type="gif", # read from gif
    ),
    vqa_webvid_qa = dict(
        anno_path=f"{anno_root_it}/video/vqa/webvid_qa/train.json", 
        data_root="p2:s3://WebVid2M",
        media_type="video",
    ),
    vqa_webvid_qa_30k = dict(
        anno_path=f"{anno_root_it}/video/vqa/webvid_qa/train_30k.json", 
        data_root="p2:s3://WebVid2M",
        media_type="video",
    ),
    vqa_perception_train = dict(
        # anno_path='/mnt/petrelfs/zhaojiahe/data/perception/mc_question_train_forchoice.json',
        anno_path='/mnt/petrelfs/share_data/wangchenting/datasets/mc_question_train_forchoice.json',
        data_root='p2:s3://perception/videos',
        media_type='video'
    ),
    egotask = dict(
        anno_path='/mnt/petrelfs/share_data/wangchenting/direct_train_qas_convertmc.json',
        data_root='',
        media_type="video",
        video_reader_type='pt'
    ),
    # 2024/04: Add detailed caption data for dynamic resolution finetuning
    caption_sharegpt4v = dict(
        anno_path=f"{anno_root_it}/image/caption/sharegpt4v/train.json", 
        data_root="p2:s3://sharegpt4v/data",
    ),
    caption_sharegpt4o = dict(
        anno_path='/mnt/petrelfs/share_data/wangchenting/datasets/sharegpt4o.json',
        data_root="p2:s3://perception/videos",
        media_type="video",
    ),
    lsmdc = dict(
        anno_path='/mnt/petrelfs/share_data/wangchenting/lsmdc.json',
        data_root="pvideo:s3://LSMDC",
        media_type="video",
        video_reader_type="av",
    ),
    caption_sharegpt4v_420k = dict(
        anno_path=f"{anno_root_it}/image/caption/sharegpt4v/train_420k.json", 
        data_root="p2:s3://sharegpt4v/data",
    ),
    caption_sharegptvideo_300k = dict(
        anno_path=f"{anno_root_it}/video/caption/sharegptvideo/train_300k.json", 
        data_root="pvideo:s3://LLaVA_DPO/train_300k",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    caption_sharegptvideo_600k = dict(
        anno_path=f"{anno_root_it}/video/caption/sharegptvideo/train_600k.json", 
        data_root="pvideo:s3://LLaVA_DPO/train_600k",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    vqa_sharegptvideo_240k = dict(
        anno_path=f"{anno_root_it}/video/vqa/sharegptvideo/train_240k.json", 
        data_root="pvideo:s3://LLaVA_DPO/train_300k",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    vqa_sharegptvideo_900k = dict(
        anno_path=f"{anno_root_it}/video/vqa/sharegptvideo/train_900k.json", 
        data_root="pvideo:s3://LLaVA_DPO/train_300k",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    caption_vidln_kinetics = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/kinetics_train.json", 
        data_root="",
        media_type="video",
    ),
    caption_vidln_oops = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/oops_train.json", 
        data_root="p2:s3://oops/oops_video/train",
        media_type="video",
    ),
    caption_vidln_oops_L15 = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/oops_train_L15.json", 
        data_root="p2:s3://oops/oops_video/train",
        media_type="video",
    ),
    caption_vidln_ovis = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/ovis_train.json", 
        data_root="p2:s3://ovis/train",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    caption_vidln_uvo_sparse = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/uvo_sparse_train.json", 
        data_root="s3://UVO/uvo_videos_sparse",
        media_type="video",
    ),
    caption_vidln_uvo_dense = dict(
        anno_path=f"{anno_root_it}/video/caption/vidln/uvo_dense_train.json", 
        data_root="sssd:s3://UVO/uvo_videos_dense",
        media_type="video",
    ),
    caption_favd = dict(
        anno_path=f"{anno_root_it}/video/caption/favd/train.json", 
        data_root="p2:s3://favd",
        media_type="video",
    ),
    grounding_didemo = dict(
        anno_path=f"{anno_root_it}/video/grounding/didemo/train.json", 
        data_root="sssd:s3://yjsBucket/DiDeMo",
        media_type="video",
    ),
    vqa_moviechat = dict(
        anno_path=f"/mnt/petrelfs/share/videointern/annotations/anno_instruction/MovieChat/train_global.json",
        data_root="p2:s3://MovieChat/real_video",
        media_type="video",
    ),
    # text
    conversation_sharegpt = dict(
        anno_path=f"{anno_root_it}/text/sharegpt/train.json", 
        data_root="",
        media_type="text",
    ),
    conversation_openhermes = dict(
        anno_path=f"{anno_root_it}/text/openhermes/train.json", 
        data_root="",
        # "text",
    ),
    conversation_openhermes_200k = dict(
        anno_path=f"{anno_root_it}/text/openhermes/train_200k.json", 
        data_root="",
        # "text",
    ),
    # add docci
    caption_docci = dict(
        anno_path=f"{anno_root_it}/image/caption/docci/train.json", 
        data_root="p2:s3://DOCCI/images",
    ),
    caption_docci_test = dict(
        anno_path=f"{anno_root_it}/image/caption/docci/test.json", 
        data_root="p2:s3://DOCCI/images",
    ),
    # TimeIT
    timeit_ANet = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/ANet/train.json", 
        data_root="p2:s3://ANet/ANet_320p_fps30/",
        media_type="video"
    ),
    timeit_COIN = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/COIN/train.json", 
        data_root="p2hdd:s3://COIN_320p",
        media_type="video"
    ),
    timeit_DiDeMo = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/DiDeMo/train.json", 
        data_root="s3://yjsBucket/DiDeMo/",
        media_type="video"
    ),
    timeit_HiREST = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/HiREST/train.json", 
        data_root="zxy_pssd:s3://HiREST/",
        media_type="video"
    ),
    timeit_QuerYD = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/QuerYD/train.json", 
        data_root="zxy_pssd:s3://QuerYD/",
        media_type="video"
    ),
    timeit_SumMe = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/SumMe/train.json", 
        data_root="zxy_pssd:s3://SumMe",
        media_type="video"
    ),

    timeit_TVSum = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/TVSum/train.json", 
        data_root="zxy_pssd:s3://TVSum",
        media_type="video"
    ),
    timeit_ViTT = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/ViTT/train.json", 
        data_root="zxy_pssd:s3://ViTT",
        media_type="video"
    ),
    timeit_yttemporal180m = dict(
        anno_path=f"{anno_root_it}/video/TimeIT/yttemporal180m/train.json", 
        data_root="p2hdd:s3://YT-Temporal-180M",
        media_type="video"
    ),
)

# for debug
available_corpus["debug_instruction"] = [
    available_corpus["caption_coco_100k"],
    available_corpus["caption_webvid_80k"],
]

# perception_data_0727
available_corpus["videochat2_instruction"] = [
    available_corpus["caption_coco"],
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["classification_imagenet"],
    available_corpus["classification_coco_itm"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"],
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa"],
    available_corpus["vqa_text_vqa"],
    available_corpus["vqa_st_vqa"],
    available_corpus["vqa_docvqa"],
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_webvid"],
    available_corpus["caption_youcook2"],
    available_corpus["classification_k710"],
    available_corpus["classification_ssv2"],
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"],
    available_corpus["reasoning_clevrer_mc"],
    available_corpus["vqa_ego_qa"],
    available_corpus["vqa_tgif_frame_qa"],
    available_corpus["vqa_tgif_transition_qa"],
    available_corpus["vqa_webvid_qa"],
]

# use coco_100k, webvid_80k, add smit
available_corpus["videochat2_instruction_2024_0204"] = [
    available_corpus["caption_coco_100k"],
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["classification_imagenet"],
    available_corpus["classification_coco_itm"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"],
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa"],
    available_corpus["vqa_text_vqa"],
    available_corpus["vqa_st_vqa"],
    available_corpus["vqa_docvqa"],
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_webvid_80k"],
    available_corpus["caption_youcook2"],
    available_corpus["caption_smit"],
    available_corpus["classification_k710"],
    available_corpus["classification_ssv2"],
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"],
    available_corpus["reasoning_clevrer_mc"],
    available_corpus["vqa_ego_qa"],
    available_corpus["vqa_tgif_frame_qa"],
    available_corpus["vqa_tgif_transition_qa"],
    available_corpus["vqa_webvid_qa"],
]

available_corpus["videochat2_stage2_sh"] = [
    # available_corpus["caption_sharegptvideo_300k"],
    # available_corpus["vqa_sharegptvideo_240k"],
    available_corpus["caption_llava"], # 23k
    # available_corpus["reasoning_llava"],
    available_corpus["reasoning_star"], # 45k
    available_corpus["caption_videochat"], # 7k
    available_corpus["reasoning_clevrer_qa"], # 40k
    available_corpus["reasoning_clevrer_mc"], # 20k
    available_corpus["vqa_perception_train"], # 8k
    # available_corpus["reasoning_next_qa"], # 34k
    available_corpus["vqa_ego_qa"], # 8k
]

# use coco_100k, webvid_80k, add smit
available_corpus["videochat2_instruction_2024_0705"] = [
    available_corpus["caption_sharegpt4o"],
    available_corpus["lsmdc"],
    available_corpus["vqa_rgbd"],
    available_corpus["vqa_perception_train"],
    available_corpus["egotask"],
    available_corpus["vqa_tvqa"],
    # available_corpus[""]
    # available_corpus["caption_coco_100k"],
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["classification_imagenet"],
    available_corpus["classification_coco_itm"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"],
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["reasoning_star"],
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa"],
    available_corpus["vqa_text_vqa"],
    available_corpus["vqa_st_vqa"],
    available_corpus["vqa_docvqa"],
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    # available_corpus["caption_webvid_80k"],
    available_corpus["caption_youcook2"],
    available_corpus["caption_smit"],
    available_corpus["caption_sharegpt4v"],
    available_corpus["classification_k710"],
    available_corpus["classification_ssv2"],
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"],
    available_corpus["reasoning_clevrer_mc"],
    available_corpus["vqa_ego_qa"],
    available_corpus["vqa_tgif_frame_qa"],
    available_corpus["vqa_tgif_transition_qa"],
    available_corpus["vqa_webvid_qa"],
    available_corpus["caption_sharegpt4v_420k"],
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["vqa_sharegptvideo_240k"],
    available_corpus["caption_vidln_kinetics"],
    available_corpus["caption_vidln_oops"],
    available_corpus["caption_vidln_ovis"],
    available_corpus["caption_vidln_uvo_sparse"],
    available_corpus["caption_vidln_uvo_dense"],
    available_corpus["caption_favd"],
    available_corpus["conversation_lvis_instruct4v"],
    available_corpus["vqa_ai2d"],
    available_corpus["vqa_chart_qa"],
    available_corpus["vqa_dvqa_80k"],
    # new 0429
    available_corpus["grounding_coco"],
    available_corpus["grounding_vg"],
    available_corpus["grounding_didemo"],
    # available_corpus["conversation_sharegpt"], # pure text
]


available_corpus["videochat2_instruction_2024_0705_no_perception"] = [
    available_corpus["caption_sharegpt4o"],
    # available_corpus["lsmdc"],
    available_corpus["vqa_rgbd"],
    # available_corpus["vqa_perception_train"],
    # available_corpus["egotask"],
    available_corpus["vqa_tvqa"],
    # available_corpus[""]
    # available_corpus["caption_coco_100k"],
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["classification_imagenet"],
    available_corpus["classification_coco_itm"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"],
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["reasoning_star"],
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa"],
    available_corpus["vqa_text_vqa"],
    available_corpus["vqa_st_vqa"],
    available_corpus["vqa_docvqa"],
    # available_corpus["vqa_tvqa"],
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    # available_corpus["caption_webvid_80k"],
    available_corpus["caption_youcook2"],
    available_corpus["caption_smit"],
    available_corpus["caption_sharegpt4v"],
    available_corpus["classification_k710"],
    available_corpus["classification_ssv2"],
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"],
    available_corpus["reasoning_clevrer_mc"],
    available_corpus["vqa_ego_qa"],
    available_corpus["vqa_tgif_frame_qa"],
    available_corpus["vqa_tgif_transition_qa"],
    available_corpus["vqa_webvid_qa"],
    available_corpus["caption_sharegpt4v_420k"],
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["vqa_sharegptvideo_240k"],
    available_corpus["caption_vidln_kinetics"],
    available_corpus["caption_vidln_oops"],
    available_corpus["caption_vidln_ovis"],
    available_corpus["caption_vidln_uvo_sparse"],
    available_corpus["caption_vidln_uvo_dense"],
    available_corpus["caption_favd"],
    available_corpus["conversation_lvis_instruct4v"],
    available_corpus["vqa_ai2d"],
    available_corpus["vqa_chart_qa"],
    available_corpus["vqa_dvqa_80k"],
    # new 0429
    available_corpus["grounding_coco"],
    available_corpus["grounding_vg"],
    available_corpus["grounding_didemo"],
]


available_corpus["videochat2_instruction_2024_0723_f16_post"] = [
    available_corpus["caption_sharegpt4o"], # 2k
    # available_corpus["vqa_rgbd"], # 110k
    available_corpus["vqa_perception_train"], # 8k
    # available_corpus["egotask"],
    # available_corpus["vqa_tvqa"], # 122k
    available_corpus["reasoning_next_qa"], # 34k
    available_corpus["reasoning_clevrer_qa"], # 40k
    available_corpus["reasoning_clevrer_mc"], # 20k
    # available_corpus["grounding_coco"], # 48k
    # available_corpus["grounding_vg"],
    # available_corpus["grounding_didemo"],
    available_corpus["reasoning_star"], # 45k
    available_corpus["caption_llava"], # 23k
    available_corpus["caption_minigpt4"], # 3k
    # available_corpus["vqa_moviechat"], # 0.8k
    # available_corpus["vqa_ego_qa"], # 8k
    # available_corpus["caption_videochat"], # 7k
    # available_corpus["caption_vidln_uvo_sparse"],
    # available_corpus["caption_vidln_uvo_dense"],
    # available_corpus["caption_favd"], # 10k
    # available_corpus["conversation_lvis_instruct4v"], # 222k
    # available_corpus["vqa_ai2d"], # 15k
    # available_corpus["vqa_chart_qa"], # 21k
    # available_corpus["vqa_ocr_vqa"], # 11k
]


# add detailed caption, keep some import data for captioning
available_corpus["dynamic_resolution_20240426"] = [
    # image
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa"],
    available_corpus["vqa_text_vqa"],
    available_corpus["vqa_st_vqa"],
    available_corpus["vqa_docvqa"],
    # video
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["vqa_webvid_qa_30k"],
    # # new
    available_corpus["caption_sharegpt4v_420k"],
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["vqa_sharegptvideo_240k"],
    available_corpus["caption_vidln_kinetics"],
    available_corpus["caption_vidln_oops"],
    available_corpus["caption_vidln_ovis"],
    available_corpus["caption_vidln_uvo_sparse"],
    available_corpus["caption_vidln_uvo_dense"],
    available_corpus["caption_favd"],
]

# add detailed caption, keep more import data for captioning
available_corpus["dynamic_resolution_20240426_plus"] = [
    # image
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"], # add
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa"],
    available_corpus["vqa_text_vqa"],
    available_corpus["vqa_st_vqa"],
    available_corpus["vqa_docvqa"],
    # video
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_youcook2"], # add
    available_corpus["caption_smit"], # add
    available_corpus["classification_ssv2"], # add
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"], # add
    available_corpus["reasoning_clevrer_mc"], # add
    available_corpus["vqa_ego_qa"], # add
    available_corpus["vqa_webvid_qa_30k"],
    # new
    available_corpus["caption_sharegpt4v_420k"],
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["vqa_sharegptvideo_240k"],
    available_corpus["caption_vidln_kinetics"],
    available_corpus["caption_vidln_oops"],
    available_corpus["caption_vidln_ovis"],
    available_corpus["caption_vidln_uvo_sparse"],
    available_corpus["caption_vidln_uvo_dense"],
    available_corpus["caption_favd"],
]

# detailed caption, keep more data for captioning
available_corpus["dynamic_resolution_20240426_plus2"] = [
    # image
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"], # add
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["reasoning_science_qa"], # add2
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa"],
    available_corpus["vqa_text_vqa"],
    available_corpus["vqa_st_vqa"],
    available_corpus["vqa_docvqa"],
    # video
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_youcook2"], # add
    available_corpus["caption_smit_40k"], # decrease2
    available_corpus["classification_k710"], # add2
    available_corpus["classification_ssv2"], # add
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"], # add
    available_corpus["reasoning_clevrer_mc"], # add
    available_corpus["vqa_ego_qa"], # add
    available_corpus["vqa_tgif_transition_qa"], # add2
    available_corpus["vqa_webvid_qa_30k"],
    # new
    available_corpus["caption_sharegpt4v_420k"],
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["vqa_sharegptvideo_240k"],
    available_corpus["caption_vidln_kinetics"],
    available_corpus["caption_vidln_oops"],
    available_corpus["caption_vidln_ovis"],
    available_corpus["caption_vidln_uvo_sparse"],
    available_corpus["caption_vidln_uvo_dense"],
    available_corpus["caption_favd"],
]


# detailed caption, keep more data for captioning, add more ocr data
available_corpus["dynamic_resolution_20240426_plus3"] = [
    # image
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"], # add
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["reasoning_science_qa"], # add2
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa"],
    available_corpus["vqa_text_vqa"],
    available_corpus["vqa_st_vqa"],
    available_corpus["vqa_docvqa"],
    # video
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_youcook2"], # add
    available_corpus["caption_smit_40k"], # decrease2
    available_corpus["classification_k710"], # add2
    available_corpus["classification_ssv2"], # add
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"], # add
    available_corpus["reasoning_clevrer_mc"], # add
    available_corpus["vqa_ego_qa"], # add
    available_corpus["vqa_tgif_transition_qa"], # add2
    available_corpus["vqa_webvid_qa_30k"],
    # new
    available_corpus["caption_sharegpt4v_420k"],
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["vqa_sharegptvideo_240k"],
    available_corpus["caption_vidln_kinetics"],
    available_corpus["caption_vidln_oops"],
    available_corpus["caption_vidln_ovis"],
    available_corpus["caption_vidln_uvo_sparse"],
    available_corpus["caption_vidln_uvo_dense"],
    available_corpus["caption_favd"],
    # new 0427
    available_corpus["conversation_lvis_instruct4v"],
    available_corpus["vqa_ai2d"],
    available_corpus["vqa_chart_qa"],
    available_corpus["vqa_dvqa_80k"],
]


# detailed caption, keep some data for captioning, add more ocr data
available_corpus["dynamic_resolution_20240426_plus4"] = [
    # image
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa"],
    available_corpus["vqa_text_vqa"],
    available_corpus["vqa_st_vqa"],
    available_corpus["vqa_docvqa"],
    # video
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["vqa_webvid_qa_30k"],
    # new
    available_corpus["caption_sharegpt4v_420k"],
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["vqa_sharegptvideo_240k"],
    available_corpus["caption_vidln_kinetics"],
    available_corpus["caption_vidln_oops"],
    available_corpus["caption_vidln_ovis"],
    available_corpus["caption_vidln_uvo_sparse"],
    available_corpus["caption_vidln_uvo_dense"],
    available_corpus["caption_favd"],
    # new 0427
    available_corpus["conversation_lvis_instruct4v"],
    available_corpus["vqa_ai2d"],
    available_corpus["vqa_chart_qa"],
    available_corpus["vqa_dvqa_80k"],
]


# detailed caption, keep more data for captioning, add more ocr data, add list data
available_corpus["dynamic_resolution_20240426_plus5"] = [
    # image
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"], # add
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["reasoning_science_qa"], # add2
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa"],
    available_corpus["vqa_text_vqa"],
    available_corpus["vqa_st_vqa"],
    available_corpus["vqa_docvqa"],
    # video
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_youcook2"], # add
    available_corpus["caption_smit_40k"], # decrease2
    available_corpus["classification_k710"], # add2
    available_corpus["classification_ssv2"], # add
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"], # add
    available_corpus["reasoning_clevrer_mc"], # add
    available_corpus["vqa_ego_qa"], # add
    available_corpus["vqa_tgif_transition_qa"], # add2
    available_corpus["vqa_webvid_qa_30k"],
    # new
    available_corpus["caption_sharegpt4v_420k"],
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["vqa_sharegptvideo_240k"],
    available_corpus["caption_vidln_kinetics"],
    available_corpus["caption_vidln_oops"],
    available_corpus["caption_vidln_ovis"],
    available_corpus["caption_vidln_uvo_sparse"],
    available_corpus["caption_vidln_uvo_dense"],
    available_corpus["caption_favd"],
    # new 0427
    available_corpus["conversation_lvis_instruct4v"],
    available_corpus["vqa_ai2d"],
    available_corpus["vqa_chart_qa"],
    available_corpus["vqa_dvqa_80k"],
    # new 0428
    available_corpus["caption_som"],
    available_corpus["conversation_som"],
]


# detailed caption, keep more data for captioning, add more ocr data, add list data
# update ocrdata
available_corpus["dynamic_resolution_20240426_plus6"] = [
    # image
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"], # add
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["reasoning_science_qa"], # add2
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa_gpt"], # use gpt
    available_corpus["vqa_text_vqa_gpt"], # use gpt
    available_corpus["vqa_st_vqa_gpt"], # use gpt
    available_corpus["vqa_docvqa_gpt"], # use gpt
    available_corpus["vqa_infovqa_gpt"], # new ocr
    # video
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_youcook2"], # add
    available_corpus["caption_smit_40k"], # decrease2
    available_corpus["classification_k710"], # add2
    available_corpus["classification_ssv2"], # add
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"], # add
    available_corpus["reasoning_clevrer_mc"], # add
    available_corpus["vqa_ego_qa"], # add
    available_corpus["vqa_tgif_transition_qa"], # add2
    available_corpus["vqa_webvid_qa_30k"],
    # new
    available_corpus["caption_sharegpt4v_420k"],
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["vqa_sharegptvideo_240k"],
    available_corpus["caption_vidln_kinetics"],
    available_corpus["caption_vidln_oops"],
    available_corpus["caption_vidln_ovis"],
    available_corpus["caption_vidln_uvo_sparse"],
    available_corpus["caption_vidln_uvo_dense"],
    available_corpus["caption_favd"],
    # new 0427
    available_corpus["conversation_lvis_instruct4v"],
    available_corpus["vqa_ai2d"],
    available_corpus["vqa_chart_qa"],
    available_corpus["vqa_dvqa_80k"],
    # new 0428
    available_corpus["caption_som"],
    available_corpus["conversation_som"],
]


# detailed caption, keep more data for captioning, add more ocr data, add list data
# update ocrdata
# add grounding data and pure text data
available_corpus["dynamic_resolution_20240426_plus7"] = [
    # image
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"], # add
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["reasoning_science_qa"], # add2
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa_gpt"], # use gpt
    available_corpus["vqa_text_vqa_gpt"], # use gpt
    available_corpus["vqa_st_vqa_gpt"], # use gpt
    available_corpus["vqa_docvqa_gpt"], # use gpt
    available_corpus["vqa_infovqa_gpt"], # new ocr
    # video
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_youcook2"], # add
    available_corpus["caption_smit_40k"], # decrease2
    available_corpus["classification_k710"], # add2
    available_corpus["classification_ssv2"], # add
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"], # add
    available_corpus["reasoning_clevrer_mc"], # add
    available_corpus["vqa_ego_qa"], # add
    available_corpus["vqa_tgif_transition_qa"], # add2
    available_corpus["vqa_webvid_qa_30k"],
    # new
    available_corpus["caption_sharegpt4v_420k"],
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["vqa_sharegptvideo_240k"],
    available_corpus["caption_vidln_kinetics"],
    available_corpus["caption_vidln_oops"],
    available_corpus["caption_vidln_ovis"],
    available_corpus["caption_vidln_uvo_sparse"],
    available_corpus["caption_vidln_uvo_dense"],
    available_corpus["caption_favd"],
    # new 0427
    available_corpus["conversation_lvis_instruct4v"],
    available_corpus["vqa_ai2d"],
    available_corpus["vqa_chart_qa"],
    available_corpus["vqa_dvqa_80k"],
    # new 0428
    available_corpus["caption_som"],
    available_corpus["conversation_som"],
    # new 0429
    available_corpus["grounding_coco"],
    available_corpus["grounding_vg"],
    available_corpus["grounding_didemo"],
    available_corpus["conversation_sharegpt"], # pure text
]


# detailed caption, keep more data for captioning, add more ocr data, add list data
# update ocrdata
# add grounding data and pure text data
# remove som
available_corpus["dynamic_resolution_20240426_plus8"] = [
    # image
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"], # add
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["reasoning_science_qa"], # add2
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa_gpt"], # use gpt
    available_corpus["vqa_text_vqa_gpt"], # use gpt
    available_corpus["vqa_st_vqa_gpt"], # use gpt
    available_corpus["vqa_docvqa_gpt"], # use gpt
    available_corpus["vqa_infovqa_gpt"], # new ocr
    # video
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_youcook2"], # add
    available_corpus["caption_smit_40k"], # decrease2
    available_corpus["classification_k710"], # add2
    available_corpus["classification_ssv2"], # add
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"], # add
    available_corpus["reasoning_clevrer_mc"], # add
    available_corpus["vqa_ego_qa"], # add
    available_corpus["vqa_tgif_transition_qa"], # add2
    available_corpus["vqa_webvid_qa_30k"],
    # new
    available_corpus["caption_sharegpt4v_420k"],
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["vqa_sharegptvideo_240k"],
    available_corpus["caption_vidln_kinetics"],
    available_corpus["caption_vidln_oops"],
    available_corpus["caption_vidln_ovis"],
    available_corpus["caption_vidln_uvo_sparse"],
    available_corpus["caption_vidln_uvo_dense"],
    available_corpus["caption_favd"],
    # new 0427
    available_corpus["conversation_lvis_instruct4v"],
    available_corpus["vqa_ai2d"],
    available_corpus["vqa_chart_qa"],
    available_corpus["vqa_dvqa_80k"],
    # new 0429
    available_corpus["grounding_coco"],
    available_corpus["grounding_vg"],
    available_corpus["grounding_didemo"],
    available_corpus["conversation_sharegpt"], # pure text
]


# detailed caption, keep more data for captioning, add more ocr data, add list data
# update ocrdata
# add grounding data and pure text data
# remove som
available_corpus["dynamic_resolution_20240426_plus9"] = [
    # image
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"], # add
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["reasoning_science_qa"], # add2
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa_gpt"], # use gpt
    available_corpus["vqa_text_vqa_gpt"], # use gpt
    available_corpus["vqa_st_vqa_gpt"], # use gpt
    available_corpus["vqa_docvqa_gpt"], # use gpt
    available_corpus["vqa_infovqa_gpt"], # new ocr
    # video
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_youcook2"], # add
    available_corpus["caption_smit_40k"], # decrease2
    available_corpus["classification_k710"], # add2
    available_corpus["classification_ssv2"], # add
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"], # add
    available_corpus["reasoning_clevrer_mc"], # add
    available_corpus["vqa_ego_qa"], # add
    available_corpus["vqa_tgif_transition_qa"], # add2
    available_corpus["vqa_webvid_qa_30k"],
    # new
    available_corpus["caption_sharegpt4v_420k"],
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["vqa_sharegptvideo_240k"],
    available_corpus["caption_vidln_kinetics"],
    available_corpus["caption_vidln_oops"],
    available_corpus["caption_vidln_ovis"],
    available_corpus["caption_vidln_uvo_sparse"],
    available_corpus["caption_vidln_uvo_dense"],
    available_corpus["caption_favd"],
    # new 0427
    available_corpus["conversation_lvis_instruct4v"],
    available_corpus["vqa_ai2d"],
    available_corpus["vqa_chart_qa"],
    available_corpus["vqa_dvqa_80k"],
    # new 0429
    available_corpus["grounding_coco"],
    available_corpus["grounding_vg"],
    available_corpus["grounding_didemo"],
    available_corpus["conversation_sharegpt"], # pure text
    # new 0509
    available_corpus["caption_docci"],
    available_corpus["caption_docci_test"], # also use test
]


# detailed caption, keep more data for captioning, add more ocr data, add list data
# update ocrdata
# add grounding data and pure text data
# remove som
# add openhermes
available_corpus["dynamic_resolution_20240426_plus10"] = [
    # image
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"], # add
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["reasoning_science_qa"], # add2
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa_gpt"], # use gpt
    available_corpus["vqa_text_vqa_gpt"], # use gpt
    available_corpus["vqa_st_vqa_gpt"], # use gpt
    available_corpus["vqa_docvqa_gpt"], # use gpt
    available_corpus["vqa_infovqa_gpt"], # new ocr
    # video
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_youcook2"], # add
    available_corpus["caption_smit_40k"], # decrease2
    available_corpus["classification_k710"], # add2
    available_corpus["classification_ssv2"], # add
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"], # add
    available_corpus["reasoning_clevrer_mc"], # add
    available_corpus["vqa_ego_qa"], # add
    available_corpus["vqa_tgif_transition_qa"], # add2
    available_corpus["vqa_webvid_qa_30k"],
    # new
    available_corpus["caption_sharegpt4v_420k"],
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["vqa_sharegptvideo_240k"],
    available_corpus["caption_vidln_kinetics"],
    available_corpus["caption_vidln_oops"],
    available_corpus["caption_vidln_ovis"],
    available_corpus["caption_vidln_uvo_sparse"],
    available_corpus["caption_vidln_uvo_dense"],
    available_corpus["caption_favd"],
    # new 0427
    available_corpus["conversation_lvis_instruct4v"],
    available_corpus["vqa_ai2d"],
    available_corpus["vqa_chart_qa"],
    available_corpus["vqa_dvqa_80k"],
    # new 0429
    available_corpus["grounding_coco"],
    available_corpus["grounding_vg"],
    available_corpus["grounding_didemo"],
    available_corpus["conversation_sharegpt"], # pure text
    # new 0510
    available_corpus["conversation_openhermes_200k"], # pure text
]


# detailed caption, keep more data for captioning, add more ocr data, add list data
# update ocrdata
# add grounding data and pure text data
# remove som
# increase vqa_sharegptvideo to 900k
available_corpus["dynamic_resolution_20240426_plus11"] = [
    # image
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"], # add
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["reasoning_science_qa"], # add2
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa_gpt"], # use gpt
    available_corpus["vqa_text_vqa_gpt"], # use gpt
    available_corpus["vqa_st_vqa_gpt"], # use gpt
    available_corpus["vqa_docvqa_gpt"], # use gpt
    available_corpus["vqa_infovqa_gpt"], # new ocr
    # video
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_youcook2"], # add
    available_corpus["caption_smit_40k"], # decrease2
    available_corpus["classification_k710"], # add2
    available_corpus["classification_ssv2"], # add
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"], # add
    available_corpus["reasoning_clevrer_mc"], # add
    available_corpus["vqa_ego_qa"], # add
    available_corpus["vqa_tgif_transition_qa"], # add2
    available_corpus["vqa_webvid_qa_30k"],
    # new
    available_corpus["caption_sharegpt4v_420k"],
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["vqa_sharegptvideo_900k"], # increase
    available_corpus["caption_vidln_kinetics"],
    available_corpus["caption_vidln_oops"],
    available_corpus["caption_vidln_ovis"],
    available_corpus["caption_vidln_uvo_sparse"],
    available_corpus["caption_vidln_uvo_dense"],
    available_corpus["caption_favd"],
    # new 0427
    available_corpus["conversation_lvis_instruct4v"],
    available_corpus["vqa_ai2d"],
    available_corpus["vqa_chart_qa"],
    available_corpus["vqa_dvqa_80k"],
    # new 0429
    available_corpus["grounding_coco"],
    available_corpus["grounding_vg"],
    available_corpus["grounding_didemo"],
    available_corpus["conversation_sharegpt"], # pure text
]


# only use video data
# add some long video data
available_corpus["dynamic_resolution_20240528"] = [
    # video
    available_corpus["caption_videochat"],
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["caption_youcook2_L20"],
    available_corpus["vqa_ego_qa_L15"],
    available_corpus["caption_vidln_oops_L15"],
    # time_it
    available_corpus["timeit_ANet"],
    available_corpus["timeit_COIN"],
    available_corpus["timeit_DiDeMo"],
    available_corpus["timeit_HiREST"],
    available_corpus["timeit_QuerYD"],
    available_corpus["timeit_SumMe"],
    available_corpus["timeit_TVSum"],
    available_corpus["timeit_ViTT"],
    available_corpus["timeit_yttemporal180m"],
]


# only use video data
# add some long video data
# add image and text
available_corpus["dynamic_resolution_20240528"] = [
    # video
    available_corpus["caption_videochat"],
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["caption_youcook2_L20"],
    available_corpus["vqa_ego_qa_L15"],
    available_corpus["caption_vidln_oops_L15"],
    # time_it
    available_corpus["timeit_ANet"],
    available_corpus["timeit_COIN"],
    available_corpus["timeit_DiDeMo"],
    available_corpus["timeit_HiREST"],
    available_corpus["timeit_QuerYD"],
    available_corpus["timeit_SumMe"],
    available_corpus["timeit_TVSum"],
    available_corpus["timeit_ViTT"],
    available_corpus["timeit_yttemporal180m"],
    # image and text
    available_corpus["grounding_coco"],
    available_corpus["grounding_vg"],
    available_corpus["conversation_sharegpt"], # pure text
]
