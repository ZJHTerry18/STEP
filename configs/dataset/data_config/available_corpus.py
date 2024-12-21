import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy

data_dir = '/mnt/petrelfs/share/videointern/annotations'
if data_dir is None:
    raise ValueError("please set environment `VL_DATA_DIR` before continue")

data_root = __os.path.join(data_dir, "videos_images")
anno_root_pt = __os.path.join(data_dir, "anno_pretrain")
anno_root_downstream = __os.path.join(data_dir, "anno_downstream")

# ============== pretraining datasets=================
available_corpus = dict(
     # pretraining image datasets
    cc_new_3m=dict(
        anno_path="/mnt/petrelfs/share_data/heyinan/data_2023_h2/cc3m.json", 
        data_root="",
        data_root_prefix="pssd:",
        media_type="image",
        jump_filter=True
    ),
    cc_new_12m=dict(
        anno_path="/mnt/petrelfs/share_data/heyinan/data_2023_h2/cc12m.json", 
        data_root="",
        data_root_prefix="pssd:",
        media_type="image",
        jump_filter=True
    ),
    sbu_new=dict(
        anno_path="/mnt/petrelfs/wangyi/share_data/vidata/sbu-1m-fuse.json", 
        data_root="",
        data_root_prefix="pssd:",
        media_type="image",
        jump_filter=True
    ),
    # pretraining video datasets
    webvid_fuse_10m=dict(
        anno_path="/mnt/petrelfs/wangyi/share_data/vidata/webvid-fuse-10m.json", 
        data_root="",
        data_root_prefix="pssd:",
        media_type="video",
        jump_filter=True
    ),
    internvid_v1_40m=dict(
        anno_path="/mnt/petrelfs/wangyi/share_data/vidata/internvid1-40m.json",
        data_root="",
        data_root_prefix="phdd:",
        media_type="video",
        jump_filter=True
    ),
    # internvid_v2_avs_69m=dict(
    #     anno_path="/mnt/petrelfs/share_data/lixinhao/internvid2.s3path.success_avs.json",
    #     data_root="",
    #     media_type="audio_video",
    #     read_clip_from_video=False,
    #     read_audio_from_video=True,
    #     zero_audio_padding_for_video=True,
    #     caption_augmentation=dict(caption_sample_type='avs_all')
    # ),
    internvid_v2_avs_50m=dict( 
        anno_path="/mnt/petrelfs/share_data/heyinan/data_2023_h2/internvid2.s3path.success_avs_50M_new.json",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        zero_audio_padding_for_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all'),
        jump_filter=True
    ),
    internvid_v2_avs_43m_av=dict( 
        anno_path="/mnt/petrelfs/share_data/yujiashuo/data-h2/internvid2.s3path.success_av.json",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        zero_audio_padding_for_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all'),
        jump_filter=True
    ),
    # pretraining datasets
    internvid_recap=dict(
        anno_path="/mnt/petrelfs/share_data/videointern/annotations/anno_pretrain_lkc/internvid_0.6M_recap.json", 
        data_root="",
        media_type="video"
    ),
    smit=dict(
        anno_path=f"/mnt/petrelfs/share/videointern/metas/S-MiT/caption_train.json", 
        data_root="pvideo:s3://S-MiT",
        media_type="video"
    ),
    smit_av=dict(
        anno_path=f"/mnt/petrelfs/share/videointern/metas/S-MiT/caption_train.json", 
        data_root="pvideo:s3://S-MiT",
        media_type="audio_video",
        read_audio_from_video=False
    ),
    allseeing_short_2M=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/as-short-2m_lxh.json", 
        data_root="changeme:s3://SA-1B",
        media_type="image",
        caption_augmentation=dict(caption_sample_type='uniform')
    ),
    cc3m=dict(
        anno_path=f"{anno_root_pt}/cc3m_train.json", 
        data_root="pssd:s3://GCC",
        media_type="image"
    ),
    cc12m=dict(
        anno_path=f"{anno_root_pt}/cc12m_train.json", 
        data_root="pssd:s3://GCC/GCC12m",
        media_type="image"
    ),
    sbu=dict(
        anno_path=f"{anno_root_pt}/sbu.json", 
        data_root="pssd:s3://SBU/images",
        media_type="image"
    ),
    vg=dict(
        anno_path=f"{anno_root_pt}/vg.json", 
        data_root="pssd:s3://VG_dataset/images",
        media_type="image"
    ),
    coco_test=dict(
        # anno_path='/mnt/petrelfs/wangchenting/umtv2_stage2/coco_test_vast_only_first.json',
        # anno_path='/mnt/petrelfs/wangchenting/umtv2_stage2/coco_test_vast.json',
        anno_path=f"/mnt/petrelfs/share_data/wangchenting/coco_5k_test_final_concat.json", 
        # anno_path='/mnt/petrelfs/wangchenting/umtv2_stage2/coco_5k_test_final.json',
        data_root="pssd:s3://coco_caption",
        media_type="image"
    ),
    coco_test_4shot=dict(
        anno_path=f"/mnt/petrelfs/share_data/wangchenting/coco_5k_test_final_concat.json", 
        data_root="pssd:s3://coco_caption",
        media_type="image",
        few_shot_config=dict(
            num_shot=4,
            few_shot_template="{visual}{caption}",
            use_rice=False
        )
    ),

    coco=dict(
        anno_path=f"{anno_root_pt}/coco.json", 
        data_root="pssd:s3://coco_caption",
        media_type="image"
    ),
    imagenet1k=dict(
        anno_path=f"{anno_root_pt}/imagenet1k_train.json", 
        data_root="/mnt/petrelfs/share/images/train",
        media_type="image",
        prompt="imagenet",
    ),
    imagenet1kval=dict(
        anno_path='/mnt/petrelfs/wangchenting/working/imagenet/imagenet_1k_val.json',
        data_root='/mnt/petrelfs/share/images/val',
        media_type="image",
        prompt="imagenet",
    ),
    webvid=dict(
        anno_path=f"{anno_root_pt}/webvid_train.json", 
        data_root="pssd:s3://WebVid2M",
        media_type="video"
    ),
    webvid2m_jiaer=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/webvid2M_jiaer.json", 
        data_root="pssd:s3://WebVid2M",
        media_type="video"
    ),
    webvid_10m=dict(
        anno_path="/mnt/petrelfs/share_data/liyizhuo/datasets/annotations/anno_pretrain/webvid_10m_train_clean_230413.json",
        data_root="pssd:s3://WebVid10M",
        media_type="video",
    ),
    summarized_230415=dict(
        anno_path="/mnt/petrelfs/share_data/liyizhuo/datasets/annotations/anno_pretrain/video_caption_summarized_list/video_caption_summarized_list_230415_processed_resized.json",
        data_root="phdd:s3://aigc_videointernsegvideos_resize/",
        media_type="video"
    ),
    intern_3m=dict(
        anno_path="/mnt/petrelfs/share_data/likunchang/video_text/intern/intern_part3M.json",
        data_root="phdd:s3://aigc_videointernsegvideos_resize/",
        media_type="video"
    ),
    intern_3m_overlap=dict(
        anno_path="/mnt/petrelfs/share_data/likunchang/video_text/intern/intern_part3M_overlap.json",
        data_root="phdd:s3://aigc_videointernsegvideos_resize/",
        media_type="video"
    ),
    intern_10m=dict(
        anno_path="/mnt/petrelfs/share_data/likunchang/video_text/intern/intern_part10M.json",
        data_root="phdd:s3://aigc_videointernsegvideos_resize/",
        media_type="video"
    ),
    intern_10m_new=dict(
        anno_path="/mnt/petrelfs/share_data/liyizhuo/datasets/annotations/anno_pretrain/video_caption_summarized_list/resized_0602_200m_10m.json",
        data_root="",
        media_type="video"
    ),
    internv2_10m_avs=dict(
        anno_path="/mnt/petrelfs/share_data/lixinhao/annos/internvid2.s3path.success_avs_random10M.json",
        data_root="",
        media_type="video"
    ),
    internv2_10m_v=dict(
        anno_path="/mnt/petrelfs/share_data/lixinhao/annos/internvid2.s3path.success_v_random10M.json",
        data_root="",
        media_type="video"
    ),
    kinetics400=dict(
        anno_path=f"/mnt/petrelfs/lixinhao/lxh_exp/data/videoclip/anno_pretrain/kinetics400_train.json",
        data_root="/mnt/petrelfs/videointern/k400",
        media_type="video",
        prompt="prompt",
    ),
    kinetics710=dict(
        anno_path=f"/mnt/petrelfs/lixinhao/lxh_exp/data/videoclip/anno_pretrain/kinetics710_train.json",
        data_root="",
        media_type="video",
        prompt="prompt",
    ),
    kinetics710_raw=dict(
        anno_path=f"/mnt/petrelfs/lixinhao/lxh_exp/data/videoclip/anno_pretrain/kinetics710_raw_train.json",
        data_root="",
        media_type="only_video", # NOTE NotImplemented
    ),
    # downstream datasets.
    videocc3m_full=dict(
        anno_path=f"/mnt/petrelfs/share_data/likunchang/video_text/videocc3m_full.json", 
        data_root="pssd:s3://videocc3m",
        media_type="video",
    ),
    videocc3m_part=dict(
        anno_path=f"/mnt/petrelfs/share_data/likunchang/video_text/videocc3m_part2.5M.json", 
        data_root="pssd:s3://videocc3m",
        media_type="video",
    ),
    youtube_full=dict(
        anno_path=f"/mnt/petrelfs/share_data/likunchang/video_text/youtube_0407/youtube_full.json", 
        data_root="",
        media_type="video",
    ),
    youtube_part=dict(
        anno_path=f"/mnt/petrelfs/share_data/likunchang/video_text/youtube_0407/youtube_part2.5M.json", 
        data_root="",
        media_type="video",
    ),
    intern10mfiltered=dict(
        anno_path="phdd:s3://video_caption_summarized_list/resized_0613_filtered_30p_10m.json",
        data_root="",
        media_type="video"
    ),
    summarized_230613_filtered_30p_random2m=dict( # internvid2m-filtered
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/resized_0613_filtered_30p_10m_random2m.json",
        data_root="",
        media_type="video" # 2129654
    ),
    summarized_230930_filtered_30p_random2m=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/resized_0930_filtered_30p_10m_random2m.json",
        data_root="",
        media_type="video" # 2089651
    ),
    youtube_5M_avscap_av=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/youtube_5M_avscap_av.json",
        data_root="",
        media_type="video"
    ),
    youtube_5M_avscap_av_audio=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/youtube_5M_avscap_av.json",
        data_root="",
        media_type="audio_video"
    ),
    youtube_5M_avscap_av_audio_full_caption=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/youtube_5M_avscap_clean_full_caption.json",
        data_root="",
        media_type="audio_video"
    ),
    youtube_5M_avscap_a=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/youtube_5M_selected_audiocap_av.json",
        data_root="",
        media_type="video"
    ),
    youtube_5M_avscap_v=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/youtube_5M_selected_vidcap_v.json",
        data_root="",
        media_type="video"
    ),
    youtube_5M_vscap_v=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/youtube_5M_vs_captions_v.json",
        data_root="",
        media_type="video"
    ),
    # ytt_10M_av=dict( 还没更新jiashuo最新版本
    #     anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/data-h2/ytt_10M_avcap.json",
    #     data_root="",
    #     media_type="audio_video",
    #     read_clip_from_video=True,
    #     read_audio_from_video=True,
    #     caption_augmentation=dict(caption_sample_type='avs_all')
    # ),
    # ytt_10M_vs=dict(
    #     anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/data-h2/ytt_10M_vscap.json",
    #     data_root="",
    #     media_type="audio_video",
    #     read_clip_from_video=True,
    #     read_audio_from_video=True,
    #     caption_augmentation=dict(caption_sample_type='avs_all')
    # ),
    # ytt_10M_avs=dict(
    #     anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/data-h2/ytt_10M_avscap.json",
    #     data_root="",
    #     media_type="audio_video",
    #     read_clip_from_video=True,
    #     read_audio_from_video=True,
    #     caption_augmentation=dict(caption_sample_type='avs_all')
    # ),
    ytt_8M_avs=dict(
        anno_path="/mnt/petrelfs/share_data/heyinan/data_2023_h2/ytt_pack1_for_exp_avscap_new.json",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all')
    ),
    ytt_8M_av=dict(
        anno_path="/mnt/petrelfs/share_data/heyinan/data_2023_h2/ytt_pack1_for_exp_7.6m_avcap_new.json",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all')
    ),
    ytt_8M_vs=dict(
        anno_path="/mnt/petrelfs/share_data/heyinan/data_2023_h2/ytt_pack1_for_exp_7.6m_vscap_new.json",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all')
    ),
    ytt_8M_avs_all=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/ytt_pack1_for_exp_7.6m_avcaps_all.json",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all')
    ),
    ytt_2M_avs_all=dict(
        anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/internvid/ytt_pack1_for_exp_2m_avcaps_all.json",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all')
    ),
    # audio-text
    wavcaps_400k_1=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/WavCaps/json_files/AudioSet_SL/as_final.json",
        data_root="phdd:s3://yjsbucket2/WavCaps/flac/AudioSet_SL_flac/",
        media_type="audio"
    ),
    wavcaps_400k_2=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/WavCaps/json_files/BBC_Sound_Effects/bbc_final.json",
        data_root="phdd:s3://yjsbucket2/WavCaps/flac/BBC_Sound_Effects_flac/",
        media_type="audio"
    ),
    wavcaps_400k_3=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/WavCaps/json_files/FreeSound/fsd_final.json",
        data_root="phdd:s3://yjsbucket2/WavCaps/flac/FreeSound_flac/",
        media_type="audio"
    ),
    wavcaps_400k_4=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/WavCaps/json_files/SoundBible/sb_final.json",
        data_root="phdd:s3://yjsbucket2/WavCaps/flac/SoundBible_flac/",
        media_type="audio"
    ),
    # interleaved-video-text
    internvid2_interleaved_1M_avscap=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_avscap.json",
        data_root="",
        media_type="interleaved_video",
        max_length=1024,
        num_sample_clips=4
    ),
    internvid2_interleaved_1M_avscap_2clip=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_avscap.json",
        data_root="",
        media_type="interleaved_video",
        max_length=1024,
        num_sample_clips=2
    ),
    internvid2_interleaved_1M_avscap_l2048=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_avscap.json",
        data_root="",
        media_type="interleaved_video",
        max_length=2048
    ),
    internvid2_interleaved_1M_asrcap=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_asrcap.json",
        data_root="",
        media_type="interleaved_video",
        max_length=1024,
        num_sample_clips=4
    ),
    internvid2_interleaved_1M_asrcap_l2048=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_asrcap.json",
        data_root="",
        media_type="interleaved_video",
        max_length=2048
    ),
    internvid2_interleaved_1M_vcap=dict(
        anno_path="/mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_vcap.json",
        data_root="",
        media_type="interleaved_video",
        max_length=1024,
        num_sample_clips=4
    ),
    howtointern7m=dict(
        anno_path="/mnt/petrelfs/wangchenting/multimodalllm/HowtoInterlink7M.json",
        data_root="",
        media_type="interleaved_video",
        max_length=1024,
        num_sample_clips=4
    ),
    caption_sharegpt4v_420k = dict(
        anno_path=f"/mnt/petrelfs/share/videointern/annotations/image/caption/sharegpt4v/train_420k.json", 
        data_root="p2:s3://sharegpt4v/data",
    ),
    caption_sharegptvideo_300k = dict(
        # anno_path=f"/mnt/petrelfs/zhaojiahe/data/VideoChat2-PT/anno_pretrain/sharegptvideo_300k_obj_filter.json",
        anno_path=f"/mnt/petrelfs/zhaojiahe/data/VideoChat2-PT/anno_pretrain/sharegptvideo_300k_obj_gpt_filter.json",
        data_root="pvideo:s3://LLaVA_DPO/train_300k",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    caption_sharegptvideo_600k = dict(
        # anno_path=f"/mnt/petrelfs/zhaojiahe/data/VideoChat2-PT/anno_pretrain/sharegptvideo_600k.json",
        anno_path=f"/mnt/petrelfs/zhaojiahe/data/VideoChat2-PT/anno_pretrain/sharegptvideo_new600k_obj_gpt_filter.json",
        data_root="pvideo:s3://LLaVA_DPO/train_600k",
        media_type="video",
        video_reader_type="img", # read from image
    ),
    caption_llava = dict(
        # anno_path=f"/mnt/petrelfs/zhaojiahe/data/VideoChat2-PT/anno_pretrain/llava_obj_filter.json",
        anno_path=f"/mnt/petrelfs/zhaojiahe/data/VideoChat2-PT/anno_pretrain/llava_obj_gpt_filter.json",
        data_root="p2:s3://coco-caption",
        media_type="image",
    )
)

# composed datasets.

available_corpus["sharegptvideo300k"] = [
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["caption_llava"],
]

available_corpus["sharegptvideo900k"] = [
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["caption_sharegptvideo_600k"],
    available_corpus["caption_llava"]    
]

available_corpus["webvid10m_image15m_smit"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["smit"],
]

available_corpus["internvid2_interleaved_1M_avscap_smit"] = [
    available_corpus["internvid2_interleaved_1M_avscap"],
    available_corpus["smit"],
]

available_corpus["internvid2_interleaved_1M_avscap_smit_clip2"] = [
    available_corpus["internvid2_interleaved_1M_avscap_2clip"],
    available_corpus["smit"],
]

available_corpus["internvid2_interleaved_1M_asrcap_smit"] = [
    available_corpus["internvid2_interleaved_1M_asrcap"],
    available_corpus["smit"],
]

available_corpus["internvid2_interleaved_1M_vcap_smit"] = [
    available_corpus["internvid2_interleaved_1M_vcap"],
    available_corpus["smit"],
]

available_corpus["howtointern7M_avscap_smit"] = [
    available_corpus["howtointern7m"],
    available_corpus["smit"]
]

available_corpus["wavcaps_400k"] = [available_corpus["wavcaps_400k_1"], available_corpus["wavcaps_400k_2"], available_corpus["wavcaps_400k_3"], available_corpus["wavcaps_400k_4"]]

available_corpus["avp_debug"] = [available_corpus["ytt_2M_avs_all"], available_corpus["cc3m"], available_corpus["summarized_230613_filtered_30p_random2m"]]

available_corpus["avp_1b"] = [available_corpus["ytt_8M_avs_all"],
                              available_corpus["cc3m"],
                              available_corpus["coco"],
                              available_corpus["vg"],
                              available_corpus["sbu"],
                              available_corpus["cc12m"],
                              available_corpus["intern10mfiltered"]]

available_corpus["ytt_8M_avs_cc3m"] = [available_corpus["ytt_8M_avs"], available_corpus["cc3m"]]
available_corpus["webvid2m_internvid_filter2m_cc3m"] = [available_corpus["webvid"], available_corpus["summarized_230613_filtered_30p_random2m"], available_corpus["cc3m"]]
available_corpus["internvid_filter2m_cc3m"] = [available_corpus["summarized_230613_filtered_30p_random2m"], available_corpus["cc3m"]]
available_corpus["internvid_chat2m_cc3m"] = [available_corpus["summarized_230930_filtered_30p_random2m"], available_corpus["cc3m"]]
available_corpus["avscap5m_av_cc3m"] = [available_corpus["youtube_5M_avscap_av"], available_corpus["cc3m"]]
available_corpus["avscap5m_av_cc3m_audio"] = [available_corpus["youtube_5M_avscap_av_audio"], available_corpus["cc3m"]]
available_corpus["avscap5m_av_full_caption_cc3m_audio"] = [available_corpus["youtube_5M_avscap_av_audio_full_caption"], available_corpus["cc3m"]]
available_corpus["internvid2m_filtered_avscap5m_av_cc3m_audio"] = [available_corpus["youtube_5M_avscap_av_audio"], available_corpus["cc3m"], available_corpus["summarized_230613_filtered_30p_random2m"]]

available_corpus["avscap5m_v_cc3m"] = [available_corpus["youtube_5M_avscap_v"], available_corpus["cc3m"]]
available_corpus["vscap5m_v_cc3m"] = [available_corpus["youtube_5M_vscap_v"], available_corpus["cc3m"]]

available_corpus["coco_vg"] = [available_corpus["coco"], available_corpus["vg"]]
available_corpus["in1k_k710"] = [
    available_corpus["imagenet1k"],
    available_corpus["kinetics710"],
]
available_corpus["webvid_cc3m"] = [available_corpus["webvid"], available_corpus["cc3m"]]
available_corpus["webvid10m_cc3m_smit"] = [available_corpus["webvid_10m"], available_corpus["cc3m"], available_corpus["smit"]]
available_corpus["webvid_cc3m_smit"] = [available_corpus["webvid"], available_corpus["cc3m"], available_corpus["smit"]]
available_corpus["webvid2m_jiaer_cc3m"] = [available_corpus["webvid2m_jiaer"], available_corpus["cc3m"]]
available_corpus["cc3m_videocc3m_part"] = [
    available_corpus["cc3m"],
    available_corpus["videocc3m_part"]
]
available_corpus["cc3m_youtube_part"] = [
    available_corpus["cc3m"],
    available_corpus["youtube_part"]
]
available_corpus["cc3m_intern_part"] = [
    available_corpus["cc3m"],
    available_corpus["intern_3m"]
]
available_corpus["cc3m_intern_part_overlap"] = [
    available_corpus["cc3m"],
    available_corpus["intern_3m_overlap"]
]
available_corpus["webvid_cc3m_in1k_k710"] = [
    available_corpus["webvid"], 
    available_corpus["cc3m"],
    available_corpus["imagenet1k"],
    available_corpus["kinetics710"],
]
available_corpus["webvid_cc3m_k710raw"] = [
    available_corpus["webvid"], 
    available_corpus["cc3m"],
    available_corpus["kinetics710_raw"],
]
available_corpus["webvid_14m"] = [
    available_corpus["webvid"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]
available_corpus["webvid12m_14m"] = [
    available_corpus["webvid"],
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]
available_corpus["webvid10m_14m"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]
available_corpus["internv2_10m_avs_14m"] = [
    available_corpus["internv2_10m_avs"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]

available_corpus["internv2_10m_avs_smit"] = [
    available_corpus["internv2_10m_avs"],
    available_corpus["smit"]
]

available_corpus["internv2_10m_v_14m"] = [
    available_corpus["internv2_10m_v"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]
available_corpus["webvid10m_14m_smit"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["smit"]
]

available_corpus["webvid10m_3m"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
]
available_corpus["simple_17m"] = [
    available_corpus["webvid"],
    available_corpus["cc3m"],
    available_corpus["cc12m"],
]
available_corpus["simple_25m"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["cc12m"],
]
available_corpus["webvid10m_14m_our230415"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["summarized_230415"],
]
available_corpus["our230415_14m"] = [
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["summarized_230415"],
]
available_corpus["intern10m_14m"] = [
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["intern_10m"],
]
available_corpus["intern10m_new_14m"] = [
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["intern_10m_new"],
]

available_corpus["intern10mfiltered_webvid10m_14m"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["intern10mfiltered"]
]

available_corpus["intern10mfiltered_14m"] = [
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["intern10mfiltered"]
]

available_corpus["intern10mfiltered_14m_smit"] = [
    available_corpus["intern10mfiltered"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["smit"]
]

available_corpus["internvid_v1_40m_14m"] = [
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["internvid_v1_40m"]
]

available_corpus["webvid_fuse_10m_14m"] = [
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
    available_corpus["webvid_fuse_10m"]
]


# ============== for validation =================
available_corpus["msrvtt_1k_test"] = dict(
    anno_path=f"{anno_root_downstream}/msrvtt_test1k.json",
    # f"{data_root}/msrvtt_2fps_224",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="video"
)

available_corpus["msrvtt_1k_test_2shot"] = dict(
    anno_path=f"{anno_root_downstream}/msrvtt_test1k.json",
    # f"{data_root}/msrvtt_2fps_224",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="video",
    few_shot_config = dict(
        num_shot=2,
        few_shot_template="{visual}{caption}",
        use_rice=False
    )
)

available_corpus["msrvtt_1k_test_4shot"] = dict(
    anno_path=f"{anno_root_downstream}/msrvtt_test1k.json",
    # f"{data_root}/msrvtt_2fps_224",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="video",
    few_shot_config = dict(
        num_shot=4,
        few_shot_template="{visual}{caption}",
        use_rice=False
    )
)


available_corpus["msrvtt_1k_vs_test"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/MSRVTT/msrvtt_ret_test1k_new.json",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="video",
    use_subtitle=True
)

available_corpus["msrvtt_1k_vs_translate_test"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/MSRVTT/msrvtt_ret_test1k_translate.json",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="video",
    use_subtitle=True
)

available_corpus["msrvtt_1k_av_test"] = dict(
    anno_path=f"{anno_root_downstream}/msrvtt_test1k.json",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="audio_video",
    read_audio_from_video=True,
    zero_audio_padding_for_video=True
)

available_corpus["msrvtt_1k_avs_test"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/MSRVTT/msrvtt_ret_test1k_new.json",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="audio_video",
    read_audio_from_video=True,
    zero_audio_padding_for_video=True,
    use_subtitle=True
)

available_corpus["msrvtt_1k_avs_translate_test"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/MSRVTT/msrvtt_ret_test1k_translate.json",
    data_root="pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="audio_video",
    read_audio_from_video=True,
    zero_audio_padding_for_video=True,
    use_subtitle=True
)



available_corpus["ssv2_ret_label_train"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_label_train.json",
    data_root="sssd:s3://video_pub/ssv2_video",
    media_type="video",
    is_paragraph_retrieval=False,
    has_multi_vision_gt=True
)

available_corpus["ssv2_ret_label_val_small"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_label_val_small.json",
    data_root="sssd:s3://video_pub/ssv2_video",
    media_type="video",
    is_paragraph_retrieval=False,
    has_multi_vision_gt=True
)

available_corpus["ssv2_ret_template_train"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_template_train.json",
    data_root="sssd:s3://video_pub/ssv2_video",
    media_type="video",
    is_paragraph_retrieval=False,
    has_multi_vision_gt=True
)

available_corpus["ssv2_ret_template_val_small"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_template_val_small.json",
    data_root="sssd:s3://video_pub/ssv2_video",
    media_type="video",
    is_paragraph_retrieval=False,
    has_multi_vision_gt=True
)

available_corpus["anet_ret_train"] = dict(
    anno_path=f"{anno_root_downstream}/anet_ret_train.json",
    data_root="sssd:s3://video_pub/ANet_320p_fps30/train",
    media_type="video",
    is_paragraph_retrieval=True,
    max_txt_l = 150
)

available_corpus["anet_ret_val"] = dict(
    anno_path=f"{anno_root_downstream}/anet_ret_val.json",
    data_root="sssd:s3://video_pub/ANet_320p_fps30/val",
    media_type="video",
    is_paragraph_retrieval=True,
    max_txt_l = 150
)

available_corpus["anet_ret_val_4shot"] = dict(
    anno_path=f"{anno_root_downstream}/anet_ret_val.json",
    data_root="sssd:s3://video_pub/ANet_320p_fps30/val",
    media_type="video",
    few_shot_config = dict(
        num_shot=4,
        few_shot_template="{visual}{caption}",
        use_rice=False
    )
)

available_corpus["didemo_ret_train"] = dict(
    anno_path=f"{anno_root_downstream}/didemo_ret_train.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64 # TODO RetTrain其实应该还没支持设置这个参数
)

available_corpus["didemo_ret_av_train"] = dict(
    anno_path=f"{anno_root_downstream}/didemo_ret_train.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="audio_video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64, # TODO RetTrain其实应该还没支持设置这个参数
    read_audio_from_video=True,
    zero_audio_padding_for_video=True
)

available_corpus["didemo_ret_val"] = dict(
    anno_path=f"{anno_root_downstream}/didemo_ret_val.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64
)

available_corpus["didemo_ret_test"] = dict(
    anno_path=f"{anno_root_downstream}/didemo_ret_test.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64
)

available_corpus["didemo_ret_vs_test"] = dict(
    anno_path=f"/mnt/petrelfs/share_data/yujiashuo/anno_downstream/DiDeMo/didemo_ret_test_new.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64,
    use_subtitle=True
)

available_corpus["didemo_ret_vs_translate_test"] = dict(
    anno_path=f"/mnt/petrelfs/share_data/yujiashuo/anno_downstream/DiDeMo/didemo_ret_test_translate.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64,
    use_subtitle=True
)

available_corpus["didemo_ret_av_test"] = dict(
    anno_path=f"{anno_root_downstream}/didemo_ret_test.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="audio_video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    read_audio_from_video=True,
    zero_audio_padding_for_video=True,
    max_txt_l=64
)

available_corpus["didemo_ret_avs_translate_test"] = dict(
    anno_path=f"/mnt/petrelfs/share_data/yujiashuo/anno_downstream/DiDeMo/didemo_ret_test_translate.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="audio_video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    read_audio_from_video=True,
    zero_audio_padding_for_video=True,
    max_txt_l=64,
    use_subtitle=True
)

available_corpus["didemo_ret_avs_test"] = dict(
    anno_path=f"/mnt/petrelfs/share_data/yujiashuo/anno_downstream/DiDeMo/didemo_ret_test_new.json",
    data_root=f"sssd:s3://yjsBucket/DiDeMo",
    media_type="audio_video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    read_audio_from_video=True,
    zero_audio_padding_for_video=True,
    max_txt_l=64,
    use_subtitle=True
)

available_corpus["lsmdc_ret_train"] = dict(
    anno_path=f"{anno_root_downstream}/lsmdc_ret_train.json",
    data_root=f"sssd:s3://video_pub/LSMDC",
    media_type="video",
    max_txt_l=96
)

available_corpus["lsmdc_ret_val"] = dict(
    anno_path=f"{anno_root_downstream}/lsmdc_ret_val.json",
    data_root=f"sssd:s3://video_pub/LSMDC",
    media_type="video",
    max_txt_l=96
)

available_corpus["lsmdc_ret_test_1000"] = dict(
    anno_path=f"{anno_root_downstream}/lsmdc_ret_test_1000.json",
    data_root=f"sssd:s3://video_pub/LSMDC",
    media_type="video",
    max_txt_l=96
)

available_corpus["msrvtt_ret_train9k"] = dict(
    anno_path=f"{anno_root_downstream}/msrvtt_ret_train9k.json",
    data_root=f"pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="video",
)

available_corpus["msrvtt_ret_av_train9k"] = dict(
    anno_path=f"{anno_root_downstream}/msrvtt_ret_train9k.json",
    data_root=f"pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="audio_video",
    read_audio_from_video=True,
    zero_audio_padding_for_video=True
)

available_corpus["msrvtt_ret_test1k"] = dict(
    anno_path=f"{anno_root_downstream}/msrvtt_ret_test1k.json",
    data_root=f"pssd:s3://MSR-VTT/MSRVTT_Videos",
    media_type="video",
)

available_corpus["msvd_ret_train"] = dict(
    anno_path=f"{anno_root_downstream}/msvd_ret_train.json",
    data_root=f"sssd:s3://video_pub/MSVD/MSVD_Videos",
    media_type="video",
    max_txt_l=64,
    has_multi_txt_gt=True
)

available_corpus["msvd_ret_val"] = dict(
    anno_path=f"{anno_root_downstream}/msvd_ret_val.json",
    data_root=f"sssd:s3://video_pub/MSVD/MSVD_Videos",
    media_type="video",
    max_txt_l=64
)

available_corpus["msvd_ret_test"] = dict(
    anno_path=f"{anno_root_downstream}/msvd_ret_test.json",
    data_root=f"sssd:s3://video_pub/MSVD/MSVD_Videos",
    media_type="video",
    max_txt_l=64
)

available_corpus["ssv2_ret_label_train"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_label_train.json",
    data_root="pssd:s3://ssv2_video",
    media_type="video",
    max_txt_l=25,
    has_multi_vision_gt=True
)


available_corpus["ssv2_ret_label_val"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_label_val_small.json",
    data_root="pssd:s3://ssv2_video",
    media_type="video",
    max_txt_l=25,
    has_multi_vision_gt = True
)

available_corpus["ssv2_ret_template_train"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_template_train.json",
    data_root="pssd:s3://ssv2_video",
    media_type="video",
    max_txt_l=25,
    has_multi_vision_gt=True
)


available_corpus["ssv2_ret_template_val"] = dict(
    anno_path=f"{anno_root_downstream}/ssv2_ret_template_val_small.json",
    data_root="pssd:s3://ssv2_video",
    media_type="video",
    max_txt_l=25,
    has_multi_vision_gt=True
)

available_corpus["vatex_en_ret_train"] = dict(
    anno_path="/mnt/petrelfs/share/videointern/annotations/anno_downstream_lkc/vatex_en_ret_train.json",
    data_root="",
    data_root_prefix="sssd:",
    media_type="video",
    has_multi_txt_gt=True
)


available_corpus["vatex_en_ret_val"] = dict(
    anno_path="/mnt/petrelfs/share/videointern/annotations/anno_downstream_lkc/vatex_en_ret_val.json",
    data_root="",
    data_root_prefix="sssd:",
    media_type="video"
)

available_corpus["youcook2_ret_train"] = dict(
    anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/youcook2/youcook2_cap_train.json",
    data_root="pssd:s3://vast-ft-dataset/srcdata/youcook/videos",
    data_root_prefix="",
    media_type="video"
)

available_corpus["youcook2_ret_test"] = dict(
    anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/youcook2/youcook2_cap_test.json",
    data_root="pssd:s3://vast-ft-dataset/srcdata/youcook/videos",
    data_root_prefix="",
    media_type="video"
)

available_corpus["youcook2_ret_merge_subtitle_test"] = dict(
    anno_path="/mnt/petrelfs/lixinhao/lxh_exp/data/youcook2/youcook2_cap_test_merge_subtitle.json",
    data_root="pssd:s3://vast-ft-dataset/srcdata/youcook/videos",
    data_root_prefix="",
    media_type="video"
)

# audio-text

available_corpus["audiocaps_ret_train"] = dict(
    anno_path=f"/mnt/petrelfs/share_data/yujiashuo/anno_downstream/audiocaps/audiocaps_ret_trainval.json",
    data_root="phdd:s3://yjsbucket2/audiocaps/full/",
    media_type="audio",
)

available_corpus["audiocaps_ret_test"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/audiocaps/audiocaps_ret_test.json",
    data_root="phdd:s3://yjsbucket2/audiocaps/full/",
    media_type="audio",
)


available_corpus["clothov1_ret_train"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/clotho_v1/clotho_ret_development.json",
    data_root="phdd:s3://yjsbucket2/clotho/clotho_audio_development/development/",
    media_type="audio",
)

available_corpus["clothov1_ret_test"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/clotho_v1/clotho_ret_evaluation.json",
    data_root="phdd:s3://yjsbucket2/clotho/clotho_audio_evaluation/evaluation/",
    media_type="audio",
)

available_corpus["clothov2_ret_train"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/clotho_v2/clotho_ret_development.json",
    data_root="phdd:s3://yjsbucket2/clotho/clotho_audio_development/development/",
    media_type="audio",
)

available_corpus["clothov2_ret_test"] = dict(
    anno_path="/mnt/petrelfs/share_data/yujiashuo/anno_downstream/clotho_v2/clotho_ret_evaluation.json",
    data_root="phdd:s3://yjsbucket2/clotho/clotho_audio_evaluation/evaluation/",
    media_type="audio",
)

available_corpus["debug"] = dict(
    anno_path=f"/mnt/petrelfs/share_data/lixinhao/smit_debug.json",
    data_root="pvideo:s3://S-MiT",
    media_type="video"
)