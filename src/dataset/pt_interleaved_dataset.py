import logging
import os
import json
import random
import io
import torch
import numpy as np
import random

from .utils import pre_text
from .base_dataset import BaseDataset
from .video_utils import VIDEO_READER_FUNCS
from ..share_utils.serialize import get_local_rank, TorchShmSerializedList

DEFAULT_IMG_TOKEN = "[IMG]"
DEFAULT_IMG_END_TOKEN = "[/IMG]"

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "[VIDEO]"

IMG_TOKEN = "[<IMG_PLH>]"
VID_TOKEN = "[<VID_PLH>]"

USER_ROLE = "USER: "
ASSISTANT_ROLE = "ASSISTANT: "

all_image_caption_prompts = [
    'Describe the following image concisely. {visual_token}',
    'Provide a brief description of the given image. {visual_token}',
    'Offer a succinct explanation of the picture presented. {visual_token}',
    'Summarize the visual content of the following image. {visual_token}',
    'Give a short and clear explanation of the subsequent image. {visual_token}',
    'Share a concise interpretation of the image provided. {visual_token}',
    'Present a compact description of the photo\'s key features. {visual_token}',
    'Relay a brief, clear account of the picture shown. {visual_token}',
    'Render a clear and concise summary of the photo below. {visual_token}',
    'Write a terse but informative summary of the following picture. {visual_token}',
    'Create a compact narrative representing the image presented. {visual_token}',
]

all_image_caption_prompts_with_role = [
    USER_ROLE+prompt+' \n' + ASSISTANT_ROLE for prompt in all_image_caption_prompts
]

all_video_caption_prompts = [
    'Describe the following video concisely. {visual_token}',
    'Provide a brief description of the given video clip. {visual_token}',
    'Offer a succinct explanation of the footage presented. {visual_token}',
    'Summarize the visual content of the following video. {visual_token}',
    'Give a short and clear explanation of the subsequent video clip. {visual_token}',
    'Share a concise interpretation of the video provided. {visual_token}',
    'Present a compact description of the clip\'s key features. {visual_token}',
    'Relay a brief, clear account of the video shown. {visual_token}',
    'Render a clear and concise summary of the video below. {visual_token}',
    'Write a terse but informative summary of the following video clip. {visual_token}',
    'Create a compact narrative representing the video presented. {visual_token}',
]

all_video_caption_prompts_with_role = [
    USER_ROLE+prompt+' \n' + ASSISTANT_ROLE for prompt in all_video_caption_prompts
]

all_asr_prompts = [
    "{visual_token} Text of automatic speech recognition: ",
    "{visual_token} Text transcription of the audio speech segment: "
    "{visual_token} Subtitle of this video: "
]

all_simple_prompts = [
    "{visual_token}"
]

'''
example from /mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_asrcap.json:
{'long_video_id': 'aOqaEiyqliY', 'interleaved_list': [{'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_0_70.mp4', 'caption': "it's getting very", 'video_start_frame': 0}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_70_372.mp4', 'caption': 'real now the world cup squad have arrived in Brazil Roy Hodgson side landed in rio de janeiro on sunday after a flight from miami whether it be based for the past week the squad', 'video_start_frame': 70}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_372_625.mp4', 'caption': 'looks relaxed getting off the plane wearing polo shirts reflecting a change of mood from four years ago what', 'video_start_frame': 372}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_625_847.mp4', 'caption': "if the last off was Wayne Rooney there was real pressure on him to perform here if he's ever to be considered one of England's all-time great the", 'video_start_frame': 625}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_847_996.mp4', 'caption': 'only injury concern is Alex oxlade-chamberlain who has a knee ligament problem the', 'video_start_frame': 847}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_996_1160.mp4', 'caption': 'squad have another flight to catch before their opening match in Group D as they face Italy in Manaus which is', 'video_start_frame': 996}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_1160_1261.mp4', 'caption': 'four hours on a plane northwest of Rio England then', 'video_start_frame': 1160}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_1261_1525.mp4', 'caption': 'face Uruguay in são Paulo before taking on costa rica in Belo Horizonte', 'video_start_frame': 1261}]}
'''
logger = logging.getLogger(__name__)


class InterleavedVidTxtPtTrainDataset(BaseDataset):
    media_type = "video"

    def __init__(self, 
                 ann_file,
                 transform,
                 tokenizer=None,
                 num_frames:int = 4,
                 video_reader_type="decord",
                 sample_type="rand",
                 num_tries=3
        ):
        super().__init__()

        logger.info(f"ann_file: {ann_file}")
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.max_length = ann_file.get("max_length", 1024)
        self.tokenizer = tokenizer
        self.media_type = ann_file.media_type
        self.label_file = ann_file.anno_path
        self.data_root = ann_file.data_root
        self.data_root_prefix = ann_file.get("data_root_prefix", "")
        self.num_sample_clips = ann_file.get("num_sample_clips", 4)
        self.fix_num_sample_clips = ann_file.get("fix_num_sample_clips", True)
        assert self.fix_num_sample_clips, "Now only support fix sample num clips!!!"
        if self.fix_num_sample_clips:
            self.min_num_clips = ann_file.get("min_num_clips", self.num_sample_clips)
        else:
            self.min_num_clips = ann_file.get("min_num_clips", 0)
        # self.clip_sample_interval = ann_file.get("clip_sample_interval", 1) # TODO not implement now
        # assert self.clip_sample_interval == 1, self.clip_sample_interval
        self.same_prompt_for_every_clip = ann_file.get("same_prompt_for_every_clip", True) # NOTE 拍脑袋想的参数
        self.reverse_ratio = ann_file.get("reverse_ratio", 0.5)
        self.start_prompt = ann_file.get("start_prompt", "")

        logger.info(f"num_sample_clips: {self.num_sample_clips}, fix_num_sample_clips: {self.fix_num_sample_clips}, min_num_clips: {self.min_num_clips}, same_prompt_for_every_clip: {self.same_prompt_for_every_clip}, reverse_ratio: {self.reverse_ratio}, start_prompt: {self.start_prompt}")

        self.transform = transform

        self.use_prompt = ann_file.get("prompt", "") != ""

        if self.use_prompt == "image_caption":
            self.prompt = all_image_caption_prompts
            logger.info(f"Use prompt for ImageNet")
        elif self.use_prompt == "video_caption":
            self.prompt = all_video_caption_prompts
            logger.info(f"Use prompt for Kinetics")
        elif self.use_prompt == "asr":
            self.prompt = all_asr_prompts
        else:
            self.prompt = all_simple_prompts
        logger.info(self.prompt)

        if '.json' in self.label_file:
            logger.info(f"Loading json file {self.label_file}")

            if get_local_rank() == 0:  # Only one rank need to read the file
                with io.BytesIO(self.client.get(self.label_file)) as f:
                # with open(self.label_file, 'r') as f:
                    old_annos = json.load(f)

                if ann_file.get("jump_filter", False):
                    logger.info("Jump filter!")
                else:
                    # Leave filter for data prepare
                    pass

                clips_num = []
                annos = []
                for anno in old_annos:
                    clips_num.append(len(anno['interleaved_list']))
                    if len(anno['interleaved_list']) >= self.min_num_clips:
                        annos.append(anno)

                logger.info(f"origin num_examples: {len(old_annos)}, num_clips: {sum(clips_num)}, mean num_clips: {(sum(clips_num) / len(clips_num)) if len(clips_num) != 0 else 0}, min num_clips:{min(clips_num)}, max num_clips:{max(clips_num)}")
            else:
                annos = []
            

            # self.anno = annos
            self.anno = TorchShmSerializedList(annos)
            self.num_examples = len(self.anno)
            
            logger.info(f"num_examples: {self.num_examples}")

        else:
            raise NotImplementedError("We need json file!!!")


    def __len__(self):
        return self.num_examples

    def sample_clips_from_video(self, index):
        video_anno = self.anno[index]
        if self.fix_num_sample_clips:
            num_samples = self.num_sample_clips
        else:
            num_samples = min(len(video_anno['interleaved_list']), self.num_sample_clips)
        start_idx = 0 + random.randint(0, len(video_anno['interleaved_list']) - num_samples + 1)
        clips_anno = video_anno['interleaved_list'][start_idx:start_idx+num_samples]

        return clips_anno


    def __getitem__(self, index):
        try:
            clips_anno = self.sample_clips_from_video(index)
            clips = []

            first_text = self.tokenizer.bos_token + self.start_prompt

            tokenized = self.tokenizer.build_input_ids(
                text=[first_text],
                max_length=self.max_length,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors='pt'
            )

            full_text = ''

            for clip_anno in clips_anno:
                clip, _ = self.load_and_transform_media_data(index, clip_anno["video"])
                clips.append(clip)
                full_text += VID_TOKEN                    
                full_text += pre_text(clip_anno["caption"])

            # full_text = full_text.replace(f"{VID_TOKEN} ", f"{VID_TOKEN}").replace(f" {VID_TOKEN} ", f"{VID_TOKEN}")
            full_text += self.tokenizer.eos_token

            toregressed = self.tokenizer.build_input_ids(
                text=[full_text],
                max_length=self.max_length - tokenized['input_ids'].shape[1],
                add_special_tokens=False,
                truncation=True,
                padding="max_length",
                return_tensors='pt'
            )
            
            # breakpoint()
            input_ids = torch.cat([tokenized['input_ids'], toregressed['input_ids']], dim=1)
            attention_mask = torch.cat([torch.ones_like(tokenized['attention_mask']), toregressed['attention_mask']], dim=1)
            labels = torch.clone(input_ids[0])
            labels[:tokenized['input_ids'][0].shape[0]] = -100
            labels[labels == 32001] = -100
            labels[labels == 32007] = -100
            labels[labels == 32002] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                'input_ids': input_ids[0], 
                'attention_mask': attention_mask[0], 
                'labels': labels, 
                'video': torch.stack(clips)
            }
        
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading clips_anno {clips_anno}")
            print(e)
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)



if __name__ == '__main__':
    from ..tokenizer.multimodal_llama_tokenizer import MultimodalLlamaTokenizer
    tokenizer = MultimodalLlamaTokenizer.from_pretrained("BAAI/Emu2", local_files_only=True) 
    from easydict import EasyDict as edict
    from .create import get_train_transform
    Myd = InterleavedVidTxtPtTrainDataset(
        ann_file=edict(
            anno_path="/mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_avscap.json",
            data_root="",
            media_type="interleaved_video",
            max_length=2048
        ),
        transform=get_train_transform(None, None),
        tokenizer=tokenizer
    )
    data = Myd[0]