import logging
import os
import json
import random
import io
import torch
import numpy as np
import random

from .base_dataset import BaseDataset
from .utils import pre_text
from .video_utils import VIDEO_READER_FUNCS
from ..share_utils.serialize import get_local_rank, TorchShmSerializedList

logger = logging.getLogger(__name__)

IMG_TOKEN = "[<IMG_PLH>]"
VID_TOKEN = "[<VID_PLH>]"

all_image_caption_prompts = [
    'Describe the following image concisely.',
    'Provide a brief description of the given image.',
    'Offer a succinct explanation of the picture presented.',
    'Summarize the visual content of the following image.',
    'Give a short and clear explanation of the subsequent image.',
    'Share a concise interpretation of the image provided.',
    'Present a compact description of the photo\'s key features.',
    'Relay a brief, clear account of the picture shown.',
    'Render a clear and concise summary of the photo below.',
    'Write a terse but informative summary of the following picture.',
    'Create a compact narrative representing the image presented.',
]

all_video_caption_prompts = [
    'Describe the following video concisely.',
    'Provide a brief description of the given video clip.',
    'Offer a succinct explanation of the footage presented.',
    'Summarize the visual content of the following video.',
    'Give a short and clear explanation of the subsequent video clip.',
    'Share a concise interpretation of the video provided.',
    'Present a compact description of the clip\'s key features.',
    'Relay a brief, clear account of the video shown.',
    'Render a clear and concise summary of the video below.',
    'Write a terse but informative summary of the following video clip.',
    'Create a compact narrative representing the video presented.',
]


class ImageCapEvalDataset(BaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform, tokenizer=None, num_tries=1):
        super().__init__()

        logger.info(f"ann_file: {ann_file}")
        self.max_length = ann_file.get("max_length", 512)
        self.tokenizer = tokenizer
        self.media_type = ann_file.media_type
        self.label_file = ann_file.anno_path
        self.data_root = ann_file.data_root
        self.data_root_prefix = ann_file.get("data_root_prefix", "")
        self.min_caption_length = ann_file.get("min_caption_length", 2)
        self.few_shot_config = ann_file.get("few_shot_config", None)
        logger.info(f"few_shot_config: {self.few_shot_config}")
        """
        few_shot_config = dict(
            num_shot=2,
            few_shot_template="Caption: {caption}",
            use_rice=False,
            rice_encoder="./assets/openai/clip-vit-large-patch14",
            cached_features_path=None
        )
        """
        self.transform = transform

        self.use_prompt = ann_file.get("use_prompt", True)
        if self.use_prompt:
            if self.media_type == "image":
                self.prompt = all_image_caption_prompts
            elif self.media_type == "video":
                self.prompt = all_video_caption_prompts
            else:
                raise NotImplementedError(self.media_type)
        else:
            self.prompt = [""]

        logger.info("Use prompt:")
        logger.info(self.prompt)

        self.role = ["", ""]
        logger.info("Use role:")
        logger.info(self.role)


        if '.json' in self.label_file:
            logger.info(f"Loading json file {self.label_file}")

            if get_local_rank() == 0:  # Only one rank need to read the file
                with io.BytesIO(self.client.get(self.label_file)) as f:
                # with open(self.label_file, 'r') as f:
                    annos = json.load(f)

                if type(annos[0]['caption']) is list:
                    logger.info(f"Num videos before flatten captions of same video: {len(annos)}")
                    old_annos = annos
                    annos = []
                    for anno in old_annos:
                        for cap in anno['caption']:
                            annos.append({self.media_type:anno[self.media_type], 'caption':cap})

                if ann_file.get("jump_filter", False):
                    logger.info("Jump filter!")
                else:
                    # filter out the caption with length less than min_caption_length
                    captions = [pre_text(anno["caption"]) for anno in annos]
                    captions_len = [len(caption.split()) for caption in captions]
                    logger.info("Num samples: {}".format(len(captions)))
                    logger.info("Num samples too short: {}".format(sum([l < self.min_caption_length for l in captions_len])))
                    annos = [anno for anno, l in zip(annos, captions_len) if l >= self.min_caption_length]

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

    def get_caption(self, index):
        if '.json' in self.label_file:
            caption = self.anno[index]["caption"]
        else:
            raise NotImplementedError

        return caption

    def get_anno(self, index):
        assert self.media_type == 'image', self.media_type
        anno = {"caption": self.get_caption(index)}
        anno["image"] = self.data_root_prefix + os.path.join(self.data_root, self.anno[index]["image"])
        return anno

    def pre_caption(self, caption):
        if type(caption) is str:
            return pre_text(caption)
        elif type(caption) is dict:
            caption_dict = {}
            for k in caption.keys():
                caption_dict[k] = pre_text(caption[k])
            return caption_dict
        else:
            raise NotImplementedError(caption)

    def get_few_shot_samples(self, query_image=None):
        if self.few_shot_config.get("num_shot", 4) == 0:
            return []
        if self.few_shot_config.get("use_rice", False):
            raise NotImplementedError
            samples = self.rice.find(query_image, self.few_shot_config.get("num_shot", 4))[0]
        else:
            idxs = random.sample(
                list(range(self.num_examples)),
                self.few_shot_config.get("num_shot", 4)
            )
            samples = [self.get_anno(i) for i in idxs]

        return samples
    
    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            caption = self.pre_caption(ann["caption"])
            image, index = self.load_and_transform_media_data(index, ann["image"])
            
            unified_tokens = self.tokenizer.bos_token + "<|im_start|>user\n" + "Describe the following image concisely" +"<img>" + IMG_TOKEN + "</img>" + "<|im_end|>\n" + "<|im_start|>assistant\n"
            
            tokenized = self.tokenizer.build_input_ids(
                text=[unified_tokens],
                max_length=self.max_length,
                add_special_tokens=False,
                truncation=False,
                require_image=True,
                padding='longest',
                return_tensors='pt'
            )
            
            return {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'image': image,
                'image_id': ann["image"],
                'caption': caption,
                'image_idx': tokenized['image_index']
            }
        
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading image {ann}")
            # raise e
            print(e)
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)



class VideoCapEvalDataset(ImageCapEvalDataset):
    media_type = "video"

    def __init__(
        self,
        ann_file,
        transform,
        tokenizer=None,
        num_frames:int = 4,
        video_reader_type="decord",
        sample_type="rand",
        num_tries=1
    ):
        super().__init__(ann_file, transform, tokenizer=tokenizer)
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.read_clip_from_video = ann_file.get("read_clip_from_video", False)
        

    def get_anno(self, index):
        assert self.media_type == "video", self.media_type
        anno = {"caption": self.get_caption(index)}
        anno["video"] = self.data_root_prefix + os.path.join(self.data_root, self.anno[index]["video"])
        if self.read_clip_from_video:
            anno["video_start_frame"] = self.anno[index]["video_start_frame"]
            anno["video_end_frame"] = self.anno[index]["video_end_frame"]

        return anno

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            caption = self.pre_caption(ann["caption"])
            
            if self.read_clip_from_video:
                data_path = {
                    "video": ann["video"],
                    "video_start_frame": ann["video_start_frame"],
                    "video_end_frame": ann["video_end_frame"],
                    "read_clip_from_video": True
                }
            else:
                data_path = ann["video"]
            video, index = self.load_and_transform_media_data(index, data_path)
            
            unified_tokens = self.tokenizer.bos_token + "<|im_start|>user\n" + "Describe the following video concisely" + "<vid>" + VID_TOKEN + "</vid>" + "<|im_end|>\n" + "<|im_start|>assistant\n"
            
            tokenized = self.tokenizer.build_input_ids(
                text=[unified_tokens],
                max_length=self.max_length,
                add_special_tokens=False,
                truncation=False,
                require_video=True,
                padding='longest',
                return_tensors='pt'
            )

            return {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'video': video,
                'image_id': data_path,
                'caption': caption,
                'video_idx': tokenized['video_index']
            }
        
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading video {ann}")
            print(e)
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)