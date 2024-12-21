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


class ImgTxtPtTrainDataset(BaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform, tokenizer=None):
        super().__init__()

        logger.info(f"ann_file: {ann_file}")
        self.max_length = ann_file.get("max_length", 512)
        self.tokenizer = tokenizer
        self.media_type = ann_file.media_type
        self.label_file = ann_file.anno_path
        self.data_root = ann_file.data_root
        self.data_root_prefix = ann_file.get("data_root_prefix", "")
        self.min_caption_length = ann_file.get("min_caption_length", 2)
        self.caption_augmentation = ann_file.get("caption_augmentation", None)
        self.transform = transform

        self.use_prompt = True
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

        if '.json' in self.label_file:
            logger.info(f"Loading json file {self.label_file}")

            if get_local_rank() == 0:  # Only one rank need to read the file
                with io.BytesIO(self.client.get(self.label_file)) as f:
                # with open(self.label_file, 'r') as f:
                    annos = json.load(f)

                if ann_file.get("jump_filter", False):
                    logger.info("Jump filter!")
                else:
                    if self.caption_augmentation is not None:
                        # filter out the caption with length less than min_caption_length
                        new_annos = []
                        if self.media_type == "audio_video" and self.caption_augmentation.caption_sample_type == 'avs_all':
                            for anno in annos:
                                ok = True
                                if not anno['video'].endswith('.mp4'): 
                                    ok = False
                                for k in anno.keys():
                                    if "caption" in k and 'asr' not in k:# TODO 避免因为asr扔多了
                                        tmp_c = pre_text(anno[k])
                                        if len(tmp_c.split()) < self.min_caption_length: 
                                            ok = False # NOTE 有一个不达标的caption就扔了
                                            break
                                if ok:
                                    new_annos.append(anno)
                        elif self.caption_augmentation.caption_sample_type == 'uniform':
                            for anno in annos:
                                if "captions" in anno.keys():
                                    caption_key = "captions"
                                else:
                                    caption_key = "caption"

                                assert type(anno[caption_key]) is list, type(anno[caption_key])
                                caption_list = []  # NOTE 用captions来区分caption
                                for c in anno[caption_key]:
                                    tmp_c = pre_text(c)
                                    if len(tmp_c.split()) >= self.min_caption_length:
                                        caption_list.append(tmp_c)

                                if len(caption_list) > 0:
                                    new_annos.append(anno)
                        else:
                            raise NotImplementedError(ann_file)
                        
                        logger.info(f"Num samples: {len(annos)}")
                        logger.info(f"Num samples not too short: {len(new_annos)} min_caption_length={self.min_caption_length}")
                        annos = new_annos
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
            if self.caption_augmentation is not None:
                if self.caption_augmentation.caption_sample_type == 'avs_all':
                    caption_dict = {}
                    for k in self.anno[index].keys():
                        if 'caption' in k:
                            caption_dict[k] = self.anno[index][k]
                else:
                    if "captions" in self.anno[index].keys():
                        captions = self.anno[index]["captions"]
                    else:
                        captions = self.anno[index]["caption"]
            else:
                caption = self.anno[index]["caption"]
        else:
            raise NotImplementedError

        if self.caption_augmentation is not None:
            if self.caption_augmentation.caption_sample_type == 'uniform':
                caption = random.choice(captions)
            elif self.caption_augmentation.caption_sample_type == 'avs_all':
                caption = caption_dict # NOTE 直接传一个dict
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
            assert self.caption_augmentation.caption_sample_type == 'avs_all'
            caption_dict = {}
            for k in caption.keys():
                caption_dict[k] = pre_text(caption[k])
            return caption_dict
        else:
            raise NotImplementedError(caption)


    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            caption = self.pre_caption(ann["caption"])
            image, index = self.load_and_transform_media_data(index, ann["image"])
            
            unified_tokens = self.tokenizer.bos_token + "<|im_start|>user\n" + random.choice(self.prompt) +'<img>' + IMG_TOKEN + '</img>' + "<|im_end|>\n" + "<|im_start|>assistant\n"
            
            tokenized = self.tokenizer.build_input_ids(
                text=[unified_tokens],
                image=[image],
                max_length=self.max_length,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors='pt'
            )
            
            toregressed = self.tokenizer.build_input_ids(
                text=[caption + "<|im_end|>\n" + '</s>'],
                max_length=self.max_length - tokenized['input_ids'].shape[0],
                add_special_tokens=False,
                truncation=True,
                padding="max_length",
                return_tensors='pt'
            )
            
            image_idx = torch.cat([tokenized['image_index'], torch.zeros_like(toregressed['input_ids'])])
            input_ids = torch.cat([tokenized['input_ids'], toregressed['input_ids']])
            attention_mask = torch.cat([torch.ones_like(tokenized['attention_mask']), toregressed['attention_mask']])
            labels = torch.clone(input_ids)
            labels[:tokenized['input_ids'].shape[0]] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                'input_ids': input_ids, 
                'attention_mask': attention_mask, 
                'labels': labels, 
                'image_idx': image_idx,
                'image': tokenized['image'][0],
            }
        
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading image {ann}")
            # raise e
            print(e)
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)

class VidTxtPtTrainDataset(ImgTxtPtTrainDataset):
    media_type = "video"

    def __init__(
        self,
        ann_file,
        transform,
        tokenizer=None,
        num_frames:int = 4,
        video_reader_type="decord",
        sample_type="rand",
        num_tries=3
    ):
        super().__init__(ann_file, transform)
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.max_length = ann_file.get("max_length", 512)
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

            unified_tokens = self.tokenizer.bos_token + "<|im_start|>user\n" + random.choice(self.prompt) + '<vid>' + VID_TOKEN + '</vid>' + "<|im_end|>\n" + "<|im_start|>assistant\n"

            tokenized = self.tokenizer.build_input_ids(
                text=[unified_tokens],
                video=[video],
                max_length=self.max_length,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors='pt'
            )
            
            toregressed = self.tokenizer.build_input_ids(
                text=[caption + "<|im_end|>\n" + '</s>'],
                max_length=self.max_length - tokenized['input_ids'].shape[0],
                add_special_tokens=False,
                truncation=True,
                padding="max_length",
                return_tensors='pt'
            )

            video_idx = torch.cat([tokenized['video_index'], torch.zeros_like(toregressed['input_ids'])])
            input_ids = torch.cat([tokenized['input_ids'], toregressed['input_ids']])
            attention_mask = torch.cat([torch.ones_like(tokenized['attention_mask']), toregressed['attention_mask']])
            labels = torch.clone(input_ids)
            labels[:tokenized['input_ids'].shape[0]] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                'input_ids': input_ids, 
                'attention_mask': attention_mask, 
                'labels': labels, 
                'video_idx': video_idx,
                'video': tokenized['video'][0],
            }
        
        except Exception as e:            
            logger.warning(f"Caught exception {e} when loading video {ann}")
            # raise e
            print(e)
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)
        