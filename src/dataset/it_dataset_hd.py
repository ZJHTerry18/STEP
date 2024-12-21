import logging
import os
import json
import random
import io
import torch
import numpy as np
import random
import torch.nn.functional as F
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

'''
example from /mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_asrcap.json:
{'long_video_id': 'aOqaEiyqliY', 'interleaved_list': [{'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_0_70.mp4', 'caption': "it's getting very", 'video_start_frame': 0}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_70_372.mp4', 'caption': 'real now the world cup squad have arrived in Brazil Roy Hodgson side landed in rio de janeiro on sunday after a flight from miami whether it be based for the past week the squad', 'video_start_frame': 70}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_372_625.mp4', 'caption': 'looks relaxed getting off the plane wearing polo shirts reflecting a change of mood from four years ago what', 'video_start_frame': 372}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_625_847.mp4', 'caption': "if the last off was Wayne Rooney there was real pressure on him to perform here if he's ever to be considered one of England's all-time great the", 'video_start_frame': 625}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_847_996.mp4', 'caption': 'only injury concern is Alex oxlade-chamberlain who has a knee ligament problem the', 'video_start_frame': 847}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_996_1160.mp4', 'caption': 'squad have another flight to catch before their opening match in Group D as they face Italy in Manaus which is', 'video_start_frame': 996}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_1160_1261.mp4', 'caption': 'four hours on a plane northwest of Rio England then', 'video_start_frame': 1160}, {'video': 'phdd:s3://InternVid2/ytt/aOqaEiyqliY_1261_1525.mp4', 'caption': 'face Uruguay in são Paulo before taking on costa rica in Belo Horizonte', 'video_start_frame': 1261}]}
'''
logger = logging.getLogger(__name__)


class ITTrainDataset(BaseDataset):

    def __init__(
        self, ann_file, transform, tokenizer,
        system="", role=("[INST]", ""),
        random_shuffle=True,
        video_reader_type='decord',
        sample_type='rand',
        num_frames=4,
        num_tries=10,
    ):
        super().__init__()
        self.local_size = 224
        self.media_type = ann_file.get('media_type', 'image')  
        
        self.label_file, self.data_root = ann_file.anno_path, ann_file.data_root

        logger.info('Load json file')
        with open(self.label_file, 'r') as f:
            self.anno = json.load(f)
        self.num_examples = len(self.anno)
        self.transform = transform

        if self.media_type == 'video':
            self.num_frames = num_frames
            self.video_reader_type = ann_file.get('video_reader_type', 'decord')  
            self.video_reader = VIDEO_READER_FUNCS[self.video_reader_type]
            self.sample_type = sample_type
            self.num_tries = num_tries
        
        self.tokenizer = tokenizer
        # prompt parameters
        if system:
            assert system[-1] == " ", "' ' should be add in the end of system, thus '###' will be tokenized into one token."
        # currently not support add start_token and end_token in the system, since the msg should be added properly
        self.begin_signal = ""
        self.end_signal = ""
        self.system = system
        self.role = ("","")
        self.random_shuffle = random_shuffle
        self.max_length = 1024
        # instruction location and number
        logger.info(f"Random shuffle: {self.random_shuffle}")

    def __len__(self):
        return self.num_examples

    def get_anno(self, index):
        filename = self.anno[index][self.media_type]
        qa = self.anno[index]["QA"]
        if "start" in self.anno[index] and "end" in self.anno[index]:
            anno = {
                "image": os.path.join(self.data_root, filename), "qa": qa,
                "start": self.anno[index]["start"], "end": self.anno[index]["end"],
            }
        else:
            anno = {"image": os.path.join(self.data_root, filename), "qa": qa}
        return anno

    def process_qa(self, qa, msg="", ilen=1):
        cur_instruction = ""
        # randomly shuffle qa for conversation
        if self.random_shuffle and len(qa) > 1:
            random.shuffle(qa)
        if "i" in qa[0].keys() and qa[0]["i"] != "":
            cur_instruction = "<|im_start|>system\n" + qa[0]["i"] + "<|im_end|>\n"

        input_ids, attention_masks, labels = [], [], []
        
        conversation = ""
        # add instruction as system message
        if cur_instruction:
            conversation += cur_instruction

        # rstrip() for the extra " " in msg
        conversation += (
            "<|im_start|>user\n"
        )
        
        if self.media_type == 'image':
            conversation += ("<img>" + IMG_TOKEN + "</img>") * ilen
        else:
            conversation += ("<vid>" + VID_TOKEN + "</vid>") * ilen
        
        conversation += (
            msg.rstrip() + "<|im_end|>\n"
        )
        
        total_len = 0
        indexs = []
        tokenized = self.tokenizer.build_input_ids(
            text=[conversation],
            max_length=self.max_length,
            add_special_tokens=False,
            truncation=False,
            require_image = (self.media_type == 'image'),
            require_video = (self.media_type == 'video'),
            padding=False,
            return_tensors='pt'
        )
        
        if self.media_type == 'image':
            indexs.append(tokenized['image_index'])
        else:
            indexs.append(tokenized['video_index'])
        
        input_ids.append(tokenized['input_ids'])
        attention_masks.append(tokenized['attention_mask'])
        labels.append(torch.ones_like(tokenized['input_ids']) * -100)
        total_len += tokenized['input_ids'].shape[0]
        
        for sentence in qa:
            if total_len >= self.max_length:
                break
            q = sentence["q"]
            a = sentence["a"]
            
            if q != "":
                conversation = ("<|im_start|>user\n" + q + "<|im_end|>\n")
            else:
                # no question, often in caption dataset
                conversation = ""
            conversation += ("<|im_start|>assistant\n")
            tokenized = self.tokenizer.build_input_ids(
                text=[conversation],
                max_length=self.max_length,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_tensors='pt'
            )
            indexs.append(torch.zeros_like(tokenized['input_ids']).to(torch.bool))
            input_ids.append(tokenized['input_ids'])
            attention_masks.append(tokenized['attention_mask'])
            labels.append(torch.ones_like(tokenized['input_ids']) * -100)
            total_len += tokenized['input_ids'].shape[0]
            
            conversation = a + "<|im_end|>\n" + '</s>'
            tokenized = self.tokenizer.build_input_ids(
                text=[conversation],
                max_length=self.max_length - total_len,
                add_special_tokens=False,
                truncation=True,
                padding="max_length",
                return_tensors='pt'
            )
            indexs.append(torch.zeros_like(tokenized['input_ids']).to(torch.bool))
            input_ids.append(tokenized['input_ids'])
            attention_masks.append(tokenized['attention_mask'])
            labels.append(tokenized['input_ids'])
            total_len += tokenized['input_ids'].shape[0]
        
        input_ids = torch.cat(input_ids)[:self.max_length]
        attention_masks = torch.cat(attention_masks)[:self.max_length]
        labels = torch.cat(labels)[:self.max_length]
        indexs = torch.cat(indexs)[:self.max_length]
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return input_ids, attention_masks, labels, indexs, cur_instruction


    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            if self.media_type == 'image':
                image, index = self.load_and_transform_media_data_image(index, ann["image"])
            else:
                clip = None
                if "start" in ann and "end" in ann:
                    clip = [ann["start"], ann["end"]]
                image, index, sec = self.load_and_transform_media_data_video(index, ann["image"], return_fps=True, clip=clip)

            T, C, H, W = image.shape
            
            sub_img = image.reshape(
                1, T, 3, H//self.local_size, self.local_size, W//self.local_size, self.local_size
            ).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, T, 3, self.local_size, self.local_size).contiguous()
            
            glb_img = F.interpolate(
                image.float(), size=(self.local_size, self.local_size), mode='bicubic', align_corners=False
            ).to(sub_img.dtype).unsqueeze(0)

            image = torch.cat([sub_img, glb_img])
        
            input_ids, attention_masks, labels, indexs, inst = self.process_qa(ann["qa"], ilen=image.shape[0])

            ret = {
                'input_ids': input_ids, 
                'attention_mask': attention_masks, 
                'labels': labels
            }
            if self.media_type == 'image':
                ret['image'] = image
                ret['image_idx'] = indexs
            else:
                ret['video'] = image     
                ret['video_idx'] = indexs
                   
            return ret 
       
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading clips_anno {ann}")
            print(e)
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


if __name__ == '__main__':
    from ..tokenizer.multimodal_llama_tokenizer import MultimodalLlamaTokenizer
    tokenizer = MultimodalLlamaTokenizer.from_pretrained("BAAI/Emu2", local_files_only=True) 
    from easydict import EasyDict as edict
    from .create import get_train_transform
    Myd = ITTrainDataset(
        ann_file=edict(
            anno_path=f"/mnt/petrelfs/share_data/videointern/annotations/anno_instruction/videochat_new/image/caption/coco/train_100k.json", 
            data_root="p2:s3://coco_caption",
        ),
        transform=get_train_transform(None, None),
        tokenizer=tokenizer
    )
    data1 = Myd[0]
    data2 = Myd[1]
    breakpoint()
    assert data1['input_ids'].shape[0] == data2['input_ids'].shape[0]