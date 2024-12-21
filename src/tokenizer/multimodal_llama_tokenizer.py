import torch.nn as nn
import torch
import os
from typing import Any, Dict, List, Optional, Union
from transformers import LlamaTokenizer

from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_IMG_TOKEN = "[IMG]"
DEFAULT_IMG_END_TOKEN = "[/IMG]"


DEFAULT_IMAGE_TOKEN = "[IMAGETOKEN]"
DEFAULT_VIDEO_TOKEN = "[VIDEOTOKEN]"

DEFAULT_IMG_PLACEHOLDER = "[<IMG_PLH>]"
DEFAULT_VID_PLACEHOLDER = "[<VID_PLH>]"



class MultimodalLlamaTokenizer(LlamaTokenizer):
    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        n_query=64,
        v_query=64,
        pad_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        device='cuda',
        **kwargs
    ):
        super().__init__(vocab_file, unk_token, bos_token, eos_token, pad_token, sp_model_kwargs, add_bos_token, add_eos_token,
                         clean_up_tokenization_spaces, **kwargs)
        
        self.device = device
        self.pad_token = self.unk_token
        
        if not self.pad_token:
            self.pad_token = self.eos_token
        # follow EMU
        # self.image_placeholder = DEFAULT_IMG_TOKEN + DEFAULT_IMAGE_TOKEN * n_query + DEFAULT_IMG_END_TOKEN
        # self.video_placeholder = DEFAULT_IMG_TOKEN + DEFAULT_VIDEO_TOKEN * v_query + DEFAULT_IMG_END_TOKEN
        
        # For mistral
        # Define the special tokens
        # special_tokens = [DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN]
        # Add the special tokens to the tokenizer
        # self.add_tokens(special_tokens)
        self.n_query = n_query
        self.v_query = v_query
        
    @property
    def processor(self):
        self._processor = None
        return self._processor


    @property
    def num_image_tokens(self):
        return 8192  # self.image_tokenizer.num_tokens # allow not load


    def to(self, device):
        self.device = device
        if hasattr(self, '_image_tokenizer'):
            self._image_tokenizer.to(device=device)


    def encode_image(
        self,
        image,
        image_size: int = 224,
    ):
        # image = self.processor(image)
        return image


    def decode_image(
        self
    ):
        return ...


    def prepare_image_input(self, images):
        # image_size: int = 224
        # images = [self.encode_image(image, image_size) for image in images]
        # return torch.stack(images, 0)
        return images


    def prepare_text_input(
        self,
        text: List[str],
        max_length,
        add_special_tokens,
        truncation,
        padding = "longest", 
        return_tensors = "pt",
        image_placeholder: str = DEFAULT_IMG_PLACEHOLDER,
        video_placeholder: str = DEFAULT_VID_PLACEHOLDER,
    ):
        text = text[0]
        start = 0
        total_len = 0
        
        input_ids = []
        attention_mask = []
        indexs = []
        
        while True:
            index1 = text.find(image_placeholder, start)
            index2 = text.find(video_placeholder, start)

            if index1 == -1 and index2 == -1:
                index = -1
            elif index1 == -1:
                index = index2
            elif index2 == -1:
                index = index1
            else:
                index = min(index1, index2)
                assert index != -1
            
            if index == -1:
                inputs = self(text[start:], max_length=max_length-total_len, add_special_tokens=add_special_tokens, truncation=truncation, padding=padding, return_tensors=return_tensors)
            else:
                inputs = self(text[start:index], max_length=max_length, add_special_tokens=add_special_tokens, truncation=truncation, padding='longest', return_tensors=return_tensors)
            
            # print(input_ids)
            input_ids += inputs.input_ids
            attention_mask += inputs.attention_mask
            indexs += torch.zeros_like(inputs.input_ids)
            total_len += inputs.input_ids[0].shape[0]
            
            if index != -1:
                input_ids += [torch.zeros(self.n_query).long()]
                attention_mask += [torch.ones(self.n_query).long()]
                indexs += [torch.ones(self.n_query)]
            
            if index == -1:
                return {
                    'input_ids': torch.cat(input_ids).long(),
                    'attention_mask': torch.cat(attention_mask).long(),
                    'index': torch.cat(indexs).to(torch.bool),
                }
            start = index + len(DEFAULT_IMG_PLACEHOLDER)


    def build_input_ids(
        self,
        text: List[str],
        max_length,
        add_special_tokens,
        truncation,
        padding,
        return_tensors,
        image = None,
        video = None,
        require_image = False,
        require_video = False,
    ):
        if image is not None:
            image = self.prepare_image_input(image)
        if video is not None:
            video = self.prepare_image_input(video)

        inputs = self.prepare_text_input(text, max_length, add_special_tokens, truncation, padding, return_tensors)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image_index': inputs['index'] if image is not None or require_image else None,
            'video_index': inputs['index'] if video is not None or require_video else None,
            'image': image if image is not None else None,
            'video': video if video is not None else None,
        }


if __name__ == '__main__':
    # image_path = 'demo_image.jpg'
    # raw_image = Image.open(image_path).convert("RGB")

    tokenizer = MultimodalLlamaTokenizer.from_pretrained(
        pretrained_model_name_or_path="BAAI/Emu2",
        local_files_only=True)

    # input_ids = tokenizer.encode_image(image_pil=raw_image)
    # print(input_ids)

    print(tokenizer.eos_token)