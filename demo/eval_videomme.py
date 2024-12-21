import io
from easydict import EasyDict
import torch
import hydra
from omegaconf import OmegaConf
import argparse
import cv2
import imageio
import os
import sys
import pyrootutils
import webvtt
import re

from transformers import StoppingCriteria, StoppingCriteriaList

from PIL import Image
import numpy as np
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms

from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms

import matplotlib.pyplot as plt

from peft import get_peft_model, LoraConfig, TaskType
import copy

import json
from collections import OrderedDict

from tqdm import tqdm

import decord
decord.bridge.set_bridge("torch")
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
from src.model import MultiModalLLM_PT, MultiModalLLM_fg_PT, MultiModalLLM_fg_IT, MultiModalLLM_fg_IT_2
from src.share_utils.py_config_utils import Config, eval_dict_leaf
from src.model.base_model import LLMConfig
from src.dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)

from petrel_client.client import Client
from decord import VideoReader, cpu
client = Client('~/petreloss.conf', enable_mc=False)

DEFAULT_IMG_TOKEN = "[IMG]"
DEFAULT_IMG_END_TOKEN = "[/IMG]"

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "[VIDEO]"

IMG_TOKEN = "[<IMG_PLH>]"
VID_TOKEN = "[<VID_PLH>]"

parser = argparse.ArgumentParser()
parser.add_argument('--model_cfg', type=str, default='configs/model/i1b_qformer_mistral7b_f8_st2_sh.py')
parser.add_argument('--tokenizer_cfg', type=str, default='configs/tokenizer/mistral_tokenizer_q96.yaml')
parser.add_argument('--llm_path', type=str, default='/mnt/petrelfs/share/videointern/MLLM/Mistral-7B-Instruct-v0.3/')
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_f', type=int, default=8)
parser.add_argument('--res', type=int, default=224)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=-1)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--sub', action='store_true')
args = parser.parse_args()

# cfg_path = dict(
#     model_cfg = "configs/model/i1b_qformer_mistral7b_f8_st2_sh.py",
#     tokenizer_cfg = "configs/tokenizer/mistral_tokenizer_q96.yaml",
#     llm_path = "/mnt/petrelfs/share/videointern/MLLM/Mistral-7B-Instruct-v0.3/",
#     ckpt_path = "/mnt/petrelfs/share_data/wangchenting/models/internvideo2_videochat2/internvideo2_1b_mistral7b_stage3",
# )

### Model Initialization
tokenizer = hydra.utils.instantiate(OmegaConf.load(args.tokenizer_cfg))
tokenizer.pad_token = tokenizer.unk_token
model_cfg = eval_dict_leaf(Config.from_file(args.model_cfg))
mllm_config = LLMConfig.from_pretrained(args.llm_path, trust_remote_code=True)
mllm_config.model_config = model_cfg.model

model_cls = model_cfg.model.get('model_cls')
model = eval(model_cls).from_pretrained(args.ckpt_path, config=mllm_config)

# peft_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM, inference_mode=False, 
#     r=16, lora_alpha=32, lora_dropout=0.,
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",
#          "gate_proj", "up_proj", "down_proj", "lm_head"
#     ]
# )
# model.lm = get_peft_model(model.lm, peft_config)

model = model.to(torch.device(args.device))
model = model.eval()

### VideoMME Load

def clean_text(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return cleaned_text


def read_vtt_and_concatenate(file_path, tokenizer, max_len=4096):
    prev = ""
    subtitles = []
    for caption in webvtt.read(file_path):
        # Split the caption text into individual lines
        lines = caption.text.split('\n')
        for line in lines:
            # Clean the text and check for repetition
            line = clean_text(line)
            if prev != line and line:
                subtitles.append(line)
                prev = line

    # Join subtitles to check length
    full_text = ' '.join(subtitles)
    tokenized_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    # If the tokenized length is within the limit, return the full text
    if len(tokenized_ids) <= max_len:
        return full_text

    # Otherwise, we need to trim the text to fit within the limit
    # We will keep the first half and the last half
    half_len = max_len // 2
    start_text = ' '.join(subtitles[:half_len])
    end_text = ' '.join(subtitles[-half_len:])
    
    # Re-tokenize to ensure the total length is within the limit
    start_tokenized_ids = tokenizer(start_text, add_special_tokens=False).input_ids
    end_tokenized_ids = tokenizer(end_text, add_special_tokens=False).input_ids

    # Adjust the lengths to fit within the max_len
    while len(start_tokenized_ids) + len(end_tokenized_ids) > max_len:
        if len(start_tokenized_ids) > len(end_tokenized_ids):
            start_tokenized_ids.pop()
        else:
            end_tokenized_ids.pop(0)
    
    # Combine the adjusted parts
    adjusted_text = tokenizer.decode(start_tokenized_ids) + ' ... ' + tokenizer.decode(end_tokenized_ids)
    
    return adjusted_text

class MME_dataset(Dataset):
    def __init__(self, data_prefix, anno_path, num_segments=16, resolution=224, max_subtitle_len=4096):
        self.data_prefix = data_prefix
        with open(anno_path, 'r') as f:
            self.data_list = json.load(f)
            
        self.num_segments = num_segments
        self.max_subtitle_len = max_subtitle_len
        
        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
    
    def __str__(self):
        task_dict = {}
        total = 0
        for data in self.data_list:
            if data['duration_category'] not in ans_dict:
                task_dict[data['duration_category']] = {}
            for q in data['questions']:
                if q['task_type'] not in ans_dict[data['duration_category']]:
                    ans_dict[data['duration_category']][q['task_type']] = 0
                ans_dict[data['duration_category']][q['task_type']] += 1
                total += 1

        res = f"There are {len(self.data_list)} videos.\n"
        res += f"There are {total} QAs.\n"
        for k, v in task_dict.items():
            res += f"------{k}------\n"
            for kk, vv in task_dict.items():
                res += f"{kk}: {vv}\n"
                
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices

    def read_frame(self, video_path, bound=None):
        video_path = os.path.join(video_path, str(self.num_segments))
        
        if os.path.exists(video_path):
            frame_list = [p for p in os.listdir(video_path)]
        else:
            raise Exception
            
        images_group = list()
        
        for frame_name in frame_list:
            img = Image.open(os.path.join(video_path, frame_name))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer = f"({answer}) {data['choices'][ord(answer) - ord('A')][3:]}"
        for idx, c in enumerate(data['choices']):
            cur_choice, cur_text = c[0], c[3:]
            question += f"({cur_choice}) {cur_text}\n"
        question = question.rstrip()
        return question, answer

    def __getitem__(self, idx):
        video_name = self.data_list[idx]['url'].split("watch?v=")[1]
        video_path = os.path.join(self.data_prefix, "frames", video_name)

        # We store the videos with only 16 or 32 frames for testing,
        # since directly reading the whold videos cost a lot of time.
        # You can also read the whole video via self.read_video(video_path)
        torch_imgs = self.read_frame(video_path)
        duration_category = self.data_list[idx]['duration_category']
        qa_list = []
        for qa in self.data_list[idx]['questions']:
            qa_list.append(self.qa_template(qa))

        subtitle = ""
        try:
            subtitle_path = os.path.join(self.data_prefix, "subtitle", video_name + ".vtt")
            if os.path.exists(subtitle_path):
                subtitle = read_vtt_and_concatenate(subtitle_path, tokenizer, self.max_subtitle_len)
        except Exception as e:
            subtitle = ""
            print(f"Error for {subtitle_path}: {e}")
            
        return {
            'subtitle': subtitle,
            'video': torch_imgs, 
            'qa_list': qa_list,
            'duration_category': duration_category
        }

### Inference
def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + " " + message + " " + conv.sep
        else:
            ret += role
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + " " + message
        else:
            if message:
                ret += role + " " + message + " " + conv.sep
            else:
                ret += role
    return ret

def ask(text, conv):
    conv.messages.append([conv.roles[0], text])
        
def infer_mme(
        data_sample, model, tokenizer,
        system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
        max_txt_length=1024,
        max_new_tokens=200,
        add_subtitle=False,
    ):
    video = data_sample["video"]
    TC, H, W = video.shape
    video = video.reshape(1, TC//3, 3, H, W).to(args.device)
    
    pred_list = []
    gt_list = []
    for idx, qa in enumerate(data_sample['qa_list']):
        # prepare prompt
        chat = EasyDict({
            "system": system,
            "roles": ("[INST]", "[/INST]"),
            "messages": [],
            "sep": ""
        })

        if add_subtitle:
            if data_sample['subtitle'] != '':
                print("has subtitle")
                subtitle = f"This video's subtitles are listed below: {data_sample['subtitle']}"
                chat.messages.append([chat.roles[0], f"{subtitle}\n<Video>{VID_TOKEN}</Video> [/INST]"])
            else:
                chat.messages.append([chat.roles[0], f"<Video>{VID_TOKEN}</Video> [/INST]"])
        else:
            chat.messages.append([chat.roles[0], f"<Video>{VID_TOKEN}</Video> [/INST]"])
        
        if system_llm:
            prompt = system + qa[0] + question_prompt
        else:
            prompt = qa[0] + question_prompt
        
        ask(prompt, chat)
        chat.messages.append([chat.roles[1], answer_prompt])
        if answer_prompt:
            prompt = get_prompt2(chat)
        else:
            prompt = get_prompt(chat)
        
        # prepare input tokens
        input_ids, attention_masks, indexs = [], [], []
        with torch.no_grad():
            tokenized = tokenizer.build_input_ids(
                text=[prompt],
                max_length=max_txt_length,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                require_image = False,
                require_video = True,
                return_tensors='pt'
            )
            input_ids.append(tokenized['input_ids'])
            attention_masks.append(tokenized['attention_mask'])
            indexs.append(tokenized['video_index'])
        
        input_ids = torch.cat(input_ids).unsqueeze(0).to(args.device)
        attention_masks = torch.cat(attention_masks).unsqueeze(0).to(args.device)
        indexs = torch.cat(indexs).unsqueeze(0).to(args.device)
        
        if system_q:
            inputs_embeds = model.pad_text_embeds(
                input_ids=input_ids,
                video=video,
                video_idx=indexs,
                # instruction = system + data_sample['question']
            )
        else:
            inputs_embeds = model.pad_text_embeds(
                input_ids=input_ids,
                video=video,
                video_idx=indexs,
                # instruction=system
            )
        if 'fg' in model_cls:
            _, inputs_embeds, _ = inputs_embeds
        
        # llm inference
        llm_message = model.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_masks,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        answer = tokenizer.decode(llm_message[0], skip_special_tokens=True)
        answer = return_prompt + answer.strip().split('\n')[0]
        # print(f"Pred: {answer}", flush=True)
        # print(f"GT: {qa[1]}", flush=True)
        pred_list.append(answer[1])
        gt_list.append(qa[1][1])
    
    return pred_list, gt_list

data_dir = "/mnt/petrelfs/share_data/likunchang/videomme"
anno_path = "/mnt/petrelfs/share_data/likunchang/videomme/Video-MME_0606.json"
dataset = MME_dataset(data_dir, anno_path, num_segments=args.num_f, resolution=args.res)

with open(anno_path, 'r') as f:
    res_json_data = json.load(f)

correct = 0
total = 0
res_list = []
acc_dict = {}

for idx, example in tqdm(enumerate(dataset), total=len(dataset)):
    duration_category = example['duration_category']
    if duration_category not in acc_dict:
        acc_dict[duration_category] = [0, 0] # correct, total
    qa_count = len(example['qa_list'])
    acc_dict[duration_category][1] += qa_count
    total += qa_count
    pred_list, gt_list = infer_mme(
        example,
        model,
        tokenizer,
        system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n", # newPrompt2
        question_prompt="\nOnly give the best option.",  # prompt3
        answer_prompt="Best option:(",
        return_prompt='(',
        system_q=False,
        # system_q=True,
        print_res=True,
        system_llm=False,
        add_subtitle=args.sub,
    )
    
    res_list.append({
        'duration': duration_category,
        'pred': pred_list,
        'gt': gt_list
    })
    qa_idx = 0
    for pred, gt in zip(pred_list, gt_list):
        if pred == gt:
            acc_dict[duration_category][0] += 1
            correct += 1
        res_json_data[idx]['questions'][qa_idx]['response'] = pred
        qa_idx += 1
    
    if (idx + 1) % 100 == 0:
        print(f"Part  Acc: {acc_dict[duration_category][0] / acc_dict[duration_category][1] * 100 :.2f}%")
        print(f"Total Acc: {correct / total * 100 :.2f}%")
        print('-' * 50, duration_category, '-' * 50)
print(f"Total Acc: {correct / total * 100 :.2f}%")

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
with open(f"{args.save_path}.json", "w") as f:
    json.dump({
        "acc_dict": acc_dict,
        "res_list": res_list
    }, f)