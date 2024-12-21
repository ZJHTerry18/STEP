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

### MVBench Load
data_list = {
    "Action Sequence": ("action_sequence.json", "p2:s3://star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "p2:s3://star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "p2:s3://ssv2-video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "p2hdd:s3://Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "p2:s3://funqa/test/", "video", False),
    "Object Existence": ("object_existence.json", "p2:s3://clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "p2:s3://star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", "p2:s3://perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "p2:s3://clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "p2:s3://sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", "p2:s3://scene-qa/video/", "video", False),
    "Action Count": ("action_count.json", "p2:s3://perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "p2:s3://clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "p2:s3://clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "p2:s3://perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "p2:s3://nturgbd/", "video", False),
    "Character Order": ("character_order.json", "p2:s3://perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "p2:s3://vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "p2:s3://tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Counterfactual Inference": ("counterfactual_inference.json", "p2:s3://clevrer/video_validation/", "video", False),
}

data_dir = "/mnt/petrelfs/share_data/likunchang/mvbench/json"

class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, num_segments=8, resolution=224):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        
        self.num_segments = num_segments
        
        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.485, 0.456, 0.406]
        input_std = [0.229, 0.224, 0.225]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
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
    
    def read_video(self, video_path, bound=None):
        if "s3://" in video_path:
            video_bytes = client.get(video_path)
            vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0), num_threads=1)
        else:
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
    
    def read_gif(self, video_path, bound=None, fps=25):
        if "s3://" in video_path:
            video_bytes = client.get(video_path)
            gif = imageio.get_reader(io.BytesIO(video_bytes))
        else:
            gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_frame(self, video_path, bound=None, fps=3):
        if os.path.exists(video_path):
            max_frame = len(os.listdir(video_path))
        else:
            max_frame = len([k for k in client.list(video_path)])
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            if "s3://" in video_path:
                img_bytes = client.get(os.path.join(video_path, f"{frame_index:05d}.jpg"))
                img = Image.open(io.BytesIO(img_bytes))
            else:
                img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video_path': self.data_list[idx]['data']['video'],
            'video': torch_imgs, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
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
        
def infer_mvbench(
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
    ):
    video = data_sample["video"]
    TC, H, W = video.shape
    video = video.reshape(1, TC//3, 3, H, W).to(args.device)
    
    # prepare prompt
    chat = EasyDict({
        "system": system,
        "roles": ("[INST]", "[/INST]"),
        "messages": [],
        "sep": ""
    })

    chat.messages.append([chat.roles[0], f"<Video>{VID_TOKEN}</Video> [/INST]"])
    
    if system_llm:
        prompt = system + data_sample['question'] + question_prompt
    else:
        prompt = data_sample['question'] + question_prompt
    
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
    
    return answer

def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], " ".join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
    elif gt_content in pred_content:
        flag = True
    elif gt_content.replace("a ", "") in pred_content:
        flag = True
    elif gt_content.replace("an ", "") in pred_content:
        flag = True
        
    return flag

dataset = MVBench_dataset(data_dir, data_list, num_segments=args.num_f, resolution=args.res)

correct = 0
total = 0
res_list = []
acc_dict = {}


for idx, example in tqdm(enumerate(dataset)):
    task_type = example['task_type']
    if task_type not in acc_dict:
        acc_dict[task_type] = [0, 0] # correct, total
    acc_dict[task_type][1] += 1
    total += 1
    pred = infer_mvbench(
        example,
        model,
        tokenizer,
#         "Carefully observe the video and choose the best option for the question. ", 
#         "Carefully watch the video and pay attention to the cause, sequence of events, and object details and movements. Based on your observations, select the best option that accurately addresses the question. ",  # newPrompt
#         "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question. ", # newPrompt2
        system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n", # newPrompt2
#         question_prompt="\nOnly give the best option without any explanation.",
#         question_prompt="\nThink it step by step. Only give the best option without any explanation.", # prompt2
        question_prompt="\nOnly give the best option.",  # prompt3
        answer_prompt="Best option:(",
        return_prompt='(',
        system_q=False,
        # system_q=True,
        print_res=True,
        system_llm=True,
    )
    gt = example['answer']
    res_list.append({
        'task_type': task_type,
        'video': example['video_path'],
        'question': example['question'],
        'pred': pred,
        'gt': gt
    })
    
    if check_ans(pred=pred, gt=gt):
        acc_dict[task_type][0] += 1
        correct += 1
    
    if (idx + 1) % 100 == 0:
        print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        print(f"Total Acc: {correct / total * 100 :.2f}%")
        print('-' * 50, task_type, '-' * 50)

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
with open(f"{args.save_path}.json", "w") as f:
    json.dump({
        "acc_dict": acc_dict,
        "res_list": res_list
    }, f)

out1 = "AS	AP	AA	FA	UA	OE	OI	OS	MD	AL	ST	AC	MC	MA	SC	FP	CO	EN	ER	CI	Avg"
out2 = ""
correct = 0
total = 0
with open(f"{args.save_path}.json", "r") as f:
    json_data = json.load(f)
    for k, v in json_data["acc_dict"].items():
        correct += v[0]
        total += v[1]    
        out2 += f"{v[0]/v[1]*100:.2f}\t"
out2 += f"{correct/total*100:.2f}"
print(out1)
print(out2)