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
import time

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
parser.add_argument('--video_base', type=str, default='p2:s3://star/Charades_v1_480/')
parser.add_argument('--ann', type=str, default='/mnt/petrelfs/share_data/likunchang/star/val.json')
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

model = model.to(torch.device(args.device))
model = model.eval()



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

def load_video_star(video_path, start, end, num_segments=8, return_msg=False, resolution=224):
    def _get_index(num_segments, start, end, fps, max_frame, first_idx=0):
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices
    
    if "s3://" in video_path:
        video_bytes = client.get(video_path)
        vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0), num_threads=1)
    else:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    
    frame_indices = _get_index(num_segments, start, end, fps, max_frame, first_idx=0) 

    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return torch_imgs, msg
    else:
        return torch_imgs

def infer_star(
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
        num_segments=32,
    ):
    vid_path = os.path.join(args.video_base, data_sample['video'])
    video, _ = load_video_star(vid_path, start=data_sample['start'], end=data_sample['end'], num_segments=num_segments, return_msg=True)
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
        prompt = system + data_sample['QA'][0]['q'] + question_prompt
    else:
        prompt = data_sample['QA'][0]['q'] + question_prompt
    
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

def check_answer_star(pred, qid):
    correct = 0
    answer_content = ans_dict[qid]['content'].lower()
    if answer_content[-1] == ".":
        answer_content = answer_content[:-1]
    if ans_dict[qid]['answer'].lower() in pred.lower():
        flag = True
        for kk in ["(A)", "(B)", "(C)", "(D)", "(E)"]:
            if kk != ans_dict[qid]['answer'].lower() and kk in pred.lower():
                flag = ans_dict
                break
        if flag:
            correct += 1
    elif answer_content in pred.lower():
        correct = 1
    elif answer_content.replace("a ", "") in pred.lower():
        correct = 1
    elif answer_content.replace("an ", "") in pred.lower():
        correct = 1
    return correct


### STAR dataset loading
json_data = json.load(open(args.ann, "r"))[args.start:args.end]
ans_data = json.load(open(args.ann, "r"))[args.start:args.end]
ans_dict = {}
for idx, data in enumerate(ans_data):
    ans = data['QA'][0]['a'].split(": ")[1].split(" ")[0].strip()
    content = data['QA'][0]['a'].split(ans)[1].strip()
    ans_dict[idx] = {
        'video': data['video'],
        'answer': ans,
        'content': content,
        'question': data['QA'][0]['q'].split('Options:')[0].strip(),
        'options': data['QA'][0]['q'].split('Options:')[1].strip()
    }

correct = 0
total = 0
res_list = []
acc_dict = {}
total_num = len(json_data)

output = ""

for idx, example in tqdm(enumerate(json_data)):
    start = time.time()
    pred = infer_star(
        example,
        model,
        tokenizer,
        system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n", 
        question_prompt="\nOnly give the best option.", 
        answer_prompt="Best option:(",
        return_prompt='(',
        system_q=False,
        print_res=False,
        system_llm=True,
        num_segments=args.num_f,
    )
    gt = ans_dict[idx]['content']
    
    duration = time.time() - start
    
    flag = 1 if check_answer_star(pred, idx) else 0
    correct += flag
    total += 1
    
    res_list.append({
        'video': example['video'],
        'question': example['QA'][0]['q'],
        'pred': pred,
        'gt': gt,
        'correct': flag,
    })
    
    if idx % 100 == 0:
        print("Acc:", correct / total)
        print('-' * 20, f'{idx+1}/{total_num} done,')
if idx % 100 == 0:
    print(f"Acc:{correct}/{total}, {correct / total}")
    print('-' * 20, f'{idx+1}/{total_num} done,')
    
os.makedirs(args.save_path, exist_ok=True)
with open(f"{args.save_path}/{args.start}_{args.end}.json", "w") as f:
    json.dump(res_list, f)