import io
import torch
import hydra
from omegaconf import OmegaConf
import argparse
import cv2
import imageio
import os
import pyrootutils
import torch.nn.functional as F

from PIL import Image
import numpy as np
import scipy
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
from peft import get_peft_model, LoraConfig, TaskType

import decord
decord.bridge.set_bridge("torch")
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
from src.model import MultiModalLLM_PT, MultiModalLLM_fg_PT, MultiModalLLM_fg_PT_2, MultiModalLLM_fg_IT, MultiModalLLM_fg_IT_2
from src.share_utils.py_config_utils import Config, eval_dict_leaf
from src.model.base_model import LLMConfig
from src.dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)

from petrel_client.client import Client
from decord import VideoReader, cpu
petrel_client = Client('~/petreloss.conf', enable_mc=False)

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
args = parser.parse_args()

### Model Initialization
tokenizer = hydra.utils.instantiate(OmegaConf.load(args.tokenizer_cfg))
tokenizer.pad_token = tokenizer.unk_token
model_cfg = eval_dict_leaf(Config.from_file(args.model_cfg))
mllm_config = LLMConfig.from_pretrained(args.llm_path, trust_remote_code=True)
mllm_config.model_config = model_cfg.model

model_cls = model_cfg.model.get('model_cls')
model = eval(model_cls).from_pretrained(args.ckpt_path, config=mllm_config)
# model.build_bridge() # load raw bert weights for q-former

model = model.to(torch.device(args.device))
model = model.eval()

### Video loading functions
def get_segment_indices(start_frame, end_frame, num_segments):
    seg_size = float(end_frame - start_frame) / num_segments
    start = start_frame + int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def read_video(video_path, num_segments=8, resolution=224, start=None, end=None):
    if 's3://' in video_path:
        video_bytes = petrel_client.get(video_path, update_cache=True)
        vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0), num_threads=1)
    else:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

    fps = vr.get_avg_fps()
    num_frames = len(vr)
    start_frame = min(max(0, int(start * fps) if start is not None else 0), num_frames - 1)
    end_frame = min(num_frames, int(end * fps) if end is not None else num_frames)
    frame_indices = get_segment_indices(start_frame=start_frame, end_frame=end_frame - 1, num_segments=num_segments)

    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = (0.485, 0.456, 0.406)
    input_std = (0.229, 0.224, 0.225)

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])
    
    img_transform = T.Compose([
        T.Resize(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(crop_size),
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img_transform(img))
    torch_imgs = transform(images_group)
    
    return images_group, torch_imgs

def read_gif(video_path, num_segments=8, resolution=224, start=None, end=None, fps=25):
    if "s3://" in video_path:
        video_bytes = petrel_client.get(video_path)
        gif = imageio.get_reader(io.BytesIO(video_bytes))
    else:
        gif = imageio.get_reader(video_path)
    num_frames = len(gif)
    start_frame = min(max(0, int(start * fps) if start is not None else 0), num_frames - 1)
    end_frame = min(num_frames, int(end * fps) if end is not None else num_frames)
    frame_indices = get_segment_indices(start_frame=start_frame, end_frame=end_frame - 1, num_segments=num_segments)
    
    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = (0.485, 0.456, 0.406)
    input_std = (0.229, 0.224, 0.225)
    
    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])
    
    img_transform = T.Compose([
        T.Resize(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(crop_size),
    ])
    
    images_group = list()
    for index, frame in enumerate(gif):
        if index in frame_indices:
            img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            img = Image.fromarray(img)
            images_group.append(img_transform(img))
    torch_imgs = transform(images_group)
    
    return images_group, torch_imgs 

def read_frames(video_path, num_segments=8, resolution=224, start=None, end=None, fps=3):
    img_list=[]
    if "s3://" in video_path:
        for path in petrel_client.list(video_path):
            # if path.startswith('img'):
            img_list.append(path)
    else:
        for path in os.listdir(video_path):
            # if path.startswith('img'):
            img_list.append(path)
    num_frames = len(img_list)
    start_frame = min(max(0, int(start * fps) if start is not None else 0), num_frames - 1)
    end_frame = min(num_frames, int(end * fps) if end is not None else num_frames)
    frame_indices = get_segment_indices(start_frame=start_frame, end_frame=end_frame - 1, num_segments=num_segments)
    
    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = (0.485, 0.456, 0.406)
    input_std = (0.229, 0.224, 0.225)
    
    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])
    
    img_transform = T.Compose([
        T.Resize(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(crop_size),
    ])
    
    images_group = list()
    for frame_index in frame_indices:
        if "s3://" in video_path:
            img_bytes = petrel_client.get(os.path.join(video_path, img_list[frame_index]))
            img = Image.open(io.BytesIO(img_bytes))
        else:
            img = Image.open(os.path.join(video_path, img_list[frame_index]))
        images_group.append(img_transform(img))
    
    torch_imgs = transform(images_group)
    return  images_group, torch_imgs

decord_methods = {
    'video': read_video,
    'gif': read_gif,
    'img': read_frames,
}

### Load sample video
# video_base = 'p2:s3://ego4d/clips'
# video_base = 'p2:s3://egoqa/split_videos'
# video_base = 'pvideo:s3://LLaVA_DPO/train_300k'
video_base = 'temp'
video_infos = [
    # {
    #     "video": "75d3fc52-3776-47d4-b7fd-8074d30b06d1.mp4",
    #     "start": 3.3400286,
    #     "end": 4.6400286,
    #     # "question": "Question: Where did I put the chopsticks?\nOptions:\n(A) in the drawer\n(B) right of the stove\n(C) on the table\n(D) left of the stove",
    #     "question": None,
    #     "choices": [
    #         "in the drawer",
    #         "right of the stove",
    #         "on the table",
    #         "left of the stove"
    #     ],
    #     "gt_answer": "(B) right of the stove",
    #     "pred_answer": "(D) left of the stove"
    # },
    # {
    #     "video": "0c169d77-43f8-402b-b86c-04c885c46294/split_0.mp4",
    #     "start": None,
    #     "end": None,
    #     "question": None
    # },
    # {
    #     "video": "v_OiL6Aj0gC14-Scene-006",
    #     "start": None,
    #     "end": None,
    #     "question": None
    # },
    {
        "video": "3460224697-preview.mp4",
        "start": None,
        "end": None,
        "question": None
    },
    # {
    #     "video": "GZqeESbWMqE",
    #     "start": None,
    #     "end": None,
    #     "question": None
    # },
    # {
    #     "video": "14106731",
    #     "start": None,
    #     "end": None,
    #     "question": None
    # },
    # {
    #     "video": "RfZPh2aP4ng",
    #     "start": None,
    #     "end": None,
    #     "question": None
    # }
]
text_list = [
    # [
    #     "stove",
    #     "chopsticks",
    #     "hand",
    #     "pot",
    #     "wall",
    #     "spoon",
    # ],
    # [
    #     "pan",
    #     "sink",
    #     "towel",
    #     "spatula",
    #     "chopsticks",
    #     "scissors",
    #     "wall"
    # ],
    # [
    #     "sky",
    #     "tree",
    #     "airplane"
    # ],
    [
        "children",
        "dog",
        "wheelcart",
        "window"
    ]
    # [
    #     "human hand", "mechanical pencil", "green keychain", "anime character illustrations", "white sheet of paper", "pre-printed lines", "white rubber eraser"
    # ],
    # [
    #     "computer", "man", "woman", "black suit", "white shirt", "tie"
    # ],
    # [
    #     "white wedding gown", "off-the-shoulder straps", "full, layered skirt", "necklace", "earrings", "long, white gloves", "styled up hair", "light gray suit", "white shirt", "blue tie", "boutonniere", "bouquet", "pastel flowers", "pink", "white", "greenery", "long ribbons", "bridesmaid in pink"
    # ]
]
text_list = text_list[0]

num_frame = 8
resolution = 224
media_type = 'video'

video_list = []
video_frame_list = []
instruction_list = []
for video_info in video_infos:
    video_path = video_info['video']
    start = video_info.get('start', None)
    end = video_info.get('end', None)
    vid_path = os.path.join(video_base, video_path)
    frame, video = decord_methods[media_type](vid_path, num_segments=num_frame, start=start, end=end)
    TC, H, W = video.shape
    video = video.reshape(1, TC//3, 3, H, W).to(args.device)
    video_list.append(video)
    video_frame_list.append(frame)
    instruction_list.append(video_info['question'])
video_tensor = torch.cat(video_list, dim=0)

### Extract q-former outputs
video_tensor = video_tensor.permute(0, 2, 1, 3, 4).to('cuda:0')
vision_query_embeds, image_embeds, avg_cross_atts = model.encode_fg_vision(
    video_tensor, None
)
B, Q, C = vision_query_embeds.shape
# vision_query_embeds = vision_query_embeds.reshape(B, -1, 4, C).mean(dim=2)
B, Q, L = avg_cross_atts.shape
# avg_cross_atts = avg_cross_atts.reshape(B, -1, 4, L).mean(dim=2)
_, pooled_text_embeds = model.encode_fg_text(
    text_list, args.device
)
# print(model.temp)

# vision_f = F.normalize(model.v_proj(vision_query_embeds), dim=-1)
# text_f = F.normalize(model.t_proj(pooled_text_embeds), dim=-1)
vision_f = F.normalize(vision_query_embeds, dim=-1)
text_f = F.normalize(pooled_text_embeds, dim=-1)
sim_scores = torch.einsum("mld,nd->mln", vision_f, text_f) # [B, num_vq, num_t]
text_sim_scores = torch.einsum('md,nd->mn', text_f, text_f)
# print(sim_scores)
# print(text_sim_scores)

# simulate the loss calculation process


model_name = args.ckpt_path.split('/')[-2]
for vi in range(len(video_infos)):
    sim_matrix = sim_scores[vi].cpu().detach().numpy()
    min_val = sim_matrix.min()
    max_val = sim_matrix.max()
    sim_matrix = (sim_matrix - min_val) / (max_val - min_val)
    plt.figure(figsize=(10, 30))
    sim_map = plt.imshow(sim_matrix, interpolation='nearest', cmap='viridis', aspect='auto')
    plt.colorbar(sim_map)
    plt.xticks(ticks=np.arange(len(text_list)), labels=text_list, rotation=60)
    plt.yticks(ticks=np.arange(sim_matrix.shape[0]), labels=list(range(sim_matrix.shape[0])))
    
    save_path = f"temp/{model_name}/{video_info['video'][:-4]}_{video_info['start']}_{video_info['end']}_sim.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

for vi in range(len(video_infos)):
    video_info = video_infos[vi]
    frame_img_list = [cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR) for frame in video_frame_list[vi]]
    cross_atts = avg_cross_atts[vi] # [num_q, num_patch]
    
    loc_atts = cross_atts.mean(dim=0)[1:].reshape(num_frame, resolution//14, resolution//14)
    frame_attn_img_list = []
    for frame_index in range(loc_atts.shape[0]):
        frame_attn_map = F.interpolate(loc_atts[frame_index].unsqueeze(0).unsqueeze(0), size=(resolution, resolution), mode='bilinear', align_corners=False)
        frame_attn_map = frame_attn_map.squeeze()
        # print(frame_attn_map.min(), frame_attn_map.max())
        
        frame_attn_map = frame_attn_map.cpu().detach().numpy()
        frame_attn_map = np.uint8(frame_attn_map * 32 * 255)
        heatmap = cv2.applyColorMap(frame_attn_map, cv2.COLORMAP_JET)
        # print(heatmap.shape, frame_img_list[frame_index].shape)
        frame_img = frame_img_list[frame_index]
        frame_attn_img = cv2.addWeighted(frame_img, 0.6, heatmap, 0.4, 0)
        frame_attn_img_list.append(frame_attn_img)
    # row_imgs = []
    # for ri in range(2):
    #     row_imgs.append(cv2.hconcat(frame_attn_img_list[ri*4:ri*4+4]))
    # full_img = cv2.vconcat(row_imgs)
    full_img = cv2.hconcat(frame_attn_img_list)
    save_path = f"temp/{model_name}/{video_info['video'][:-4]}_{video_info['start']}_{video_info['end']}/agg.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, full_img)
    
    full_img = cv2.hconcat(frame_img_list)
    save_path = f"temp/{model_name}/{video_info['video'][:-4]}_{video_info['start']}_{video_info['end']}/raw.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, full_img)
    
    for query_index in range(cross_atts.shape[0]):
        attn_maps = cross_atts[query_index][1:].reshape(num_frame, resolution//14, resolution//14)
        frame_attn_img_list = []
        for frame_index in range(attn_maps.shape[0]):
            frame_attn_map = F.interpolate(attn_maps[frame_index].unsqueeze(0).unsqueeze(0), size=(resolution, resolution), mode='bilinear', align_corners=False)
            frame_attn_map = frame_attn_map.squeeze()
            # print(frame_attn_map.min(), frame_attn_map.max())
            
            frame_attn_map = np.uint8(frame_attn_map.cpu().detach().numpy() * 4 * 255)
            heatmap = cv2.applyColorMap(frame_attn_map, cv2.COLORMAP_JET)
            # print(heatmap.shape, frame_img_list[frame_index].shape)
            frame_img = frame_img_list[frame_index]
            frame_attn_img = cv2.addWeighted(frame_img, 0.6, heatmap, 0.4, 0)
            frame_attn_img_list.append(frame_attn_img)
        
        # row_imgs = []
        # for ri in range(2):
        #     row_imgs.append(cv2.hconcat(frame_attn_img_list[ri*4:ri*4+4]))
        # full_img = cv2.vconcat(row_imgs)
        full_img = cv2.hconcat(frame_attn_img_list)
        
        save_path = f"temp/{model_name}/{video_info['video'][:-4]}_{video_info['start']}_{video_info['end']}/q{query_index}.jpg"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, full_img)