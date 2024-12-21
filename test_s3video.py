import os
import json
import torch

import decord
from decord import VideoReader, cpu
from tqdm import tqdm
import io
import random
import numpy as np
import cv2
decord.bridge.set_bridge("torch")
from petrel_client.client import Client

petrel_client = Client('~/petreloss.conf')

# video_root = 'pvideo:s3://LLaVA_DPO/train_300k'
# video_name = 'IbV6dCrXCkk'
# video_root = 'p2:s3://clevrer/video_validation/'
# video_name = 'video_11596.mp4'
# video_root = 'p2:s3://perception/videos'
# video_name = 'video_4144.mp4'
video_root = 'p2:s3://nturgbd/'
video_name = 'S018C002P045R001A097_rgb.avi'

# video_root = 'p2:s3://perception/videos'
# video_json = '/mnt/petrelfs/share_data/wangchenting/datasets/mc_question_train_forchoice.json'
# video_root = 'pvideo:s3://LLaVA_DPO/train_300k'
# video_json = '/mnt/petrelfs/share_data/videointern/annotations/anno_instruction/videochat_new/video/caption/sharegptvideo/train_300k.json'
# video_paths = []
# with open(video_json, 'r') as f:
#     data = json.load(f)
#     for item in data[::999]:
#         video_paths.append(item['video'])
# video_paths = [os.path.join(video_root, x) for x in video_paths]

# video_root = 'p2:s3://MovieChat/real_video'
# video_name = 's02e10-2.mp4'

video_path = os.path.join(video_root, video_name)

# img_bytes = petrel_client.Get(video_path)
# img_np = np.frombuffer(img_bytes, np.uint8)
# img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
# cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
# cv2.imwrite('llava.jpg', img)
# exit()

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    # num_frames = int(num_frames)
    # vlen = int(vlen)
    # print(num_frames, vlen, sample)
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        # print(intervals)
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        # print(ranges)
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        # print(frame_indices)
        
        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
        
        # print(frame_indices)
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices

def read_frames_img(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, clip=None,
    ):
    img_list=[]
    if "s3://" in video_path:
        for path in client.list(video_path):
            # if path.startswith('img'):
            img_list.append(path)
    else:
        for path in os.listdir(video_path):
            # if path.startswith('img'):
            img_list.append(path)
    # print('video readed')
    vlen = len(img_list)
    # print(vlen)

    def _get_index_by_time(start_sec, end_sec, num_segments=8, fps=3, max_frame=9999):
        start_idx = max(1, round(start_sec * fps))
        end_idx = min(round(end_sec * fps), max_frame)
        seg_size = float(end_idx - start_idx) / (num_segments - 1)
        offsets = np.array([start_idx + int(np.round(seg_size * idx)) for idx in range(num_segments)])
        return offsets
    
    if clip is not None:
        frame_indices = _get_index_by_time(float(clip[0]), float(clip[1]), num_segments=num_frames, max_frame=vlen)
    else:
        frame_indices = get_frame_indices(
            num_frames, vlen, sample=sample, fix_start=fix_start,
            max_num_frames=max_num_frames
        )

    imgs = []
    for idx in frame_indices:
        frame_fname = os.path.join(video_path, img_list[idx])
        if "s3://" in video_path:
            img_bytes = client.get(frame_fname)
        else:
            with open(frame_fname, 'rb') as f:
                img_bytes = f.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        # cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        imgs.append(img)

    return imgs

def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, 
        max_num_frames=-1, client=None, clip=None
    ):
    # print("Reading decord")
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0), num_threads=1)
    else:
        video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    # print("Finish Reading decord")
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    print(vlen)

    if clip:
        start, end = clip
        start_index = min(max(0, int(start * fps)), vlen - 1)
        end_index = min(vlen, int(end * fps))
        vlen = max(end_index - start_index, 1)
        print(start_index, start_index + vlen)

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    if clip:
        frame_indices = [f + start_index for f in frame_indices]

    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames#, frame_indices, float(fps), start_index, start_index + vlen, len(video_reader)

# video_file = '/mnt/petrelfs/share_data/videointern/annotations/anno_instruction/videochat_new/video/reasoning/star/train.json'
# with open(video_file, 'r') as f:
#     video_datas = json.load(f)

# for vd in video_datas:
#     video_path = os.path.join(video_root, vd['video'])
#     clip = [vd['start'], vd['end']]
#     frames, frame_indices, fps, start_index, end_index, max_len = read_frames_decord(video_path, num_frames=8, client=petrel_client, clip=clip)
#     print("success", frame_indices, fps, start_index, end_index, max_len)

frame_tensors = read_frames_decord(video_path, num_frames=8, client=petrel_client) # (T, C, H, W)
frame_imgs = []
for ft in frame_tensors:
    ft_img = ft.permute(1, 2, 0).numpy()
    ft_img = cv2.cvtColor(ft_img, cv2.COLOR_BGR2RGB)
    frame_imgs.append(ft_img)

temp_dir = f"temp/mvbench/{video_path.split('/')[-1]}"
os.makedirs(temp_dir, exist_ok=True)
for i, fimg in enumerate(frame_imgs):
    cv2.imwrite(f"{temp_dir}/{i}.jpg", fimg)

# for video_path in tqdm(video_paths):   
#     frame_imgs = read_frames_img(video_path, num_frames=8, client=petrel_client)

#     temp_dir = f"temp/sharegptvideo/{video_path.split('/')[-1]}"
#     os.makedirs(temp_dir, exist_ok=True)
#     for i, fimg in enumerate(frame_imgs):
#         cv2.imwrite(f"{temp_dir}/{i}.jpg", fimg)