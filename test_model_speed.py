from src.tokenizer.multimodal_llama_tokenizer import MultimodalLlamaTokenizer
from src.share_utils.my_easydict import MyEasyDict as edict
from src.dataset.pt_dataset import VidTxtPtTrainDataset
from src.dataset.pt_interleaved_dataset import InterleavedVidTxtPtTrainDataset
from src.share_utils.py_config_utils import Config, eval_dict_leaf
from src.model import *
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import io
import torch
import time
from petrel_client.client import Client
from torch.utils.data import DataLoader

tokenizer = MultimodalLlamaTokenizer.from_pretrained("BAAI/Emu2", local_files_only=True, 
        n_query=96,
        v_query=96) 

def get_transform():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean, std)

    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

    aug_transform = transforms.Lambda(lambda x: x)

    train_transform = transforms.Compose(
        [
            aug_transform,
            transforms.RandomResizedCrop(
                224,
                scale=(0.5, 1.0),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.RandomHorizontalFlip(),
            type_transform,
            normalize,
        ]
    )

    return train_transform


dataset = VidTxtPtTrainDataset(
    ann_file=edict(
        anno_path=f"/mnt/petrelfs/share/videointern/metas/S-MiT/caption_train.json", 
        data_root="pvideo:s3://S-MiT",
        media_type="video"
    ),
    transform=get_transform(),
    tokenizer=tokenizer
)



dataloader = DataLoader(
            dataset,
            # sampler=sampler,
            batch_size=4,
            # drop_last=self.args.dataloader_drop_last,
            num_workers=16,
            pin_memory=True,
        )


model_config_path = "configs/model/umt-L_qformer_vicuna7b_f4.py"
model_cfg = eval_dict_leaf(Config.from_file(model_config_path))
model = eval(model_cfg.model.get('model_cls'))(config=model_cfg.model, tokenizer=tokenizer)
model.cuda()
model.half()

torch.cuda.synchronize()
start_data_time = time.perf_counter()

data_times = []
compute_times = []

for inputs in dataloader:

    torch.cuda.synchronize()
    data_elapsed = time.perf_counter() - start_data_time
    data_times.append(data_elapsed)
    print(f'mean data time: {sum(data_times)/len(data_times)} now data time: {data_times[-1]}')

    labels = None
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs.get('attention_mask', None).to(model.device)
    input_labels = inputs.get('labels', None).to(model.device)

    # image = inputs['image']
    video = inputs['video'].to(model.device).half()

    torch.cuda.synchronize()
    start_compute_time = time.perf_counter()

    outputs = model(video=video, input_ids=input_ids, attention_mask=attention_mask, labels=input_labels)

    torch.cuda.synchronize()
    compute_elapsed = time.perf_counter() - start_compute_time
    compute_times.append(compute_elapsed)
    print(f'mean compute time: {sum(compute_times)/len(compute_times)} now compute time: {compute_times[-1]}')
    print('max cuda mem:', torch.cuda.max_memory_allocated() /(1024*1024))


    torch.cuda.synchronize()
    start_data_time = time.perf_counter()