from src.tokenizer.multimodal_llama_tokenizer import MultimodalLlamaTokenizer
tokenizer = MultimodalLlamaTokenizer.from_pretrained("/mnt/petrelfs/wangchenting/Mistral-7B-Instruct-v0.2", local_files_only=True) 
from src.share_utils.my_easydict import MyEasyDict as edict
from torchvision import transforms

# from src.dataset.pt_interleaved_dataset import InterleavedVidTxtPtTrainDataset

# Myd = InterleavedVidTxtPtTrainDataset(
#     ann_file=edict(
#         anno_path="/mnt/petrelfs/share_data/yujiashuo/data24-h1/interleaved/internvid2_interleaved_1M_avscap.json",
#         data_root="",
#         media_type="video",
#     ),
#     transform=transforms.Compose([
#         transforms.RandomResizedCrop(
#             224,
#             scale=(0.5, 1.0),
#             antialias=True
#         ),
#         transforms.RandomHorizontalFlip(),
#     ]),
#     tokenizer=tokenizer
# )
# for i in range(10):
#     data = Myd[i]
#     print(data['clips'].shape)
#     exit(0)

from src.dataset.cap_dataset import VideoCapEvalDataset
from src.dataset.pt_dataset import VidTxtPtTrainDataset
from src.dataset.it_eval import ITEvalDataset
from src.dataset.it_dataset import ITTrainDataset

Myd = ITTrainDataset(
    ann_file=edict(
        anno_path=f"/mnt/petrelfs/share_data/videointern/annotations/anno_instruction/videochat_new/image/caption/coco/train.json", 
        data_root="p2:s3://coco_caption",
    ),
    transform=transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.5, 1.0),
            antialias=True
        ),
        transforms.RandomHorizontalFlip(),
    ]),
    tokenizer=tokenizer
)

for i in range(10):
    data = Myd[i]
    breakpoint()
    exit(0)