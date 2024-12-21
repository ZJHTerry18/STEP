from src.tokenizer.multimodal_llama_tokenizer import MultimodalLlamaTokenizer
from src.share_utils.my_easydict import MyEasyDict as edict
from src.dataset.cap_dataset import VideoCapEvalDataset
from src.share_utils.py_config_utils import Config, eval_dict_leaf
from src.model import *
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import io
import torch
from petrel_client.client import Client
from src.dataset.pt_dataset import VidTxtPtTrainDataset
tokenizer = MultimodalLlamaTokenizer.from_pretrained(
    "/mnt/petrelfs/share/videointern/MLLM/Mistral-7B-Instruct-v0.3/", 
    local_files_only=True,
    n_query=96,
    v_query=96
) 

model_config_path = "/mnt/petrelfs/wangchenting/multimodalllm/configs/model/i1b_qformer_mistral7b_f8_hd.py"
model_cfg = eval_dict_leaf(Config.from_file(model_config_path))
from src.model.base_model import LLMConfig
config = LLMConfig.from_pretrained('/mnt/petrelfs/share/videointern/MLLM/Mistral-7B-Instruct-v0.3/', trust_remote_code=True)
config.model_config = model_cfg.model
model = MultiModalLLM_PT.from_pretrained('/mnt/lustre/share_data/wangchenting/models/internvideo2_videochat2/internvideo2_1b_mistral7b_stage4_hd_post_f16', config=config, trust_remote_code=True)

torch.save(model.state_dict(), 'internvideo2_1b_mistral_7b_stage4_hd_post_f8_interpolate_to_f16.pth')