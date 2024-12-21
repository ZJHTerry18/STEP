import torch
from safetensors import safe_open
from safetensors.torch import save_file

# tensors = {
#    "weight1": torch.zeros((1024, 1024)),
#    "weight2": torch.zeros((1024, 1024))
# }
# save_file(tensors, "model.safetensors")

tensors = {}
with safe_open("/mnt/petrelfs/wangchenting/multimodalllm/logs/scripts/pt/1b_qformer_mistral/stage1.sh_20240611_224630/checkpoint-4000/model.safetensors", framework="pt", device="cpu") as f:
   for key in f.keys():
       tensors[key] = f.get_tensor(key)

torch.save(tensors, 'model.bin')