import torch
from IPython import embed

ckpt_path = '/mnt/petrelfs/share_data/lixinhao/avp_1b_qformer_new_smit_e04.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')
embed()