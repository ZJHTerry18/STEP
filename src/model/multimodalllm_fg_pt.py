import io
import logging
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import MSELoss
import torch.nn.functional as F
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from timm.models.layers import trunc_normal_
from typing import List, Optional, Tuple, Union
# from transformers import LlamaForCausalLM
from .llm.llama_xformer import LlamaForCausalLM

from petrel_client.client import Client
from torch.cuda.amp import autocast as autocast

from ..share_utils.constants import *
from ..share_utils.func_utils import freeze_module

from .vision_encoder import pretrain_internvideo2_giant_patch14_224_clean, build_vit
from .vision_encoder.pos_embed import interpolate_pos_embed_internvideo2_new
from .bridge import build_qformer, build_causal_qformer

from .base_model import BaseMLLM
from .sa_layer import SelfAttentionBlock

logger = logging.getLogger(__name__)


class MultiModalLLM_fg_PT(BaseMLLM):
    
    def __init__(
        self,
        config
    ):
        super().__init__(config=config)
        
        ### configs for fine-grained training
        self.num_qformer_token = self.model_config.bridge.num_query_token + self.extra_num_query_token
        self.num_global_token = self.model_config.bridge.num_global_token
        self.num_token_group = self.model_config.bridge.num_token_group
        self.token_per_group = self.num_qformer_token // self.num_token_group
        self.fix_text_feat = self.model_config.bridge.get('fix_text_feat', True)
        self.qformer_text_input = self.model_config.bridge.get('qformer_text_input', False)
        
        ### initialize query tokens
        self.global_query_tokens = nn.Parameter(
            torch.zeros(1, self.num_global_token, self.query_tokens.shape[-1])
        )
        
        ### new modules for fine-grained training
        self.temp = nn.parameter.Parameter(torch.ones([]) * self.model_config.bridge.temp)
        self.v_proj = nn.Linear(self.qformer.config.hidden_size, self.model_config.bridge.embed_dim)
        self.t_proj = nn.Linear(self.qformer.config.hidden_size, self.model_config.bridge.embed_dim)
        self.vtm_w = self.model_config.loss.get('vtm_w', 0.0)
        self.vtc_w = self.model_config.loss.get('vtc_w', 0.0)
        self.vidtc_w = self.model_config.loss.get('vidtc_w', 0.0)
        if self.vtm_w > 0:
            vtm_layers = [
                nn.Linear(2 * self.qformer.config.hidden_size, self.qformer.config.hidden_size),
                nn.Linear(self.qformer.config.hidden_size, 2)
            ]
            self.vtm_head = nn.Sequential(*vtm_layers)
            for i in range(len(self.vtm_head)):
                self.vtm_head[i].apply(self._init_weights)
        self.llm_w = self.model_config.loss.get('llm_w', 1.0)
        
        self.v_proj.apply(self._init_weights)
        self.t_proj.apply(self._init_weights)
            
        self.sa_block = SelfAttentionBlock(
            embed_size=self.qformer.config.hidden_size, num_heads=self.model_config.bridge.sa_heads, num_layers=self.model_config.bridge.sa_layers
        )
        self.sa_block._param_init()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def build_bridge(self):
        # ViT to LM: 1792 -> 6656 NOTE 768 is qformer dim
        self.project_up = nn.Linear(768, self.lm.config.hidden_size) # whether bias is needed?
        # LM to ViT: 6656 -> 1792
        self.project_down = nn.Linear(self.lm.config.hidden_size, 768)
        
        if 'qformer' in self.model_config.bridge.name.lower():
            from transformers import BertTokenizer
            self.qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="left", local_files_only=True)
            self.qformer_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            self.qformer_tokenizer.padding_side = "left"
            if self.model_config.bridge.name == 'qformer':
                self.qformer, self.query_tokens = build_qformer(
                        self.model_config.bridge.num_query_token, self.model_config.vision_encoder.encoder_embed_dim,
                        qformer_hidden_dropout_prob=self.model_config.bridge.qformer_hidden_dropout_prob,
                        qformer_attention_probs_dropout_prob=self.model_config.bridge.qformer_attention_probs_dropout_prob,
                        qformer_drop_path_rate=self.model_config.bridge.qformer_drop_path_rate,
                )
            elif self.model_config.bridge.name == 'causal_qformer':
                self.qformer, self.query_tokens = build_causal_qformer(
                        self.model_config.bridge.num_query_token, self.model_config.vision_encoder.encoder_embed_dim,
                        qformer_hidden_dropout_prob=self.model_config.bridge.qformer_hidden_dropout_prob,
                        qformer_attention_probs_dropout_prob=self.model_config.bridge.qformer_attention_probs_dropout_prob
                )
            print('len(self.qformer_tokenizer): ', len(self.qformer_tokenizer))
            self.qformer.bert.resize_token_embeddings(len(self.qformer_tokenizer))
            self.qformer.resize_token_embeddings(len(self.qformer_tokenizer))
            self.qformer.cls = None
            self.extra_num_query_token = self.model_config.bridge.extra_num_query_token
            if self.model_config.bridge.extra_num_query_token > 0:
                logger.info(f"Add extra {self.model_config.bridge.extra_num_query_token} tokens in QFormer")
                self.extra_query_tokens = nn.Parameter(
                    torch.zeros(1, self.model_config.bridge.extra_num_query_token, self.query_tokens.shape[-1])
                )
            
            self.freeze_bridge = self.model_config.get("freeze_bridge", False)
            if self.freeze_bridge:
                logger.info("freeze bridge")
                freeze_module(self.qformer)
                self.query_tokens.requires_grad = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        instruction = None,
        fg_text = None,
        video_idx = None,
        image_idx = None,
        **kwargs,
    ):  
        ### pre-process fine-grained text
        B = input_ids.size(0)
        fg_text_list = [x.split('|') for x in fg_text]
                
        assert len(fg_text_list) == B
        assert all([len(x) >= 2 for x in fg_text_list])
        
        ### llm input embedding
        group_vq_embeds, llm_text_embeds, video_frame_att_feats = self.pad_text_embeds(
            input_ids=input_ids, image=image, video=video, 
            image_idx=image_idx, video_idx=video_idx, instruction=instruction
        )
        
        # LLM forwards
        outputs = self.lm(
            inputs_embeds=llm_text_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        
        ### fine-grained visual-text alignment loss
        loss_dict = self.fg_loss(
            group_vq_embeds, fg_text_list, video_frame_att_feats
        )
        # loss aggregation
        loss_dict['llm'] = outputs.loss * self.llm_w
        total_loss = 0.0
        for k, v in loss_dict.items():
            total_loss += v
        outputs['loss'] = total_loss
        outputs['loss_dict'] = loss_dict
        
        return outputs

    def fg_loss(
        self,
        group_vq_embeds,
        fg_text_list,
        vf_att_feats, # [B, n_g, T, C]
    ):
        B = len(fg_text_list)
        
        ### filter out text that exceed query number
        for i in range(len(fg_text_list)):
            if len(fg_text_list[i]) > self.num_token_group:
                fg_text_list[i] = fg_text_list[i][:self.num_token_group]
        
        ### q-former fine-grained text encoding
        fg_text_nums = [len(tl) for tl in fg_text_list] # store the number of text segments in each instance
        fg_text = list(itertools.chain(*fg_text_list)) # flatten all text segments into a single list, for more efficient computing
        fg_text_tokenize, fg_text_embeds = self.encode_fg_text(fg_text, device=group_vq_embeds.device) # [BM, c_t]
        if self.fix_text_feat: # Not sure, should the text features back propagate to the q-former?
            fg_text_embeds = fg_text_embeds.detach() 
        
        ### bipartite matching of vision & fine-grained text
        group_vq_mean_embeds = group_vq_embeds.mean(dim=1) # [Bxn_g, c_q]
        vq_nums = [self.num_token_group for _ in range(B)]
        vq_proj = F.normalize(self.v_proj(group_vq_mean_embeds), dim=-1)
        tq_proj = F.normalize(self.t_proj(fg_text_embeds), dim=-1)
        visual_indices, text_indices, sim_mat = self.batch_bipartite_matching(
            vq_proj, tq_proj, vq_nums, fg_text_nums
        ) # [[1,3],[6,7],[9,11,15]]; [[0,1],[2,3],[4,5,6]]
        # print('visual indices', visual_indices)
        # print('text indices', text_indices)
        
        ### vision & fine-grained text alignment loss
        loss_dict = dict()
        visual_indices_all = list(itertools.chain(*visual_indices))
        text_indices_all = list(itertools.chain(*text_indices))
        sim_mat_match = sim_mat[visual_indices_all, :][:, text_indices_all]
        vt_nums = [len(l) for l in visual_indices]
        # visual-text contrastive learning
        if self.vtc_w > 0:
            vq_proj_match = vq_proj[visual_indices_all] # [N, c_q]
            tq_proj_match = tq_proj[text_indices_all] # [N, c_q]
            vtc_loss = self.vtc_loss(
                vq_proj_match, tq_proj_match, vt_nums, temp=self.temp, sim_mat=sim_mat_match
            )
            loss_dict['vtc'] = vtc_loss * self.vtc_w
        
        # visual-text matching
        if self.vtm_w > 0:
            # vq_embeds_match = group_vq_mean_embeds[visual_indices_all]
            # tq_embeds_match = fg_text_embeds[text_indices_all]
            vtm_loss = self.fg_vtm_loss(
                vq_proj_match, tq_proj_match, vt_nums, temp=self.temp, sim_mat=sim_mat_match
            )
            loss_dict['vtm'] = vtm_loss * self.vtm_w
        
        ### vision token temporal consistency loss
        if self.vidtc_w > 0:
            vidtc_loss = self.vid_tc_loss(
                vf_att_feats, vt_nums
            )
            loss_dict['vidtc'] = vidtc_loss * self.vidtc_w
        
        return loss_dict
    
    def pad_text_embeds(
        self,
        input_ids: torch.LongTensor = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        image_idx = None,
        video_idx = None,
        instruction = None,
    ):
        ### q-former vision encoding
        if image is not None:
            B, T, C, H, W = image.shape
            image = image.permute(0, 2, 1, 3, 4)
            vision_query_embeds, vit_embeds, visual_attention_maps = self.encode_fg_vision(image, instruction=instruction)
            B, L, C = vit_embeds.shape
        elif video is not None:
            if len(video.shape) == 5:
                B, T, C, H, W = video.shape
                N = 1
            else:
                B, N, T, C, H, W = video.shape

            video = video.reshape(-1, T, C, H, W).permute(0, 2, 1, 3, 4) # [B, C, T, H, W]
            B = video.size(0)
            vision_query_embeds, vit_embeds, visual_attention_maps = self.encode_fg_vision(
                video, instruction=instruction
            ) # [B, n_q, c_q], [B, L, C], [B, n_q, L]
            B, L, C = vit_embeds.shape
        else:
            logger.error(f"don't get visual input, input_ids: {input_ids}")
        
        ### video frame-wise attention-weighted features
        vit_embeds = vit_embeds.permute(0, 2, 1)[:, :, 1:] # [B, C, HW]
        video_att_maps = visual_attention_maps[:, :, 1:].view(B, self.num_qformer_token, T, -1) # [B, n_q, T, HW]
        video_att_feats = torch.einsum('bctn,bqtn->bqctn', vit_embeds.view(B, C, T, -1), video_att_maps) # [B, n_q, C, T, HW]
        video_frame_att_feats = video_att_feats.mean(dim=-1).permute(0, 1, 3, 2) # [B, n_q, T, C]
        
        ### global vision token aggregation
        group_vq_embeds = vision_query_embeds.reshape(B * self.num_token_group, self.token_per_group, -1)
        local_vision_embeds = group_vq_embeds.reshape(B, self.num_qformer_token, -1) # [B, n_q, c_q]
        global_vision_query = self.global_query_tokens.expand(B, -1, -1)
        sa_query = torch.cat([global_vision_query, local_vision_embeds], dim=1) # [B, n_global + n_q, c_q]
        sa_output = self.sa_block(sa_query) # [B, n_global + n_q, c_q]
        
        # pad text embedding for LLM input
        prompt_visual_embeds = torch.cat([sa_output[:, :global_vision_query.size(1), :], local_vision_embeds], dim=1) # [B, n_global + n_q, c_q]
        prompt_visual_embeds = self.project_up(prompt_visual_embeds)
        prompt_visual_embeds = prompt_visual_embeds.view(-1, prompt_visual_embeds.shape[-1])
        text_embeds = self.lm.get_input_embeddings()(input_ids.long()).detach()
        if image is not None:
            visual_idx = image_idx
        elif video is not None:
            visual_idx = video_idx
        text_embeds[visual_idx == 1] = text_embeds[visual_idx == 1] * 0 + prompt_visual_embeds.to(text_embeds.device).to(text_embeds.dtype)
        
        return group_vq_embeds, text_embeds, video_frame_att_feats

    def encode_fg_vision(
        self,
        image,
        instruction
    ):
        device = image.device
        B = image.shape[0]
        T = image.shape[2]
        use_image = True if T == 1 else False
        image_embeds = self.vision_encoder(image, use_image=use_image)
        C = image_embeds.shape[-1]
        image_embeds = image_embeds.reshape(B, -1, C)
        image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, T*L, C]
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        if self.extra_num_query_token > 0:
            query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
        query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input and instruction is not None:
            # print(instruction)
            text_Qformer = self.qformer_tokenizer(
                instruction,
                padding='longest',
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(image_embeds.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
            # print(query_atts.shape, text_Qformer.attention_mask.shape)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
            query_output = self.qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                output_attentions=True,
            )
        else:
            query_output = self.qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                output_attentions=True,
            )
        
        cross_attentions = query_output.cross_attentions
        all_cross_attentions = []
        for i in range(len(cross_attentions)):
            ca = cross_attentions[i]
            if isinstance(ca, torch.Tensor):
                layer_ca = ca.mean(dim=1, keepdim=True) # [B, 1, num_query, num_patches]
                all_cross_attentions.append(layer_ca)

        avg_cross_atts = torch.cat(all_cross_attentions, dim=1).sum(dim=1) # [B, num_query, num_patches]
        
        return query_output.last_hidden_state[:, :query_tokens.size(1), :], avg_cross_atts

    def encode_fg_text(
        self,
        fg_text,
        device
    ):
        """encode text.
        Args:
            - fg_text: List of fine-grained text segments in each video.
        Returns: tuple.
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [BL,C].

        """
        BL = len(fg_text)
        fg_text_Qformer = self.qformer_tokenizer(
            fg_text,
            padding='longest',
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        fg_text_output = self.qformer.bert(
            fg_text_Qformer.input_ids,
            attention_mask=fg_text_Qformer.attention_mask,
            return_dict=True,
        )
        fg_text_embeds = fg_text_output.last_hidden_state
        pooled_fg_text_embeds = fg_text_embeds[:, 0]
        return fg_text_Qformer, pooled_fg_text_embeds
    
    def batch_bipartite_matching(
        self,
        visual_feats,
        text_feats,
        visual_nums,
        text_nums,
    ):
        bs = len(visual_nums)
        sim_mat = torch.matmul(visual_feats, text_feats.T)
        cost_mat = -sim_mat.detach().to(torch.float).cpu().numpy()
        visual_accu_nums = np.zeros(bs + 1, dtype=int)
        text_accu_nums = np.zeros(bs + 1, dtype=int)
        visual_accu_nums[1:] = np.cumsum(visual_nums, dtype=int)
        text_accu_nums[1:] = np.cumsum(text_nums, dtype=int)
        visual_indices = []
        text_indices = []
        for bi in range(len(visual_nums)):
            v_s, v_e = visual_accu_nums[bi], visual_accu_nums[bi + 1]
            t_s, t_e = text_accu_nums[bi], text_accu_nums[bi + 1]
            sample_cost_mat = cost_mat[v_s:v_e, t_s:t_e]
            v_inds, t_inds = linear_sum_assignment(sample_cost_mat)
            v_inds = v_inds + v_s
            t_inds = t_inds + t_s
            
            visual_indices.append(v_inds)
            text_indices.append(t_inds)
        
        return visual_indices, text_indices, sim_mat
    
    def vtc_loss(
        self,
        visual_feats,
        text_feats,
        vt_nums,
        temp=1.0,
        sim_mat=None,
    ):
        if sim_mat is None:
            raise NotImplementedError
        else:
            sim_v2t = sim_mat / temp
            sim_t2v = sim_mat.T / temp
        
        vt_startends = np.zeros(len(vt_nums) + 1, dtype=int)
        vt_startends[1:] = np.cumsum(vt_nums)
        with torch.no_grad():
            label_v2t = torch.zeros_like(sim_v2t).fill_diagonal_(1)
            label_t2v = label_v2t

            mask_v2t = torch.zeros_like(sim_v2t)
            for bi in range(len(vt_nums)):
                mask_v2t[vt_startends[bi]:vt_startends[bi + 1], vt_startends[bi]:vt_startends[bi + 1]] = 1
            mask_t2v = mask_v2t.T
        
        sim_v2t = sim_v2t.masked_fill(mask_v2t == 0, -1e9)
        sim_t2v = sim_t2v.masked_fill(mask_t2v == 0, -1e9)
        # print("sim_v2t:")
        # print(sim_v2t[:20, :20])
        # print("sim_v2t logsoftmax")
        # print((F.log_softmax(sim_v2t, dim=1) * label_v2t)[:20, :20])
        # print("label_v2t:")
        # print(label_v2t[:20, :20])
        loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1) * label_v2t, dim=1).mean()
        loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1) * label_t2v, dim=1).mean()
        
        loss_vtc = (loss_v2t + loss_t2v) / 2.0
        return loss_vtc
    
    def fg_vtm_loss(
        self,
        visual_feats, # query
        text_feats,
        vt_nums,
        temp=1.0,
        sim_mat=None,
    ):
        if sim_mat is None:
            raise NotImplementedError
        else:
            sim_v2t = sim_mat / temp
            sim_t2v = sim_mat.T / temp
        
        vt_startends = np.zeros(len(vt_nums) + 1, dtype=int)
        vt_startends[1:] = np.cumsum(vt_nums)
        with torch.no_grad():
            label_v2t = torch.zeros_like(sim_v2t).fill_diagonal_(1)
            label_t2v = label_v2t
            
            mask_v2t = torch.zeros_like(sim_v2t)
            for bi in range(len(vt_nums)):
                mask_v2t[vt_startends[bi]:vt_startends[bi + 1], vt_startends[bi]:vt_startends[bi + 1]] = 1
            mask_t2v = mask_v2t.T
            sim_v2t.masked_fill_(mask_v2t == 0, -1e9)
            sim_t2v.masked_fill_(mask_t2v == 0, -1e9)
            
            weights_v2t = F.softmax(sim_v2t, dim=1) + 1e-4
            weights_v2t.masked_fill_(mask_v2t == 0, 0).masked_fill_(label_v2t == 1, 0)
            weights_v2t = torch.nan_to_num_(
                weights_v2t, nan=1e-2, posinf=1e-2, neginf=1e-2
            )
            weights_t2v = F.softmax(sim_t2v, dim=1) + 1e-4
            weights_t2v.masked_fill_(mask_t2v == 0, 0).masked_fill_(label_t2v == 1, 0)
            weights_t2v = torch.nan_to_num_(
                weights_t2v, nan=1e-2, posinf=1e-2, neginf=1e-2
            )
        
        # select a negative video feature for each text
        visual_neg_indices = torch.multinomial(weights_t2v, 1).squeeze()
        # select a negative text for each video feature
        text_neg_indices = torch.multinomial(weights_v2t, 1).squeeze()
        
        visual_feats_neg = visual_feats[visual_neg_indices] # [N, c_q]
        text_feats_neg = text_feats[text_neg_indices] # [N, c_t]
        
        # concat video pos and neg
        visual_feats_pos_neg = torch.cat(
            [visual_feats, visual_feats_neg, visual_feats], dim=0
        ) # [3N, c_q]  
        # concat text pos and neg
        text_feats_pos_neg = torch.cat(
            [text_feats, text_feats, text_feats_neg], dim=0
        ) # [3N, c_t]
        vtm_feats = torch.cat([visual_feats_pos_neg, text_feats_pos_neg], dim=1) # [3N, c_q + c_t]
        
        logits = self.vtm_head(vtm_feats)
        bs = logits.size(0) // 3
        labels = logits.new_ones(logits.size(0), dtype=torch.long)
        labels[bs:] = 0
        # print("logits")
        # print(logits.size(), logits)
        # print("labels")
        # print(labels.size(), labels)
        loss_vtm = F.cross_entropy(logits, labels)
        
        return loss_vtm
    
    def vid_tc_loss(
        self,
        video_feats, # [B, n_g, T, C]
        vt_nums,
    ):
        video_feats = F.normalize(video_feats, dim=-1)
        video_feats_c = video_feats.mean(dim=2) # feature center for each query group: [B, n_g, C]
        sim_vf_vfc = torch.einsum('bntc,bnc->bntn', video_feats, video_feats_c).permute(0, 2, 1, 3) # [B, T, n_g, n_g]
        sim_vfc_vf = sim_vf_vfc.transpose(2, 3)
        num_g = sim_vf_vfc.size(-1)
        label_vf_vfc = torch.zeros_like(sim_vf_vfc)
        label_vf_vfc[:, :, :, :] = torch.eye(num_g)
        label_vfc_vf = label_vf_vfc.transpose(2, 3)
        
        loss_vf_vfc = -torch.sum(F.log_softmax(sim_vf_vfc, dim=3) * label_vf_vfc, dim=2).mean()
        loss_vfc_vf = -torch.sum(F.log_softmax(sim_vfc_vf, dim=3) * label_vfc_vf, dim=2).mean()
        loss_vid_tc = (loss_vf_vfc + loss_vfc_vf) / 2.0
        
        return loss_vid_tc
    
    def generate_caption(
        self,
        input_ids,
        attention_mask,
        image_idx = None,
        video_idx = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        num_beams=1,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.9,
        top_k=None,
        temperature=1.0,
        length_penalty=1,
        repetition_penalty=1.0,
    ):
        ### llm input embedding
        _, llm_text_embeds, _ = self.pad_text_embeds(
            input_ids=input_ids, image=image, video=video, 
            image_idx=image_idx, video_idx=video_idx
        )
        outputs = self.lm.generate(
            inputs_embeds=llm_text_embeds,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            min_length=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )

        return outputs

class MultiModalLLM_fg_PT_2(BaseMLLM):
    
    def __init__(
        self,
        config
    ):
        super().__init__(config=config)
        
        ### configs for fine-grained training
        self.num_qformer_token = self.model_config.bridge.num_query_token + self.extra_num_query_token
        self.num_global_token = self.model_config.bridge.num_global_token
        self.num_local_token = self.num_qformer_token - self.num_global_token
        self.num_token_group = self.model_config.bridge.num_token_group
        self.token_per_group = self.num_local_token // self.num_token_group
        self.fix_text_feat = self.model_config.bridge.get('fix_text_feat', True)
        self.qformer_text_input = self.model_config.bridge.get('qformer_text_input', False)
        
        ### new modules for fine-grained training
        self.temp = nn.parameter.Parameter(torch.ones([]) * self.model_config.bridge.vtc_temp)
        self.vidtc_temp = nn.parameter.Parameter(torch.ones([]) * self.model_config.bridge.vidtc_temp)
        self.v_proj = nn.Linear(self.qformer.config.hidden_size, self.model_config.bridge.embed_dim)
        self.t_proj = nn.Linear(self.qformer.config.hidden_size, self.model_config.bridge.embed_dim)
        self.vtm_w = self.model_config.loss.get('vtm_w', 0.0)
        self.vtc_w = self.model_config.loss.get('vtc_w', 0.0)
        self.vidtc_w = self.model_config.loss.get('vidtc_w', 0.0)
        if self.vtm_w > 0:
            vtm_layers = [
                nn.Linear(2 * self.qformer.config.hidden_size, self.qformer.config.hidden_size),
                nn.Linear(self.qformer.config.hidden_size, 2)
            ]
            self.vtm_head = nn.Sequential(*vtm_layers)
            for i in range(len(self.vtm_head)):
                self.vtm_head[i].apply(self._init_weights)
        self.llm_w = self.model_config.loss.get('llm_w', 1.0)
        
        self.v_proj.apply(self._init_weights)
        self.t_proj.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def build_bridge(self):
        # ViT to LM: 1792 -> 6656 NOTE 768 is qformer dim
        self.project_up = nn.Linear(768, self.lm.config.hidden_size) # whether bias is needed?
        # # LM to ViT: 6656 -> 1792
        # self.project_down = nn.Linear(self.lm.config.hidden_size, 768)
        
        if 'qformer' in self.model_config.bridge.name.lower():
            from transformers import BertTokenizer
            self.qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="left", local_files_only=True)
            self.qformer_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            self.qformer_tokenizer.padding_side = "left"
            if self.model_config.bridge.name == 'qformer':
                self.qformer, self.query_tokens = build_qformer(
                        self.model_config.bridge.num_query_token, self.model_config.vision_encoder.encoder_embed_dim,
                        qformer_hidden_dropout_prob=self.model_config.bridge.qformer_hidden_dropout_prob,
                        qformer_attention_probs_dropout_prob=self.model_config.bridge.qformer_attention_probs_dropout_prob,
                        qformer_drop_path_rate=self.model_config.bridge.qformer_drop_path_rate,
                )
            elif self.model_config.bridge.name == 'causal_qformer':
                self.qformer, self.query_tokens = build_causal_qformer(
                        self.model_config.bridge.num_query_token, self.model_config.vision_encoder.encoder_embed_dim,
                        qformer_hidden_dropout_prob=self.model_config.bridge.qformer_hidden_dropout_prob,
                        qformer_attention_probs_dropout_prob=self.model_config.bridge.qformer_attention_probs_dropout_prob
                )
            print('len(self.qformer_tokenizer): ', len(self.qformer_tokenizer))
            self.qformer.bert.resize_token_embeddings(len(self.qformer_tokenizer))
            self.qformer.resize_token_embeddings(len(self.qformer_tokenizer))
            self.qformer.cls = None
            self.extra_num_query_token = self.model_config.bridge.extra_num_query_token
            if self.model_config.bridge.extra_num_query_token > 0:
                logger.info(f"Add extra {self.model_config.bridge.extra_num_query_token} tokens in QFormer")
                self.extra_query_tokens = nn.Parameter(
                    torch.zeros(1, self.model_config.bridge.extra_num_query_token, self.query_tokens.shape[-1])
                )
            
            self.freeze_bridge = self.model_config.get("freeze_bridge", False)
            if self.freeze_bridge:
                logger.info("freeze bridge")
                freeze_module(self.qformer)
                self.query_tokens.requires_grad = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        instruction = None,
        fg_text = None,
        video_idx = None,
        image_idx = None,
        **kwargs,
    ):  
        ### pre-process fine-grained text
        B = input_ids.size(0)
        fg_text_list = [x.split('|') for x in fg_text]
                
        assert len(fg_text_list) == B
        assert all([len(x) >= 2 for x in fg_text_list])
        
        ### llm input embedding
        group_vq_embeds, llm_text_embeds, video_frame_att_feats = self.pad_text_embeds(
            input_ids=input_ids, image=image, video=video, 
            image_idx=image_idx, video_idx=video_idx, instruction=instruction
        )
        
        # LLM forwards
        outputs = self.lm(
            inputs_embeds=llm_text_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        
        ### fine-grained visual-text alignment loss
        loss_dict = self.fg_loss(
            group_vq_embeds, fg_text_list, video_frame_att_feats
        )
        # loss aggregation
        loss_dict['llm'] = outputs.loss * self.llm_w
        total_loss = 0.0
        for k, v in loss_dict.items():
            total_loss += v
        outputs['loss'] = total_loss
        outputs['loss_dict'] = loss_dict
        
        return outputs

    def fg_loss(
        self,
        group_vq_embeds,
        fg_text_list,
        vf_att_feats,
    ):
        B = len(fg_text_list)
        
        ### filter out text that exceed query number
        for i in range(len(fg_text_list)):
            if len(fg_text_list[i]) > self.num_token_group:
                fg_text_list[i] = fg_text_list[i][:self.num_token_group]
        
        ### q-former fine-grained text encoding
        fg_text_nums = [len(tl) for tl in fg_text_list] # store the number of text segments in each instance
        fg_text = list(itertools.chain(*fg_text_list)) # flatten all text segments into a single list, for more efficient computing
        fg_text_tokenize, fg_text_embeds = self.encode_fg_text(fg_text, device=group_vq_embeds.device) # [BM, c_t]
        if self.fix_text_feat: # Not sure, should the text features back propagate to the q-former?
            fg_text_embeds = fg_text_embeds.detach() 
        
        ### bipartite matching of vision & fine-grained text
        group_vq_mean_embeds = group_vq_embeds.mean(dim=1) # [Bxn_g, c_q]
        vq_nums = [self.num_token_group for _ in range(B)]
        vq_proj = F.normalize(self.v_proj(group_vq_mean_embeds), dim=-1)
        tq_proj = F.normalize(self.t_proj(fg_text_embeds), dim=-1)
        visual_indices, text_indices, sim_mat = self.batch_bipartite_matching(
            vq_proj, tq_proj, vq_nums, fg_text_nums
        ) # [[1,3],[6,7],[9,11,15]]; [[0,1],[2,3],[4,5,6]]
        # print('visual indices', visual_indices)
        # print('text indices', text_indices)
        
        ### vision & fine-grained text alignment loss
        loss_dict = dict()
        visual_indices_all = list(itertools.chain(*visual_indices))
        text_indices_all = list(itertools.chain(*text_indices))
        sim_mat_match = sim_mat[visual_indices_all, :][:, text_indices_all]
        vt_nums = [len(l) for l in visual_indices]
        # visual-text contrastive learning
        if self.vtc_w > 0:
            vq_proj_match = vq_proj[visual_indices_all] # [N, c_q]
            tq_proj_match = tq_proj[text_indices_all] # [N, c_q]
            vtc_loss = self.vtc_loss(
                vq_proj_match, tq_proj_match, vt_nums, temp=self.temp, sim_mat=sim_mat_match
            )
            loss_dict['vtc'] = vtc_loss * self.vtc_w
        
        # visual-text matching
        if self.vtm_w > 0:
            # vq_embeds_match = group_vq_mean_embeds[visual_indices_all]
            # tq_embeds_match = fg_text_embeds[text_indices_all]
            vtm_loss = self.fg_vtm_loss(
                vq_proj_match, tq_proj_match, vt_nums, temp=self.temp, sim_mat=sim_mat_match
            )
            loss_dict['vtm'] = vtm_loss * self.vtm_w
        
        ### vision token temporal consistency loss
        if self.vidtc_w > 0:
            vidtc_loss = self.vid_tc_loss(
                vf_att_feats, vt_nums, temp=self.vidtc_temp,
            )
            loss_dict['vidtc'] = vidtc_loss * self.vidtc_w
        
        return loss_dict
    
    def pad_text_embeds(
        self,
        input_ids: torch.LongTensor = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        image_idx = None,
        video_idx = None,
        instruction = None,
    ):
        ### q-former vision encoding
        if image is not None:
            B, T, C, H, W = image.shape
            image = image.permute(0, 2, 1, 3, 4)
            vision_query_embeds, vit_embeds, visual_attention_maps = self.encode_fg_vision(image, instruction=instruction)
            B, L, C = vit_embeds.shape
        elif video is not None:
            if len(video.shape) == 5:
                B, T, C, H, W = video.shape
                N = 1
            else:
                B, N, T, C, H, W = video.shape

            video = video.reshape(-1, T, C, H, W).permute(0, 2, 1, 3, 4) # [B, C, T, H, W]
            B = video.size(0)
            vision_query_embeds, vit_embeds, visual_attention_maps = self.encode_fg_vision(
                video, instruction=instruction
            ) # [B, n_q, c_q], [B, L, C], [B, n_q, L]
            B, L, C = vit_embeds.shape
        else:
            logger.error(f"don't get visual input, input_ids: {input_ids}")
        
        ### video frame-wise attention-weighted features
        vit_embeds = vit_embeds.permute(0, 2, 1)[:, :, 1:] # [B, C, HW]
        video_att_maps = visual_attention_maps[:, :, 1:].view(B, self.num_qformer_token, T, -1) # [B, n_q, T, HW]
        video_att_feats = torch.einsum('bctn,bqtn->bqctn', vit_embeds.view(B, C, T, -1), video_att_maps) # [B, n_q, C, T, HW]
        video_frame_att_feats = video_att_feats.mean(dim=-1).permute(0, 1, 3, 2) # [B, n_q, T, C]
        
        ### local global feature
        local_vq_embeds = vision_query_embeds[:, :self.num_local_token, :]
        group_vq_embeds =  \
            local_vq_embeds.reshape(B, self.num_token_group, self.token_per_group, -1).reshape(B * self.num_token_group, self.token_per_group, -1)
        
        # pad text embedding for LLM input
        prompt_visual_embeds = self.project_up(vision_query_embeds)
        prompt_visual_embeds = prompt_visual_embeds.view(-1, prompt_visual_embeds.shape[-1])
        text_embeds = self.lm.get_input_embeddings()(input_ids.long()).detach()
        if image is not None:
            visual_idx = image_idx
        elif video is not None:
            visual_idx = video_idx
        text_embeds[visual_idx == 1] = text_embeds[visual_idx == 1] * 0 + prompt_visual_embeds.to(text_embeds.device).to(text_embeds.dtype)
        
        return group_vq_embeds, text_embeds, video_frame_att_feats

    def encode_fg_vision(
        self,
        image,
        instruction
    ):
        device = image.device
        B = image.shape[0]
        T = image.shape[2]
        use_image = True if T == 1 else False
        image_embeds = self.vision_encoder(image, use_image=use_image)
        C = image_embeds.shape[-1]
        image_embeds = image_embeds.reshape(B, -1, C)
        image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, T*L, C]
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        if self.extra_num_query_token > 0:
            query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
        query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input and instruction is not None:
            # print(instruction)
            text_Qformer = self.qformer_tokenizer(
                instruction,
                padding='longest',
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(image_embeds.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
            # print(query_atts.shape, text_Qformer.attention_mask.shape)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
            query_output = self.qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                output_attentions=True,
            )
        else:
            query_output = self.qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                output_attentions=True,
            )
        
        cross_attentions = query_output.cross_attentions
        all_cross_attentions = []
        for i in range(len(cross_attentions)):
            ca = cross_attentions[i]
            if isinstance(ca, torch.Tensor):
                layer_ca = ca.mean(dim=1, keepdim=True) # [B, 1, num_query, num_patches]
                all_cross_attentions.append(layer_ca)

        avg_cross_atts = torch.cat(all_cross_attentions, dim=1).sum(dim=1) # [B, num_query, num_patches]
        
        return query_output.last_hidden_state[:, :query_tokens.size(1), :], image_embeds, avg_cross_atts

    def encode_fg_text(
        self,
        fg_text,
        device
    ):
        """encode text.
        Args:
            - fg_text: List of fine-grained text segments in each video.
        Returns: tuple.
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [BL,C].

        """
        BL = len(fg_text)
        fg_text_Qformer = self.qformer_tokenizer(
            fg_text,
            padding='longest',
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        fg_text_output = self.qformer.bert(
            fg_text_Qformer.input_ids,
            attention_mask=fg_text_Qformer.attention_mask,
            return_dict=True,
        )
        fg_text_embeds = fg_text_output.last_hidden_state
        pooled_fg_text_embeds = fg_text_embeds[:, 0]
        return fg_text_Qformer, pooled_fg_text_embeds
    
    def batch_bipartite_matching(
        self,
        visual_feats,
        text_feats,
        visual_nums,
        text_nums,
    ):
        bs = len(visual_nums)
        sim_mat = torch.matmul(visual_feats, text_feats.T)
        cost_mat = -sim_mat.detach().to(torch.float).cpu().numpy()
        visual_accu_nums = np.zeros(bs + 1, dtype=int)
        text_accu_nums = np.zeros(bs + 1, dtype=int)
        visual_accu_nums[1:] = np.cumsum(visual_nums, dtype=int)
        text_accu_nums[1:] = np.cumsum(text_nums, dtype=int)
        visual_indices = []
        text_indices = []
        for bi in range(len(visual_nums)):
            v_s, v_e = visual_accu_nums[bi], visual_accu_nums[bi + 1]
            t_s, t_e = text_accu_nums[bi], text_accu_nums[bi + 1]
            sample_cost_mat = cost_mat[v_s:v_e, t_s:t_e]
            v_inds, t_inds = linear_sum_assignment(sample_cost_mat)
            v_inds = v_inds + v_s
            t_inds = t_inds + t_s
            
            visual_indices.append(v_inds)
            text_indices.append(t_inds)
        
        return visual_indices, text_indices, sim_mat
    
    def vtc_loss(
        self,
        visual_feats,
        text_feats,
        vt_nums,
        temp=1.0,
        sim_mat=None,
    ):
        if sim_mat is None:
            raise NotImplementedError
        else:
            sim_v2t = sim_mat / temp
            sim_t2v = sim_mat.T / temp
        
        vt_startends = np.zeros(len(vt_nums) + 1, dtype=int)
        vt_startends[1:] = np.cumsum(vt_nums)
        with torch.no_grad():
            label_v2t = torch.zeros_like(sim_v2t).fill_diagonal_(1)
            label_t2v = label_v2t

            mask_v2t = torch.zeros_like(sim_v2t)
            for bi in range(len(vt_nums)):
                mask_v2t[vt_startends[bi]:vt_startends[bi + 1], vt_startends[bi]:vt_startends[bi + 1]] = 1
            mask_t2v = mask_v2t.T
        
        sim_v2t = sim_v2t.masked_fill(mask_v2t == 0, -1e9)
        sim_t2v = sim_t2v.masked_fill(mask_t2v == 0, -1e9)
        # print("sim_v2t:")
        # print(sim_v2t[:20, :20])
        # print("sim_v2t logsoftmax")
        # print((F.log_softmax(sim_v2t, dim=1) * label_v2t)[:20, :20])
        # print("label_v2t:")
        # print(label_v2t[:20, :20])
        loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1) * label_v2t, dim=1).mean()
        loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1) * label_t2v, dim=1).mean()
        
        loss_vtc = (loss_v2t + loss_t2v) / 2.0
        return loss_vtc
    
    def fg_vtm_loss(
        self,
        visual_feats, # query
        text_feats,
        vt_nums,
        temp=1.0,
        sim_mat=None,
    ):
        if sim_mat is None:
            raise NotImplementedError
        else:
            sim_v2t = sim_mat / temp
            sim_t2v = sim_mat.T / temp
        
        vt_startends = np.zeros(len(vt_nums) + 1, dtype=int)
        vt_startends[1:] = np.cumsum(vt_nums)
        with torch.no_grad():
            label_v2t = torch.zeros_like(sim_v2t).fill_diagonal_(1)
            label_t2v = label_v2t
            
            mask_v2t = torch.zeros_like(sim_v2t)
            for bi in range(len(vt_nums)):
                mask_v2t[vt_startends[bi]:vt_startends[bi + 1], vt_startends[bi]:vt_startends[bi + 1]] = 1
            mask_t2v = mask_v2t.T
            sim_v2t.masked_fill_(mask_v2t == 0, -1e9)
            sim_t2v.masked_fill_(mask_t2v == 0, -1e9)
            
            weights_v2t = F.softmax(sim_v2t, dim=1) + 1e-4
            weights_v2t.masked_fill_(mask_v2t == 0, 0).masked_fill_(label_v2t == 1, 0)
            weights_v2t = torch.nan_to_num_(
                weights_v2t, nan=1e-2, posinf=1e-2, neginf=1e-2
            )
            weights_t2v = F.softmax(sim_t2v, dim=1) + 1e-4
            weights_t2v.masked_fill_(mask_t2v == 0, 0).masked_fill_(label_t2v == 1, 0)
            weights_t2v = torch.nan_to_num_(
                weights_t2v, nan=1e-2, posinf=1e-2, neginf=1e-2
            )
        
        # select a negative video feature for each text
        visual_neg_indices = torch.multinomial(weights_t2v, 1).squeeze()
        # select a negative text for each video feature
        text_neg_indices = torch.multinomial(weights_v2t, 1).squeeze()
        
        visual_feats_neg = visual_feats[visual_neg_indices] # [N, c_q]
        text_feats_neg = text_feats[text_neg_indices] # [N, c_t]
        
        # concat video pos and neg
        visual_feats_pos_neg = torch.cat(
            [visual_feats, visual_feats_neg, visual_feats], dim=0
        ) # [3N, c_q]  
        # concat text pos and neg
        text_feats_pos_neg = torch.cat(
            [text_feats, text_feats, text_feats_neg], dim=0
        ) # [3N, c_t]
        vtm_feats = torch.cat([visual_feats_pos_neg, text_feats_pos_neg], dim=1) # [3N, c_q + c_t]
        
        logits = self.vtm_head(vtm_feats)
        bs = logits.size(0) // 3
        labels = logits.new_ones(logits.size(0), dtype=torch.long)
        labels[bs:] = 0
        # print("logits")
        # print(logits.size(), logits)
        # print("labels")
        # print(labels.size(), labels)
        loss_vtm = F.cross_entropy(logits, labels)
        
        return loss_vtm
    
    def vid_tc_loss(
        self,
        video_feats, # [B, n_g, T, C]
        vt_nums,
        temp=1.0
    ):
        video_feats = F.normalize(video_feats, dim=-1)
        video_feats_c = video_feats.mean(dim=2) # feature center for each query group: [B, n_g, C]
        sim_vf_vfc = torch.einsum('bntc,bmc->bntm', video_feats, video_feats_c).permute(0, 2, 1, 3) # [B, T, n_g, n_g]
        sim_vfc_vf = sim_vf_vfc.transpose(2, 3)
        sim_vf_vfc = sim_vf_vfc / temp
        sim_vfc_vf = sim_vfc_vf / temp
        num_g = sim_vf_vfc.size(-1)
        label_vf_vfc = torch.zeros_like(sim_vf_vfc)
        label_vf_vfc[:, :, :, :] = torch.eye(num_g)
        label_vfc_vf = label_vf_vfc.transpose(2, 3)
        
        loss_vf_vfc = -torch.sum(F.log_softmax(sim_vf_vfc, dim=3) * label_vf_vfc, dim=2).mean()
        loss_vfc_vf = -torch.sum(F.log_softmax(sim_vfc_vf, dim=3) * label_vfc_vf, dim=2).mean()
        loss_vid_tc = (loss_vf_vfc + loss_vfc_vf) / 2.0
        
        return loss_vid_tc
 
    def generate_caption(
        self,
        input_ids,
        attention_mask,
        image_idx = None,
        video_idx = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        num_beams=1,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.9,
        top_k=None,
        temperature=1.0,
        length_penalty=1,
        repetition_penalty=1.0,
    ):
        ### llm input embedding
        _, llm_text_embeds = self.pad_text_embeds(
            input_ids=input_ids, image=image, video=video, 
            image_idx=image_idx, video_idx=video_idx
        )
        outputs = self.lm.generate(
            inputs_embeds=llm_text_embeds,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            min_length=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )

        return outputs

class MultiModalLLM_fg_PT_test(BaseMLLM):
    
    def __init__(
        self,
        config
    ):
        super().__init__(config=config)
        
        ### configs for fine-grained training
        self.num_qformer_token = self.model_config.bridge.num_query_token + self.extra_num_query_token
        self.num_global_token = self.model_config.bridge.num_global_token
        self.num_local_token = self.num_qformer_token - self.num_global_token
        self.num_token_group = self.model_config.bridge.num_token_group
        self.token_per_group = self.num_local_token // self.num_token_group
        self.fix_text_feat = self.model_config.bridge.get('fix_text_feat', True)
        self.qformer_text_input = self.model_config.bridge.get('qformer_text_input', False)
        
        ### new modules for fine-grained training
        self.temp = nn.parameter.Parameter(torch.ones([]) * self.model_config.bridge.vtc_temp)
        self.vidtc_temp = nn.parameter.Parameter(torch.ones([]) * self.model_config.bridge.vidtc_temp)
        self.v_proj = nn.Linear(self.qformer.config.hidden_size, self.model_config.bridge.embed_dim)
        self.t_proj = nn.Linear(self.qformer.config.hidden_size, self.model_config.bridge.embed_dim)
        self.vtm_w = self.model_config.loss.get('vtm_w', 0.0)
        self.vtc_w = self.model_config.loss.get('vtc_w', 0.0)
        self.vidtc_w = self.model_config.loss.get('vidtc_w', 0.0)
        if self.vtm_w > 0:
            vtm_layers = [
                nn.Linear(2 * self.qformer.config.hidden_size, self.qformer.config.hidden_size),
                nn.Linear(self.qformer.config.hidden_size, 2)
            ]
            self.vtm_head = nn.Sequential(*vtm_layers)
            for i in range(len(self.vtm_head)):
                self.vtm_head[i].apply(self._init_weights)
        self.llm_w = self.model_config.loss.get('llm_w', 1.0)
        
        self.v_proj.apply(self._init_weights)
        self.t_proj.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def build_bridge(self):
        # ViT to LM: 1792 -> 6656 NOTE 768 is qformer dim
        self.project_up = nn.Linear(768, self.lm.config.hidden_size) # whether bias is needed?
        # # LM to ViT: 6656 -> 1792
        # self.project_down = nn.Linear(self.lm.config.hidden_size, 768)
        
        if 'qformer' in self.model_config.bridge.name.lower():
            from transformers import BertTokenizer
            self.qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="left", local_files_only=True)
            self.qformer_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            self.qformer_tokenizer.padding_side = "left"
            if self.model_config.bridge.name == 'qformer':
                self.qformer, self.query_tokens = build_qformer(
                        self.model_config.bridge.num_query_token, self.model_config.vision_encoder.encoder_embed_dim,
                        qformer_hidden_dropout_prob=self.model_config.bridge.qformer_hidden_dropout_prob,
                        qformer_attention_probs_dropout_prob=self.model_config.bridge.qformer_attention_probs_dropout_prob,
                        qformer_drop_path_rate=self.model_config.bridge.qformer_drop_path_rate,
                )
            elif self.model_config.bridge.name == 'causal_qformer':
                self.qformer, self.query_tokens = build_causal_qformer(
                        self.model_config.bridge.num_query_token, self.model_config.vision_encoder.encoder_embed_dim,
                        qformer_hidden_dropout_prob=self.model_config.bridge.qformer_hidden_dropout_prob,
                        qformer_attention_probs_dropout_prob=self.model_config.bridge.qformer_attention_probs_dropout_prob
                )
            print('len(self.qformer_tokenizer): ', len(self.qformer_tokenizer))
            self.qformer.bert.resize_token_embeddings(len(self.qformer_tokenizer))
            self.qformer.resize_token_embeddings(len(self.qformer_tokenizer))
            self.qformer.cls = None
            self.extra_num_query_token = self.model_config.bridge.extra_num_query_token
            if self.model_config.bridge.extra_num_query_token > 0:
                logger.info(f"Add extra {self.model_config.bridge.extra_num_query_token} tokens in QFormer")
                self.extra_query_tokens = nn.Parameter(
                    torch.zeros(1, self.model_config.bridge.extra_num_query_token, self.query_tokens.shape[-1])
                )
            
            self.freeze_bridge = self.model_config.get("freeze_bridge", False)
            if self.freeze_bridge:
                logger.info("freeze bridge")
                freeze_module(self.qformer)
                self.query_tokens.requires_grad = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        instruction = None,
        fg_text = None,
        video_idx = None,
        image_idx = None,
        **kwargs,
    ):  
        ### pre-process fine-grained text
        B = input_ids.size(0)
        fg_text_list = [x.split('|') for x in fg_text]
                
        assert len(fg_text_list) == B
        assert all([len(x) >= 2 for x in fg_text_list])
        
        ### llm input embedding
        group_vq_embeds, llm_text_embeds, video_frame_att_feats = self.pad_text_embeds(
            input_ids=input_ids, image=image, video=video, 
            image_idx=image_idx, video_idx=video_idx, instruction=instruction
        )
        
        # LLM forwards
        outputs = self.lm(
            inputs_embeds=llm_text_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        
        ### fine-grained visual-text alignment loss
        loss_dict = self.fg_loss(
            group_vq_embeds, fg_text_list, video_frame_att_feats
        )
        # loss aggregation
        loss_dict['llm'] = outputs.loss * self.llm_w
        total_loss = 0.0
        for k, v in loss_dict.items():
            total_loss += v
        outputs['loss'] = total_loss
        outputs['loss_dict'] = loss_dict
        
        return outputs

    def fg_loss(
        self,
        group_vq_embeds,
        fg_text_list,
        vf_att_feats,
    ):
        B = len(fg_text_list)
        
        ### filter out text that exceed query number
        for i in range(len(fg_text_list)):
            if len(fg_text_list[i]) > self.num_token_group:
                fg_text_list[i] = fg_text_list[i][:self.num_token_group]
        
        ### q-former fine-grained text encoding
        fg_text_nums = [len(tl) for tl in fg_text_list] # store the number of text segments in each instance
        fg_text = list(itertools.chain(*fg_text_list)) # flatten all text segments into a single list, for more efficient computing
        fg_text_tokenize, fg_text_embeds = self.encode_fg_text(fg_text, device=group_vq_embeds.device) # [BM, c_t]
        if self.fix_text_feat: # Not sure, should the text features back propagate to the q-former?
            fg_text_embeds = fg_text_embeds.detach() 
        
        ### bipartite matching of vision & fine-grained text
        group_vq_mean_embeds = group_vq_embeds.mean(dim=1) # [Bxn_g, c_q]
        vq_nums = [self.num_token_group for _ in range(B)]
        vq_proj = F.normalize(self.v_proj(group_vq_mean_embeds), dim=-1)
        tq_proj = F.normalize(self.t_proj(fg_text_embeds), dim=-1)
        visual_indices, text_indices, sim_mat = self.batch_bipartite_matching(
            vq_proj, tq_proj, vq_nums, fg_text_nums
        ) # [[1,3],[6,7],[9,11,15]]; [[0,1],[2,3],[4,5,6]]
        # print('visual indices', visual_indices)
        # print('text indices', text_indices)
        
        ### vision & fine-grained text alignment loss
        loss_dict = dict()
        visual_indices_all = list(itertools.chain(*visual_indices))
        text_indices_all = list(itertools.chain(*text_indices))
        sim_mat_match = sim_mat[visual_indices_all, :][:, text_indices_all]
        vt_nums = [len(l) for l in visual_indices]
        # visual-text contrastive learning
        if self.vtc_w > 0:
            vq_proj_match = vq_proj[visual_indices_all] # [N, c_q]
            tq_proj_match = tq_proj[text_indices_all] # [N, c_q]
            vtc_loss = self.vtc_loss(
                vq_proj_match, tq_proj_match, vt_nums, temp=self.temp, sim_mat=sim_mat_match
            )
            loss_dict['vtc'] = vtc_loss * self.vtc_w
        
        # visual-text matching
        if self.vtm_w > 0:
            # vq_embeds_match = group_vq_mean_embeds[visual_indices_all]
            # tq_embeds_match = fg_text_embeds[text_indices_all]
            vtm_loss = self.fg_vtm_loss(
                vq_proj_match, tq_proj_match, vt_nums, temp=self.temp, sim_mat=sim_mat_match
            )
            loss_dict['vtm'] = vtm_loss * self.vtm_w
        
        ### vision token temporal consistency loss
        if self.vidtc_w > 0:
            vidtc_loss = self.vid_tc_loss(
                vf_att_feats, vt_nums, temp=self.vidtc_temp,
            )
            loss_dict['vidtc'] = vidtc_loss * self.vidtc_w
        
        return loss_dict
    
    def pad_text_embeds(
        self,
        input_ids: torch.LongTensor = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        image_idx = None,
        video_idx = None,
        instruction = None,
    ):
        ### q-former vision encoding
        if image is not None:
            B, T, C, H, W = image.shape
            image = image.permute(0, 2, 1, 3, 4)
            vision_query_embeds, vit_embeds, visual_attention_maps = self.encode_fg_vision(image, instruction=instruction)
            B, L, C = vit_embeds.shape
        elif video is not None:
            if len(video.shape) == 5:
                B, T, C, H, W = video.shape
                N = 1
            else:
                B, N, T, C, H, W = video.shape

            video = video.reshape(-1, T, C, H, W).permute(0, 2, 1, 3, 4) # [B, C, T, H, W]
            B = video.size(0)
            vision_query_embeds, vit_embeds, visual_attention_maps = self.encode_fg_vision(
                video, instruction=instruction
            ) # [B, n_q, c_q], [B, L, C], [B, n_q, L]
            B, L, C = vit_embeds.shape
        else:
            logger.error(f"don't get visual input, input_ids: {input_ids}")
        
        ### video frame-wise attention-weighted features
        vit_embeds = vit_embeds.permute(0, 2, 1)[:, :, 1:] # [B, C, HW]
        video_att_maps = visual_attention_maps[:, :, 1:].view(B, self.num_qformer_token, T, -1) # [B, n_q, T, HW]
        video_att_feats = torch.einsum('bctn,bqtn->bqctn', vit_embeds.view(B, C, T, -1), video_att_maps) # [B, n_q, C, T, HW]
        video_frame_att_feats = video_att_feats.mean(dim=-1).permute(0, 1, 3, 2) # [B, n_q, T, C]
        
        # only align local cross-attention feature
        local_vf_att_feats = video_frame_att_feats[:, :self.num_local_token, :, :]
        group_vf_att_feats = \
            local_vf_att_feats.reshape(B, self.num_token_group, self.token_per_group, T, -1).mean(dim=2) # [B, n_g, T, C]
        
        ### local global feature
        local_vq_embeds = vision_query_embeds[:, :self.num_local_token, :]
        group_vq_embeds = \
            local_vq_embeds.reshape(B, self.num_token_group, self.token_per_group, -1).reshape(B * self.num_token_group, self.token_per_group, -1)
        
        # pad text embedding for LLM input
        prompt_visual_embeds = self.project_up(vision_query_embeds)
        prompt_visual_embeds = prompt_visual_embeds.view(-1, prompt_visual_embeds.shape[-1])
        text_embeds = self.lm.get_input_embeddings()(input_ids.long()).detach()
        if image is not None:
            visual_idx = image_idx
        elif video is not None:
            visual_idx = video_idx
        text_embeds[visual_idx == 1] = text_embeds[visual_idx == 1] * 0 + prompt_visual_embeds.to(text_embeds.device).to(text_embeds.dtype)
        
        return group_vq_embeds, text_embeds, group_vf_att_feats

    def encode_fg_vision(
        self,
        image,
        instruction
    ):
        device = image.device
        B = image.shape[0]
        T = image.shape[2]
        use_image = True if T == 1 else False
        image_embeds = self.vision_encoder(image, use_image=use_image)
        C = image_embeds.shape[-1]
        image_embeds = image_embeds.reshape(B, -1, C)
        image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, T*L, C]
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        if self.extra_num_query_token > 0:
            query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
        query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
        if self.qformer_text_input and instruction is not None:
            # print(instruction)
            text_Qformer = self.qformer_tokenizer(
                instruction,
                padding='longest',
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(image_embeds.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
            # print(query_atts.shape, text_Qformer.attention_mask.shape)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
            query_output = self.qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                output_attentions=True,
            )
        else:
            query_output = self.qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                output_attentions=True,
            )
        
        cross_attentions = query_output.cross_attentions
        all_cross_attentions = []
        for i in range(len(cross_attentions)):
            ca = cross_attentions[i]
            if isinstance(ca, torch.Tensor):
                layer_ca = ca.mean(dim=1, keepdim=True) # [B, 1, num_query, num_patches]
                all_cross_attentions.append(layer_ca)

        avg_cross_atts = torch.cat(all_cross_attentions, dim=1).sum(dim=1) # [B, num_query, num_patches]
        
        return query_output.last_hidden_state[:, :query_tokens.size(1), :], image_embeds, avg_cross_atts

    def encode_fg_text(
        self,
        fg_text,
        device
    ):
        """encode text.
        Args:
            - fg_text: List of fine-grained text segments in each video.
        Returns: tuple.
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [BL,C].

        """
        BL = len(fg_text)
        fg_text_Qformer = self.qformer_tokenizer(
            fg_text,
            padding='longest',
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        fg_text_output = self.qformer.bert(
            fg_text_Qformer.input_ids,
            attention_mask=fg_text_Qformer.attention_mask,
            return_dict=True,
        )
        fg_text_embeds = fg_text_output.last_hidden_state
        pooled_fg_text_embeds = fg_text_embeds[:, 0]
        return fg_text_Qformer, pooled_fg_text_embeds
    
    def batch_bipartite_matching(
        self,
        visual_feats,
        text_feats,
        visual_nums,
        text_nums,
    ):
        bs = len(visual_nums)
        sim_mat = torch.matmul(visual_feats, text_feats.T)
        cost_mat = -sim_mat.detach().to(torch.float).cpu().numpy()
        visual_accu_nums = np.zeros(bs + 1, dtype=int)
        text_accu_nums = np.zeros(bs + 1, dtype=int)
        visual_accu_nums[1:] = np.cumsum(visual_nums, dtype=int)
        text_accu_nums[1:] = np.cumsum(text_nums, dtype=int)
        visual_indices = []
        text_indices = []
        for bi in range(len(visual_nums)):
            v_s, v_e = visual_accu_nums[bi], visual_accu_nums[bi + 1]
            t_s, t_e = text_accu_nums[bi], text_accu_nums[bi + 1]
            sample_cost_mat = cost_mat[v_s:v_e, t_s:t_e]
            v_inds, t_inds = linear_sum_assignment(sample_cost_mat)
            v_inds = v_inds + v_s
            t_inds = t_inds + t_s
            
            visual_indices.append(v_inds)
            text_indices.append(t_inds)
        
        return visual_indices, text_indices, sim_mat
    
    def vtc_loss(
        self,
        visual_feats,
        text_feats,
        vt_nums,
        temp=1.0,
        sim_mat=None,
    ):
        if sim_mat is None:
            raise NotImplementedError
        else:
            sim_v2t = sim_mat / temp
            sim_t2v = sim_mat.T / temp
        
        vt_startends = np.zeros(len(vt_nums) + 1, dtype=int)
        vt_startends[1:] = np.cumsum(vt_nums)
        with torch.no_grad():
            label_v2t = torch.zeros_like(sim_v2t).fill_diagonal_(1)
            label_t2v = label_v2t

            mask_v2t = torch.zeros_like(sim_v2t)
            for bi in range(len(vt_nums)):
                mask_v2t[vt_startends[bi]:vt_startends[bi + 1], vt_startends[bi]:vt_startends[bi + 1]] = 1
            mask_t2v = mask_v2t.T
        
        sim_v2t = sim_v2t.masked_fill(mask_v2t == 0, -1e9)
        sim_t2v = sim_t2v.masked_fill(mask_t2v == 0, -1e9)
        # print("sim_v2t:")
        # print(sim_v2t[:20, :20])
        # print("sim_v2t logsoftmax")
        # print((F.log_softmax(sim_v2t, dim=1) * label_v2t)[:20, :20])
        # print("label_v2t:")
        # print(label_v2t[:20, :20])
        loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1) * label_v2t, dim=1).mean()
        loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1) * label_t2v, dim=1).mean()
        
        loss_vtc = (loss_v2t + loss_t2v) / 2.0
        return loss_vtc
    
    def fg_vtm_loss(
        self,
        visual_feats, # query
        text_feats,
        vt_nums,
        temp=1.0,
        sim_mat=None,
    ):
        if sim_mat is None:
            raise NotImplementedError
        else:
            sim_v2t = sim_mat / temp
            sim_t2v = sim_mat.T / temp
        
        vt_startends = np.zeros(len(vt_nums) + 1, dtype=int)
        vt_startends[1:] = np.cumsum(vt_nums)
        with torch.no_grad():
            label_v2t = torch.zeros_like(sim_v2t).fill_diagonal_(1)
            label_t2v = label_v2t
            
            mask_v2t = torch.zeros_like(sim_v2t)
            for bi in range(len(vt_nums)):
                mask_v2t[vt_startends[bi]:vt_startends[bi + 1], vt_startends[bi]:vt_startends[bi + 1]] = 1
            mask_t2v = mask_v2t.T
            sim_v2t.masked_fill_(mask_v2t == 0, -1e9)
            sim_t2v.masked_fill_(mask_t2v == 0, -1e9)
            
            weights_v2t = F.softmax(sim_v2t, dim=1) + 1e-4
            weights_v2t.masked_fill_(mask_v2t == 0, 0).masked_fill_(label_v2t == 1, 0)
            weights_v2t = torch.nan_to_num_(
                weights_v2t, nan=1e-2, posinf=1e-2, neginf=1e-2
            )
            weights_t2v = F.softmax(sim_t2v, dim=1) + 1e-4
            weights_t2v.masked_fill_(mask_t2v == 0, 0).masked_fill_(label_t2v == 1, 0)
            weights_t2v = torch.nan_to_num_(
                weights_t2v, nan=1e-2, posinf=1e-2, neginf=1e-2
            )
        
        # select a negative video feature for each text
        visual_neg_indices = torch.multinomial(weights_t2v, 1).squeeze()
        # select a negative text for each video feature
        text_neg_indices = torch.multinomial(weights_v2t, 1).squeeze()
        
        visual_feats_neg = visual_feats[visual_neg_indices] # [N, c_q]
        text_feats_neg = text_feats[text_neg_indices] # [N, c_t]
        
        # concat video pos and neg
        visual_feats_pos_neg = torch.cat(
            [visual_feats, visual_feats_neg, visual_feats], dim=0
        ) # [3N, c_q]  
        # concat text pos and neg
        text_feats_pos_neg = torch.cat(
            [text_feats, text_feats, text_feats_neg], dim=0
        ) # [3N, c_t]
        vtm_feats = torch.cat([visual_feats_pos_neg, text_feats_pos_neg], dim=1) # [3N, c_q + c_t]
        
        logits = self.vtm_head(vtm_feats)
        bs = logits.size(0) // 3
        labels = logits.new_ones(logits.size(0), dtype=torch.long)
        labels[bs:] = 0
        # print("logits")
        # print(logits.size(), logits)
        # print("labels")
        # print(labels.size(), labels)
        loss_vtm = F.cross_entropy(logits, labels)
        
        return loss_vtm
    
    def vid_tc_loss(
        self,
        video_feats, # [B, n_g, T, C]
        vt_nums,
        temp=1.0
    ):
        video_feats = F.normalize(video_feats, dim=-1)
        video_feats_c = video_feats.mean(dim=2) # feature center for each query group: [B, n_g, C]
        sim_vf_vfc = torch.einsum('bntc,bmc->bntm', video_feats, video_feats_c).permute(0, 2, 1, 3) # [B, T, n_g, n_g]
        sim_vfc_vf = sim_vf_vfc.transpose(2, 3)
        sim_vf_vfc = sim_vf_vfc / temp
        sim_vfc_vf = sim_vfc_vf / temp
        num_g = sim_vf_vfc.size(-1)
        label_vf_vfc = torch.zeros_like(sim_vf_vfc)
        label_vf_vfc[:, :, :, :] = torch.eye(num_g)
        label_vfc_vf = label_vf_vfc.transpose(2, 3)
        
        loss_vf_vfc = -torch.sum(F.log_softmax(sim_vf_vfc, dim=3) * label_vf_vfc, dim=2).mean()
        loss_vfc_vf = -torch.sum(F.log_softmax(sim_vfc_vf, dim=3) * label_vfc_vf, dim=2).mean()
        loss_vid_tc = (loss_vf_vfc + loss_vfc_vf) / 2.0
        
        return loss_vid_tc
 
    def generate_caption(
        self,
        input_ids,
        attention_mask,
        image_idx = None,
        video_idx = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        num_beams=1,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.9,
        top_k=None,
        temperature=1.0,
        length_penalty=1,
        repetition_penalty=1.0,
    ):
        ### llm input embedding
        _, llm_text_embeds = self.pad_text_embeds(
            input_ids=input_ids, image=image, video=video, 
            image_idx=image_idx, video_idx=video_idx
        )
        outputs = self.lm.generate(
            inputs_embeds=llm_text_embeds,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            min_length=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )

        return outputs