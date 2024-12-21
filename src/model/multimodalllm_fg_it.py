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


class MultiModalLLM_fg_IT(BaseMLLM):
    
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
        self.qformer_text_input = self.model_config.bridge.get('qformer_text_input', False)
        
        ### initialize query tokens (must be after pretrained weights loaded)
        self.global_query_tokens = nn.Parameter(
            torch.zeros(1, self.num_global_token, self.query_tokens.shape[-1])
        )
        
        ### new modules for fine-grained training  
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
        # fg_text_list = None,
        video_idx = None,
        image_idx = None,
        **kwargs,
    ):  
        ### llm input embedding
        _, llm_text_embeds, _ = self.pad_text_embeds(
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
        
        return outputs

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
            vision_query_embeds, _, visual_attention_maps = self.encode_fg_vision(image, instruction=instruction)
        elif video is not None:
            if len(video.shape) == 5:
                B, T, C, H, W = video.shape
                N = 1
            else:
                B, N, T, C, H, W = video.shape

            video = video.reshape(-1, T, C, H, W).permute(0, 2, 1, 3, 4) # [B, C, T, H, W]
            B = video.size(0)
            vision_query_embeds, _, visual_attention_maps = self.encode_fg_vision(
                video, instruction=instruction
            ) # [B, n_q, c_q], [B, num_head, n_q, T*H*W]
        else:
            logger.error(f"don't get visual input, input_ids: {input_ids}")
        
        group_vq_embeds = vision_query_embeds.reshape(B * self.num_token_group, self.token_per_group, -1)
        
        ### global vision token aggregation
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
        
        return group_vq_embeds, text_embeds, visual_attention_maps

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
        llm_text_embeds = self.pad_text_embeds(
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

class MultiModalLLM_fg_IT_2(BaseMLLM):
    
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
        self.qformer_text_input = self.model_config.bridge.get('qformer_text_input', False)
    
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
        # fg_text_list = None,
        video_idx = None,
        image_idx = None,
        **kwargs,
    ):  
        ### llm input embedding
        _, llm_text_embeds, _ = self.pad_text_embeds(
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
        
        return outputs

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
            vision_query_embeds, _, visual_attention_maps = self.encode_fg_vision(image, instruction=instruction)
        elif video is not None:
            if len(video.shape) == 5:
                B, T, C, H, W = video.shape
                N = 1
            else:
                B, N, T, C, H, W = video.shape

            video = video.reshape(-1, T, C, H, W).permute(0, 2, 1, 3, 4) # [B, C, T, H, W]
            B = video.size(0)
            vision_query_embeds, _, visual_attention_maps = self.encode_fg_vision(
                video, instruction=instruction
            ) # [B, n_q, c_q], [B, num_head, n_q, T*H*W]
        else:
            logger.error(f"don't get visual input, input_ids: {input_ids}")
        
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
        
        return group_vq_embeds, text_embeds, visual_attention_maps

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
        llm_text_embeds = self.pad_text_embeds(
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