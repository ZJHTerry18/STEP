import io
import logging
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import MSELoss
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from typing import List, Optional, Tuple, Union
# from transformers import LlamaForCausalLM
from .llm.llama_xformer import LlamaForCausalLM

from petrel_client.client import Client
from torch.cuda.amp import autocast as autocast

from ..share_utils.constants import *
from ..share_utils.func_utils import freeze_module

from .vision_encoder.vit_scale import teacher_internvideo2_stage2_1B
from .vision_encoder import pretrain_internvideo2_giant_patch14_224_clean, build_vit
from .vision_encoder.pos_embed import interpolate_pos_embed_internvideo2_new
from .bridge import build_qformer, build_causal_qformer

logger = logging.getLogger(__name__)

from transformers import LlamaTokenizer,AutoTokenizer,AutoModel,AutoModelForCausalLM,AutoProcessor
from transformers import AutoConfig, PreTrainedModel


class LLMConfig(AutoConfig):
    model_type = "20b"


class BaseMLLM(PreTrainedModel):
    config_class = LLMConfig
    def __init__(self, config):
        # m_config = LLMConfig.from_pretrained('/mnt/petrelfs/share_data/likunchang/model/llm/internlm2-chat-20b', trust_remote_code=True)
        # super().__init__(config)
        self.model_config = config.model_config
        # self.tokenizer = config.model_tokenizer
        config.model_config = None
        # config.model_tokenizer = None
        super().__init__(config)
        self.build_vision_encoder()
        self.build_llm()
        self.build_bridge()
        self.build_loss()
        self.load_pretrained_weights()
        # NOTE place it after freeze llm
        # logger.info(f'Length of tokenizer and resize embedding: {len(self.tokenizer)}')
        for n, p in self.named_parameters():
            if p.requires_grad:
                logger.info(f'{n} requires_grad')
        
    
    def build_vision_encoder(self):
        # load pretrained internvideo2-1b here, simplified as it receives no args
        # note that we haven't load the internvideo pretrained version
        if 'internvideo2' in self.model_config.vision_encoder.name.lower():
            encoder_name = self.model_config.vision_encoder.name
            logger.info(f"Build vision_encoder: {encoder_name}")
            if encoder_name == 'internvideo2-1B':
                self.vision_encoder = pretrain_internvideo2_giant_patch14_224_clean(self.model_config)
            else:
                raise ValueError(f"Not implemented: {encoder_name}")
        elif 'vit' in self.model_config.vision_encoder.name.lower():
            self.vision_encoder = build_vit(self.model_config)
        else:
            raise NotImplementedError(self.model_config.vision_encoder.name)

        if self.model_config.vision_encoder.vit_add_ln:
            self.vision_layernorm = nn.LayerNorm(self.model_config.vision_encoder.encoder_embed_dim, eps=1e-12)
        else:
            self.vision_layernorm = nn.Identity()

        self.freeze_vision_encoder = self.model_config.get("freeze_vision_encoder", False)

        if self.freeze_vision_encoder:
            logger.info("freeze vision encoder")
            freeze_module(self.vision_encoder)
            freeze_module(self.vision_layernorm)


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

    def build_llm(self):
        self.lm_name = self.model_config.llm.name
        if self.model_config.llm.name == "vicuna1.5_7b":
            self.lm = LlamaForCausalLM.from_pretrained(self.model_config.llm.pretrained_llm_path)
            self.lm.gradient_checkpointing = self.model_config.llm.get("use_llama_gradient_checkpointing", True)
        elif self.model_config.llm.name == 'mistral_7b':
            from transformers import AutoModelForCausalLM
            
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.model_config.llm.pretrained_llm_path,
                torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",
            )
        elif self.model_config.llm.name == 'internlm_20b':
            from transformers import AutoModelForCausalLM
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.model_config.llm.pretrained_llm_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self.lm.gradient_checkpointing = True
            self.lm._set_gradient_checkpointing()
        elif self.model_config.llm.name == 'internlm2_5_7b':
            from transformers import AutoModelForCausalLM
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.model_config.llm.pretrained_llm_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
                local_files_only=True,
            )
            self.lm.gradient_checkpointing = True
            self.lm._set_gradient_checkpointing()
        else:
            raise NotImplementedError(self.model_config.llm.name)

        self.freeze_llm = self.model_config.get("freeze_llm", True)
        logger.info(f'freeze_llm: {self.freeze_llm}')
        if self.freeze_llm and not self.model_config.llm.use_lora:
            logger.info("freeze llm")
            freeze_module(self.lm)
        
        if self.model_config.llm.use_lora:
            self.use_lora = True
            from peft import get_peft_model, LoraConfig, TaskType
            logger.info("Use lora")
            if self.model_config.llm.name == 'internlm_20b':
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                    r=self.model_config.llm.lora_r, lora_alpha=self.model_config.llm.lora_alpha, lora_dropout=self.model_config.llm.lora_dropout,
                    target_modules=['wqkv', 'wo', 'w1', 'w2', 'w3', 'output']
                )
            elif self.model_config.llm.name == 'internlm2_5_7b':
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                    r=self.model_config.llm.lora_r, lora_alpha=self.model_config.llm.lora_alpha, lora_dropout=self.model_config.llm.lora_dropout,
                    target_modules=['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
                )
            else:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                    r=self.model_config.llm.lora_r, lora_alpha=self.model_config.llm.lora_alpha, lora_dropout=self.model_config.llm.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj", "lm_head"]
                )
            self.lm.train()
            self.lm = get_peft_model(self.lm, peft_config)
            self.lm.enable_input_require_grads()
            self.lm.print_trainable_parameters()
        else:
            self.use_lora = False


    def build_loss(self):
        self.use_vision_regression_loss = self.model_config.loss.get("use_vision_regression_loss", False)
        if self.use_vision_regression_loss:
            self.image_loss_fct = MSELoss()
        
        
    def load_pretrained_weights(self):
        if self.model_config.pretrained_paths.get('pretrained_vit_qformer_path', None):
            if 'safetensor' in self.model_config.pretrained_paths.pretrained_vit_qformer_path:
                from safetensors import safe_open
                from safetensors.torch import save_file
                state_dict = {}
                with safe_open(self.model_config.pretrained_paths.pretrained_vit_qformer_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            else:
                with io.BytesIO(Client().get(self.model_config.pretrained_paths.pretrained_vit_qformer_path)) as buffer:
                    state_dict = torch.load(buffer, map_location="cpu")
                    
                    if "model" in state_dict.keys():
                        state_dict = state_dict["model"]
                    elif "module" in state_dict.keys(): 
                        state_dict = state_dict["module"] # for deepspeed
            new_state_dict = {}
            for k in state_dict.keys():
                new_state_dict[k.replace('.gamma', '')] = state_dict[k]            
            
            self.check_temp_emb(new_state_dict)
            msg = self.load_state_dict(new_state_dict, strict=False)
            print('Loading vit: ', msg)
            logger.info(f"Load ViT and QFormer from {self.model_config.pretrained_paths.pretrained_vit_qformer_path}: {msg}")

        if self.model_config.pretrained_paths.get('pretrained_videochat2', None):
            with io.BytesIO(Client().get(self.model_config.pretrained_paths.pretrained_videochat2)) as buffer:
                    state_dict = torch.load(buffer, map_location="cpu")

            new_state_dict = {}
            for k in state_dict.keys():
                if 'bert.embeddings' not in k:
                    new_state_dict[k] = state_dict[k]
            state_dict = new_state_dict
            # self.check_temp_emb(state_dict)
            msg = self.load_state_dict(state_dict, strict=False)
            print('Loading videochat2: ', msg)
            logger.info(f"Load pretrained model from {self.model_config.pretrained_paths.pretrained_videochat2}, msg: {msg}")


    def check_temp_emb(self, state_dict):
        # if "vision_temp_embed" in state_dict.keys():
        #     raise NotImplementedError
        old_num_frames = self.model_config.vision_encoder.get('origin_num_frames', None)
        new_num_frames = self.model_config.vision_encoder.num_frames
        if old_num_frames is not None and old_num_frames != new_num_frames:
            logger.info(f"interpolate_pos_embed_internvideo2 to {new_num_frames} (origin_num_frames={old_num_frames})!!!")
            a = len(state_dict)
            interpolate_pos_embed_internvideo2_new(state_dict, self.vision_encoder, orig_t_size=4)
            assert a == len(state_dict), state_dict.keys()


    @property
    def dtype(self):
        return self.lm.dtype


    @property
    def device(self):
        return self.lm.device