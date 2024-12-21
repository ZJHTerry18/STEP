import transformers
import hydra
import pyrootutils
import logging
import torch
import torch.distributed as dist
import wandb
import io
import os
from petrel_client.client import Client
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
from src.model import *
from src.train.trainer import CustomTrainer
from typing import Optional, List
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from src.share_utils.py_config_utils import Config, eval_dict_leaf
from src.train.cap_utils import compute_metrics_cider
from src.train.qa_utils import compute_metrics_vqa
from src.share_utils.distributed import is_main_process
from src.share_utils.logger import setup_logger
from src.train.utils import init_distributed_mode, setup_output_dir
from src.share_utils.serialize import local_broadcast_process_authkey
import deepspeed
os.environ["WANDB_DISABLED"] = "true"
logger = logging.getLogger(__name__)


@dataclass
class ConfigPathArguments:
    model: Optional[str] = field(default=None, metadata={"help": "config path of model used to initialize LM model"})
    tokenizer: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})
    train_data: Optional[str] = field(default=None, metadata={"help": "config path of train dataset"})
    eval_data: Optional[str] = field(default=None, metadata={"help": "config path of eval dataset"})
    pretrained_path: Optional[str] = field(default=None, metadata={"help": "config path of pretrained model"})
    is_eval: Optional[bool] = field(default=None, metadata={"help": "train or eval"})
    mode: str = field(
        default="none", metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    llm_path: str = field(
        default="/mnt/petrelfs/share/videointern/MLLM/Mistral-7B-Instruct-v0.3/")
    model_path: str = field(
        default='/mnt/petrelfs/share_data/wangchenting/models/internvideo2_videochat2/internvideo2_1b_mistral7b_stage3'
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(
        default=None, metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    overwrite_output_dir: bool = field(default=False, metadata={"help": "Overwrite the content of the output directory"})
    optim: str = field(default="adamw_hf")
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    min_lr_ratio: float = field(
        default=0.1, metadata={"help": "The min lr ratio reqpect to the learning rate, only used to cosine lr scheduler"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1, metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."})

    lr_scheduler_type: str = field(default='cosine', metadata={"help": "The scheduler type to use."})
    report_to: Optional[str] = field(default=None,
                                           metadata={"help": "The list of integrations to report the results and logs to."})
    save_steps: int = field(default=1000, metadata={"help": "The interval between saving the model checkpoint."})
    bf16: bool = field(default=False,
                       metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"})
    fp16: bool = field(default=False,
                       metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"})
    dataloader_num_workers: int = field(default=8, metadata={"help": "The number of workers to use for data loading."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/CPU for training."})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/CPU for evaluation."})
    run_name: str = field(default=None, metadata={"help": "The name of the run."})
    torch_compile: bool = field(default=False, metadata={"help": "Whether to use torch.jit.trace to compile the model."})
    custom_lr_wd_dict: Optional[dict] = field(default=None, metadata={"help": "custom lr and wd for special params."})
    custom_per_device_train_bs_dict: Optional[dict] = field(default=None, metadata={"help": "custom bs for special media."})



def train_or_eval():
    parser = transformers.HfArgumentParser((ConfigPathArguments, TrainingArguments))
    
    cfg_path, training_args = parser.parse_args_into_dataclasses()

    if is_main_process():
        setup_output_dir(training_args.output_dir, excludes=["code"])
        setup_logger(output=training_args.output_dir, color=True, name="mllm")
    
    dist.barrier()
    local_broadcast_process_authkey()

    tokenizer = hydra.utils.instantiate(OmegaConf.load(cfg_path.tokenizer))
    tokenizer.pad_token = tokenizer.unk_token
    model_cfg = eval_dict_leaf(Config.from_file(cfg_path.model))
 
    from src.model.base_model import LLMConfig
    config = LLMConfig.from_pretrained(cfg_path.llm_path, trust_remote_code=True)
    
    config.model_config = model_cfg.model
    # config.model_tokenizer = tokenizer # if you need to add special tokens, un-comment this and remember to delete it in the base_model init func after loading it
    
    model = eval(model_cfg.model.get('model_cls')).from_pretrained(cfg_path.model_path, config=config)
    logger.info(f'Init tokenizer: {tokenizer}')
    
    if cfg_path.mode == 'pt':
        train_data = hydra.utils.instantiate(OmegaConf.load('configs/dataset/dataset_pt_train.yaml'), tokenizer=tokenizer, config_path=cfg_path.train_data)
    elif cfg_path.mode == 'it_hd':
        train_data = hydra.utils.instantiate(OmegaConf.load('configs/dataset/dataset_it_hd_train.yaml'), tokenizer=tokenizer, config_path=cfg_path.train_data)
    elif cfg_path.mode == 'it' :
        train_data = hydra.utils.instantiate(OmegaConf.load('configs/dataset/dataset_it_train.yaml'), tokenizer=tokenizer, config_path=cfg_path.train_data)
    elif cfg_path.mode == 'pt_fg':
        train_data = hydra.utils.instantiate(OmegaConf.load('configs/dataset/dataset_pt_fg_train.yaml'), tokenizer=tokenizer, config_path=cfg_path.train_data)
    else:
        print('Please Give A Correct Training Mode')
        raise NotImplementedError
    
    train_data_cfg = eval_dict_leaf(Config.from_file(cfg_path.train_data))
    training_args.custom_per_device_train_bs_dict = train_data_cfg.get('custom_per_device_train_bs_dict', None)
    logger.info(f'Init train data: {train_data} config_path: {cfg_path.train_data}')
    
    training_args.custom_lr_wd_dict = model_cfg.get('custom_lr_wd_dict', None)

    if cfg_path.pretrained_path is not None:
        raise NotImplementedError # To support ZerO3, we do not support loading from pretrained path here, please use from_pretrained and test_model function
        state_dict = {}
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Load pretrained model from {cfg_path.pretrained_path}, msg: {msg}")
    
    if cfg_path.eval_data is not None:
        eval_data_cfg = OmegaConf.load('configs/dataset/dataset_cap_eval.yaml')
        # eval_data_cfg = OmegaConf.load('configs/dataset/dataset_vqa_eval.yaml')
        eval_data = hydra.utils.instantiate(eval_data_cfg, tokenizer=tokenizer, config_path=cfg_path.eval_data)
        if type(eval_data) is list:
            assert len(eval_data) == 1, len(eval_data) # NOTE only support one eval_dataset now!!!
            eval_data = eval_data[0]
    else:
        eval_data = None

    logger.info("training_args:")
    logger.info(training_args)

    if cfg_path.is_eval:
        trainer = CustomTrainer(
            self_defined_logging_dir=training_args.output_dir,
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_cider
        )
        trainer.evaluate()
    else:
        # if is_main_process():
        #     wandb.init(entity="terryzhao", name=os.path.basename(training_args.output_dir), reinit=True)
        model.lm.config.use_cache = False
        model.train()
        trainer = CustomTrainer(
            self_defined_logging_dir=training_args.output_dir,
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=tokenizer,
            compute_metrics=None
        )
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model(output_dir=os.path.join(training_args.output_dir, "checkpoint-last"))


if __name__ == "__main__":
    init_distributed_mode()
    train_or_eval()
