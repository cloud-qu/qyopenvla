"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
# from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from torchvision import transforms
import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.newdataloader import DataLoader
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

import clip
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from experiments.robot.robot_utils import (
    set_seed_everywhere,
)
import time

# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = False                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    use_wandb: bool = True                                          # Whether to log to Weights & Biases
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases

    # fmt: on
    algo_name: str = 'random' # random; mpts; diverse_mpts
    diversity_type: str = 'msdsum' # rs, msdmin
    sampler_multiplier: int = 1
    mpts_training_steps: int = 1
    mpts_training_lr: float = 0.00001
    seed: int = 0

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
class MPTS(object):
    def __init__(self, algo_name, tokenizer, device_ids, diversity_type, mpts_training_steps, mpts_training_lr):
        self.tokenizer = tokenizer
        self.algo_name = algo_name
        self.no_add_random = False
        self.batch_norm = False
        self.gamma_0 = 1
        self.gamma_1 = 3
        if 'diverse' in algo_name:
            self.gamma_2 = 5
        else:
            self.gamma_2 = 0
        self.warmup_steps = 0
        self.current_steps = 0
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device='cuda')
        self.normalized_mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).reshape(1, 3, 1, 1).to('cuda')
        self.normalized_std = torch.tensor((0.26862954, 0.26130258, 0.27577711)).reshape(1, 3, 1, 1).to('cuda')
        from wordllama import WordLlama
        risklearner_input_size = 512
        self.wl = WordLlama.load(dim=64)
        self.mpts_training_steps = mpts_training_steps


        from MPModel.risklearner import RiskLearner
        from MPModel.new_trainer_risklearner import RiskLearnerTrainer
        mp_device = torch.device('cuda')
        self.risklearner = RiskLearner(512*2, 1, 128, 128, 128).to(mp_device)
        # self.risklearner = DDP(self.risklearner, device_ids=device_ids, find_unused_parameters=True, gradient_as_bucket_view=True, static_graph=True)
        risklearner_optimizer = torch.optim.Adam(self.risklearner.parameters(), lr=mpts_training_lr)
        if 'diverse' in algo_name:
            posterior_sampling = True
            diversity_type = diversity_type
        else:
            posterior_sampling = False
            diversity_type = None
        self.risklearner_trainer = RiskLearnerTrainer(mp_device, self.risklearner, risklearner_optimizer,  posterior_sampling=posterior_sampling, 
                                        diversity_type=diversity_type)
        self.text_encodings = {}
        
    def identifier_decode(self, batch):
        raw_instruction = batch['input_ids']
        raw_imgs = batch['raw_imgs']
        reshaped_imgs = raw_imgs.permute(0, 3, 1, 2).to('cuda')#B, 3, 224, 224
        # pil_imgs = []
        # for i in range(len(reshaped_imgs)):
        #     # pil_img = transforms.ToPILImage()(reshaped_imgs[i])
        #     # pil_imgs.append(self.clip_preprocess(pil_img))
        #     pil_imgs.append(self.simple_clip_preprocess(reshaped_imgs[i]/255.0))
        # normalized_imgs = torch.stack(pil_imgs).to('cuda')
        normalized_imgs = (reshaped_imgs/255.0 - self.normalized_mean) / self.normalized_std
        # normalized_imgs = self.simple_clip_preprocess(reshaped_imgs/255.0)
        img_embeddings = self.clip_model.encode_image(normalized_imgs).to(batch['input_ids'].device)#B, 512
        # img_embeddings = batch['img_embeddings'].to(batch['input_ids'].device)
        # img_embeddings = torch.rand(raw_instruction.shape[0], 512).to(batch['input_ids'].device)

        lang_have_encodings = True
        lang = []
        for i in range(len(raw_instruction)):
            lang.append(self.tokenizer.decode(raw_instruction[i]).split('What action should the robot take to ')[1].split('Out:')[0])
            if lang[-1] not in self.text_encodings:
                lang_have_encodings = False
        if not lang_have_encodings:
            text_embeddings = self.clip_model.encode_text(clip.tokenize(lang).to('cuda'))
            for i in range(len(raw_instruction)):
                self.text_encodings[lang[i]] = text_embeddings[i]
        else:
            text_embeddings = torch.stack([self.text_encodings[lang[i]] for i in range(len(raw_instruction))])

        # # text_embeddings = torch.tensor(self.wl.embed(lang, norm=True)).to(batch['input_ids'].device)

        img_embeddings = img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = (text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)).to(batch['input_ids'].device)
        embeddings = torch.cat((img_embeddings, text_embeddings), dim=1).detach()
        # embeddings = torch.rand(raw_instruction.shape[0], 1024).to(batch['input_ids'].device)
        return embeddings
    
    def get_acquisition_score(self, tasks, real_batch_size=None, diversified=False):
        if real_batch_size is None:
            real_batch_size = int(tasks.shape[0])
        if diversified:
            best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score, acquisition_score = self.risklearner_trainer.acquisition_function(tasks, self.gamma_0, self.gamma_1, self.gamma_2, real_batch_size=real_batch_size)
            return best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score, acquisition_score
        else:
            acquisition_score, acquisition_mean, acquisition_std = self.risklearner_trainer.acquisition_function(tasks, self.gamma_0, self.gamma_1, self.gamma_2, pure_acquisition=True, real_batch_size=real_batch_size)
            return acquisition_score, acquisition_mean, acquisition_std
        
    def select_batch(self, batch, selected_ids):
        selected_batch = {}
        for key in batch:
            if key == 'dataset_names':
                selected_batch[key] = [batch[key][i] for i in selected_ids]
            else:
                selected_batch[key] = batch[key][selected_ids]
        return selected_batch

    def sample_tasks(self, batch, batch_size):
        #libero: batch: {'pixel_values':(bs, 6, 224, 224), 'input_ids': (bs, len), 'attention_mask': (bs, len), 'labels': (bs, len), 'dataset_names': []*bs,
        #'raw_imgs': tensor(bs, 224, 224, 3)}
        with torch.no_grad():
            candidate_tasks = self.identifier_decode(batch) #B, dim
            if not 'diverse' in self.algo_name:
                acquisition_score, acquisition_mean, acquisition_std = self.get_acquisition_score(candidate_tasks) # candidate tasks 15 * loss 1
                if self.current_steps < self.warmup_steps:
                    rand_idx = np.random.choice(candidate_tasks.shape[0], batch_size, replace=False)
                    return self.select_batch(batch, rand_idx), acquisition_score[rand_idx]
                acquisition_score = acquisition_score.squeeze(1) # candidate tasks 15
                if not self.no_add_random:
                    selected_values, selected_index = torch.topk(acquisition_score, k=batch_size//2)
                else:
                    selected_values, selected_index = torch.topk(acquisition_score, k=batch_size)
                mask = ~torch.isin(torch.arange(0, int(len(candidate_tasks))), selected_index.cpu())
                unselected_index = torch.arange(0, int(len(candidate_tasks)))[mask]
                index=torch.cat((selected_index.cpu(),unselected_index),dim=0)[:batch_size][torch.randperm(batch_size)] # num_tasks 10
                return self.select_batch(batch, index), acquisition_score[index]
            else:
                best_batch_id, diversified_score, combine_local_diverse_score, combine_local_acquisition_score, acquisition_score = self.get_acquisition_score(candidate_tasks, real_batch_size=int(batch_size), diversified=True)
                if self.current_steps < self.warmup_steps:
                    rand_idx = np.random.choice(candidate_tasks.shape[0], batch_size, replace=False)
                    return self.select_batch(batch, rand_idx), acquisition_score[rand_idx]
                return self.select_batch(batch, best_batch_id), acquisition_score[best_batch_id]
    
    def gather_all_tensors(self, tensor):
        """
        Gather tensors from all GPUs and concatenate them on the first GPU.
        """
        if not dist.is_initialized():
            raise RuntimeError("Distributed environment is not initialized")
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensors_gather, tensor)
        
        if rank == 0:
            output = torch.cat(tensors_gather, dim=0)
            return output
        else:
            return None
        
    def train(self, batch, batch_loss):
        batch['input_ids'] = batch['input_ids'].to(batch_loss.device)
        batch['raw_imgs'] = batch['raw_imgs'].to(batch_loss.device)
        if dist.is_initialized():
            raw_instruction = self.gather_all_tensors(batch['input_ids'])
            raw_imgs = self.gather_all_tensors(batch['raw_imgs'])
            batch_loss = self.gather_all_tensors(batch_loss)
        loss, recon_loss, kl_loss = None, None, None
        if dist.get_rank() == 0:
            gathered_batch = {}
            gathered_batch['input_ids'] = raw_instruction
            gathered_batch['raw_imgs'] = raw_imgs
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # gathered_batch = batch
                self.current_steps += 1
                tasks = self.identifier_decode(gathered_batch).to(batch_loss.device)
                # if self.batch_norm:
                #     batch_loss = (batch_loss - batch_loss.mean()) / (batch_loss.std() + 1e-6)
                for _ in range(self.mpts_training_steps):
                    loss, recon_loss, kl_loss = self.risklearner_trainer.train(tasks.detach(), batch_loss.detach())
                # print(f"Step {self.current_steps}: Loss {loss}, Recon Loss {recon_loss}, KL Loss {kl_loss}")
        # dist.barrier()
        for param in self.risklearner_trainer.risklearner.parameters():
            dist.broadcast(param.data, src=0)
        return loss, recon_loss, kl_loss

from torch.nn import CrossEntropyLoss
def cal_batch_loss(outputs, inputs):
    logits = outputs['logits'].float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].contiguous()
    shift_logits = shift_logits[:, -shift_labels.size(1):].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    shift_labels = shift_labels.to(shift_logits.device)
    batch_size, sentence_len, vocab_size = shift_logits.size(0), shift_logits.size(1), shift_logits.size(2)
    element_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    batch_element_loss = element_loss.view(batch_size, sentence_len)
    batch_loss = batch_element_loss.sum(dim=1)/(torch.sum(shift_labels>=0, dim=1))
    return batch_loss


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # Set Seed
    set_seed_everywhere(cfg.seed)
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    # Configure Unique Experiment ID & Log Directory
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"
    exp_id += f"--{cfg.algo_name}"
    if 'mpts' in cfg.algo_name:
        exp_id += f"--{cfg.sampler_multiplier}"
        exp_id += f"--{cfg.mpts_training_steps}"
        exp_id += f"--{cfg.mpts_training_lr}"
    if cfg.algo_name == 'diverse_mpts':
        exp_id += f"--{cfg.diversity_type}"

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True, static_graph=True)
    vla.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":True})
    if 'mpts' in cfg.algo_name:
        mpts_sampler = MPTS(cfg.algo_name, processor.tokenizer, device_ids=[device_id], diversity_type=cfg.diversity_type, mpts_training_steps=cfg.mpts_training_steps, mpts_training_lr=cfg.mpts_training_lr)
    else:
        cfg.sampler_multiplier = 1

    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    torch.multiprocessing.set_sharing_strategy('file_system')
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size * cfg.sampler_multiplier,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process and cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=f"ft+{exp_id}", config=vars(cfg))

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)
    local_steps = 0

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            #libero: batch: {'pixel_values':(bs, 6, 224, 224), 'input_ids': (bs, len), 'attention_mask': (bs, len), 'labels': (bs, len), 'dataset_names': []*bs,
            #'raw_img': (bs, 3, 224, 224)}
            for_start_time = time.time()
            start_time = time.time()
            if 'mpts' in cfg.algo_name:
                batch, acquisition_score = mpts_sampler.sample_tasks(batch, cfg.batch_size)
            print(f"Time to sample tasks: {time.time() - start_time}")
            # for batch_data in 
            
            start_time = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                
                loss = output.loss
                batch_loss = cal_batch_loss(output, batch)
            print(f"Time to forward pass: {time.time() - start_time}")
                

            # Normalize loss to account for gradient accumulation
            normalized_loss = loss / cfg.grad_accumulation_steps

            # Backward pass
            normalized_loss.backward()

            # Compute Accuracy and L1 Loss for Logging
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            # Store recent train metrics
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            # Compute gradient step index
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

            # Compute smoothened train metrics
            #   =>> Equal to current step metrics when not using gradient accumulation
            #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)
            
            start_time = time.time()
            if 'mpts' in cfg.algo_name:
                mpts_loss, recon_loss, kl_loss = mpts_sampler.train(batch, batch_loss)
            print(f"Time to train mpts: {time.time() - start_time}")
                

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0 and cfg.use_wandb:
                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "action_accuracy": smoothened_action_accuracy,
                        "l1_loss": smoothened_l1_loss,
                    },
                    step=gradient_step_idx,
                )
                if 'mpts' in cfg.algo_name:
                    if mpts_loss is not None:
                        wandb.log({'mpts_loss': mpts_loss, 'recon_loss': recon_loss, 'kl_loss': kl_loss}, step=gradient_step_idx)
                    if acquisition_score is not None:
                        wandb.log({'acquisition_score': acquisition_score.mean().item(),
                                   'corr':np.corrcoef(acquisition_score.squeeze().cpu().detach().numpy(), batch_loss.squeeze().cpu().detach().numpy())[0,1],  }, step=gradient_step_idx)

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()
            torch.cuda.synchronize()
            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                    save_dir = adapter_dir if cfg.use_lora else run_dir

                    # Save Processor & Weights
                    processor.save_pretrained(run_dir)
                    vla.module.save_pretrained(save_dir)

                # Wait for processor and adapter weights to be saved by main process
                dist.barrier()

                # Merge LoRA weights into model backbone for faster inference
                #   =>> Note that merging is slow and can be done post-hoc to speed up training
                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:
                            # Overwrite latest checkpoint
                            merged_vla.save_pretrained(run_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:
                            # Prepare to save checkpoint in new directory
                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)

                            # Save dataset statistics to new directory
                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                            # Save processor and model weights to new directory
                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                # Block on Main Process Checkpointing
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break

            print(f"Time for one batch: {time.time() - for_start_time}")


if __name__ == "__main__":
    finetune()
