"""Shared Direct-Forcing setup: model, FSDP, optimizer, data, and conditioning."""
import os
import time
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from model.causal_video import CausalVideoModel
from utils.dataset import PromptLMDBDataset, cycle
from utils.distributed import fsdp_wrap, get_fsdp_wrap_kwargs, launch_distributed_job
from utils.misc import resolve_base_seed, set_seed
from utils.prompt_embedding_cache import PromptEmbeddingLMDBCache
from model.wan.modules.attention import require_flash_attention_2

class BaseTrainer:
    model_cls = CausalVideoModel
    dataset_cls = PromptLMDBDataset

    def __init__(self, config):
        self._validate_static_config(config)
        world = int(os.environ.get("WORLD_SIZE", "1"))
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if world != int(config.rdm_expected_world_size) or local_world != int(config.rdm_expected_local_world_size):
            raise RuntimeError("Launch topology disagrees with the configured world size")
        self._initialize_training_runtime(config)

    def _initialize_training_runtime(self, config) -> None:
        """Initialize the shared causal generator and distributed optimizer."""
        self.config = config
        self.step = 0
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.attention_backend = self._validate_attention_backend(config)
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        if self.is_main_process:
            print(f"ATTENTION_BACKEND {self.attention_backend}", flush=True)
        self.disable_wandb = config.disable_wandb
        self.max_steps = int(config.max_steps)
        base_seed = (
            [resolve_base_seed(int(config.seed))] if self.is_main_process else [None]
        )
        dist.broadcast_object_list(base_seed, src=0)
        config.seed = int(base_seed[0])
        set_seed(int(config.seed) + global_rank)
        if self.is_main_process and (not self.disable_wandb):
            wandb_mode = str(getattr(config, "wandb_mode", "online"))
            wandb_host = str(config.wandb_host).strip() or None
            wandb_key = config.wandb_key
            if isinstance(wandb_key, str):
                wandb_key = wandb_key.strip() or None
            if wandb_mode == "online":
                wandb.login(host=wandb_host, key=wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=os.environ.get("DIRECT_FORCING_RUN_NAME", config.config_name),
                mode=wandb_mode,
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir,
            )
        self.output_path = config.logdir
        self.model = self.model_cls(config, device=self.device)
        self.prompt_embedding_cache = None
        if getattr(config, "prompt_embedding_cache_path", ""):
            self.prompt_embedding_cache = PromptEmbeddingLMDBCache(
                config.prompt_embedding_cache_path
            )
        self.model.generator = fsdp_wrap(
            self.model.generator,
            **get_fsdp_wrap_kwargs(
                config, "generator", default_transformer_modules=["causal_wan_block"]
            ),
        )
        if self.model.text_encoder is not None:
            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                **get_fsdp_wrap_kwargs(
                    config,
                    "text_encoder",
                    default_cpu_offload=getattr(
                        config, "text_encoder_cpu_offload", False
                    ),
                ),
            )
            if getattr(
                config, "text_encoder_fsdp_wrap_strategy", "none"
            ) == "none" and (not getattr(config, "text_encoder_cpu_offload", False)):
                self.model.text_encoder = self.model.text_encoder.to(
                    device=self.device, dtype=self.dtype
                )
        self.generator_optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in self.model.generator.parameters()
                if parameter.requires_grad
            ],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
        self.dataset = self.dataset_cls(
            config.data_path,
            max_pair=int(getattr(config, "max_pair", 100000000.0)),
            readahead=bool(getattr(config, "lmdb_readahead", False)),
        )
        sampler = torch.utils.data.distributed.DistributedSampler(
            self.dataset, shuffle=True, drop_last=True, seed=int(config.seed)
        )
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=getattr(config, "dataloader_num_workers", 8),
        )
        self.dataloader = cycle(dataloader)
        if self.is_main_process:
            print(f"DATASET SIZE {len(self.dataset)}", flush=True)
        self.max_grad_norm_generator = float(config.max_grad_norm_generator)


    def _get_conditioning(self, text_prompts):
        if self.prompt_embedding_cache is not None:
            return {
                "prompt_embeds": self.prompt_embedding_cache.get_batch(
                    text_prompts, device=self.device, dtype=self.dtype
                )
            }
        with torch.no_grad():
            return self.model.text_encoder(text_prompts=text_prompts)


    @staticmethod
    def _validate_attention_backend(config) -> str:
        """Fail closed when reproducing the reported RDM training recipe."""
        expected_backend = str(config.rdm_attention_backend)
        expected_version = str(config.rdm_flash_attn_version)
        if expected_backend != "flash_attn_2":
            raise ValueError(
                "the released RDM training recipe requires flash_attn_2"
            )
        if not bool(config.rdm_require_flash_attention):
            raise ValueError(
                "the released RDM training recipe must require FlashAttention"
            )
        return require_flash_attention_2(expected_version)


    def _sync_time(self, start: float) -> float:
        torch.cuda.synchronize(self.device)
        return time.perf_counter() - start


    def _global_max_scalar(self, value: float) -> float:
        tensor = torch.tensor([value], device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())


    def _progress(
        self, phase: str, *, accum_index: int | None = None, detail: str = ""
    ) -> None:
        if not self.is_main_process:
            return
        fields = [
            "DIRECT_FORCING_PROGRESS",
            f"step={self.step}",
            f"update={self.generator_updates + int(phase != 'generator_complete')}",
            f"phase={phase}",
        ]
        if accum_index is not None:
            fields.append(f"accum={accum_index}/{self.grad_accum}")
        if detail:
            fields.append(detail)
        fields.extend(
            [
                f"gpu_allocated_gib={torch.cuda.memory_allocated(self.device) / 2 ** 30:.3f}",
                f"gpu_reserved_gib={torch.cuda.memory_reserved(self.device) / 2 ** 30:.3f}",
            ]
        )
        print(" ".join(fields), flush=True)
