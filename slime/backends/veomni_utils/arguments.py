import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Literal, Optional

import yaml


from slime.utils.arguments_type import FullArgs


@dataclass
class VeOmniArgs:
    # Optim
    optimizer: str = "adamw"
    lr: float = 2e-5
    lr_decay_style: str = "constant"
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_ratio: float = 0.0

    # FSDP specific
    # fsdp_wrap: str = "transformer_blocks"  # future use: auto wrap policy
    # fsdp_sharding_strategy: str = "FULL_SHARD"
    # fsdp_cpu_offload: bool = False
    # fsdp_limit_all_gathers: bool = False
    # fsdp_sync_module_states: bool = True
    # fsdp_forward_prefetch: bool = True
    # fsdp_backward_prefetch: bool = True

    enable_full_shard: bool = True
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    enable_fsdp_offload: bool = False
    enable_forward_prefetch: bool = True
    enable_reentrant: bool = False

    # Currently broken
    compile: bool = False

    data_parallel_mode: str = field(
        default="fsdp2",
        metadata={"help": "Data parallel mode."},
    )
    data_parallel_replicate_size: int = field(
        default=1,
        metadata={"help": "Data parallel replicate size."},
    )
    data_parallel_shard_size: int = field(
        default=-1,
        metadata={"help": "Data parallel shard degree."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size."},
    )
    expert_parallel_size: int = field(
        default=1,
        metadata={"help": "Expert parallel size."},
    )
    pipeline_parallel_size: int = field(
        default=1,
        metadata={"help": "Pipeline parallel size."},
    )
    ulysses_parallel_size: int = field(
        default=1,
        metadata={"help": "Ulysses sequence parallel size."},
    )
    context_parallel_size: int = field(
        default=1,
        metadata={"help": "Ring-attn context parallel size."},
    )
    enable_activation_offload: bool = field(
        default=False,
        metadata={"help": "Enable activation offload to CPU."},
    )
    activation_gpu_limit: float = field(
        default=0.0,
        metadata={
            "help": "When enabling activation offload, `activation_gpu_limit` GB activations are allowed to reserve on GPU."
        },
    )

    basic_modules: Optional[list[str]] = field(
        default_factory=list,
        metadata={"help": "Basic modules beyond model._no_split_modules to be sharded in FSDP."},
    )

    # Logging
    wandb_project: str = "slime-veomni"
    wandb_run_name: Optional[str] = None

    # YAML bookkeeping
    config: Optional[str] = None
    output_dir: str = "output"


@dataclass
class VeOmnniFullArgs(VeOmniArgs, FullArgs):
    pass


def parse_veomni_cli(extra_args_provider=None):
    parser = argparse.ArgumentParser("FSDP SFT Training (slime)")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    for f in dataclasses.fields(VeOmniArgs):
        if f.name == "config":
            continue
        arg_type = f.type if f.type != Optional[str] else str
        if arg_type is bool:
            # Get default value
            default_value = f.default
            parser.add_argument(f"--{f.name.replace('_', '-')}", default=default_value)
        else:
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=arg_type, default=f.default)

    if extra_args_provider is not None:
        parser = extra_args_provider(parser)
    args = parser.parse_args()
    return args


def load_veomni_args(extra_args_provider=None):
    args = parse_veomni_cli(extra_args_provider)
    if args.config:
        with open(args.config, "r") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if not hasattr(args, k):
                setattr(args, k, v)
    return args
