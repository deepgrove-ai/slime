import torch


from veomni.distributed.parallel_state import init_parallel_state
from veomni.models.auto import build_foundation_model
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.optim.optimizer import build_optimizer
from .arguments import VeOmnniFullArgs

from dataclasses import dataclass, field


from veomni.utils.arguments import ModelArguments, TrainingArguments

from slime.backends.fsdp_utils.actor import FSDPTrainRayActor


def data_parallel_size(args: VeOmnniFullArgs):
    return (
        args.actor_num_gpus_per_node
        * args.actor_num_nodes
        // (
            args.pipeline_parallel_size
            * args.ulysses_parallel_size
            * args.context_parallel_size
            * args.tensor_parallel_size
        )
    )


@dataclass
class VeOmniInternalArgs:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)


def dump_param_and_state_dtypes(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n, p in model.named_parameters():
        key = f"param:{p.dtype}"
        counts[key] = counts.get(key, 0) + p.numel()
    for st in optimizer.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                key = f"opt_state:{k}:{v.dtype}"
                counts[key] = counts.get(key, 0) + v.numel()
    return counts


class VeOmniTrainRayActor(FSDPTrainRayActor):
    """Simplified TrainRayActor for pure HF+FSDP training.

    Responsibilities:
      * Initialize model/tokenizer on rank0 sequentially to avoid race on cache
      * Wrap model with FSDP
      * Provide minimal train / save / update_weights hooks compatible with existing RayTrainGroup

    Weight update strategy:
      * Rank0 gathers state_dict (full) and broadcasts tensor-by-tensor.
      * For small models this is fine; for larger models consider sharded state_dict type.
    """

    def init_model(self, args: VeOmnniFullArgs):
        # TODO: Configure torch seperately
        torch.set_float32_matmul_precision("high")
        if args.data_parallel_shard_size == -1:
            args.data_parallel_shard_size = data_parallel_size(args) // args.data_parallel_replicate_size
        assert args.data_parallel_shard_size > 0, "data_parallel_shard_size should be greater than 0."
        init_parallel_state(
            dp_size=data_parallel_size(args),
            dp_replicate_size=args.data_parallel_replicate_size,
            dp_shard_size=args.data_parallel_shard_size,
            tp_size=args.tensor_parallel_size,
            ep_size=args.expert_parallel_size,
            pp_size=args.pipeline_parallel_size,
            cp_size=args.context_parallel_size,
            ulysses_size=args.ulysses_parallel_size,
            dp_mode=args.data_parallel_mode,
        )
        # with torch.device(f"cuda:{torch.cuda.current_device()}"):
        model = build_foundation_model(
            config_path=args.hf_checkpoint,
            quantize=False,  # Student model quantization
            weights_path=args.hf_checkpoint,
            torch_dtype="bfloat16",
            init_device="meta",
            # attn_implementation=args.model.attn_implementation,
            # moe_implementation=args.model.moe_implementation,
            # force_use_huggingface=args.model.force_use_huggingface,
        )
        basic_modules = model._no_split_modules if model._no_split_modules is not None else []
        if isinstance(args.basic_modules, list):
            basic_modules.extend(args.basic_modules)
        model = build_parallelize_model(
            model,
            init_device="meta",
            weights_path=args.hf_checkpoint,
            enable_full_shard=args.enable_full_shard,
            enable_mixed_precision=args.enable_mixed_precision,
            enable_gradient_checkpointing=args.enable_gradient_checkpointing,
            enable_fsdp_offload=args.enable_fsdp_offload,
            # For some reason this messes with grad norms
            basic_modules=model._no_split_modules,
            enable_reentrant=args.enable_reentrant,
            enable_forward_prefetch=args.enable_forward_prefetch,
        )

        assert not args.enable_activation_offload, "Activation offload is not supported yet."
        if args.compile:
            print("Compiling model")
            model = torch.compile(model, mode="default", fullgraph=False)

        # self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
        #     args.enable_activation_offload,
        #     args.enable_gradient_checkpointing,
        #     args.activation_gpu_limit,
        # )
        return model

    def load_ref_model(self, ref_load_path):
        """Load reference model parameters once and store in CPU memory (like Megatron backend)"""
        assert ref_load_path == self.args.hf_checkpoint, "Reference model path must be the same as the model path"
        self.weights["ref"] = {}
        self.update_cpu_params_dict(self.weights["ref"])
        print("Reference model parameters loaded and stored in CPU memory")

    @property
    def original_model(self):
        if self.args.compile:
            assert hasattr(self.model, "_orig_mod"), "Model is not compiled with torch.compile"
            return self.model._orig_mod
        return self.model

    def init_optimizer(self, args: VeOmnniFullArgs, model: torch.nn.Module):
        get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)

        optimizer = build_optimizer(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            fused=True,
            optimizer_type=args.optimizer,
        )
        if get_optimizer_pre_hook is not None:
            optimizer_pre_hook = get_optimizer_pre_hook(model, self.hf_config, args.data_parallel_mode)
            optimizer.register_step_pre_hook(optimizer_pre_hook)
        return optimizer
