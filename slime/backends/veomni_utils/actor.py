from typing import cast
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from transformers import PreTrainedModel
from slime.backends.fsdp_utils.update_weight_utils import UpdateWeightFromTensor
import torch
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from veomni.models.transformers.qwen3_moe.modeling_qwen3_moe import quantize
import torch.distributed.fsdp._fully_shard._fsdp_param_group

from veomni.distributed.parallel_state import init_parallel_state
from veomni.models.auto import build_foundation_model
from veomni.distributed.torch_parallelize import build_parallelize_model, MixedPrecisionConfig, DType
from veomni.optim.optimizer import MultiOptimizer, build_optimizer
from .arguments import VeOmnniFullArgs
from veomni.utils.convert_moe import get_full_ep_shard_state, resplit_experts_tensor
from functools import partial


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


def quantize_proj_weights(name: str, tensor: torch.Tensor) -> tuple[tuple[str, torch.Tensor], ...]:
    if name.endswith("_proj.weight"):
        assert len(tensor.shape) == 2, "Quantized weight should be 2D"
        return ((name, quantize(tensor)),)
    if name.endswith("_proj"):
        # MOE EP weights
        assert len(tensor.shape) == 3, "EP Quantized weight should be 3D"
        return ((name, quantize(tensor)),)
    return ((name, tensor),)


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

    @classmethod
    def make_weight_updater(cls, args: VeOmnniFullArgs, model: PreTrainedModel) -> UpdateWeightFromTensor:
        postprocess = quantize_proj_weights if args.quantize else None
        preprocess_state_dict = None
        # TODO: Make this less cursed
        if args.moe_implementation == "fused":
            preprocess_state_dict = partial(get_full_ep_shard_state, config=model.config, prefix="", model=model)

        num_experts = int(model.config.num_experts) if hasattr(model.config, "num_experts") else None
        if num_experts is not None:
            resplit_experts_tensor_partial = partial(resplit_experts_tensor, num_experts=num_experts)
            if postprocess is not None:
                og_postprocess = postprocess

                def reprocess(name: str, tensor: torch.Tensor) -> tuple[tuple[str, torch.Tensor], ...]:
                    post_processed = og_postprocess(name, tensor)
                    reprocessed = tuple(k for n, t in post_processed for k in resplit_experts_tensor_partial(n, t))
                    return reprocessed

                postprocess = reprocess
            else:
                postprocess = resplit_experts_tensor_partial
        weight_updater = UpdateWeightFromTensor(
            args, model, postprocess_tensor_func=postprocess, preprocess_state_dict_func=preprocess_state_dict
        )

        # monkey_patch_torch_reductions()
        # sharded_state_dict = cast(
        #     dict[str, torch.Tensor],
        #     get_model_state_dict(model, options=StateDictOptions(full_state_dict=False, cpu_offload=False)),
        # )
        # if preprocess_state_dict is not None:
        #     print("Preprocessing state dict for weight updater")
        #     sharded_state_dict = preprocess_state_dict(sharded_state_dict)
        #     print("Done preprocessing state dict for weight updater")
        return weight_updater

    @torch.no_grad()
    def offload_optimizer(self):
        if not isinstance(self.optimizer, MultiOptimizer):
            return super().offload_optimizer()
        self.optimizer_state_dict = self.optimizer.state_dict(
            options=StateDictOptions(cpu_offload=True, full_state_dict=False)
        )
        self.optimizer.clear_state()

    @torch.no_grad()
    def update_gpu_optimizer_dict(self):
        if not isinstance(self.optimizer, MultiOptimizer):
            return super().update_gpu_optimizer_dict()
        if not self.optimizer_state_dict:
            print("No optimizer state dict to update")
            return
        self.optimizer.load_state_dict(
            self.optimizer_state_dict,
            options=StateDictOptions(full_state_dict=False, strict=True),
        )

    def get_model_state_dict(self) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            state_dict = cast(
                dict[str, torch.Tensor],
                get_model_state_dict(self.original_model, options=StateDictOptions(full_state_dict=False)),
            )
            if self.args.moe_implementation == "fused":
                state_dict = get_full_ep_shard_state(
                    config=self.original_model.config, prefix="", model=self.original_model, state_dict=state_dict
                )
        return state_dict

    def init_model(self, args: VeOmnniFullArgs):
        # TODO: Configure torch seperately
        torch.set_float32_matmul_precision("high")
        if args.data_parallel_shard_size == -1:
            args.data_parallel_shard_size = data_parallel_size(args) // args.data_parallel_replicate_size
        assert args.data_parallel_shard_size > 0, "data_parallel_shard_size should be greater than 0."

        self.mixed_precision_config = MixedPrecisionConfig(
            forward_dtype=DType(args.forward_dtype),
            reduce_dtype=DType(args.reduce_dtype),
            model_dtype=DType(args.model_dtype),
        )
        print(f"Mixed precision config: {self.mixed_precision_config}")
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
            config_path=args.load,
            quantize=args.quantize,  # Student model quantization
            weights_path=args.load,
            torch_dtype=self.mixed_precision_config.model_dtype.value,
            init_device="meta",
            # attn_implementation=args.model.attn_implementation,
            moe_implementation=args.moe_implementation,
            # force_use_huggingface=True,
        )
        basic_modules = model._no_split_modules if model._no_split_modules is not None else []
        if isinstance(args.basic_modules, list):
            basic_modules.extend(args.basic_modules)

        model = build_parallelize_model(
            model,
            init_device="meta",
            weights_path=args.load,
            enable_full_shard=args.enable_full_shard,
            mixed_precision_config=self.mixed_precision_config,
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
        assert ref_load_path == self.args.load, "Reference model path must be the same as the model path"
        self.weights["ref"] = {}
        self.update_cpu_params_dict(self.weights["ref"])
        print("Reference model parameters loaded and stored in CPU memory")

    @property
    def original_model(self) -> PreTrainedModel:
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
