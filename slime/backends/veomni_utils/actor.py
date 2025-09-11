from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch_memory_saver import torch_memory_saver
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

import wandb
from slime.ray.train_actor import TrainRayActor
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss
from slime.utils.timer import Timer, timer

from slime.utils.wandb_utils import init_wandb_secondary
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
from veomni.models.auto import build_foundation_model
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.optim.optimizer import build_optimizer
from .update_weight_utils import UpdateWeightFromTensor
from .arguments import VeOmnniFullArgs

from dataclasses import dataclass, field


from veomni.utils.arguments import ModelArguments, TrainingArguments
from dataclasses import field


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


class VeOmniTrainRayActor(TrainRayActor):
    """Simplified TrainRayActor for pure HF+FSDP training.

    Responsibilities:
      * Initialize model/tokenizer on rank0 sequentially to avoid race on cache
      * Wrap model with FSDP
      * Provide minimal train / save / update_weights hooks compatible with existing RayTrainGroup

    Weight update strategy:
      * Rank0 gathers state_dict (full) and broadcasts tensor-by-tensor.
      * For small models this is fine; for larger models consider sharded state_dict type.
    """

    def init(self, args: VeOmnniFullArgs, role, wandb_run_id, with_ref: bool = False):  # type: ignore[override]
        super().init(args, role, wandb_run_id, with_ref)
        self.args = args
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
        # TODO: Fix global rank stuffs for when colocate is disabled
        if dist.get_rank() == 0:
            init_wandb_secondary(args, wandb_run_id)

        torch.manual_seed(args.seed)
        with torch.device(f"cuda:{torch.cuda.current_device()}"):
            self.model = build_foundation_model(
                config_path=args.hf_checkpoint,
                quantize=True,  # Student model quantization
                weights_path=args.hf_checkpoint,
                torch_dtype="bfloat16",
                # attn_implementation=args.model.attn_implementation,
                # moe_implementation=args.model.moe_implementation,
                # init_device=args.train.init_device,
                # force_use_huggingface=args.model.force_use_huggingface,
            )
        self.hf_config = self.model.config
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

        get_optimizer_pre_hook = getattr(self.model, "get_optimizer_pre_hook", None)
        basic_modules = self.model._no_split_modules if self.model._no_split_modules is not None else []
        if isinstance(args.basic_modules, list):
            basic_modules.extend(args.basic_modules)
        self.model = build_parallelize_model(
            self.model,
            # init_device=args.train.init_device,
            weights_path=args.hf_checkpoint,
            enable_full_shard=args.enable_full_shard,
            enable_mixed_precision=args.enable_mixed_precision,
            enable_gradient_checkpointing=args.enable_gradient_checkpointing,
            enable_fsdp_offload=args.enable_fsdp_offload,
            basic_modules=self.model._no_split_modules,
            enable_reentrant=args.enable_reentrant,
            enable_forward_prefetch=args.enable_forward_prefetch,
        )

        self.optimizer = build_optimizer(
            self.model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            fused=True,
            optimizer_type=args.optimizer,
        )
        if get_optimizer_pre_hook is not None:
            optimizer_pre_hook = get_optimizer_pre_hook(self.model, self.hf_config, args.data_parallel_mode)
            self.optimizer.register_step_pre_hook(optimizer_pre_hook)

        assert not args.enable_activation_offload, "Activation offload is not supported yet."

        # self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
        #     args.enable_activation_offload,
        #     args.enable_gradient_checkpointing,
        #     args.activation_gpu_limit,
        # )
        self.model.train()
        # Load model

        self.ref_model = None
        # TODO: support ref model
        if with_ref:
            raise NotImplementedError()

        self.weight_updator = UpdateWeightFromTensor(self.args, self.model)

        if self.args.offload:
            self.sleep(("model"))

        Timer().start("train_wait")
        self.global_step = 0
        self.micro_step = 0
        return 0

    def sleep(self, tags):
        if not getattr(self.args, "offload", False):
            return
        if torch_memory_saver is not None:
            torch_memory_saver.pause()

    def wake_up(self, tags):
        if not getattr(self.args, "offload", False):
            return
        if torch_memory_saver is not None:
            torch_memory_saver.resume()

    def save_model(self, iteration, with_optimizer=True):
        if self.args.debug_rollout_only:
            return

        raise NotImplementedError()

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        self.rollout_engines = rollout_engines

        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        self.weight_updator.connect_rollout_engines(rollout_engines, rollout_engine_lock)
        dist.barrier(group=get_gloo_group())

    def compute_log_prob(
        self,
        model_tag,
        padded_batches,
        store_prefix="",
    ):
        rollout_data = {f"{store_prefix}log_probs": []}
        with timer(f"{store_prefix}log_probs") and torch.no_grad():
            for batch in padded_batches:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.model(input_ids=batch["tokens"]).logits
                batch[f"{store_prefix}log_probs"] = gather_log_probs(logits, batch["tokens"])
        return rollout_data

    def pad_and_move_to_device(self, rollout_data):
        tokens = rollout_data["tokens"]
        loss_masks = rollout_data["loss_masks"]

        padded_batches = []
        for i in range(0, len(tokens), self.args.micro_batch_size):
            batch_tokens = tokens[i : i + self.args.micro_batch_size]
            batch_loss_masks = loss_masks[i : i + self.args.micro_batch_size]
            max_len = max(len(t) for t in batch_tokens)
            padded_tokens = [t + [self.tokenizer.pad_token_id] * (max_len - len(t)) for t in batch_tokens]
            padded_loss_masks = [
                # -1 because its the loss mask for logprob
                [0] * (len(t) - len(l) - 1) + l + [0] * (max_len - len(t))
                for l, t in zip(batch_loss_masks, batch_tokens)
            ]
            padded_batches.append(
                {
                    "tokens": torch.tensor(padded_tokens, dtype=torch.long, device=torch.cuda.current_device()),
                    "loss_masks": torch.tensor(padded_loss_masks, dtype=torch.int, device=torch.cuda.current_device()),
                    "rewards": torch.tensor(
                        rollout_data["rewards"][i : i + self.args.micro_batch_size],
                        dtype=torch.float,
                        device=torch.cuda.current_device(),
                    ),
                    "raw_reward": rollout_data["raw_reward"][i : i + self.args.micro_batch_size],
                }
            )
        return padded_batches

    def train(self, rollout_id, rollout_data_ref):  # type: ignore[override]
        Timer().end("train_wait")

        if self.args.offload:
            self.wake_up(("model"))

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        rollout_data = process_rollout_data(self.args, rollout_data_ref, rank, world_size)
        padded_batches = self.pad_and_move_to_device(rollout_data)

        grad_accum = self.args.global_batch_size // (self.args.micro_batch_size * world_size)
        assert grad_accum > 0, (
            f"Invalid grad_accum {grad_accum} for micro_batch_size {self.args.micro_batch_size} and global_batch_size {self.args.global_batch_size}"
        )

        if self.ref_model is not None:
            self.compute_log_prob("ref", padded_batches, store_prefix="ref_")

        self.compute_log_prob("actor", padded_batches)

        # TODO: compute rewards and adv for t
        for batch in padded_batches:
            if self.args.advantage_estimator in ["grpo", "gspo"]:
                batch["advantages"] = batch["returns"] = batch["rewards"].expand_as(batch["log_probs"])
            else:
                raise NotImplementedError(f"Unsupported advantage_estimator {self.args.advantage_estimator}")

        # TODO: finish log rollout_data
        log_dict = {}
        for key in ["log_probs", "ref_log_probs", "advantages", "returns", "raw_reward"]:
            if key not in padded_batches[0]:
                continue
            val = torch.tensor([0.0], device=torch.cuda.current_device())
            for batch in padded_batches:
                if isinstance(batch[key], torch.Tensor):
                    val += per_sample_mean(batch[key], batch["loss_masks"]).item()
                else:
                    val += sum(batch[key])
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            log_dict[f"rollout/{key}"] = (val / len(padded_batches) / world_size).item()
        if dist.get_rank() == 0:
            print(f"rollout {rollout_id}: {log_dict}")
            if self.args.use_wandb:
                log_dict["rollout/step"] = (
                    rollout_id
                    if not self.args.wandb_always_use_train_step
                    else rollout_id
                    * self.args.rollout_batch_size
                    * self.args.n_samples_per_prompt
                    // self.args.global_batch_size
                )
                wandb.log(log_dict)

        reported_accum: dict[str, list[torch.Tensor]] = {}
        self.optimizer.zero_grad(set_to_none=True)
        for mbs_id, batch in enumerate(padded_batches):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(input_ids=batch["tokens"]).logits
            log_probs = gather_log_probs(logits, batch["tokens"])

            if self.args.advantage_estimator == "gspo":
                raise NotImplementedError("implement GSPO")

            ppo_kl = batch["log_probs"] - log_probs
            pg_loss, pg_clipfrac = compute_policy_loss(
                ppo_kl, batch["advantages"], self.args.eps_clip, self.args.eps_clip_high
            )

            pg_loss = per_sample_mean(pg_loss, batch["loss_masks"])
            pg_clipfrac = per_sample_mean(pg_clipfrac, batch["loss_masks"])
            ppo_kl = per_sample_mean(ppo_kl.abs(), batch["loss_masks"])

            loss = pg_loss

            if self.args.use_tis:
                raise NotImplementedError("implement TIS")

            if self.args.entropy_coef != 0:
                raise NotImplementedError("implement entropy bonus")

            if self.args.use_kl_loss:
                kl = compute_approx_kl(
                    log_probs,
                    batch["ref_log_probs"],
                    kl_loss_type=self.args.kl_loss_type,
                )
                kl_loss = per_sample_mean(kl, batch["loss_masks"])

                loss = loss + self.args.kl_loss_coef * kl_loss

            reported = {
                "loss": pg_loss.detach(),
                "pg_loss": pg_loss.detach(),
                "pg_clipfrac": pg_clipfrac.detach(),
                "ppo_kl": ppo_kl.detach(),
            }
            if self.args.use_kl_loss:
                reported["kl_loss"] = kl_loss.detach()

            # Scale loss for gradient accumulation
            loss = loss / grad_accum
            loss.backward()

            # Accumulate reported metrics (store tensors for later mean)
            for k, v in reported.items():
                reported_accum.setdefault(k, []).append(v)

            if (mbs_id + 1) % grad_accum == 0:
                # TODO: check if the grad norm is global grad norm.
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                # Aggregate logs
                aggregated = {k: torch.stack(v).mean().item() for k, v in reported_accum.items()}
                # TODO: change this, this is slow.
                reduced_aggregated = [None] * world_size
                dist.all_gather_object(reduced_aggregated, aggregated)
                # Mean across dp ranks
                aggregated = {}
                for k in reported_accum.keys():
                    aggregated[k] = sum([r[k] for r in reduced_aggregated]) / world_size
                reported_accum = {}
                if dist.get_rank() == 0:
                    log_dict = {
                        f"train/{k}": (val.item() if torch.is_tensor(val) else val) for k, val in aggregated.items()
                    }
                    log_dict["train/grad_norm"] = grad_norm.item() if not isinstance(grad_norm, float) else grad_norm
                    for gid, group in enumerate(self.optimizer.param_groups):
                        if "lr" in group:
                            log_dict[f"train/lr-pg_{gid}"] = group["lr"]
                    print(f"step {self.global_step}: {log_dict}")
                    if self.args.use_wandb:
                        log_dict["train/step"] = self.global_step
                        wandb.log(log_dict)
                self.global_step += 1

        Timer().start("train_wait")
        return

    def update_weights(self):  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.offload:
            # TODO: don't wake up here
            self.wake_up(("model"))

        with torch_memory_saver.disable() if self.args.offload and not torch.version.hip else nullcontext():
            self.weight_updator.update_weights()

        if self.args.offload:
            # TODO: don't wake up here
            self.sleep(("model"))


def gather_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    # log_probs: [B, T-1, V]; input_ids: [B, T]
    pred_logits = logits[:, :-1]
    log_probs_all = torch.log_softmax(pred_logits, dim=-1)
    tgt = input_ids[:, 1:].contiguous()
    log_probs = log_probs_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return log_probs


def per_sample_mean(x, loss_mask):
    # TODO: impl per token loss
    return ((x * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1)).mean()
