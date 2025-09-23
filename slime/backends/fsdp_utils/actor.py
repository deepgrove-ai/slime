from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, StateDictType
from torch.distributed.tensor import DTensor
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    get_optimizer_state_dict,
)
import wandb
from slime.ray.registry import get_actors
from slime.ray.train_actor import TrainRayActor
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss
from slime.utils.timer import Timer, timer
from slime.utils.memory_utils import available_memory, clear_memory, print_memory
from slime.utils.wandb_utils import init_wandb_secondary

from .update_weight_utils import UpdateWeightFromTensor, PreprocessTensorFunc
from slime.utils.logging import configure_logging
from transformers import PreTrainedModel

# TODO:
# from torch.distributed.checkpoint.state_dict import get_state_dict

logger = configure_logging(__name__)


class FSDPTrainRayActor(TrainRayActor):
    """Simplified TrainRayActor for pure HF+FSDP training.

    Responsibilities:
      * Initialize model/tokenizer on rank0 sequentially to avoid race on cache
      * Wrap model with FSDP
      * Provide minimal train / save / update_weights hooks compatible with existing RayTrainGroup

    Weight update strategy:
      * Rank0 gathers state_dict (full) and broadcasts tensor-by-tensor.
      * For small models this is fine; for larger models consider sharded state_dict type.
    """

    def __init__(self, world_size, rank, master_addr, master_port, wandb_run_id):
        torch_memory_saver.hook_mode = "torch"
        super().__init__(world_size, rank, master_addr, master_port, wandb_run_id)

    def init_model(self, args) -> PreTrainedModel:
        with torch.device(f"cuda:{torch.cuda.current_device()}"):
            model = AutoModelForCausalLM.from_pretrained(
                args.hf_checkpoint,
                trust_remote_code=True,
            )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        auto_wrap_policy = None

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            sharding_strategy=ShardingStrategy[args.fsdp_sharding_strategy],
            cpu_offload=args.fsdp_cpu_offload,
            forward_prefetch=args.fsdp_forward_prefetch,
            backward_prefetch=args.fsdp_backward_prefetch,
            limit_all_gathers=args.fsdp_limit_all_gathers,
        )
        return model

    def init_optimizer(self, args, model: torch.nn.Module) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        return optimizer

    @classmethod
    def build_model_weights_post_process(cls, args) -> PreprocessTensorFunc | None:
        return None

    def init(self, args, role, wandb_run_id, with_ref: bool = False):  # type: ignore[override]
        super().init(args, role, wandb_run_id, with_ref)

        if dist.get_rank() == 0:
            init_wandb_secondary(args, wandb_run_id)
            world_size = dist.get_world_size()
            grad_accum = self.args.global_batch_size // (self.args.micro_batch_size * world_size)
            print(f"{grad_accum=},{self.args.global_batch_size=},{self.args.micro_batch_size=},{world_size=}")

        self.args = args
        torch.manual_seed(args.seed)

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        if self.args.multimodal_keys:
            self.vlm_processor = AutoProcessor.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)

        # Load model
        print_memory("before init model")
        self.model = self.init_model(args)
        self.model.train()
        print_memory("after init model")
        self.optimizer = self.init_optimizer(args, self.model)

        self.weights = {"actor": {}}

        self.ref_model = None
        if with_ref:
            self.load_ref_model(args.ref_load)
        print_memory("after load ref model")

        self.update_cpu_params_dict(self.weights["actor"])

        self.weight_updator = UpdateWeightFromTensor(
            self.args, self.original_model, self.build_model_weights_post_process(args)
        )
        self.connected = False

        self.optimizer_state_dict = {}

        if self.args.offload:
            self.offload_optimizer()
            self.sleep(("model"))

        Timer().start("train_wait")
        self.global_step = 0
        self.micro_step = 0
        return 0

    @property
    def original_model(self) -> torch.nn.Module:
        return self.model

    @timer
    def sleep(self, tags):
        assert self.args.offload
        assert "model" in tags
        if isinstance(tags, str):
            tags = (tags,)

        clear_memory()
        print_memory(f"before offload model")
        assert torch_memory_saver is not None
        torch_memory_saver.pause()
        print_memory(f"after offload model")

    @timer
    def wake_up(self, tags):
        assert self.args.offload
        print_memory("before wake_up model")

        if isinstance(tags, str):
            tags = (tags,)

        assert torch_memory_saver is not None
        torch_memory_saver.resume()
        print_memory("after wake_up model")

    def save_model(self, iteration):
        if self.args.debug_rollout_only:
            return

        import os

        os.makedirs(f"{self.args.save}/iter_{iteration:07}/hf", exist_ok=True)

        self.model.save_pretrained(f"{self.args.save}/iter_{iteration:07}/hf", save_dtype=torch.bfloat16)

    def compute_log_prob(
        self,
        model_tag,
        padded_batches,
        store_prefix="",
    ):
        """
        Compute log probabilities using specified model.

        Args:
            model_tag: "actor" for current model, "ref" for reference model
            padded_batches: Input batches
            store_prefix: Prefix for storing results (e.g., "ref_")
        """
        need_restore = False
        if model_tag != "actor" and model_tag in self.weights:
            self.update_cpu_params_dict(self.weights["actor"])
            self.update_gpu_params_dict(self.weights[model_tag])
            need_restore = True
        self.model.eval()

        try:
            rollout_data = {f"{store_prefix}log_probs": []}
            with timer(f"{store_prefix}log_probs") and torch.no_grad():
                for batch in padded_batches:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        model_args = {"input_ids": batch["tokens"]}
                        if "pixel_values" in batch:
                            model_args["pixel_values"] = batch["pixel_values"]
                    logits = self.model(**model_args).logits
                    batch[f"{store_prefix}log_probs"] = gather_log_probs(
                        logits, batch["tokens"], self.args.rollout_temperature
                    )
            return rollout_data

        finally:
            if need_restore:
                self.update_gpu_params_dict(self.weights["actor"])
                torch.cuda.synchronize()
            self.model.train()

    def pad_and_move_to_device(self, rollout_data):
        tokens = rollout_data["tokens"]
        loss_masks = rollout_data["loss_masks"]
        prompts = rollout_data.get("prompt", [[] for _ in range(len(tokens))])

        padded_batches = []
        for i in range(0, len(tokens), self.args.micro_batch_size):
            batch_tokens = tokens[i : i + self.args.micro_batch_size]
            batch_loss_masks = loss_masks[i : i + self.args.micro_batch_size]
            batch_prompts = prompts[i : i + self.args.micro_batch_size]
            max_len = max(len(t) for t in batch_tokens)
            padded_tokens = [t + [self.tokenizer.pad_token_id] * (max_len - len(t)) for t in batch_tokens]
            padded_loss_masks = [
                # -1 because its the loss mask for logprob
                [0] * (len(t) - len(l) - 1) + l + [0] * (max_len - len(t))
                for l, t in zip(batch_loss_masks, batch_tokens)
            ]

            batch_rollout_log_probs = rollout_data["rollout_log_probs"][i : i + self.args.micro_batch_size]
            padded_rollout_log_probs = [
                [0] * (len(t) - len(l) - 1) + l + [0] * (max_len - len(t))
                for l, t in zip(batch_rollout_log_probs, batch_tokens)
            ]
            batch = {
                "tokens": torch.tensor(padded_tokens, dtype=torch.long, device=torch.cuda.current_device()),
                "loss_masks": torch.tensor(padded_loss_masks, dtype=torch.int, device=torch.cuda.current_device()),
                "rewards": torch.tensor(
                    rollout_data["rewards"][i : i + self.args.micro_batch_size],
                    dtype=torch.float,
                    device=torch.cuda.current_device(),
                ),
                "raw_reward": rollout_data["raw_reward"][i : i + self.args.micro_batch_size],
                "rollout_log_probs": torch.tensor(
                    padded_rollout_log_probs,
                    dtype=torch.float,
                    device=torch.cuda.current_device(),
                ),
            }

            if self.args.multimodal_keys:
                processed_media = {}
                for sample_prompt in batch_prompts:
                    for media_part in sample_prompt:
                        media_type = media_part.get("type")

                        if media_type == "image":
                            path = media_part.get("path")
                            if path:
                                if "pixel_values" not in processed_media:
                                    processed_media["pixel_values"] = []
                                image = Image.open(path).convert("RGB")
                                inputs = self.vlm_processor(images=image, return_tensors="pt")
                                processed_media["pixel_values"].append(inputs.pixel_values)

                # Stack and move all processed media to the GPU for the batch
                for key, tensor_list in processed_media.items():
                    if tensor_list:
                        batch[key] = torch.cat(tensor_list).to(
                            device=torch.cuda.current_device(), dtype=torch.bfloat16
                        )

            padded_batches.append(batch)
        return padded_batches

    @classmethod
    def check_logprobs(cls, logprobs: torch.Tensor, loss_masks: torch.Tensor, rollout_log_probs: torch.Tensor):
        assert logprobs.shape == loss_masks.shape == rollout_log_probs.shape, (
            f"{logprobs.shape=} != {loss_masks.shape=} != {rollout_log_probs.shape=}"
        )
        with torch.no_grad():
            l2_dist = torch.nn.functional.mse_loss(logprobs * loss_masks, rollout_log_probs * loss_masks)
        if l2_dist > 5e-3:
            logger.warning(f"L2 distance between logprobs and rollout_log_probs is {l2_dist:.4f}")
        return l2_dist

    def train(self, rollout_id, rollout_data_ref):  # type: ignore[override]
        Timer().end("train_wait")
        Timer().start("train_loop")

        if self.args.offload:
            self.wake_up(("model"))
            print_memory("before load_optimizer")
            with timer("load_optimizer"):
                self.update_gpu_optimizer_dict()
            print_memory("after load_optimizer")

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        rollout_data = process_rollout_data(self.args, rollout_data_ref, rank, world_size)
        padded_batches = self.pad_and_move_to_device(rollout_data)

        grad_accum = self.args.global_batch_size // (self.args.micro_batch_size * world_size)
        assert grad_accum > 0, (
            f"Invalid grad_accum {grad_accum} for micro_batch_size {self.args.micro_batch_size} and global_batch_size {self.args.global_batch_size}"
        )

        with timer("llh_actor"):
            self.compute_log_prob("actor", padded_batches)

        if "ref" in self.weights:
            with timer("llh_ref"):
                self.compute_log_prob("ref", padded_batches, store_prefix="ref_")

        # TODO: compute rewards and adv for t
        for batch in padded_batches:
            # check log_probs
            self.check_logprobs(batch["log_probs"], batch["loss_masks"], batch["rollout_log_probs"])
            if self.args.advantage_estimator in ["grpo", "gspo"]:
                batch["advantages"] = batch["returns"] = batch["rewards"].expand_as(batch["log_probs"])
            else:
                raise NotImplementedError(f"Unsupported advantage_estimator {self.args.advantage_estimator}")

        log_dict = {}

        for key in ["log_probs", "ref_log_probs", "advantages", "returns", "raw_reward", "rollout_log_probs"]:
            if key not in padded_batches[0]:
                continue
            val = torch.tensor([0.0], device=torch.cuda.current_device())
            for batch in padded_batches:
                data = batch[key]
                batch_mask = batch["loss_masks"]
                if isinstance(data, torch.Tensor):
                    # check that first dim matches
                    assert data.shape[0] == batch_mask.shape[0], f"{data.shape[0]=} != {batch_mask.shape[0]=}"
                    # If this is an output token stat, it should be [batch, seq_len]
                    if data.shape[1] == batch_mask.shape[1]:
                        val += per_sample_mean(data, batch_mask).item()
                    else:
                        # Otherwise take the mean
                        val += data.mean().item()
                else:
                    val += sum(data)
            dist.all_reduce(val, op=dist.ReduceOp.SUM)
            log_dict[f"rollout/{key}"] = (val / len(padded_batches) / world_size).item()
        for key in [
            "raw_reward",
            # "rollout_log_probs",
            "response_lengths",
        ]:
            assert key in rollout_data, f"key {key} not in rollout_data"
            val = torch.tensor([0.0], device=torch.cuda.current_device())
            data = rollout_data[key]

            if isinstance(data, list):
                val += sum(data) / len(data)
            else:
                logger.warning(f"Unsupported type: {key}: {type(data)}")
                val += sum(data)
            dist.all_reduce(val, op=dist.ReduceOp.AVG)
            log_dict[f"rollout/{key}"] = val.item()
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

        print_memory("before train")
        with timer("train"):
            reported_accum: dict[str, list[torch.Tensor]] = {}
            self.optimizer.zero_grad(set_to_none=True)
            for mbs_id, batch in enumerate(padded_batches):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.model(input_ids=batch["tokens"]).logits
                log_probs = gather_log_probs(logits, batch["tokens"], self.args.rollout_temperature)

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

                # TODO: report entropy

                reported = {
                    "loss": loss.detach(),
                    "pg_loss": pg_loss.detach(),
                    "pg_clipfrac": pg_clipfrac.detach(),
                    "ppo_kl": ppo_kl.detach(),
                }

                if self.args.use_kl_loss:
                    reported["kl_loss"] = kl_loss.detach()

                loss = loss / grad_accum
                loss.backward()

                for k, v in reported.items():
                    reported_accum.setdefault(k, []).append(v)

                if (mbs_id + 1) % grad_accum == 0:
                    # TODO: check if the grad norm is global grad norm.
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    aggregated = {}
                    for k, v in reported_accum.items():
                        if k in ["kl_loss"]:
                            kl_values = torch.stack(v)
                            aggregated[k] = (kl_values * self.args.micro_batch_size).sum().item()
                        else:
                            aggregated[k] = torch.stack(v).mean().item()
                    # TODO: change this, this is slow.
                    reduced_aggregated = [None] * world_size
                    dist.all_gather_object(reduced_aggregated, aggregated)
                    aggregated = {}
                    for k in reported_accum.keys():
                        if k in ["kl_loss"]:
                            total_kl = sum([r[k] for r in reduced_aggregated])
                            aggregated[k] = total_kl / self.args.global_batch_size
                        else:
                            aggregated[k] = sum([r[k] for r in reduced_aggregated]) / world_size
                    reported_accum = {}
                    if dist.get_rank() == 0:
                        log_dict = {
                            f"train/{k}": (val.item() if torch.is_tensor(val) else val)
                            for k, val in aggregated.items()
                        }
                        log_dict["train/grad_norm"] = (
                            grad_norm.item() if not isinstance(grad_norm, float) else grad_norm
                        )

                        for gid, group in enumerate(self.optimizer.param_groups):
                            if "lr" in group:
                                log_dict[f"train/lr-pg_{gid}"] = group["lr"]

                        kl_info = ""
                        if self.args.use_kl_loss and "kl_loss" in aggregated:
                            kl_info = f", kl_loss: {aggregated['kl_loss']:.4f}, kl_penalty: {aggregated['kl_loss'] * self.args.kl_loss_coef:.4f}"

                        print(
                            f"step {self.global_step}: loss: {aggregated.get('loss', 0):.4f}, pg_loss: {aggregated.get('pg_loss', 0):.4f}{kl_info}"
                        )
                        print(f"step {self.global_step} full: {log_dict}")

                        if self.args.use_wandb:
                            log_dict["train/step"] = self.global_step
                            wandb.log(log_dict)
                    self.global_step += 1

        train_memory_stats = available_memory()
        # Prefix with train_mem/
        train_memory_stats = {f"train_mem/{k}": v for k, v in train_memory_stats.items()}
        train_memory_stats["rollout/step"] = rollout_id
        # TODO: All reduce train_memory_stats
        if dist.get_rank() == 0 and self.args.use_wandb:
            wandb.log(train_memory_stats)
                    for gid, group in enumerate(self.optimizer.param_groups):
                        if "lr" in group:
                            log_dict[f"train/lr-pg_{gid}"] = group["lr"]
                    
                    kl_info = ""
                    if self.args.use_kl_loss and "kl_loss" in aggregated:
                        kl_info = f", kl_loss: {aggregated['kl_loss']:.4f}, kl_penalty: {aggregated['kl_loss'] * self.args.kl_loss_coef:.4f}"
                    
                    print(f"step {self.global_step}: loss: {aggregated.get('loss', 0):.4f}, pg_loss: {aggregated.get('pg_loss', 0):.4f}{kl_info}")
                    print(f"step {self.global_step} full: {log_dict}")
                    
                    if self.args.use_wandb:
                        log_dict["train/step"] = self.global_step
                        wandb.log(log_dict)
                self.global_step += 1


        self.update_cpu_params_dict(self.weights["actor"])
        if self.args.offload:
            with timer("offload_optimizer"):
                self.offload_optimizer()
        Timer().end("train_loop")
        log_perf_data(rollout_id, self.args)
        Timer().start("train_wait")

    @timer
    def update_weights(self):  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if not self.connected:
            self.connected = True
            rollout_engines = get_actors("rollout")
            rollout_engine_lock = get_actors("rollout_lock", 0)
            self.weight_updator.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())

        if self.args.offload:
            # TODO: don't wake up here
            self.wake_up(("model"))

        with torch_memory_saver.disable() if self.args.offload and not torch.version.hip else nullcontext():
            print_memory("before update_weights")
            self.weight_updator.update_weights()
            print_memory("after update_weights")

        if self.args.offload:
            # TODO: don't wake up here
            self.sleep(("model"))

    @torch.no_grad()
    def offload_optimizer(self):
        self.optimizer_state_dict = get_optimizer_state_dict(
            self.model, self.optimizer, options=StateDictOptions(cpu_offload=True, full_state_dict=False)
        )

        # 2) Drop the live state from the optimizer to free GPU memory
        #    (safe: PyTorch will lazily re-create on next .step() if you reload later)
        self.optimizer.state.clear()

        # optimizer_sd: dict[str, Any] = get_optimizer_state_dict(
        #     self.model,
        #     self.optimizer,
        #     options=StateDictOptions(full_state_dict=False, cpu_offload=False),
        # )
        # for name, val in optimizer_sd.items():
        #     if isinstance(val, DTensor):
        #         local = val.to_local()
        #     elif isinstance(val, torch.Tensor):
        #         local = val
        #     else:
        #         continue
        #     buf = self.optimizer_state_dict.get(name)
        #     if buf is None or buf.shape != local.shape or buf.dtype != local.dtype:
        #         self.optimizer_state_dict[name] = torch.empty_like(local, device=torch.device("cpu"), pin_memory=True)
        #     self.optimizer_state_dict[name].copy_(local, non_blocking=True)
        # torch.cuda.synchronize()


    @torch.no_grad()
    def update_gpu_optimizer_dict(self):
        if not self.optimizer_state_dict:
            print("No optimizer state dict to update")
            return
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            optim_state_dict=self.optimizer_state_dict,
            options=StateDictOptions(full_state_dict=False, strict=True),
        )
        # template = get_optimizer_state_dict(
        #     self.original_model,
        #     self.optimizer,
        #     options=StateDictOptions(full_state_dict=False, cpu_offload=False),
        # )
        # device = torch.cuda.current_device()
        # rebuild: dict[str, Any] = {}
        # for name, val in template.items():
        #     if isinstance(val, DTensor):
        #         local_cpu = optimizer_dict[name]
        #         local_dev = local_cpu.to(device, non_blocking=True)
        #         rebuild[name] = DTensor.from_local(
        #             local_dev,
        #             device_mesh=val.device_mesh,
        #             placements=val.placements,
        #             shape=val.size(),
        #             stride=val.stride(),
        #         )
        #     elif isinstance(val, torch.Tensor):
        #         rebuild[name] = optimizer_dict[name].to(device, non_blocking=True)
        #     else:
        #         continue
        # set_optimizer_state_dict(
        #     self.model,
        #     self.optimizer,
        #     optim_state_dict=rebuild,
        #     options=StateDictOptions(full_state_dict=False, strict=True),
        # )
        # torch.cuda.synchronize()

    @torch.no_grad()
    def update_cpu_params_dict(self, params_dict: dict[str, torch.Tensor]) -> None:
        """
        Store ONLY this rank's shards on pinned CPU.
        Works with DTensor+DeviceMesh by extracting local shards.
        """
        model_sd: dict[str, Any] = get_model_state_dict(
            self.original_model,
            options=StateDictOptions(full_state_dict=False, cpu_offload=False),
        )
        # print(f"{model_sd=}")

        for name, val in model_sd.items():
            if isinstance(val, DTensor):
                local = val.to_local()  # this rank's shard on device
            elif isinstance(val, torch.Tensor):
                local = val  # unsharded tensor on device
            else:
                continue  # skip non-tensors/metadata if any

            buf = params_dict.get(name)
            if buf is None or buf.shape != local.shape or buf.dtype != local.dtype:
                params_dict[name] = torch.empty_like(local, device=torch.device("cpu"), pin_memory=True)
            params_dict[name].copy_(local, non_blocking=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @torch.no_grad()
    def update_gpu_params_dict(self, params_dict: dict[str, torch.Tensor], strict: bool = True) -> None:
        """
        Rebuild DTensors from the saved local CPU shards using the model's CURRENT
        mesh/placements, then load with DCP state-dict setters (no all-gather).
        """
        # Use a template sharded state_dict to read mesh/placements/global sizes
        template: dict[str, Any] = get_model_state_dict(
            self.original_model,
            options=StateDictOptions(full_state_dict=False, cpu_offload=False),
        )

        device = torch.cuda.current_device()
        rebuild: dict[str, Any] = {}

        for name, spec in template.items():
            if isinstance(spec, DTensor):
                local_cpu = params_dict[name]
                local_dev = local_cpu.to(device, non_blocking=True)
                # Recreate a DTensor with the SAME global spec as the template
                dt = DTensor.from_local(
                    local_dev,
                    device_mesh=spec.device_mesh,
                    placements=spec.placements,
                    shape=spec.size(),
                    stride=spec.stride(),
                )
                rebuild[name] = dt
            elif isinstance(spec, torch.Tensor):
                rebuild[name] = params_dict[name].to(device, non_blocking=True)
            else:
                continue

        # Load back into the module using DCP's setter (strict by default)
        missing_unexpected = set_model_state_dict(
            self.original_model,
            model_state_dict=rebuild,
            options=StateDictOptions(full_state_dict=False, strict=strict),
        )
        if (missing_unexpected.missing_keys or missing_unexpected.unexpected_keys) and strict:
            raise RuntimeError(
                f"load mismatch: missing={missing_unexpected.missing_keys}, "
                f"unexpected={missing_unexpected.unexpected_keys}"
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def load_ref_model(self, ref_load_path):
        """Load reference model parameters once and store in CPU memory (like Megatron backend)"""
        if ref_load_path is None:
            raise ValueError("ref_load_path must be provided when loading reference model")

        print(f"Loading reference model from {ref_load_path}")

        current_weights = {}
        self.update_cpu_params_dict(current_weights)

        try:
            import os

            if os.path.isdir(ref_load_path):
                temp_ref_model = AutoModelForCausalLM.from_pretrained(
                    ref_load_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )

                with FSDP.state_dict_type(self.original_model, StateDictType.FULL_STATE_DICT):
                    self.original_model.load_state_dict(temp_ref_model.state_dict(), strict=True)

                del temp_ref_model
                torch.cuda.empty_cache()
            else:
                raise NotImplementedError(f"Loading from checkpoint file {ref_load_path} not yet implemented")

            self.weights["ref"] = {}
            self.update_cpu_params_dict(self.weights["ref"])

            print(f"Reference model parameters loaded and stored in CPU memory")

        finally:
            self.update_gpu_params_dict(current_weights)


def log_perf_data(rollout_id, args):
    timer_instance = Timer()
    if dist.get_rank() == 0:
        log_dict = {f"perf/{key}_time": val for key, val in timer_instance.log_dict().items()}

        if "perf/train_wait_time" in log_dict and "perf/train_loop_time" in log_dict:
            total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_loop_time"]
            if total_time > 0:
                log_dict["perf/total_train_loop_time"] = total_time
                log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time
        print(f"perf {rollout_id}: {log_dict}")
        if args.use_wandb:
            log_dict["rollout/step"] = (
                rollout_id
                if not args.wandb_always_use_train_step
                else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
            )
            wandb.log(log_dict)
    timer_instance.reset()

def gather_log_probs(logits: torch.Tensor, input_ids: torch.Tensor, rollout_temperature: float = 1.0) -> torch.Tensor:
    # log_probs: [B, T-1, V]; input_ids: [B, T]
    assert rollout_temperature > 0, f"rollout_temperature must be greater than 0, but got {rollout_temperature}"
    pred_logits = logits[:, :-1]
    # haoran: whether to apply temperature shifting here?
    if rollout_temperature != 1.0:
        pred_logits = pred_logits / rollout_temperature
    log_probs_all = torch.log_softmax(pred_logits, dim=-1)

def per_sample_mean(x, loss_mask):
    return ((x * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1)).mean()
