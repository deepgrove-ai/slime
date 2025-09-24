from typing import Callable, Iterable, Iterator, Optional, cast
import ray
from slime.backends.fsdp_utils.fsdp_version_utils import preprocess_tensor_for_update_weights
from slime.utils.memory_utils import print_memory
import torch
import torch.distributed as dist
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict


def get_named_tensor_buckets(
    iterable: Iterable[tuple[str, torch.Tensor]], bucket_bytes: int, as_dtype: Optional[torch.dtype] = None
) -> Iterable[list[tuple[str, torch.Tensor]]]:
    """
    Group tensors into buckets based on a specified size in megabytes.

    Args:
        iterable: An iterator of tuples containing tensor names and tensors.
        bucket_bytes: The maximum size of each bucket in bytes.

    Yields:
        Lists of tuples, where each tuple contains a tensor name and its corresponding tensor.

    Example:
        >>> tensors = [('tensor1', torch.randn(1000, 1000)), ('tensor2', torch.randn(2000, 2000))]
        >>> for bucket in get_named_tensor_buckets(tensors, bucket_size_mb=10):
        ...     print(bucket)
        [('tensor1', tensor(...)), ('tensor2', tensor(...))]

    """
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    current_bucket = []
    current_size = 0
    for name, tensor in iterable:
        # Instead of converting the actual tensor, we compute tensor size with the as_dtype if provided
        element_size = tensor.element_size() if as_dtype is None else as_dtype.itemsize
        tensor_size = element_size * tensor.numel()
        if current_size + tensor_size > bucket_bytes:
            if current_bucket:
                yield current_bucket
            current_bucket = [(name, tensor)]
            current_size = tensor_size
        else:
            current_bucket.append((name, tensor))
            current_size += tensor_size

    if current_bucket:
        yield current_bucket


ONE_GB = 1024 * 1024 * 1024
MAX_UPDATE_WEIGHTS_SIZE = 8 * ONE_GB  # 1GB


PostprocessTensorFunc = Callable[[str, torch.Tensor], tuple[tuple[str, torch.Tensor], ...]]
PreprocessStateDictFunc = Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]


class UpdateWeightFromTensor:
    def __init__(
        self,
        args,
        model,
        postprocess_tensor_func: Optional[PostprocessTensorFunc] = None,
        preprocess_state_dict_func: Optional[PreprocessStateDictFunc] = None,
    ):
        self.args = args
        self.model = model
        self.postprocess_tensor_func = postprocess_tensor_func
        self.preprocess_state_dict_func = preprocess_state_dict_func

    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        self.rollout_engines = rollout_engines

        # Here we assume the gpu id of rollout engines and train actors are the same.
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(
                ranks=group_ranks,
                backend="gloo",
            )
            if dist.get_rank() in group_ranks:
                self._ipc_gather_src = start_rank
                self._ipc_gather_group = new_group
                self._ipc_engine = engine

    @torch.no_grad()
    def update_weights(self):
        monkey_patch_torch_reductions()
        sharded_state_dict = cast(
            dict[str, torch.Tensor],
            get_model_state_dict(self.model, options=StateDictOptions(full_state_dict=False, cpu_offload=False)),
        )
        if self.preprocess_state_dict_func is not None:
            sharded_state_dict = self.preprocess_state_dict_func(sharded_state_dict)
        # Zero copy here
        named_tensors = [(name, param) for name, param in sharded_state_dict.items()]
        # print_memory(f"before weight update")
        for i, params_batch in enumerate(
            get_named_tensor_buckets(named_tensors, MAX_UPDATE_WEIGHTS_SIZE, as_dtype=torch.bfloat16)
        ):
            print(f"Update weights from tensor {i}")
            # Detach and convert to the same dtype
            params_batch = [(name, param.detach().to(dtype=torch.bfloat16)) for name, param in params_batch]
            # Then gather the tensor
            gathered_named_tensors = [
                (name, preprocess_tensor_for_update_weights(param)) for name, param in params_batch
            ]

            if self.postprocess_tensor_func is not None:
                gathered_named_tensors = [
                    k for name, param in gathered_named_tensors for k in self.postprocess_tensor_func(name, param)
                ]

            serialized_tensors = MultiprocessingSerializer.serialize(gathered_named_tensors, output_str=True)

            serialized_named_tensors = (
                [None] * dist.get_world_size(self._ipc_gather_group)
                if self._ipc_gather_src == dist.get_rank()
                else None
            )
            dist.gather_object(
                serialized_tensors,
                object_gather_list=serialized_named_tensors,
                dst=self._ipc_gather_src,
                group=self._ipc_gather_group,
            )
            if dist.get_rank() == self._ipc_gather_src:
                kwargs = {
                    "serialized_named_tensors": serialized_named_tensors,
                }

                ref = self._ipc_engine.update_weights_from_tensor.remote(**kwargs)
                ray.get(ref)
            dist.barrier(group=self._ipc_gather_group)
            del (
                serialized_named_tensors,
                serialized_tensors,
                params_batch,
                gathered_named_tensors,
                # flattened_tensor_bucket,
                # flattened_tensor_data,
            )
            # Expected peak memory usage is (memory of model in fp32 / shard + MAX_UPDATE_WEIGHTS_SIZE)
        # print_memory(f"after update")
