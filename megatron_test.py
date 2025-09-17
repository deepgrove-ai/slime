from slime.backends.megatron_utils.model import forward_only, initialize_model_and_optimizer
from slime.backends.megatron_utils.initialize import init, is_megatron_main_rank
from slime.utils.arguments import parse_args
import torch


def load_data():
    # Load a simple sample from file
    sample = torch.load("veomni_first_sample.pt")
    # {'sample': [8192], 'response_lengths': 8192}
    sample["tokens"] = sample["tokens"].reshape(1, -1)
    sample["total_lengths"] = torch.tensor([sample["tokens"].shape[1]])  # [[8192]]
    sample["response_lengths"] = torch.tensor([[sample["response_lengths"]]])  # [[8192]]

    # Wrap the sample in a simple iterator with a .reset() and .get_next(keys) method
    class SimpleDataIterator:
        def __init__(self, sample):
            self.sample = sample
            self.used = False

        def reset(self):
            self.used = False

        def get_next(self, keys):
            # Return only the requested keys
            # INSERT_YOUR_CODE
            # Print a dict with the same keys but values as the shape of the tensor
            shapes = {k: v.shape if hasattr(v, "shape") else type(v) for k, v in self.sample.items() if k in keys}
            print("Sample shapes:", shapes)
            return {k: self.sample[k] for k in keys if k in self.sample}

    data_iterator = [SimpleDataIterator(sample)]
    num_microbatches = [1]
    return data_iterator, num_microbatches


def main():
    # INSERT_YOUR_CODE
    import torch
    import torch.distributed as dist
    import os

    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=0,
            world_size=1,
        )
    args = parse_args()
    init(args)
    (model, optimizer, opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(args)
    print("initialized model and optimizer")
    data_iterator, num_microbatches = load_data()
    log_probs = forward_only(
        args,
        model,
        data_iterator,
        num_microbatches,
        store_prefix="",
    )
    # save log_probs to file
    torch.save(log_probs, "megatron_log_probs.pt")
    print(log_probs)


if __name__ == "__main__":
    main()
