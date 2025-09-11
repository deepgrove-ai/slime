from dataclasses import dataclass, field
from typing import Literal, Optional, List


@dataclass
class DataArguments:
    train_path: str = field(
        default=None,
        metadata={"help": "Path of the training data. Use comma to separate multiple datasets."},
    )
    train_size: int = field(
        default=10_000_000,
        metadata={"help": "Number of tokens for training to compute training steps for dynamic batch dataloader."},
    )
    data_type: Literal["plaintext", "conversation", "diffusion"] = field(
        default="conversation",
        metadata={"help": "Type of the training data."},
    )
    dataloader_type: Literal["native"] = field(
        default="native",
        metadata={"help": "Type of the dataloader."},
    )
    datasets_type: Literal["mapping", "iterable"] = field(
        default="mapping",
        metadata={"help": "Type of the datasets."},
    )
    data_name: str = field(
        default=None,
        metadata={"help": "Dataset name for multimodal training."},
    )
    global_batch_size: int = field(
        default=None,
        metadata={"help": "Global batch size. If None, use `micro_batch_size` * `data_parallel_size`."},
    )
    micro_batch_size: int = field(
        default=1,
        metadata={"help": "Micro batch size. The number of samples per iteration on each device."},
    )
    data_tag: Literal["default", "mmtag"] = field(
        default="default",
        metadata={"help": "Dataset tag for multimodal training."},
    )
    text_keys: str = field(
        default=None,
        metadata={"help": "Key to get text from the training data."},
    )
    image_keys: str = field(
        default="images",
        metadata={"help": "Key to get images from the training data."},
    )
    chat_template: str = field(
        default="default",
        metadata={"help": "Chat template to use."},
    )
    max_seq_len: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length in training."},
    )
    num_workers: int = field(
        default=2,
        metadata={"help": "Number of workers to load data."},
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "Number of batches loaded in advance by each worker."},
    )
    drop_last: bool = field(
        default=True,
        metadata={"help": "Whether to drop the last incomplete batch."},
    )


@dataclass
class RolloutArguments:
    hf_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "The huggingface checkpoint of the trained model. This is used to initialize sglang and also provide the tokenizer. Note that, we will always update the parameters in sglang with that of megatron before training, so you only need to provide a huggingface checkpoint that has the same architecture as the model you want to train. It doesn't necessary need to contain the most up-to-date parameters."
        },
    )
    use_hf_config_for_megatron: bool = field(
        default=False,
        metadata={"help": "Whether to use HF config for Megatron core to define the model architecture."},
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the model, this is used to convert the megatron weights into huggingface format. If not set, we will use `type(AutoConfig.from_pretrained(args.hf_checkpoint)).__name__.lower()` as model_name. Also, sometimes this will help alleviate the bug that transformers cannot find certain model."
        },
    )
    rollout_function_path: str = field(
        default="slime.rollout.sglang_rollout.generate_rollout",
        metadata={
            "help": "Path to the rollout generation function. You should use this model to create your own custom rollout function, and then set this to the path of your custom rollout function. The signature of the function should be `def generate_rollout(args, rollout_id, *, evaluation=False) -> list[list[Sample]]` and within the output sample, you should at least set `tokens`, `response_length`, `reward` and `truncated`."
        },
    )
    rollout_temperature: float = field(
        default=1.0, metadata={"help": "the temperature for the inference engine during rollout."}
    )
    rollout_top_p: float = field(default=1.0, metadata={"help": "the top-p for the inference engine during rollout."})
    rollout_top_k: int = field(default=-1, metadata={"help": "the top-k for the inference engine during rollout."})
    rollout_max_context_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum context size for the inference engine during rollout. It should no exceed the `max_position_embeddinds` in Huggingface model's `config.json`"
        },
    )
    rollout_max_prompt_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum length of the prompt for the inference engine during rollout. If set, we will filter out the long prompts during initialization of the global dataset. This is not recommended if the dataset is large."
        },
    )
    rollout_max_response_len: int = field(
        default=1024,
        metadata={
            "help": "The maximum length of the response for the inference engine during rollout. It is basically `max_tokens` in sglang."
        },
    )
    rollout_skip_special_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to skip special tokens in the response during rollout. This is useful when you want to use the response as a prompt for the next rollout."
        },
    )
    rollout_stop: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "The stop words for the inference engine during rollout. It can be a list of strings or a single string. It may be hard to pass special tokens in command line, in that case rollout_stop_token_ids can be used."
        },
    )
    rollout_stop_token_ids: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "The stop token ids for the inference engine during rollout. It can be a list of integers or a single integer."
        },
    )
    rollout_shuffle: bool = field(default=False, metadata={"help": "Whether to shuffle the prompts during rollout."})
    rollout_seed: int = field(
        default=42,
        metadata={
            "help": "The seed for the random number generator during rollout. This is used to shuffle the prompts and also for the random sampling of the prompts."
        },
    )
    over_sampling_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "This defines the granularity of the sampling batch in the rollout function. When the number of available samples falls below the target, a sampling operation of size over_sampling_batch_size will be triggered. Regardless of whether partial rollout is used or filters are applied, the sampling granularity is always determined by this value. If this value is None, rollout_batch_size will be used as the default over_sampling_batch_size."
        },
    )
    dynamic_sampling_filter_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "This is the filter function for dynamic sampling. It should be able to judge whether the result of a prompt should be selected or not. We will do dynamic filter for sampling as in DAPO. e.g. not all correct or all wrong samples. You could use `slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` as an example."
        },
    )
    partial_rollout: bool = field(
        default=False,
        metadata={
            "help": "Whether to use partial rollout. If set, the unfinished samples during dynamic sampling will be recycled back to data buffer. This is useful for long responses."
        },
    )
    custom_generate_function_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Only substitue the `def generate(args, sample, sampling_params)` function within the example rollout function. This should be useful if you need to implement some special rollout logic, e.g. multi-turn, function calling."
        },
    )
    buffer_filter_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the buffer filter function. It should be able to select the samples in the buffer. The function should take list[list[Sample]] and return list[list[Sample]]."
        },
    )
    update_weight_buffer_size: int = field(
        default=512 * 1024**2,
        metadata={
            "help": "buffer size for update weight, in bytes. This is used for updating weights by chunk and should be useful for MoE models."
        },
    )
    update_weights_interval: int = field(default=1, metadata={"help": "Interval for updating the weights"})
    keep_old_actor: bool = field(
        default=False, metadata={"help": "Whether to keep the rollout model on training process"}
    )
    rollout_data_postprocess_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The called after we have all the rollout data including log_probs. It may be helpful for updating loss mask."
        },
    )


@dataclass
class ClusterArguments:
    actor_num_nodes: int = field(default=1, metadata={"help": "Number of nodes for training actor"})
    actor_num_gpus_per_node: int = field(default=8, metadata={"help": "Number of gpus per node for training actor"})
    rollout_num_gpus: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of GPUs for inference. Note that when using --colocate, i.e. the training and the inference engines are on the same gpus, this param will be ignored and will be set as actor_num_gpus_per_node * actor_num_nodes."
        },
    )
    rollout_num_gpus_per_engine: int = field(
        default=1, metadata={"help": "Number of GPUs per inference engine, just like the tp_size in sglang."}
    )
    num_gpus_per_node: int = field(default=8, metadata={"help": "Number of gpus per node for rollout."})


@dataclass
class AlgoArguments:
    ref_load: Optional[str] = field(
        default=None,
        metadata={
            "help": "The checkpoint for reference model. When --load is not set, this will be used as the initial checkpoint for training."
        },
    )
    ref_ckpt_step: Optional[int] = field(default=None, metadata={"help": "The checkpoint step for reference model."})
    load: Optional[str] = field(default=None, metadata={"help": "The checkpoint for training."})
    save: Optional[str] = field(default=None, metadata={"help": "The checkpoint for saving."})
    save_interval: Optional[int] = field(default=None, metadata={"help": "The interval for saving."})
    seed: int = field(default=1234, metadata={"help": "The seed for training."})
    clip_grad: float = field(default=1.0, metadata={"help": "The clip grad for training."})
    calculate_per_token_loss: bool = field(
        default=False, metadata={"help": "Whether to calculate the per token loss."}
    )
    eps_clip: float = field(default=0.2, metadata={"help": "The eps clip for training."})
    eps_clip_high: Optional[float] = field(default=None, metadata={"help": "The eps clip high for training."})
    eps_clip_c: Optional[float] = field(default=None, metadata={"help": "The eps clip c for training."})
    kl_coef: float = field(default=0.00, metadata={"help": "The kl coef for training."})
    kl_loss_type: str = field(default="kl", metadata={"help": "The kl loss type for training."})
    advantage_estimator: str = field(default="grpo", metadata={"help": "The advantage estimator for training."})
    disable_compute_advantages_and_returns: bool = field(
        default=False, metadata={"help": "Whether to disable computing advantages and returns."}
    )
    use_kl_loss: bool = field(default=False, metadata={"help": "Whether to use kl loss."})
    kl_loss_coef: float = field(default=0.0, metadata={"help": "The kl loss coef for training."})
    entropy_coef: float = field(default=0.0, metadata={"help": "The entropy coef for training."})
    gamma: float = field(default=1.0, metadata={"help": "The gamma for training."})
    normalize_advantages: bool = field(default=False, metadata={"help": "Whether to normalize advantages."})
    disable_grpo_std_normalization: bool = field(
        default=False, metadata={"help": "Whether to disable grpo std normalization."}
    )
    disable_rewards_normalization: bool = field(
        default=False, metadata={"help": "Whether to disable rewards normalization."}
    )
    use_rollout_entropy: bool = field(default=False, metadata={"help": "Whether to use rollout entropy."})
    use_tis: bool = field(default=False, metadata={"help": "Whether to use tis."})
    tis_clip: float = field(default=2.0, metadata={"help": "The tis clip for training."})
    tis_clip_low: float = field(default=0, metadata={"help": "The tis clip low for training."})
    load: Optional[str] = field(default=None, metadata={"help": "The checkpoint for training."})
    save: Optional[str] = field(default=None, metadata={"help": "The checkpoint for saving."})
    save_interval: Optional[int] = field(default=None, metadata={"help": "The interval for saving."})
    seed: int = field(default=1234, metadata={"help": "The seed for training."})
    clip_grad: float = field(default=1.0, metadata={"help": "The clip grad for training."})
    calculate_per_token_loss: bool = field(
        default=False, metadata={"help": "Whether to calculate the per token loss."}
    )


@dataclass
class WandbArguments:
    use_wandb: bool = field(default=False, metadata={"help": "Whether to use wandb."})
    wandb_mode: str = field(
        default=None,
        metadata={
            "help": "Wandb mode: online (default), offline (local only), or disabled. Overrides WANDB_MODE env var."
        },
    )
    wandb_project: str = field(default=None, metadata={"help": "Wandb project."})
    wandb_dir: str = field(
        default=None, metadata={"help": "Directory to store wandb logs. Default is ./wandb in current directory."}
    )
    wandb_key: str = field(default=None, metadata={"help": "Wandb key."})
    wandb_host: str = field(default=None, metadata={"help": "Wandb host."})
    wandb_team: str = field(default=None, metadata={"help": "Wandb team."})
    wandb_group: str = field(default=None, metadata={"help": "Wandb group."})
    wandb_project: str = field(default=None, metadata={"help": "Wandb project."})
    wandb_random_suffix: bool = field(
        default=True, metadata={"help": "Whether to add a random suffix to the wandb run name."}
    )
    wandb_always_use_train_step: bool = field(
        default=False, metadata={"help": "Whether to always use train step as the step metric in wandb."}
    )
    log_multi_turn: bool = field(
        default=False, metadata={"help": "Whether to log information for multi-turn rollout."}
    )
    log_passrate: bool = field(
        default=False,
        metadata={
            "help": "Whether to turn on passrate logging, which will log the pass@n of the responses in the rollout."
        },
    )
    wandb_run_id: str = field(default=None, metadata={"help": "Wandb run id."})


@dataclass
class DebugArguments:
    save_debug_rollout_data: Optional[str] = field(
        default=None,
        metadata={
            "help": "Save the rollout data to this path for debugging. The file will be saved to `save_debug_rollout_data.format(rollout_id)`."
        },
    )
    load_debug_rollout_data: Optional[str] = field(
        default=None,
        metadata={
            "help": "Load the rollout data from this path for debugging. The file will be loaded from `load_debug_rollout_data.format(rollout_id)`. When this is enabled, slime will not instantiate sglang servers."
        },
    )
    debug_rollout_only: bool = field(
        default=False, metadata={"help": "Whether to only run the rollout generation without training."}
    )
    debug_train_only: bool = field(
        default=False, metadata={"help": "Whether to only run the training without sglang servers."}
    )
    save_debug_train_data: Optional[str] = field(
        default=None,
        metadata={
            "help": "Save the train data to this path for debugging. The file will be saved to `save_debug_train_data.format(rollout_id)`."
        },
    )
    dump_details: Optional[str] = field(
        default=None, metadata={"help": "Dump all details of training for post-hoc analysis and visualization."}
    )


@dataclass
class FullArgs(DataArguments, RolloutArguments, ClusterArguments, AlgoArguments, WandbArguments, DebugArguments):
    pass
