import os
import json
from dataclasses import dataclass, field, asdict
from transformers.training_args import TrainingArguments
from typing import Optional, List, Tuple, Union, Dict

@dataclass
class ModelArgs:
    model_cache_dir: str = field(
        default='',
        metadata={'help': 'Default path to save language models.'}
    )
    dataset_cache_dir: str = field(
        default='',
        metadata={'help': 'Default path to save huggingface datasets.'}
    )
    data_root: str = field(
        default="", 
        metadata={'help': 'The base directory storing all data used for training and evaluation. If specified, make sure all train_data, eval_data, and corpus are path relative to data_root!'},
    )
    train_data: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'Training json file or glob to match a list of files.'},
    )
    eval_data: Optional[str] = field(
        default=None,
        metadata={'help': 'Evaluation json file.'},
    )
    index_path: Optional[str] = field(
        default="",
        metadata={'help': 'Evaluation json file.'},
    )
    model_name_or_path: str = field(
        default='',
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    padding_side: str = field(
        default="left",
        metadata={'help': 'Tokenizer padding side.'}
    )
    access_token: Optional[str] = field(
        default="",
        metadata={'help': 'Huggingface access token.'}
    )
    attn_impl: str = field(
        default="flash_attention_2",
        metadata={'help': 'The implementation of attention.'}
    )
    max_length: int = field(
        default=4096,
        metadata={'help': 'How many tokens at maximum for each input.'},
    )
    chat_template: str = field(
        default="",
        metadata={'help': 'Instruction template name in fastchat.'}
    )
    lora: Optional[str] = field(
        default=None,
        metadata={'help': 'LoRA ID.'},
    )
    lora_unload: bool = field(
        default=True,
        metadata={'help': 'Merge and unload LoRA?'},
    )

    dtype: str = field(
        default="bf16",
        metadata={'help': 'Data type for embeddings.'}
    )
    device_map: Optional[str] = field(
        default=None,
        metadata={'help': 'Device map for loading the model. Set to auto to load across devices.'}
    )
    batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation batch size.'},
    )
    cpu: bool = field(
        default=False,
        metadata={'help': 'Use cpu?'}
    )
    cache_implementation: str = field(
        default=None,
        metadata={'help': 'use cache?'}
    )

    cache_backend: str = field(
        default=None,
        metadata={'help': 'cache backend'}
    )

    cache_nbits: int = field(
        default=None,
        metadata={'help': 'quant size'}
    )

    load_in_4bit: bool = field(
        default=False,
        metadata={'help': 'quant size'}
    )

    enable_tp: bool = field(
        default=False,
        metadata={'help': 'Use tensor parallel to wrap the model?'}
    )

    l1_l2_ratio: List[int] = field(
        default_factory=lambda: [8],
        metadata={'help': 'Compression ratios for ultragist memory.'}
    )

    ret_model: str = field(
        default="3",
        metadata={'help': 'Model name or path for retrieval.'}
    )
    ret_bm25_k1: float = field(
        default=0.9,
        metadata={'help': 'BM25 k1.'}
    )
    acre_window: Optional[int] = field(
        default=None,
        metadata={'help': 'The initial sliding window size.'}
    )
    acre_stride: Optional[int] = field(
        default=None,
        metadata={'help': 'The stride of the sliding window.'}
    )
    acre_attn: Optional[str] = field(
        default=None,
        metadata={'help': 'How to assign attention masks of acre tokens? {segmentation, step-expansion, full-converage}'}
    )
    acre_ratio_mix: Optional[str] = field(
        default=None,
        metadata={'help': 'How to determine the acre_ratio for each input. {step-random, instance-random, adapt-x}'}
    )
    acre_param: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'The introduced parameters for acre. {q, k, v, o}'}
    )
    acre_embed_init: str = field(
        default="eos",
        metadata={'help': 'Initialize acre embedding from eos/bos embedding.'}
    )
    acre_sink_size: Optional[int] = field(
        default=None,
        metadata={'help': 'The number of activations that are always kept in the head of the sequence according to StreamingLLM.'}
    )
    acre_attend_prev: Optional[bool] = field(
        default=None,
        metadata={'help': 'Can acre tokens attend to previous acre tokens?'}
    )
    acre_pos: Optional[str] = field(
        default=None,
        metadata={'help': 'Where to put acre tokens? {append, interleave}'}
    )
    gen_model: str = field(
        default="",
        metadata={'help': 'Model name or path for generation.'}
    )
    gen_max_new_tokens: Optional[int] = field(
        default=512,
        metadata={'help': 'How many tokens at maximum to return?'},
    )
    gen_do_sample: Optional[bool] = field(
        default=False,
        metadata={'help': 'Do sampling when decoding?'},
    )
    gen_temperature: Optional[float] = field(
        default=None,
        metadata={'help': 'Sampling temperature.'},
    )
    gen_top_p: Optional[float] = field(
        default=None,
        metadata={'help': "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation."}
    )
    note: str = field(
        default="",
        metadata={'help': 'experiment note'}
    )
    conv: bool = field(
        default=False,
        metadata={'help': 'Merge and unload LoRA?'},
    )
    kv_strategy: str = field(
        default="",
        metadata={'help': 'Merge and unload LoRA?'},
    )
    acre_strategy: str = field(
        default="layer_same",
        metadata={'help': 'Number of retrieval candidates.'}
    )

    acre_max_window: int = field(
        default=16384,
        metadata={'help': 'Number of retrieval candidates.'}
    )

    def resolve_path(self, path):
        """Resolve any path starting with 'long-llm:' to relative path against data_root."""
        pattern = "long-llm:"
        # resolve relative data paths when necessary
        if isinstance(path, list):
            for i, x in enumerate(path):
                if x.startswith(pattern):
                    path[i] = os.path.join(self.data_root, x.replace(pattern, ""))
        else:
            if path.startswith(pattern):
                path = os.path.join(self.data_root, path.replace(pattern, ""))

        return path

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    def __post_init__(self):
        if self.train_data is not None:
            self.train_data = self.resolve_path(self.train_data)

        if self.eval_data is not None:
            self.eval_data = self.resolve_path(self.eval_data)

        if hasattr(self, "output_dir") and self.output_dir is not None:
            self.output_dir = self.resolve_path(self.output_dir)

        if hasattr(self, "result_dir"):
            if self.result_dir is None: 
                self.result_dir = "tmp"