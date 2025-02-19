import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math

from src.models import init_args, HuggingFaceModel
from src.utils import *
from src.prompts import prompts
from typing import Dict, Union, List, Optional
from itertools import chain
from semantic_text_splitter import TextSplitter
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def get_pipeline(model_args, device="cpu", **kwargs):
    model_kwargs, tokenizer_kwargs = init_args(
        model_args, 
        model_args.gen_model, device)
    
    model_args_dict = model_args.to_dict()

    if model_args_dict["gen_model"].find("acre") != -1:
        model_kwargs["l1_l2_ratio"] = model_args_dict["l1_l2_ratio"]
        model_kwargs["max_window"] = model_args_dict["max_window"]
        model_kwargs["max_refilling"] = model_args_dict["max_refilling"]

    pipeline_name = model_args_dict["pipeline"]
    index_path = model_args_dict["index_path"]

    ### initialize generation model 
    
    gen_model_name = model_args_dict["gen_model"]
        
    print(model_kwargs)
        
    gen_model = HuggingFaceModel(
            gen_model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs
        )

    generation_kwargs = {}
    if model_args_dict["gen_max_new_tokens"]:
        generation_kwargs["max_new_tokens"] = model_args_dict["gen_max_new_tokens"]
    if model_args_dict["gen_do_sample"]:
        generation_kwargs["do_sample"] = model_args_dict["gen_do_sample"]
    if model_args_dict["gen_temperature"]:
        generation_kwargs["temperature"] = model_args_dict["gen_temperature"]
    if model_args_dict["gen_top_p"]:
        generation_kwargs["top_p"] = model_args_dict["gen_top_p"]


    pipeline = ACREPipeline(
            generator=gen_model,
            generation_kwargs=generation_kwargs,
            max_window=model_args_dict["max_window"],
            l1_l2_ratio=model_kwargs["l1_l2_ratio"],
            max_refilling=model_args_dict["max_refilling"],
            strategy=model_args_dict["mega_strategy"],
    )

    return pipeline

class ACREPipeline:
    def __init__(
        self, 
        generator: Union[HuggingFaceModel], 
        generation_kwargs: Dict={}, 
        max_window: int=16384,
        l1_l2_ratio: int=16,
        max_refilling=4096,
        strategy="layer_same"):

        self.strategy = strategy
        self.generator = generator
        self.generation_kwargs = generation_kwargs
        self.repeat_num = self.generator.model.config.num_attention_heads // self.generator.model.config.num_key_value_heads
        self.max_refilling = max_refilling
        self.l1_l2_ratio = l1_l2_ratio
        self.topk = max_refilling // l1_l2_ratio
        self.max_window = max_window
        print(f"using max window: {max_window}")
        print(f"using topk: {self.topk}")
        print(f"using strategy: {strategy}")
        self.reset()

    def reset(self):
        # internal attributes that facilitates inspection
        if self.generator.model_name_or_path.find("acre") != -1:
            self.generator.model.memory.reset()

    def __call__(self, context:str, question:str, prompt:str="", task="", cache_id="", conv=False):
        self.reset()
        # build bi-layer kv cache
        self.memorize(context)
        # query-guided refilling
        if task not in prompts:
            _prompt_q = prompts["general_qa"].format(input=question)
        else:
            _prompt_q = prompts[task].format(input=question)

        question_inputs = self.generator.template2ids([[{"role": "user", "content": _prompt_q}]])
        if self.generator.model.memory.l1_activations[-1][0] is not None:
            topk = self.get_topk(question_inputs)
            highlight_raw_activations = self.get_highlight_kv(topk)
        else:
            highlight_raw_activations = None

        if self.generator.model_name_or_path.find("acre") != -1:
            self.generator.model.memory.reset(**self.mem_state)
        if highlight_raw_activations is not None:

            nested_cache_activation = [(None, None) for _ in range(self.generator.model.config.num_hidden_layers)]
            for layer_idx in range(self.generator.model.config.num_hidden_layers):
                layer_l1_key_activation = self.generator.model.memory.l1_activations[layer_idx][0]
                layer_l1_value_activation = self.generator.model.memory.l1_activations[layer_idx][1]
                layer_highlight_key = highlight_raw_activations[layer_idx][0]
                layer_highlight_value = highlight_raw_activations[layer_idx][1]

                nest_cache_key_activation = self.insert_tensor(layer_l1_key_activation, layer_highlight_key)
                nest_cache_value_activation = self.insert_tensor(layer_l1_value_activation, layer_highlight_value)
                nested_cache_activation[layer_idx] = (nest_cache_key_activation, nest_cache_value_activation)
            self.generator.model.memory.l1_activations = nested_cache_activation

        # answer decoding
        answer_output = self.generator.generate(_prompt_q, **self.generation_kwargs)
            
        return answer_output, ""

    def get_topk(self, question_inputs):
        
        self.generator.model(**question_inputs)
        if self.strategy == "layer_same":
            last_query_states = self.generator.model.memory.raw_query_states[-1]
            expanded_key_last_layer = repeat_kv(self.generator.model.memory.l1_activations[-1][0], self.repeat_num)
            topk = self.compute_topk(last_query_states, expanded_key_last_layer)
            
        elif self.strategy == "layer_diff":
            topk = []
            for layer_idx in range(self.generator.model.config.num_hidden_layers):
                query_states_at_layer = self.generator.model.memory.raw_query_states[layer_idx]
                expanded_key_at_layer = repeat_kv(self.generator.model.memory.l1_activations[layer_idx][0], self.repeat_num)
                topk.append(self.compute_topk(query_states_at_layer, expanded_key_at_layer)) 
                if layer_idx > 0:
                    assert len(topk[-1]) == len(topk[-2]) 
        return topk

    def compute_topk(self, query_states, key_states):
        head_dim = query_states.shape[-1]
        qk = torch.einsum("bhqd,bhkd->bhqk", query_states, key_states) // math.sqrt(head_dim)
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32) # bs n_head q_len k_len
        vertical = qk.sum(-2)
        vertical = vertical.sum(-2)
        l1_size = key_states.shape[-2]

        dynamic_topk = (self.max_window - l1_size - 4096) // self.l1_l2_ratio[0]
        topk = min(dynamic_topk, self.topk)

        if vertical.shape[-1] < topk:
            topk = torch.topk(vertical, vertical.shape[-1], dim=-1).indices[0].tolist()
        else:
            topk = torch.topk(vertical, self.topk, dim=-1).indices[0].tolist()
        topk.sort()  
        return topk   

    def get_highlight_kv(self, topk):
        highlight_raw_activations = [(None, None) for _ in range(self.generator.model.config.num_hidden_layers)]
        for layer_idx in range(self.generator.model.config.num_hidden_layers):
            layer_highlight_key = []
            layer_highlight_value = []
            if self.strategy == "layer_same":
                layer_topk = topk
            elif self.strategy == "layer_diff":
                # if layer_idx < self.generator.model.config.num_hidden_layers // 2:
                if layer_idx == 0:
                    layer_topk = topk[-1]
                else:
                    layer_topk = topk[layer_idx]
            
            for i in layer_topk:
                _start, _end, _step = self.generator.model.memory.cache_activation_indices[i]
   
                _highlight_key = self.generator.model.memory.cache_activations[layer_idx][0][_step][:,:,_start:_end,:]
                _highlight_value = self.generator.model.memory.cache_activations[layer_idx][1][_step][:,:,_start:_end,:]
                layer_highlight_key.append((_highlight_key, i))
                layer_highlight_value.append((_highlight_value, i))

            highlight_raw_activations[layer_idx] = (layer_highlight_key, layer_highlight_value)
        return highlight_raw_activations 


    @staticmethod
    def insert_tensor(raw_tensor, new_tensor_list):
        _raw_start_idx = 0
        _raw_end_idx = 0
        expanded_tensor_list = []
        for new_tensor, insert_idx in new_tensor_list:
            if insert_idx == 0:
                expanded_tensor_list.append(new_tensor)
            else:
                _raw_end_idx = insert_idx
                expanded_tensor_list.append(raw_tensor[:,:,_raw_start_idx:_raw_end_idx])
                _raw_start_idx = _raw_end_idx
                expanded_tensor_list.append(new_tensor)
        if _raw_start_idx < raw_tensor.shape[2]:
            expanded_tensor_list.append(raw_tensor[:,:,_raw_start_idx:])
        return torch.cat(expanded_tensor_list, dim=2)                

    def memorize(self, context):
        _prompt = [
            {"role": "user", "content": prompts["memorize"].format(context=context)},
            {"role": "assistant", "content": "I have read the article. Please provide your question."}]
        input_ids = self.generator.template2ids(_prompt)

        self.generator.model(**input_ids)
        self.mem_state = self.generator.model.memory.export()
    