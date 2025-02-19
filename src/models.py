from typing import Dict, Union, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, SinkCache
from transformers.utils import logging
import torch
from transformers.integrations import is_deepspeed_zero3_enabled

logger = logging.get_logger(__name__)

class HuggingFaceModel:
    def __init__(self, model_name_or_path, model_kwargs:Dict={}, tokenizer_kwargs:Dict={}):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            **model_kwargs,
        ).eval()

        self.model_name_or_path = model_name_or_path
        # use eos as pad
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Model config: {self.model.config}")

    def template2ids(self, template, remove_symbol=None):
        to_encode = self.tokenizer.apply_chat_template(
                template, 
                tokenize=False, 
                add_generation_prompt=True)

        if remove_symbol:
            to_encode = to_encode.replace(remove_symbol, "")

        inputs = self.tokenizer(
                to_encode, add_special_tokens=False, return_tensors="pt", padding=True
                ).to(self.model.device)

        return inputs

    def generate_conv(self, query, context, prompt, instruct:Union[str,list], **generation_kwargs):
        if isinstance(instruct, str):
            instruct = [instruct]
        context = [
            {"role": "user", "content": prompt.format(context=context)},
            {"role": "assistant", "content": "I have read the article. Please provide your question."}]
        inputs = self.template2ids(context)
        self.model(**inputs)
        mem_state = self.model.memory.export()

        outputs = []
        if query:
            for i,inst in enumerate(instruct):
                if i > 0:
                    self.model.memory.reset(**mem_state)
                sample = [
                        {"role": "user", "content": inst.format(question=query)}]
                inputs = self.template2ids(sample)
                res = self.ids2text(inputs, **generation_kwargs)
                outputs.append(res)
        else:
            sample = [
                    {"role": "user", "content": instruct[0]}]
            inputs = self.template2ids(sample)
            res = self.ids2text(inputs, **generation_kwargs)
            outputs.append(res)
        return outputs

    def ids2text(self, inputs, **generation_kwargs):
        outputs = self.model.generate(
            **inputs,
            **generation_kwargs,
            pad_token_id=self.tokenizer.eos_token_id)
        outputs = outputs[:, inputs["input_ids"].shape[1]:]
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return outputs

    def generate(self, prompts:Union[str, List[str]], batch_size:int=4, past_key_values=None, **generation_kwargs):
        all_outputs = []

        if isinstance(prompts, str):
            squeeze = True
            prompts = [prompts]
        else:
            squeeze = False

        for i in range(0, len(prompts), batch_size):
            batch_prompts = []
            for prompt in prompts[i: i + batch_size]:
                prompt = self.tokenizer.apply_chat_template([{"role":"user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                batch_prompts.append(prompt)
            
            inputs = self.tokenizer(batch_prompts, add_special_tokens=False, return_tensors="pt", padding=True).to(self.model.device)
            
            if past_key_values:
                generation_kwargs["past_key_values"] = past_key_values
            outputs = self.model.generate(
                **inputs, 
                **generation_kwargs,
                pad_token_id=self.tokenizer.eos_token_id)
            outputs = outputs[:, inputs["input_ids"].shape[1]:]

            # decode to string
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            all_outputs.extend(outputs)

        if squeeze:
            all_outputs = all_outputs[0]

        return all_outputs


def init_args(model_args, model_name, device):
    model_args_dict = model_args.to_dict()
    dtype = model_args_dict["dtype"]
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    if model_args_dict["device_map"] is None and not is_deepspeed_zero3_enabled():
        device_map = {"": device}

    model_kwargs = {
        "cache_dir": model_args_dict["model_cache_dir"],
        "token": model_args_dict["access_token"],
        "device_map": model_args_dict["device_map"],
        "attn_implementation": model_args_dict["attn_impl"],
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }

    tokenizer_kwargs = {
        "cache_dir": model_args_dict["model_cache_dir"],
        "token": model_args_dict["access_token"],
        "padding_side": model_args_dict["padding_side"],
        "trust_remote_code": True,
    }
    return model_kwargs, tokenizer_kwargs

