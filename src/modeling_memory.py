import os
import torch
import time
import numpy as np
import torch.distributed as dist
from transformers.utils import logging
from transformers import AutoTokenizer
from itertools import cycle
from typing import List
from copy import deepcopy


logger = logging.get_logger(__name__)

class Memory(torch.nn.Module):
    def __init__(
        self,
        model_config,
        k_seq_dim: int,
        v_seq_dim: int
    ):
        super().__init__()
        self.config = model_config

        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.rng = np.random.default_rng(42)

        self.acre_token =  torch.tensor([self.config.vocab_size])
        self.reset()

    def reset(self, **kwargs):
        """Initialize attributes for a new sequence."""
        # the cursor pointing to the start of the current window
        self._start_idx = 0
        # the cursor pointing to the end of the current window
        self._end_idx = 0
        # the acre sizes of all strides
        self._all_acre_sizes = []
        # the loss per batch
        self._batch_loss = None
        # the valid token number per batch
        self._valid_token_num = None
        # the step index for processing the input_ids
        self._step_idx = 0
        # used in set_compression_ratio
        self._compression_ratio = None
        # the previous inputs is a full window or not, defaults to True
        self._is_full_window = True
        # the number of raw activations to preserve in update_memory (only useful when acre_stride < acre_window)
        self._raw_size_to_cache = 0

        # the number of tokens in previous stride that should be compressed by the upcoming acre
        self._interleave_remainder = 0
        # compression ratio for the unfinished window
        self._interleave_compression_ratio = None
        self._acre_indices = None

        self.all_input_ids = None
        self.all_attention_mask = None
        self.all_labels = None

        # NOTE: will be reset in prepare()
        self.acre_skip_first = None
        self.acre_skip_last = None

        # the raw activations of recent tokens
        self.raw_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        # the attention sink activations
        self.sink_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        # the acre activations
        self.acre_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]
        self.highlight_raw_activations = [(None, None) for _ in range(self.config.num_hidden_layers)]

        self.raw_query_states = [None for _ in range(self.config.num_hidden_layers)]

        self.cache_activations = [(dict(), dict()) for _ in range(self.config.num_hidden_layers)]
        self.cache_activation_indices = dict()
        self.cache_idx = 0 
        # NOTE: in case we want to resume the memory from a particular state

        for k, v in kwargs.items():
            # NOTE: deepcopy to untie the memory state and the model memory
            setattr(self, deepcopy(k), deepcopy(v))

    def prepare(self, input_ids, attention_mask, labels, skip_first=None, skip_last=None):
        """
        Prepare inputs for the model. These inputs belong to the same sequence.
        """
        # assert input_ids.shape[0] == 1, "Make sure the batch size is 1!"
        # assert attention_mask is None or (attention_mask == 1).all(), "Make sure there is no padding!"

        self._device = input_ids.device

        # accumulate input_ids
        if self.all_input_ids is None:
            self.all_input_ids = input_ids.cpu()
        else:
            self.all_input_ids = torch.cat([self.all_input_ids, input_ids.cpu()], dim=1)

        # accumulate attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=torch.device("cpu"))
        if self.all_attention_mask is None:
            self.all_attention_mask = attention_mask.cpu()
        else:
            self.all_attention_mask = torch.cat([self.all_attention_mask, attention_mask.cpu()], dim=1)

        # accumulate labels if exisits
        if labels is not None:
            # rotate labels in advance so that the loss of the last token is not ignored in every window
            labels = torch.cat([labels[:, 1:].cpu(), torch.tensor([-100]).expand(labels.shape[0], 1)], dim=1)
            if self.all_labels is None:
                self.all_labels = labels.cpu()
            else:
                self.all_labels = torch.cat([self.all_labels, labels], dim=1)
            assert self.all_input_ids.shape[1] == self.all_labels.shape[1], f"Found inconsistent all_input_ids {self.all_input_ids.shape} and all_labels {self.all_labels.shape}!"
        
        # how many tokens to skip at the beginning of the sequence? (They will be packed in a single chunk and processed by the model, after which their activations will be cached in sink_activations.)
        if skip_first is not None:
            assert self.config.acre_parallel_window == 1, f"Make sure the parallel window is set to 1 when using acre_skip!"
            assert self.config.acre_window == self.config.acre_stride, f"Make sure the acre_window equals to acre_stride when using acre_skip."
            assert self.config.acre_sink_size == 0, f"Make sure the acre_sink_size is set to 0 when using acre_skip!"
        # stop compression after how many tokens
        if skip_last is not None:
            skip_first = skip_first if skip_first is not None else 0
            assert (skip_last - skip_first) % self.config.acre_window == 0, f"skip_last ({skip_last}) - skip_first ({skip_first}) = {skip_last - skip_first} is not divisible by window size {self.config.acre_window}"
            assert self.config.acre_sink_size == 0, "Make sure the acre_sink_size is zero when using skip_last!"
        self.acre_skip_first = skip_first
        self.acre_skip_last = skip_last

    def step(self):
        start_idx = self._start_idx
        end_idx = start_idx + self.config.acre_window

        if end_idx > self.all_sequence_length:
            end_idx = self.all_sequence_length
            is_full_window = False
        else:
            is_full_window = True

        if self.training and end_idx == self.all_sequence_length:
            next_start_idx = start_idx
            raw_size_to_cache = -1
            acre_size = 0
            compression_ratio = -1

        elif self._step_idx == 0 and self.acre_skip_first is not None:
            end_idx = start_idx + self.acre_skip_first
            assert end_idx < self.all_sequence_length
            next_start_idx = end_idx
            is_full_window = True
            raw_size_to_cache = -1
            acre_size = 0
            compression_ratio = -1
        
        elif self.acre_skip_last is not None and start_idx >= self.acre_skip_last:
            end_idx = min(start_idx + self.config.acre_window, self.all_sequence_length)
            next_start_idx = end_idx
            is_full_window = False
            raw_size_to_cache = -1
            acre_size = 0
            compression_ratio = -1
        else:
            # set compression ratio
            if self.config.acre_pos == "interleave":
                current_window_size = end_idx - self._end_idx
                if self._is_full_window:
                    compression_ratio = self.set_compression_ratio(start_idx=start_idx, end_idx=end_idx)
                    self._interleave_compression_ratio = compression_ratio
                else:
                    compression_ratio = self._interleave_compression_ratio

                if compression_ratio > 0:
                    acre_size = (current_window_size + self._interleave_remainder) // compression_ratio
                else:
                    acre_size = -1

                if is_full_window:
                    next_start_idx = start_idx + self.config.acre_stride
                    raw_size_to_cache = 0 # TODO
                else:
                    next_start_idx = start_idx
                    raw_size_to_cache = -1
        
        input_ids = self.all_input_ids[:, self._end_idx: end_idx].to(self._device)
        attention_mask = self.all_attention_mask[:, self._end_idx: end_idx].to(self._device)
        if self.all_labels is not None:
            labels = self.all_labels[:, self._end_idx: end_idx].to(self._device)
        else:
            labels = None
        batch_size = input_ids.size(0)

        if self.config.acre_pos == "interleave":
            input_len = input_ids.shape[1]
            if acre_size > 0:
                # prepare input_ids with acres
                input_ids_with_acres = input_ids.new_full((batch_size, input_len+acre_size), fill_value=self.acre_token.item())

                raw_token_indices = torch.arange(input_ids_with_acres.shape[1], device=input_ids.device)

                interleave_start_idx = compression_ratio - self._interleave_remainder
                raw_token_indices = raw_token_indices[raw_token_indices % (compression_ratio + 1) != interleave_start_idx].unsqueeze(0).expand_as(input_ids)

                input_ids_with_acres = input_ids_with_acres.scatter(dim=1, index=raw_token_indices, src=input_ids)
                input_ids = input_ids_with_acres

                # prepare attention_mask with acres
                attention_mask_with_acres = attention_mask.new_full((attention_mask.shape[0], attention_mask.shape[1] + acre_size), 1)
                attention_mask_with_acres = attention_mask_with_acres.scatter(dim=1, index=raw_token_indices, src=attention_mask)
                attention_mask = attention_mask_with_acres
                # labels
                if labels is not None:
                    labels_with_acres = labels.new_full((labels.shape[0], labels.shape[1] + acre_size), -100)
                    labels_with_acres = labels_with_acres.scatter(dim=1, index=raw_token_indices, src=labels)
                    labels = labels_with_acres

            if compression_ratio > 0:
                self._interleave_remainder = (input_len + self._interleave_remainder) % compression_ratio

        if self.training and self._step_idx == 0 and \
            not (self.config.acre_pos == 'interleave' and \
                self.config.acre_attn == 'full-coverage'):
            labels[:] = -100

        acre_indices = (input_ids[0] == self.acre_token.item()).long()
        if self._is_full_window:
            self._acre_indices = torch.tensor([], dtype=torch.long, device=input_ids.device)
        acre_indices = torch.cat([self._acre_indices, acre_indices])
        self._acre_indices = acre_indices
        if is_full_window and acre_size == -1:
            # NOTE: the first acre_stride raw tokens serve as acre tokens
            # we use -1 to indicate these raw tokens, so that the attention mask and position ids will not be modified
            acre_indices[:self.config.acre_stride] = -1


        past_key_values = []
        for layer_idx in range(self.config.num_hidden_layers):
            sink_key, sink_value = self.sink_activations[layer_idx]
            acre_key, acre_value = self.acre_activations[layer_idx]
            raw_key, raw_value = self.raw_activations[layer_idx]
            highlight_raw_key, highlight_raw_value = self.highlight_raw_activations[layer_idx]
            # if highlight_raw_key is not None:
            #     print(highlight_raw_key.shape)
                
            key = cat_tensor([
                sink_key, acre_key, highlight_raw_key, raw_key,
            ], dim=self.k_seq_dim)
            value = cat_tensor([
                sink_value, acre_value, highlight_raw_value, raw_value,
            ], dim=self.v_seq_dim)
            query_states = None
            layer_past_key_values = (key, value, acre_size, acre_indices, query_states)
            past_key_values.append(layer_past_key_values)
        
        # Prepare attention_mask and position_ids.
        first_key = past_key_values[0][0]
        mem_size = first_key.shape[self.k_seq_dim] if first_key is not None else 0
        if mem_size > 0:
            attention_mask = torch.cat([attention_mask.new_ones(batch_size, mem_size), attention_mask], dim=1)

        input_length = input_ids.shape[1]
        position_ids = torch.arange(attention_mask.shape[-1], dtype=torch.long, device=self._device).repeat(batch_size, 1)

        if self.config._attn_implementation == "flash_attention_2":
            assert self.config.acre_attn == "full-coverage", f"Make sure to set acre_attn='full-coverage' when using flash attention! Found {self.config.acre_attn}."
            if 0 in attention_mask:
                pass
            else:
                attention_mask = None

        else:
            raise NotImplementedError("Not implemented yet!")

        self._is_full_window = is_full_window
        self._raw_size_to_cache = raw_size_to_cache
        self._all_acre_sizes.append(acre_size)
        self._start_idx = next_start_idx
        self._end_idx = end_idx
        self._step_idx += 1
        return input_ids, attention_mask, position_ids, past_key_values, labels

    def set_compression_ratio(self, start_idx, end_idx):
        def filter_ratio(ratios, stride):
            valid_ratios = []
            for ratio in ratios:
                # stride must be bigger than condensing ratio because we there must be at least one acre
                if stride < ratio:
                    continue
                # the stride must be evenly divisible by condensing ratio
                if ratio > 0 and (stride % ratio) != 0:
                    continue
                # when training, ratio=0 is valid if previous windows contain acre or later windows contain acre
                if ratio == 0 and self.training:
                    previous_has_zero = -1 in self._all_acre_sizes
                    following_has_nonzero = (start_idx + stride + self.config.acre_window) <= self.all_sequence_length
                    if previous_has_zero or (not following_has_nonzero):
                        continue
                valid_ratios.append(ratio)
            assert len(valid_ratios), f"Cannot find valid condensing ratio (among {ratios}) for stride {stride}!"
            return valid_ratios

        if len(self.config.l1_l2_ratio) == 1:
            return self.config.l1_l2_ratio[0]

        ratio_mix = self.config.l1_l2_ratio_mix

        l1_l2_ratio = filter_ratio(self.config.l1_l2_ratio, self.config.acre_stride)

        if ratio_mix == "instance-random":
            if self._compression_ratio is None:
                l1_l2_ratio = self.rng.choice(l1_l2_ratio).tolist()
                self._compression_ratio = l1_l2_ratio
            else:
                l1_l2_ratio = self._compression_ratio

        elif ratio_mix == "step-random":
            l1_l2_ratio = self.rng.choice(l1_l2_ratio).tolist()
        
        elif ratio_mix == "sequence":
            if self._compression_ratio is None:
                self._compression_ratio = cycle(l1_l2_ratio)
            l1_l2_ratio = next(self._compression_ratio)

        else:
            raise NotImplementedError(f"Invalid l1_l2_ratio_mix: {ratio_mix}")

        return l1_l2_ratio

    def update_cache_indices(self, acre_indices):
        # slice the key and value activations by acre_indices and store them in self.cache_activations by layer_idx and self.cache_idx
        start_idx = 0
        for i,acre_idx in enumerate(acre_indices):
            if acre_idx == 1:
                self.cache_activation_indices[self.cache_idx] = (start_idx, i, self._step_idx)
                self.cache_idx += 1
                start_idx = i + 1

    def update_memory(self, past_key_values):
       for layer_idx, (key, value, acre_size, acre_indices, query_states) in enumerate(past_key_values):

            # NOTE: the past_key_values are incrementally returned (only the new keys and values are returned)
            previous_raw_key, previous_raw_value = self.raw_activations[layer_idx]

            if self.acre_skip_first is not None and self.sink_activations[layer_idx][0] is None:
                assert key.shape[self.k_seq_dim] == self.acre_skip_first
                assert value.shape[self.k_seq_dim] == self.acre_skip_first
                self.sink_activations[layer_idx] = [
                    key,
                    value,
                ]
                # NOTE: no need to update raw activations and acre activations as all activations are kept as sink activations
                continue

            if self.acre_activations[layer_idx][0] is None and self.config.acre_sink_size > 0:
                # save the sink activations
                # NOTE: we do not slice the key/value activations, which may cause duplication when l1_l2_ratio=-1 for the first window, but it's okay
                self.sink_activations[layer_idx] = [
                    slice_tensor(key, end=self.config.acre_sink_size, dim=self.k_seq_dim),
                    slice_tensor(value, end=self.config.acre_sink_size, dim=self.v_seq_dim),
                ]

            if not self._is_full_window:
                # this means the current input does not fulfill a window
                # thus, the key and value are all raw activations, and we accumulate them until the window is fulfilled
                assert self._raw_size_to_cache == -1
                raw_key = cat_tensor([
                    previous_raw_key,
                    key
                ], dim=self.k_seq_dim)
                raw_value = cat_tensor([
                    previous_raw_value, 
                    value
                ], dim=self.v_seq_dim)
                self.raw_activations[layer_idx] = (raw_key, raw_value)
                self.raw_query_states[layer_idx] = query_states

            else:
                self.cache_activations[layer_idx][0][self._step_idx] = cat_tensor([
                    previous_raw_key,
                    key
                ], dim=self.k_seq_dim)
                self.cache_activations[layer_idx][1][self._step_idx] =  cat_tensor([
                    previous_raw_value, 
                    value
                ], dim=self.v_seq_dim)
                if layer_idx == 0:
                    self.update_cache_indices(acre_indices)

                # NOTE: use the correct previous_acre_key and value!
                previous_acre_key, previous_acre_value = self.acre_activations[layer_idx]
                acre_key, acre_value, raw_key, raw_value = self._extract_acre_and_raw_memory(
                    key, 
                    value, 
                    previous_acre_key, 
                    previous_acre_value, 
                    previous_raw_key, 
                    previous_raw_value, 
                    acre_indices,
                )

                self.acre_activations[layer_idx] = (acre_key, acre_value)
                self.raw_activations[layer_idx] = (raw_key, raw_value)

    def update_loss(self, loss, valid_token_num):
        if self._batch_loss is None:
            # NOTE: multiply valid_token_num because batch_loss is divided by it in advance
            self._batch_loss = batch_loss * valid_token_num
            self._valid_token_num = valid_token_num
        else:
            # NOTE: avoid in-place operations, otherwise there will be gradient errors in training
            self._batch_loss = self._batch_loss + batch_loss * valid_token_num
            self._valid_token_num = self._valid_token_num + valid_token_num

    def output(self, model_outputs):
        if self._batch_loss is not None:
            # here the batch_loss is the summation of all token losses in each element
            loss = self._batch_loss.sum() / self._valid_token_num.sum()

            # NOTE: prevent nan
            batch_loss = self._batch_loss / self._valid_token_num
            if (self._valid_token_num == 0).any():
                batch_loss = batch_loss.masked_fill(self._valid_token_num == 0, 0.)

            # NOTE: we must use dict to override values, otherwise trainer cannot find loss
            model_outputs["loss"] = loss
            model_outputs["batch_loss"] = batch_loss

        # override last_hidden_states (used in generation)
        acre_size = self._all_acre_sizes[-1]
        # remove logits corresponding to acre tokens
        if acre_size > 0:
            logits = model_outputs["logits"]
            acre_indices = self._acre_indices[-logits.shape[1]:]
            model_outputs["logits"] = logits[:, acre_indices == 0]

        return model_outputs

    def export(self):
        """Export all necessary attributes of the memory module."""
        return {
            "_start_idx": self._start_idx,
            "_end_idx": self._end_idx,
            "_all_acre_sizes": self._all_acre_sizes,
            "_batch_loss": self._batch_loss,
            "_valid_token_num": self._valid_token_num,
            "_step_idx": self._step_idx,
            "_compression_ratio": self._compression_ratio,
            "_is_full_window": self._is_full_window,
            "_raw_size_to_cache": self._raw_size_to_cache,
            "_interleave_remainder": self._interleave_remainder,
            "_interleave_compression_ratio": self._interleave_compression_ratio,
            "_acre_indices": self._acre_indices,
            "all_input_ids": self.all_input_ids,
            "all_attention_mask": self.all_attention_mask,
            "all_labels": self.all_labels,
            "acre_skip_first": self.acre_skip_first,
            "acre_skip_last": self.acre_skip_last,
            # NOTE: deepcopy to untie the memory state and the model memory
            "sink_activations": deepcopy(self.sink_activations),
            "acre_activations": deepcopy(self.acre_activations),
            "raw_activations": deepcopy(self.raw_activations),
        }

    def _extract_acre_and_raw_memory(self, 
        key, 
        value, 
        previous_acre_key, 
        previous_acre_value, 
        previous_raw_key, 
        previous_raw_value, 
        acre_indices,):
        key = cat_tensor([
            previous_raw_key, 
            key
        ], dim=self.k_seq_dim)
        value = cat_tensor([
            previous_raw_value, 
            value
        ], dim=self.v_seq_dim)

        # NOTE: we use magic slice instead of boolean index here for efficiency
        acre_key = slice_tensor(key, index=torch.logical_or(acre_indices == 1, acre_indices == -1), dim=self.k_seq_dim)
        acre_key = cat_tensor([previous_acre_key, acre_key], dim=self.k_seq_dim)
        acre_value = slice_tensor(value, index=torch.logical_or(acre_indices == 1, acre_indices == -1), dim=self.v_seq_dim)
        acre_value = cat_tensor([previous_acre_value, acre_value], dim=self.v_seq_dim)

        if self._raw_size_to_cache > 0:
            raw_key = slice_tensor(key, index=acre_indices == 0, dim=self.k_seq_dim)
            raw_key = slice_tensor(raw_key, start=-raw_size_to_cache, dim=self.k_seq_dim)

            raw_value = slice_tensor(value, index=acre_indices == 0, dim=self.v_seq_dim)
            raw_value = slice_tensor(raw_value, start=-raw_size_to_cache, dim=self.v_seq_dim)        

        else:
            raw_key = None
            raw_value = None

        return acre_key, acre_value, raw_key, raw_value

    @property
    def all_sequence_length(self):
        if self.all_input_ids is None:
            return 0
        else:
            return self.all_input_ids.shape[1]

    @property
    def batch_size(self):
        if self.all_input_ids is None:
            return 0
        else:
            return self.all_input_ids.shape[0]

    @property
    def finish(self):
        is_finish = self._end_idx == self.all_sequence_length
        return is_finish

    @property
    def dtype(self):
        return self.config.torch_dtype

    @property
    def min_value(self):
        return torch.finfo(self.dtype).min

    @property
    def max_position_embeddings(self):
        max_position_embeddings = self.config.max_position_embeddings
        if getattr(self.config, "rope_scaling", None) is not None:
            scaling_factor = self.config.rope_scaling["factor"]
            max_position_embeddings = max_position_embeddings * scaling_factor
        return max_position_embeddings
 
    def get_memory_size(self):
        """
        Sink memory size, acre memory size and raw memory size.
        """
        sink_memory_size = 0
        acre_memory_size = 0
        raw_memory_size = 0
        if self.sink_activations[0][0] is not None:
            sink_memory_size += self.sink_activations[0][0].shape[self.k_seq_dim]
        if self.acre_activations[0][0] is not None:
            acre_memory_size += self.acre_activations[0][0].shape[self.k_seq_dim]
        if self.raw_activations[0][0] is not None:
            raw_memory_size += self.raw_activations[0][0].shape[self.k_seq_dim]
        return sink_memory_size, acre_memory_size, raw_memory_size

def slice_tensor(x, start=None, end=None, step=None, index=None, dim=2):
    if x is None:
        return None
    if end == 0:
        return None
    if start == x.shape[dim]:
        return None
    if start is not None and start == end:
        return None
    if dim == 2:
        if index is not None:
            return x[:, :, index]
        elif start is None and end is not None:
            if step is None:
                return x[:, :, :end, ...]
            else:
                return x[:, :, :end:step, ...]
        elif start is not None and end is None:
            if step is None:
                return x[:, :, start:, ...]
            else:
                return x[:, :, start::step, ...]
        elif start is not None and end is not None:
            if step is None:
                return x[:, :, start:end, ...]
            else:
                return x[:, :, start:end:step, ...]
    elif dim == 1:
        if index is not None:
            return x[:, :, index]
        elif start is None and end is not None:
            if step is None:
                return x[:, :end, ...]
            else:
                return x[:, :end:step, ...]
        elif start is not None and end is None:
            if step is None:
                return x[:, start:, ...]
            else:
                return x[:, start::step, ...]
        elif start is not None and end is not None:
            if step is None:
                return x[:, start:end, ...]
            else:
                return x[:, start:end:step, ...]
    else:
        raise NotImplementedError

def cat_tensor(list_of_tensors, dim=-1):
    list_of_tensors = [t for t in list_of_tensors if t is not None]
    if len(list_of_tensors) > 1:
        result = torch.cat(list_of_tensors, dim=dim)
    elif len(list_of_tensors) == 1:
        result = list_of_tensors[0]
    else:
        result = None
    return result

def slice_activations(activations, start=None, end=None, k_seq_dim=2, v_seq_dim=2):
    new_activations = []
    for key, value in activations:
        new_key = slice_tensor(key, start=start, end=end, dim=k_seq_dim)
        new_value = slice_tensor(value, start=start, end=end, dim=v_seq_dim)
        new_activations.append([new_key, new_value])
    return new_activations

def cat_activations(list_of_activations, k_seq_dim=2, v_seq_dim=2):
    assert all(len(x) == len(list_of_activations[0]) for x in list_of_activations), f"Make sure all activations have the same number of layers! Found {[len(x) for x in list_of_activations]}."

    new_activations = []
    for layer_idx in range(len(list_of_activations[0])):
        keys = [x[layer_idx][0] for x in list_of_activations]
        values = [x[layer_idx][1] for x in list_of_activations]

        new_key = cat_tensor(keys, dim=k_seq_dim)
        new_value = cat_tensor(values, dim=v_seq_dim)
        new_activations.append([new_key, new_value])
    return new_activations
