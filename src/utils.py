import pandas as pd
import json
from collections import defaultdict
from statistics import quantiles
from math import ceil
from typing import Optional, List, Dict, Any, Mapping, Iterable
from dataclasses import dataclass
import shutil
import torch
import pathlib
from tqdm import tqdm
import sys
import re
import pytz
from datetime import datetime
from transformers.tokenization_utils import PreTrainedTokenizer
from sacremoses import MosesTokenizer, MosesDetokenizer
import tiktoken
import nltk
encoder = tiktoken.get_encoding("cl100k_base")
import random
import numpy as np

def set_seed(seed: int):
    """
    Set a fixed seed for reproducibility across multiple libraries.

    Parameters:
    seed (int): The seed value to use for random number generation.
    """
    # Set the seed for Python's random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    
    # Ensures deterministic behavior when using CUDA (may affect performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_sentences(text, sentence_indices):
    # Dictionary to store the sentences with their corresponding index
    sentences_dict = {}
    
    # Regular expression to find sentences starting with [sX]
    pattern = r'(\[s\d+\])\s*(.*?)(?=\[\s*s\d+\]|\Z)'
    for match in re.finditer(pattern, text, re.DOTALL):
        index = match.group(1)
        sentence = match.group(2).strip()
        sentences_dict[index] = sentence
    print(sentences_dict)
    # Retrieve the requested sentences based on the input list of indices
    extracted_sentences = [sentences_dict.get(index) for index in sentence_indices if index in sentences_dict]
    
    return extracted_sentences

def csv2json(file_path, sep=','):
    df = pd.read_csv(file_path, sep=sep, engine='python')
    json_data = df.to_dict(orient='records')
    return json_data

def json2csv(json_list, out_path):
    df = pd.DataFrame(json_list)
    df.to_csv(out_path, index=False)

def load_jsonl(path: str) -> List:
    rtn = []
    print(f"Begin to load {path}")
    for line in tqdm(open(path)):
        line = json.loads(line)
        rtn.append(line)
    return rtn

def save_jsonl(data: list, path: str) -> None:
    with open(path, "w") as f:
        for line in data:
            f.write(
                json.dumps(
                    line, 
                    ensure_ascii=False
                    )+"\n")

def load_json(path: str):
    with open(path, "r") as f:
        rtn = json.load(f)
    return rtn

def save_json(data, path: str):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_txt(path):
    return open(path).read()

def save_txt(content, path):
    with open(path, "w") as f:
        f.write(content, path)

@dataclass
class DefaultDataCollator:
    """
    Data collator that can:
    1. Dynamically pad all inputs received. The inputs must be dict of lists.
    2. Add position_ids based on attention_mask if required.
    """
    tokenizer: Optional[PreTrainedTokenizer] = None
    padding_side: str = "left"
    input_padding_value: int = 0
    attention_padding_value: int = 0
    label_padding_value: int = -100

    keys_to_tensorize = {"input_ids", "attention_mask", "labels", "position_ids", "token_type_ids", "length", "depth", "index"}

    def __call__(self, batch_elem: List) -> Dict[str, Any]:
        first_elem = batch_elem[0]
        return_batch = {}
        
        for key, value in first_elem.items():
            # HACK: any key containing attention_mask must be attention_mask
            # important to assign different pad token for different types of inputs
            if "attention_mask" in key:
                pad_token_id = self.attention_padding_value
            elif "label" in key:
                pad_token_id = self.label_padding_value
            else:
                pad_token_id = self.input_padding_value

            batch_value = [elem[key] for elem in batch_elem]
            # pad all lists and nested lists
            if isinstance(value, list) and key in self.keys_to_tensorize:
                max_length = get_max_length_in_nested_lists(batch_value)
                batch_value, _ = pad_nested_lists(batch_value, max_length, pad_token_id, self.padding_side)

            if key in self.keys_to_tensorize:
                return_batch[key] = torch.tensor(batch_value)
            else:
                # handle strings and None
                return_batch[key] = batch_value
        return return_batch

class FileLogger:
    def __init__(self, log_file) -> None:
        self.log_file = log_file
    
    def log(self, metrics, **kwargs):
        with open(self.log_file, "a+") as f:
            # get current time
            tz = pytz.timezone('')
            time = f"{'Time': <10}: {json.dumps(datetime.now(tz).strftime('%Y-%m-%d, %H:%M:%S'), ensure_ascii=False)}\n"
            print(time)
            command = f"{'Command': <10}: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}\n"
            print(command)
            metrics = f"{'Metrics': <10}: {json.dumps(metrics, ensure_ascii=False)}\n"
            msg = time + command

            for key, value in kwargs.items():
                x = f"{key: <10}: {json.dumps(value, ensure_ascii=False)}\n"
                print(x)
                msg += x
            msg += metrics
            print(metrics)
            f.write(str(msg) + "\n")

def makedirs(path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return path

def clear_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def split_file_dir_name_ext(path):
    """Return the directory, name, and extension of a given file."""
    p = pathlib.Path(path)
    assert p.is_file(), f"{path} is not a valid file!"
    return p.parent, p.stem, p.suffix

def save_pickle(obj, path:str):
    """
    Save pickle file.
    """
    if not os.path.exists(path):
        makedirs(path)
    with open(path, "wb") as f:
        return pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
