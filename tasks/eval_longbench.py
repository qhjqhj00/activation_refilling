import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets
import json
import torch
import time
from tqdm import tqdm
from typing import Optional, Dict, List
from functools import partial
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from torch.utils.data import DataLoader

from src import ModelArgs, DefaultDataCollator, FileLogger, makedirs, get_pipeline
from src.utils import set_seed
from tasks.longbench_utils import DATASET2CATEGORY, scorer, DATASET2PROMPT, scorer_compare, DATASET2MAXNEWTOKENS

logger = logging.get_logger(__name__)
set_seed(42)

@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="long-llm:longbench/test.json",
        metadata={'help': 'The evaluation json data path.'}
    )
    output_dir: str = field(
        default="",
        metadata={'help': 'The base directory for saving results and logs.'}
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )

    dataset_names: List[str] = field(
        default_factory=lambda: ['lcc', 'samsum', 'narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique', 'gov_report', 'qmsum', 'multi_news'],
        metadata={'help': 'Which dataset to evaluate?'}
    )

    max_length: Optional[int] = field(
        default=None,
        metadata={'help': 'Max input length.'}
    )
    truncate_from_middle: bool = field(
        default=True,
        metadata={'help': 'Truncate inputs from the middle.'}
    )
    load_result: bool = field(
        default=False,
        metadata={'help': 'Load result from saved files?'}
    )
    mini: Optional[int] = field(
        default=None,
        metadata={'help': 'How many samples for each dataset?'}
    )

def process_longbench(data, indices, tokenizer, max_length=3500, truncate_from_middle=True):
    outputs = {'context': [], 'question': [], "dataset": [], "index": [], "length": []}

    for input, context, dataset, index in zip(data['input'], data['context'], data['dataset'], indices):
        if dataset.endswith("_e"):
            dataset = dataset[:-2]

        if dataset in ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique', 'samsum']:
            question = input
        elif dataset in ["gov_report", "lcc", "multi_news"]:
            question = ""
        else:
            continue
        
        if max_length is not None:
            if truncate_from_middle:
                try:
                    tokenized_context = tokenizer.encode(context, add_special_tokens=False)
                except:
                    tokenized_context = tokenizer.encode(context)
                if len(tokenized_context) > max_length:
                    half = int(max_length / 2)
                    context = tokenizer.decode(tokenized_context[:half]) + tokenizer.decode(tokenized_context[-half:])
            else:
                tokenized_context = tokenizer.encode(context)
                context = tokenizer.decode(tokenized_context[-max_length:])

        length = len(tokenizer.encode(context))

        outputs["context"].append(context)
        outputs["question"].append(question)
        outputs["dataset"].append(dataset)
        outputs["index"].append(index)
        outputs["length"].append(length)

    return outputs

@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]
    accelerator = Accelerator(cpu=args.cpu)

    pipe = get_pipeline(args, device=accelerator.device)
    
    tokenizer = pipe.generator.tokenizer

    with accelerator.main_process_first():
        process_fn = partial(
            process_longbench, 
            tokenizer=tokenizer,
            max_length=args.max_length,
            truncate_from_middle=args.truncate_from_middle
        )

        raw_dataset = datasets.load_dataset("json", data_files=args.eval_data, cache_dir=args.dataset_cache_dir, split="train")
        dataset = raw_dataset.map(process_fn, batched=True, num_proc=32, with_indices=True, remove_columns=raw_dataset.column_names)

    groupby_dataset = dataset.to_pandas().groupby("dataset")

    metrics = {}
    if args.dataset_names is None:
        dataset_names = [key for key, _ in groupby_dataset]
    else:
        dataset_names = args.dataset_names

    result_dir = os.path.join(args.output_dir, args.result_dir)

    for i, dataset_name in enumerate(dataset_names):
        if accelerator.process_index == 0:
            logger.info(f"Evaluating {dataset_name} ({i + 1} / {len(dataset_names)})...")

        result_path = os.path.join(result_dir, f"{dataset_name}.json")
        
        if args.load_result and os.path.exists(result_path):
            if accelerator.process_index == 0:
                with open(result_path, encoding="utf-8") as f:
                    score = json.loads(f.readline())
                logger.info(f"{dataset_name}: {score}")
                metrics[dataset_name] = score

        else:
            dataset = datasets.Dataset.from_pandas(groupby_dataset.get_group(dataset_name), preserve_index=False)

            # the first 32 samples
            if args.mini is not None and len(dataset) - args.mini > 0:
                dataset = dataset.train_test_split(len(dataset) - args.mini, shuffle=False)["train"]

            data_collator = DefaultDataCollator(padding_side=args.padding_side)
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                collate_fn=data_collator,
                # only pin memory when no gpu
                pin_memory=not args.cpu,
            )

            # NOTE: prepare dataloader so the data moves to GPU automatically
            dataloader = accelerator.prepare(dataloader)

            indices = []
            preds = []
            memory_results = []
            _prompt = DATASET2PROMPT[dataset_name]
            pipe.generation_kwargs["max_new_tokens"]=DATASET2MAXNEWTOKENS[dataset_name]
            
            for i, x in enumerate(tqdm(dataloader, desc="Generating")):
                x.pop("dataset")
                index = x.pop("index")[0]

                # pipe.reset()

                # NOTE: output should be a list
                
                if "QA" in DATASET2CATEGORY[dataset_name]:
                    
                    if hasattr(pipe, "memory"):
                        pipe.mem_generation_kwargs["max_new_tokens"] = 256
                    output = [pipe(x["context"][0], x["question"][0], prompt=_prompt, task=dataset_name, conv=args.conv)]
                else:
                    if hasattr(pipe, "memory"):
                        pipe.mem_generation_kwargs["max_new_tokens"] = 512
                    output = [pipe(x["context"][0], x["question"][0], prompt=_prompt, task=dataset_name, conv=args.conv)]

                if accelerator.num_processes > 1:
                    # pad across device to the same length
                    output = accelerator.gather_for_metrics(output)
                    index = accelerator.gather_for_metrics(index)

                accelerator.print([line[0] for line in output])

                index = index.tolist()

                if accelerator.process_index == 0:
                    pred = [line[0] for line in output]
                    preds.extend(pred)
                    memory_res = [line[1] for line in output]
                    memory_results.extend(memory_res)
                    if isinstance(index, list):
                        indices.extend(index)
                    else:
                        # single process
                        indices.append(index)

            if accelerator.process_index == 0:
                raw_dataset_subset = raw_dataset[indices]
                answers = raw_dataset_subset["answers"]
                # lengths = raw_dataset_subset["length"]
                all_classes = []
                # all_classes = raw_dataset_subset["all_classes"][0]
                # TODO
                score = scorer(dataset_name, preds, answers, all_classes)        
                
                logger.info(f"{dataset_name}: {score}")
                metrics[dataset_name] = score

                with open(makedirs(result_path), "w", encoding="utf-8") as f:
                    f.write(json.dumps(score, ensure_ascii=False) + "\n")
                    for index, pred, memory_res in zip(indices, preds, memory_results):
                        sample = raw_dataset[index]
                        # del sample["all_classes"]
                        del sample["context"]
                        # del sample["language"]
                        # del sample["_id"]
                        sample["pred"] = pred
                        sample["memory_res"] = memory_res
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if accelerator.process_index == 0:
        # save config
        args.save(os.path.join(result_dir, "config.json"))

        # compute category score
        category_metrics = defaultdict(list)
        for dataset, metric in metrics.items():
            category = DATASET2CATEGORY[dataset]
            category_metrics[category].append(metric)
        for k, v in category_metrics.items():
            # when evaluating on longbench_e, each metric is a dict of float
            if isinstance(v[0], dict):
                category_metric = {}
                for kk in v[0].keys():
                    vv = [v[j][kk] for j in range(len(v))]
                    category_metric[kk] = round(sum(vv) / len(vv), 2)
                category_metrics[k] = category_metric
            else:
                category_metrics[k] = round(sum(v) / len(v), 2)
        
        # compute average score
        if isinstance(next(iter(metrics.values())), dict):
            avg = defaultdict(list)
            for k, v in metrics.items():
                for kk, vv in v.items():
                    avg[kk].append(vv)
            for k, v in avg.items():
                avg[k] = round(sum(v) / len(v), 2)
        else:
            avg = round(sum(metrics.values()) / len(metrics), 2)
        metrics["avg"] = avg

        file_logger = FileLogger(makedirs(os.path.join(args.output_dir, "metrics.log")))
        file_logger.log(metrics, Args=asdict(args), Category_Metrics=category_metrics)
        with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as f:
            save_args = asdict(args)
            save_args["metrics"] = metrics
            save_args["category_metrics"] = category_metrics
            f.write(json.dumps(save_args)+"\n")

if __name__ == "__main__":
    main()

    