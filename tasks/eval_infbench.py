import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import datasets
import json
import torch
import pandas as pd
from tqdm import tqdm
from functools import partial
from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
from transformers import HfArgumentParser, AutoTokenizer
from transformers.utils import logging
from torch.utils.data import DataLoader

from src import ModelArgs, DefaultDataCollator, FileLogger, makedirs, apply_chat_template, get_pipeline
from tasks.infbench_utils import TASK_TO_PATH, TASK_TO_MAX_NEW_TOKENS, get_score_one, MODEL_TO_PROMPT_TEMPLATE, get_answer


logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="long-llm:infbench",
        metadata={'help': 'The directory of all infbench evaluation data.'}
    )
    output_dir: str = field(
        default="",
        metadata={'help': 'The base directory for saving results and logs.'}
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )

    tasks: List[str] = field(
        default_factory=lambda: ['longbook_qa_eng'],
        metadata={'help': 'Which dataset to evaluate?'}
    )
    prompt_template: str = field(
        default="mistral",
        metadata={'help': 'Which prompt template to use? (See infbench_utils.py for reference.)'}
    )

    max_length: int = field(
        default=128000,
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

    delay: int = field(
        default=0,
        metadata={'help': 'How many seconds to wait for each forward call?'}
    )
    mini: Optional[int] = field(
        default=None,
        metadata={'help': 'How many samples for each dataset?'}
    )


def process_infbench(data, indices, tokenizer, chat_template, task:str, prompt_template:str="mistral", max_length=100000, truncate_from_middle=True):
    outputs = {'context': [], 'question': [], "answer": [], "index": []}

    # NOTE: high version datasets use LazyBatch to wrap data, which cannot be reverted to list of dicts, thus, we need to convert it to dict first
    data = pd.DataFrame(dict(data)).to_dict(orient="records")

    for sample, index in zip(data, indices):
        context = sample['context']
        question = sample['input']
        if task == "longbook_sum_eng":
            question = ""
        answer = get_answer(sample, task)

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

        outputs["context"].append(context)
        outputs["question"].append(question)
        outputs["answer"].append(answer)
        outputs["index"].append(index)

    return outputs


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)

    pipe = get_pipeline(args, device=accelerator.device)
    tokenizer = pipe.generator.tokenizer

    with accelerator.main_process_first():
        all_datasets = {}

        for task in args.tasks:
            process_fn = partial(
                process_infbench, 
                tokenizer=tokenizer,
                chat_template=args.chat_template,
                max_length=args.max_length,
                task=task,
                prompt_template=args.prompt_template,
                truncate_from_middle=args.truncate_from_middle,
            )

            path = os.path.join(args.eval_data, TASK_TO_PATH[task])
            raw_dataset = datasets.load_dataset("json", data_files=path, cache_dir=args.dataset_cache_dir, split="train")
            dataset = raw_dataset.map(process_fn, batched=True, num_proc=32, batch_size=10, with_indices=True, remove_columns=raw_dataset.column_names)

            all_datasets[task] = dataset

    result_dir = os.path.join(args.output_dir, args.result_dir)

    metrics = {}

    for i, (task, dataset) in enumerate(all_datasets.items()):
        if accelerator.process_index == 0:
            logger.info(f"Evaluating {task} ({i + 1} / {len(all_datasets)})...")

        result_path = os.path.join(result_dir, f"{task}.json")
        
        if args.load_result and os.path.exists(result_path):
            if accelerator.process_index == 0:
                scores = []
                preds = []
                labels = []
                indices = []
                with open(result_path, encoding="utf-8") as f:
                    # the first line is metric
                    f.readline()

                    for line in f:
                        item = json.loads(line)
                        pred = item["pred"]
                        label = item["label"]
                        index = item["index"]
                        # NOTE: here we explicitly input model_name=None
                        score = get_score_one(pred, label, task, None)
                        scores.append(score)

                        preds.append(pred)
                        labels.append(label)
                        indices.append(index)

                    score = round(sum(scores) / len(scores), 4)

                    logger.info(f"{task}: {score}")
                    metrics[task] = score

                with open(makedirs(result_path), "w", encoding="utf-8") as f:
                    f.write(json.dumps(score, ensure_ascii=False) + "\n")
                    for index, pred, label in zip(indices, preds, labels):
                        item = {
                            "index": index,
                            "pred": pred,
                            "label": label,
                        }
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

        else:
            # get answers in advance
            labels = dataset["answer"]
            dataset = dataset.remove_columns(["answer"])

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
            max_new_tokens = TASK_TO_MAX_NEW_TOKENS[task]
            _prompt = MODEL_TO_PROMPT_TEMPLATE[args.prompt_template][task]

            for j, x in enumerate(tqdm(dataloader, desc="Generating")):
                index = x.pop("index")[0]

                # pipe.reset()

                output = [pipe(x["context"][0], x["question"][0], prompt=_prompt, conv=args.conv)]

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
                scores = []
                for label, pred in tqdm(zip(labels, preds)):
                    # NOTE: here we explicitly input model_name=None
                    score = get_score_one(pred, label, task, None)
                    scores.append(score)
                score = round(sum(scores) / len(scores), 4)

                logger.info(f"{task}: {score}")
                metrics[task] = score

                with open(makedirs(result_path), "w", encoding="utf-8") as f:
                    f.write(json.dumps(score, ensure_ascii=False) + "\n")
                    for index, pred, label, memory_res in zip(indices, preds, labels, memory_results):
                        item = {
                            "index": index,
                            "pred": pred,
                            "label": label,
                            "memory_res": memory_res
                        }
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if accelerator.process_index == 0:
        # save config
        args.save(os.path.join(result_dir, "config.json"))

        avg = round(sum(metrics.values()) / len(metrics), 4)
        metrics["avg"] = avg

        file_logger = FileLogger(makedirs(os.path.join(args.output_dir, "metrics.log")))
        file_logger.log(metrics, Args=asdict(args))
        with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as f:
            save_args = asdict(args)
            save_args["metrics"] = metrics
            f.write(json.dumps(save_args)+"\n")


if __name__ == "__main__":
    main()
