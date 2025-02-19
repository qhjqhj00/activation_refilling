#!/bin/bash

source /opt/conda/bin/activate path_to_env
PWD="$(pwd)" 
cd $PWD

result_dir=acre/

max_length=200000
l1_l2_ratio=16
max_window=32000
stride=2048
gen_max_new_tokens=1200
checkpoint=acre-qwen2.5-3B

torchrun --nproc_per_node 8 -m tasks.eval_longbench \
            --dataset_names narrativeqa hotpotqa qasper multifieldqa_en 2wikimqa musique \
            --result_dir $result_dir \
            --ret_hits $hits \
            --gen_model  $checkpoint\
            --gen_max_new_tokens $gen_max_new_tokens \
            --max_length $max_length \
            --max_window $max_window \
            --l1_l2_ratio $l1_l2_ratio \
            --stride $stride 

torchrun --nproc_per_node 8 -m tasks.eval_infbench \
            --dataset_names longbook_qa_eng \
            --result_dir $result_dir \
            --ret_hits $hits \
            --gen_model  $checkpoint\
            --gen_max_new_tokens $gen_max_new_tokens \
            --max_length $max_length \
            --max_window $max_window \
            --l1_l2_ratio $l1_l2_ratio \
            --stride $stride 

torchrun --nproc_per_node 8 -m tasks.eval_domain \
            --dataset_names fin legal\
            --eval_data_files fin.dev.jsonl legal.dev.jsonl \
            --result_dir $result_dir \
            --ret_hits $hits \
            --gen_model  $checkpoint\
            --gen_max_new_tokens $gen_max_new_tokens \
            --max_length $max_length \
            --max_window $max_window \
            --l1_l2_ratio $l1_l2_ratio \
            --stride $stride 

torchrun --nproc_per_node 8 -m tasks.eval_domain_18 \
            --dataset_names cs mathematics physics philosophy history biology \
            --eval_data_files cs.dev.jsonl mathematics.dev.jsonl physics.dev.jsonl philosophy.dev.jsonl history.dev.jsonl biology.dev.jsonl \
            --result_dir $result_dir \
            --ret_hits $hits \
            --gen_model  $checkpoint\
            --gen_max_new_tokens $gen_max_new_tokens \
            --max_length $max_length \
            --max_window $max_window \
            --l1_l2_ratio $l1_l2_ratio \
            --stride $stride 
