#!/bin/bash

########### make bash recognize aliases ##########
shopt -s expand_aliases
source ~/.bashrc

pta llm
python --version

if [[ $WORLD_SIZE ]]; then
DDP="--nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR"
else
DDP=""
fi

output_name=stage1-qwen2.5

torchrun --nproc_per_node 8 $DDP -m tasks.train \
--output_dir data/outputs/$output_name \
--model_name_or_path qwen2.5-3b-instruct \
--train_data redpajama/train.json \
--min_length 4096 \
--max_length 65536 \
--group_by_stride strict \
--enable_acre \
--acre_window 4096 \
--acre_stride 2048 \
--acre_attn full-coverage \
--acre_attend_prev True \
--acre_sink_size 0 \
--l1_l2_ratio 8 16 32 64 128 \
--l1_l2_ratio_mix step-random \
--acre_param q k v \
--acre_pos interleave \
--attn_impl flash_attention_2 \
--gradient_checkpointing \
--gradient_checkpointing_kwargs '{"use_reentrant": false}' \
--save_only_model \
--save_strategy epoch \
--evaluation_strategy steps \
--num_train_epochs 1 \
--logging_steps 50 \
--bf16 \
--deepspeed data/deepspeed/stage2.json
