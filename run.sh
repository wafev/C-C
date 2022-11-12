#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS="ignore"

# evaluate
python finetune.py --eval --model deit_tiny_patch16_224  --batch-size 256 --data-path /path/to/imagenet --resume /path/to/checkpoint.pth

# throughput test
python finetune.py --speed-only --model deit_tiny_patch16_224  --batch-size 256 --data-path /path/to/imagenet --resume /path/to/checkpoint.pth

# prune
python finetune.py --prune --model deit_tiny_patch16_224 \
--resume /path/to/pre-trained/model --data-path path/to/imagenet \
--batch-size 256 --prune_mode attn --prune_rate 0.25

# finetune
python -m torch.distributed.launch --nproc_per_node=2 --use_env \
finetune.py --model deit_tiny_patch16_224 --batch-size 256 --data-path /path/to/imagenet \
--resume /path/to/pruned/model \
--teacher-path  /path/to/pretrained/model \
--teacher-model deit_tiny_patch16_224 --distillation-type soft \
--prune_mode attn --final_finetune 30 --decay-epochs 6 \
--batch-size 256 --lr 1e-4 --weight-decay 0.001 --distillation-alpha 0.2 --distillation-tau 20 \
--output_dir /path/to/save
