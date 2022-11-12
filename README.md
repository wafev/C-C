# Clip & Concat: Efficient Token Pruning For Vision Transformer With Spatial Information Preserved 

This repository contains PyTorch evaluation code, pruning code, finetuning code and pruned models of our our method for DeiT:
 
The framework of our Clip & Concat:

![cc-vit](.github/framework.jpg)

# Model Zoo

We provide baseline DeiT models pretrained on ImageNet 2012 and the pruned model.

| name | acc@1 | FLOPS | Throghput | url |
| --- | --- | --- | --- | --- |
| DeiT-tiny | 72.2 | 1.3 | 2234 | [model](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth) |
| DeiT-small | 79.8 | 4.6 | 1153| [model](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) |
| DeiT-base | 81.8 | 17.6 | - | [model](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth) |
| C&C-tiny-25 | 73.0 | 1.0 | 2876 | [model](https://drive.google.com/file/d/1FTpJ7r0ihgjhGnwQUh5dfcdiYlkEVm7n/view?usp=share_link) |
| C&C-tiny-40 | 71.4 | 0.7 | 3685 | [model](https://drive.google.com/file/d/1-UKN2C5juWRMCjdZm1JZq1oDEyT3mlgQ/view?usp=share_link) |
| C&C-small-25 | 79.5  | 3.4 | 1587 | [model](https://drive.google.com/file/d/1wLAInKP0cIvCJXRsJM5OA7TFmZ1Bz3C0/view?usp=share_link) |
| C&C-small-30 | 79.1  | 3.2 | 1711 | [model](https://drive.google.com/file/d/1yingnN-qSPGcJfL7BJIf2e4PBhegV063/view?usp=share_link) |

# Usage

First, clone the repository locally:
```
git clone https://github.com/wafev/C-C.git
```
Then, create the conda environment and install PyTorch 1.10.1+ and torchvision 0.11.2+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda create --n ccvit python==3.7
conda install pytorch==1.10.1 torchvision==0.11.2 -c pytorch
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Evaluation
To evaluate a pruned DeiT-Tiny on ImageNet val with a single GPU run:
```
python finetune.py --eval --model deit_tiny_patch16_224  --batch-size 256 --data-path /path/to/imagenet --resume /path/to/checkpoint.pth
```

For Throughput test, run:
```
python finetune.py --speed-only --model deit_tiny_patch16_224  --batch-size 256 --data-path /path/to/imagenet --resume /path/to/checkpoint.pth
```

Note that the resume path is necessary because the pruning setting is saved in the checkpoint. 

## Pruning

For pruning DeiT-Tiny without finetuning, run:
```
python finetune.py --prune --model deit_tiny_patch16_224 \
--resume /path/to/pre-trained/model --data-path path/to/imagenet \
--batch-size 256 --prune_mode attn --prune_rate 0.25
```
## Finetuning

To finetune the pruned DeiT-small and Deit-tiny on ImageNet on a single node with 2 gpus for 30 epochs run:

DeiT-Tiny
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env \
finetune.py --model deit_tiny_patch16_224 --batch-size 256 --data-path /path/to/imagenet \
--resume /path/to/pruned/model \
--teacher-path  /path/to/pretrained/model \
--teacher-model deit_tiny_patch16_224 --distillation-type soft \
--prune_mode attn --final_finetune 30 --decay-epochs 6 \
--batch-size 256 --lr 1e-4 --weight-decay 0.001 --distillation-alpha 0.2 --distillation-tau 20 \
--output_dir /path/to/save
```

DeiT-Small
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env \
finetune.py --model deit_small_patch16_224 --batch-size 256 --data-path /path/to/imagenet \
--resume /path/to/pruned/model \
--teacher-path  /path/to/pretrained/model \
--teacher-model deit_small_patch16_224 --distillation-type soft \
--prune_mode attn --final_finetune 30 --decay-epochs 6 \
--batch-size 256 --lr 1e-4 --weight-decay 0.001 --distillation-alpha 0.5 --distillation-tau 20 \
--output_dir /path/to/save
```

# Acknowledge
1. https://github.com/facebookresearch/deit
2. https://github.com/youweiliang/evit
