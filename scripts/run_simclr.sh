#!/bin/bash
export PYTHONPATH=/mnt/bn/mllm_playground/dinov2

torchrun \
    dinov2/dinov2/train/train_simclr.py \
    --config-file /mnt/bn/eason-hl/yuanliang/datasets/mllm_datasets/sj_mllm_application/sft_config/dinov2_yaml/sj_vitg14_global_size_518_local_size_98.yaml \
    --output-dir /mnt/bn/eason-hl/yuanliang/model_ckpts/sft_ckpts/simclr \
    --root /mnt/bn/eason-hl/yuanliang/datasets/mllm_datasets/sj_mllm_application/batch_videos/24_25_dataset/simclr/ \
    --pretrain_backbone_weight /mnt/bn/eason-hl/yuanliang/model_ckpts/sft_ckpts/teacher_checkpoint_124999.pth \
    --hdfs_loading True \
    --b 256 \
    --crop_size 98 \
    --epochs 100
