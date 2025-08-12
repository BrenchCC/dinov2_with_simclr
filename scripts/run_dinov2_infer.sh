#!/bin/bash

export PYTHONPATH=/opt/tiger/mllm_playground/dinov2

MASTER_PORT=${ARNOLD_WORKER_0_PORT}

# 定义需要加载的训练步骤列表
steps=("124999")
# 定义数据集类型列表
modes=("train" "val")

# 遍历每个训练步骤
for step in "${steps[@]}"; do
    # 遍历 train 和 val 数据集
    for mode in "${modes[@]}"; do
        torchrun \
            --nproc_per_node=8 \
            --master_port=$MASTER_PORT \
            dinov2/dinov2/infer/inference.py \
            --config-file "/mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/sft_config/dinov2_yaml/sj_vitg14_global_size_518_local_size_98.yaml" \
            --pretrained-weights "/mnt/bn/yuanliang-llm-gpu-train/model_ckpts/sft_ckpts/dinov2_vitg14_epoch100_bz1024_lr2e4_gcz_518_lcz_98/eval/training_124999/simclr_checkpoint_0100.pth" \
            --images_fp "/mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/batch_videos/24_25_dataset_with_video_part2/index_frames.txt" \
            --emb_out_fp "/mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/evaluation/image_recognition_v2/dinov2/dinov2_vitg14_gcz_518_lcz_98_epoch100_24_25_entries/dinov2_mlp/24_25_dataset_with_video_part2_embeddings_step_${step}.pth" \
            --label_out_fp "/mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/evaluation/image_recognition_v2/dinov2/dinov2_vitg14_gcz_518_lcz_98_epoch100_24_25_entries/dinov2_mlp/24_25_dataset_with_video_part2_labels.txt" \
            --model_type dinov2_mlp
    done
done