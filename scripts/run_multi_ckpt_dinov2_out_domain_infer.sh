#!/bin/bash

export PYTHONPATH=/opt/tiger/mllm_playground/dinov2

MASTER_PORT=${ARNOLD_WORKER_0_PORT}
BASE_INFER_DIR="/mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/evaluation/image_recognition_v2/dinov2"

# 定义成对出现的参数：配置文件路径和预训练权重路径前缀
# 格式："config_path_prefix|pretrained_weights_prefix"
param_pairs=(
    "sj_vitg14_gs_224_ls_98_reg4|dinov2_vitg14_epoch100_gs224_ls98_reg4_lr2e4"
    "sj_vitg14_gs_224_ls_98|dinov2_vitg14_epoch100_gs224_ls98_lr2e4"
    "sj_vitg14_global_size_518_local_size_98|dinov2_vitg14_epoch100_bz1024_lr2e4_gcz_518_lcz_98"
)

# 定义需要加载的训练步骤列表
steps=("124999")
# 定义数据集类型列表
out_domain_dir="24_25_dataset_with_video_part2"
fnames=("24_25_val" "index_frame" "videos_part2_val")

model_type=("dinov2")

# 最外层循环：遍历成对参数
for pair in "${param_pairs[@]}"; do
    # 拆分配置文件前缀和预训练权重前缀
    IFS="|" read -r config_prefix weights_prefix <<< "$pair"
    
    # 遍历模型类型
    for model in "${model_type[@]}"; do
        base_output_dir="${BASE_INFER_DIR}/${weights_prefix}_24_25_entries/${model}/${out_domain_dir}"
        
        # 检查路径是否存在，不存在则创建（-p确保父目录也会被创建）
        if [ ! -d "$base_output_dir" ]; then
            echo "创建目录: $base_output_dir"
            mkdir -p "$base_output_dir"
        fi

        # 遍历每个训练步骤
        for step in "${steps[@]}"; do
            # 遍历 train 和 val 数据集
            for fname in "${fnames[@]}"; do
                emb_out_file="${base_output_dir}/${fname}_embeddings_step_${step}.pth"
                # 检查输出文件是否存在，存在则跳过当前循环
                if [ -f "$emb_out_file" ]; then
                    echo "输出文件已存在，跳过: $emb_out_file"
                    continue  # 跳出当前fname循环，进入下一个迭代
                fi
                torchrun \
                    --nproc_per_node=8 \
                    --master_port=$MASTER_PORT \
                    dinov2/dinov2/infer/inference.py \
                    --config-file "/mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/sft_config/dinov2_yaml/${config_prefix}.yaml" \
                    --pretrained-weights "/mnt/bn/yuanliang-llm-gpu-train/model_ckpts/sft_ckpts/${weights_prefix}/eval/training_${step}/teacher_checkpoint.pth" \
                    --images_fp "/mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/batch_videos/${out_domain_dir}/${fname}_frame_hdfs_path.txt" \
                    --emb_out_fp $emb_out_file \
                    --label_out_fp "${base_output_dir}/${fname}_labels.txt" \
                    --model_type $model
            done
        done
    done
done