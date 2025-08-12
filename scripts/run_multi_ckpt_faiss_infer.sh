#!/bin/bash

export PYTHONPATH=/opt/tiger/mllm_playground/dinov2
# pip3.9 install "numpy<2" 
# pip3.9 install torch==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
# pip3.9 install faiss-gpu --extra-index-url https://download.pytorch.org/whl/cu121
# pip3.9 install tqdm

# 定义枚举的目录数组（请替换为实际目录名）
dirs=(
    "dinov2_vitg14_epoch100_gs224_ls98_lr2e4_24_25_entries" 
    "dinov2_vitg14_epoch100_gs224_ls98_reg4_lr2e4_24_25_entries" 
    "dinov2_vitg14_epoch100_bz1024_lr2e4_gcz_518_lcz_98_24_25_entries")

# 定义枚举的step值数组
steps=("12499" "24999" "37499" "49999" "62499" "74999" "87499" "99999" "112499" "124999")

# 外层循环：遍历目录
for dir in "${dirs[@]}"; do
    # 内层循环：遍历step
    for step in "${steps[@]}"; do
        echo "########start processing ${dir} ${step}#######"
        output_path=/mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/evaluation/image_recognition_v2/dinov2/${dir}/dinov2/val_infer_ip_step_${step}.jsonl

        if [ -f "$output_path" ]; then
            echo "输出文件已存在，跳过: $output_path"
            continue  # 跳出当前mode循环，进入下一个迭代
        fi
        python3.9 dinov2/dinov2/infer/faiss_retrieval.py \
        --embedding-path /mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/evaluation/image_recognition_v2/dinov2/${dir}/dinov2/train_embeddings_step_${step}.pth \
        --label-path /mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/evaluation/image_recognition_v2/dinov2/${dir}/dinov2/train_labels.txt \
        --test-embedding-path /mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/evaluation/image_recognition_v2/dinov2/${dir}/dinov2/val_embeddings_step_${step}.pth \
        --test-labels-path /mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/evaluation/image_recognition_v2/dinov2/${dir}/dinov2/val_labels.txt \
        --output-path $output_path \
        --index-type FlatIP
    done
done