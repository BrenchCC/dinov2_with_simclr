#!/bin/bash

# pip3.9 install "numpy<2"
# pip3.9 install torch==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
# pip3.9 install torchvision==0.16.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
# pip3.9 install faiss-gpu --extra-index-url https://download.pytorch.org/whl/cu121
# pip3.9 install tqdm
# pip3.9 install pyarrow
# pip3.9 install pillow
# pip3.9 install omegaconf
# pip3.9 install torchmetrics
# pip3.9 install bytedtos
# pip3.9 install volcengine-python-sdk[ark]
# pip3.9 install byted_overpass_toutiao_article_article
# pip3.9 install imagehash
# pip3.9 install opencv-python==4.9.0
# pip3.9 install bytedeuler==2.6.4 -i https://bytedpypi.byted.org/simple

pip3.9 install -r requirements.txt

IMAGE_PATHS="hdfs://haruna/home/byte_toutiao_client_ai/toutiao_playlet_materials/20250730/frames_data/7216247515274808355_狂野小神医/7523576732541501492_狂野小神医_frames/frame_0000.jpg"

GID="7516129515760864553"

python3.9 ai_resource_recognition/main.py \
    --dino-model-type dinov2 \
    --dino-config-file /mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/sft_config/dinov2_yaml/sj_vitg14_gs224_pretrain_toutiao_playlet_v1.yaml \
    --dino-pretrained-weight /mnt/bn/yuanliang-llm-gpu-train/model_ckpts/sft_ckpts/dinov2_gs224_with_pretrain_toutiao_playlet_v1/eval/training_249999/teacher_checkpoint.pth \
    --faiss-embedding-path /mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/evaluation/image_recognition_v2/dinov2/dinov2_gs224_with_pretrain_toutiao_playlet_v1/dinov2/toutiao_playlet/train_index_embeddings_step_249999.pth \
    --faiss-label-path /mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/evaluation/image_recognition_v2/dinov2/dinov2_gs224_with_pretrain_toutiao_playlet_v1/dinov2/toutiao_playlet/train_labels.txt \
    --faiss-index-type FlatIP \
    --doubao-model-ep ep-20250811202750-kzd92 \
    --doubao-system-prompt-fp "./ai_resource_recognition/prompts/resource_recognition_sp.md" \
    --video-gid $GID \
    --title2id-fp "/mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/batch_videos/toutiao_playlet_data/title2id.json" \
    --id2title-fp "/mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/batch_videos/toutiao_playlet_data/id2title.json" \
    --freq-thresh 10 \
    --score-thresh 0 \
    --num-frame 48 \
    --doubao-num-images 3
    # --image-paths $IMAGE_PATHS \