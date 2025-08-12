#!/bin/bash

export PYTHONPATH=/opt/tiger/mllm_playground/dinov2
pip3.9 install "numpy<2" 
pip3.9 install torch==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip3.9 install faiss-gpu --extra-index-url https://download.pytorch.org/whl/cu121
pip3.9 install tqdm


python3.9 dinov2/dinov2/infer/faiss_retrieval.py \
--embedding-path /mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/evaluation/image_recognition_v2/dinov2/dinov2_vitg14_gcz_518_lcz_98_epoch100_24_25_entries/train_embeddings_step_99999.pth \
--label-path /mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/evaluation/image_recognition_v2/dinov2/dinov2_vitg14_gcz_518_lcz_98_epoch100_24_25_entries/train_labels.txt \
--test-embedding-path /mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/evaluation/image_recognition_v2/dinov2/dinov2_vitg14_gcz_518_lcz_98_epoch100_24_25_entries/val_embeddings_step_99999.pth \
--test-labels-path /mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/evaluation/image_recognition_v2/dinov2/dinov2_vitg14_gcz_518_lcz_98_epoch100_24_25_entries/val_labels.txt \
--output-path /mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/evaluation/image_recognition_v2/dinov2/dinov2_vitg14_gcz_518_lcz_98_epoch100_24_25_entries/val_infer_ip_step99999.jsonl \
--index-type FlatIP