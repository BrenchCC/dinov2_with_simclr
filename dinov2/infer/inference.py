import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os

import io

import torch
import torch.nn.functional as F
from pyarrow.fs import FileSystem
from PIL import Image
from tqdm import tqdm
import numpy as np

from dinov2.data.transforms import make_classification_eval_transform
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model, get_autocast_dtype
from dinov2.eval.utils import ModelWithNormalize
from dinov2.models.vision_transformer import vit_giant2_with_mlp
from dinov2.utils.config import setup_for_simclr
from dinov2.utils.config import default_setup

import logging

# 初始化分布式环境
def parse_args(description):
    args_parser = get_setup_args_parser(description=description)
    args_parser.add_argument('--images_fp', type=str, default="", help='path to file with all image filepaths')
    args_parser.add_argument('--emb_out_fp', type=str, default='embeddings.pth', help='path to embedding.pth')
    args_parser.add_argument('--label_out_fp', type=str, default='labels.txt', help='path to label.pth')
    args_parser.add_argument('--model_type', type=str, default='dinov2', choices=['dinov2', 'dinov2_mlp'], help='type of model')
    args_parser.add_argument('--out_dim', type=int, default=128, help='output dimension of the model')
    
    
    args = args_parser.parse_args()
    # # 从环境变量获取 local_rank（适配 torchrun）
    # local_rank = int(os.environ["LOCAL_RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    # torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl', init_method='env://')
    # return local_rank, world_size, args
    return args

class HDFSDataset(Dataset):
    def __init__(self, filepaths, transform):
        self.filepaths = filepaths
        self.transform = transform
        self.hdfs_client, _ = FileSystem.from_uri('hdfs://haruna/home/')

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image_path = self.filepaths[idx]
        if image_path.startswith("hdfs"):
            try:
                with self.hdfs_client.open_input_file(image_path) as f:
                    image_data = f.read()
            except:
                print(f"Error loading image {image_path}, use the first image instead")
                image_path = self.filepaths[0]
                with self.hdfs_client.open_input_file(image_path) as f:
                    image_data = f.read()
            try:
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            except Exception as e:
                print(f"Load image error {e}")
                random_array = np.random.randint(0, 256, size=(720, 960, 3), dtype=np.uint8)
                random_image = Image.fromarray(random_array, mode='RGB')
                return self.transform(random_image), "NULL"
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except:
                print(f"Error loading image {image_path}, use the first image instead")
                image_path = self.filepaths[0]
                image = Image.open(image_path).convert('RGB')
        return self.transform(image), image_path

def main():
    # 初始化分布式
    description = "DINOv2 k-NN evaluation"
    # rank, world_size, args = init_distributed(description=description)
    args = parse_args(description=description)
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 初始化日志
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    logger = logging.getLogger(__name__)

    # 模型加载
    if args.model_type == "dinov2":
        print("Loading dinov2 model")
        model, autocast_dtype = setup_and_build_model(args)  #  Here execute dist.init_process_group()
        cfg = setup_for_simclr(args)
    elif args.model_type == "dinov2_mlp":
        default_setup(args)  # ddp env initializing
        print("Loading dinov2_mlp model")
        cfg = setup_for_simclr(args)
        vit_config = cfg.student
        vit_kwargs = dict(
            img_size=cfg.crops.global_crops_size,
            patch_size=vit_config.patch_size,
            init_values=vit_config.layerscale,
            ffn_layer=vit_config.ffn_layer,
            block_chunks=vit_config.block_chunks,
            qkv_bias=vit_config.qkv_bias,
            proj_bias=vit_config.proj_bias,
            ffn_bias=vit_config.ffn_bias,
            num_register_tokens=vit_config.num_register_tokens,
            interpolate_offset=vit_config.interpolate_offset,
            interpolate_antialias=vit_config.interpolate_antialias,
        )
        mlp_kwargs = dict(
            out_dim=args.out_dim
        )
        model = vit_giant2_with_mlp(**vit_kwargs, **mlp_kwargs)
        model.load_state_dict(torch.load(args.pretrained_weights, map_location="cpu")["state_dict"], strict=False)
        autocast_dtype = get_autocast_dtype(cfg)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    model = ModelWithNormalize(model)
    model = DDP(
        model.cuda(),
        device_ids=[rank],
        output_device=rank
    ).eval()
    
    # 数据加载
    images_path_fp = args.images_fp
    with open(images_path_fp, "r") as f:
        filepaths = [l.strip() for l in f]
    
    # 添加开始日志
    if rank == 0:
        logger.info("开始分布式推理任务")
        logger.info(f"总样本数: {len(filepaths)}")
        logger.info(f"使用GPU数量: {world_size}")
    # change image_resize to global size. default is 224
    dataset = HDFSDataset(filepaths, make_classification_eval_transform(resize_size=cfg.crops.global_crops_size, crop_size=cfg.crops.global_crops_size))
    sampler = DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler, 
                        num_workers=4, pin_memory=True)

    # 分布式推理（保持原修改，确保embedding实时转移到CPU）
    embeddings = []
    filenames = []
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=autocast_dtype):
        for batch_imgs, image_paths in tqdm(loader, disable=rank!=0):
            batch_imgs = batch_imgs.cuda()
            emb = model(batch_imgs)
            embeddings.append(emb.cpu())  # 实时转移到CPU，释放GPU内存
            filenames.extend(image_paths)
    
    # 收集所有结果（修改为：本地落盘 → 主进程合并，避免GPU内存峰值）
    local_emb = torch.cat(embeddings)
    local_filename = f"local_emb_{rank}.pth"
    local_label_filename = f"local_labels_{rank}.txt"
    
    # 1. 每个进程将本地数据保存到临时文件（CPU上操作，无GPU占用）
    torch.save(local_emb, local_filename)
    with open(local_label_filename, "w") as f:
        for fn in filenames:
            f.write(fn + "\n")
    
    # 2. 等待所有进程完成本地落盘
    dist.barrier()
    
    # 3. 仅主进程（rank=0）合并所有临时文件
    if rank == 0:
        logger.info("推理完成，开始合并结果")
        # 合并embedding
        final_emb = []
        for i in range(world_size):
            final_emb.append(torch.load(f"local_emb_{i}.pth"))
        final_emb = torch.cat(final_emb)
        
        # 合并文件名
        final_filenames = []
        for i in range(world_size):
            with open(f"local_labels_{i}.txt", "r") as f:
                final_filenames.extend([l.strip() for l in f])
        
        # 保存最终结果
        logger.info(f"最终嵌入矩阵形状: {final_emb.shape}")
        logger.info(f"保存结果到 {args.emb_out_fp}")
        torch.save(final_emb, args.emb_out_fp)
        with open(args.label_out_fp, "w") as f:
            for fn in final_filenames:
                f.write(fn + "\n")
        logger.info(f"保存完成：{args.emb_out_fp} & {args.label_out_fp}")
        
        # 清理临时文件
        for i in range(world_size):
            os.remove(f"local_emb_{i}.pth")
            os.remove(f"local_labels_{i}.txt")


if __name__ == "__main__":
    """
    launch command:
    torchrun \
        --nproc_per_node=8 \
        --master_port=$MASTER_PORT \
        dinov2/infer.py \
        --config-file "/mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/sft_config/dinov2_yaml/sj_vitg14_global_size_518_local_size_98.yaml" \
        --pretrained-weights "/mnt/bn/yuanliang-llm-gpu-train/model_ckpts/DinoV2/dinov2_vitg14_pretrain.pth" \
        --images_fp "/mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/batch_videos/24_25_dataset/train/frame_hdfs_path.txt" \
        --emb_out_fp "./embeddings.pth" \
        --label_out_fp "./labels.txt"
    """
    main()
