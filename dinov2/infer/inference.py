import os
import io
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pyarrow.fs import FileSystem
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import ModelWithNormalize
import logging


# 初始化分布式环境
def init_distributed(description):
    args_parser = get_setup_args_parser(description=description)
    args = args_parser.parse_args()
    # 从环境变量获取 local_rank（适配 torchrun）
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print("Local:", local_rank, world_size)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    return local_rank, world_size, args


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
            with self.hdfs_client.open_input_file(image_path) as f:
                image_data = f.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        return self.transform(image), image_path


def main():
    # 初始化分布式
    description = "DINOv2 k-NN evaluation"
    rank, world_size, args = init_distributed(description=description)
    # 初始化日志
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    logger = logging.getLogger(__name__)

    # 模型加载
    model, autocast_dtype = setup_and_build_model(args)
    model = ModelWithNormalize(model)
    model = DDP(
        model.cuda(),
        device_ids=[rank],
        output_device=rank
    ).eval()

    # 数据加载
    train_fp = "/mnt/bn/yuanliang-llm-gpu-train/datasets/mllm_datasets/sj_mllm_application/batch_videos/24_25_dataset/val/frame_hdfs_path.txt"
    with open(train_fp, "r") as f:
        filepaths = [l.strip() for l in f]

    # 添加开始日志
    if rank == 0:
        logger.info("开始分布式推理任务")
        logger.info(f"总样本数: {len(filepaths)}")
        logger.info(f"使用GPU数量: {world_size}")

    dataset = HDFSDataset(filepaths, make_classification_eval_transform())
    sampler = DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler,
                        num_workers=4, pin_memory=True)
    # 分布式推理
    embeddings = []
    filenames = []
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=autocast_dtype):
        for batch_imgs, image_paths in tqdm(loader, disable=rank != 0):
            batch_imgs = batch_imgs.cuda()
            emb = model(batch_imgs)
            # embeddings.append(emb.cpu())
            embeddings.append(emb)
            filenames.extend(image_paths)

    # 收集所有结果
    # embeddings = torch.cat(embeddings).cuda()
    gathered = [torch.zeros_like(torch.cat(embeddings)) for _ in range(world_size)]
    dist.all_gather(gathered, torch.cat(embeddings))
    final_emb = torch.cat(gathered).cpu()
    gathered_filenames = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_filenames, filenames)
    final_filenames = sum(gathered_filenames, [])  # flatten

    if rank == 0:
        logger.info("推理完成，开始收集结果")
        logger.info(f"最终嵌入矩阵形状: {final_emb.shape}")
        logger.info("保存结果到 embeddings.pth")
        torch.save(final_emb, "embeddings.pth")
        with open("filename.txt", "w") as f:
            for fn in final_filenames:
                f.write(fn + "\n")
        logger.info("保存完成：embeddings.pth + filename.txt")


if __name__ == "__main__":
    main()
