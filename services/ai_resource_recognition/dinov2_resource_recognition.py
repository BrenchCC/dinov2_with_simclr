import argparse
import torch
import torch.nn.functional as F
from pyarrow.fs import FileSystem
from PIL import Image
import io
import numpy as np
from tqdm import tqdm
import os

from utils.transforms import make_classification_eval_transform
from models.setup import get_args_parser as get_setup_args_parser
from models.setup import setup_and_build_model, get_autocast_dtype
from models.eval.utils import ModelWithNormalize
from models.vision_transformer import vit_giant2_with_mlp
from models.utils.config import setup_for_simclr
from models.utils.config import default_setup


def parse_args():
    parser = argparse.ArgumentParser(description="dinvov2 inference")
    
    # 必须参数
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["dinov2", "dinov2_mlp"], 
        help="模型类型 (必须): 'dinov2' 或 'dinov2_mlp'"
    )
    parser.add_argument(
        "--cfg_file", type=str, required=True, help="配置文件路径 (必须)"
    )
    parser.add_argument(
        "--pretrain_weight", type=str, required=True, help="预训练权重路径 (必须)"
    )
    
    # 可选参数
    parser.add_argument(
        "--out_dim", type=int, default=128, help="输出维度 (可选，默认: 128)"
    )
    
    # 图像路径参数（支持多图逗号分隔）
    parser.add_argument(
        "--image_path", type=str, required=True, 
        help="图像路径，单图或多图(用英文逗号分隔) (必须)"
    )
    
    args = parser.parse_args()
    
    # 处理多图路径
    args.image_paths = args.image_path.split(',')
    args.image_paths = [p.strip() for p in args.image_paths if p.strip()]
    
    # 验证文件存在
    if not os.path.exists(args.cfg_file):
        raise FileNotFoundError(f"config not found: {args.cfg_file}")
    if not os.path.exists(args.pretrain_weight):
        raise FileNotFoundError(f"pretrained weight not found: {args.pretrain_weight}")
    
    return args

class DinoV2Encoder:
    def __init__(self, model_type='dinov2', config_file=None, pretrained_weights=None, out_dim=128, device='cuda'):
        self.model_type = model_type
        self.cfg_file = config_file
        self.pretrained_weights = pretrained_weights
        self.out_dim = out_dim
        self.device = device
        self.model = None
        self.transform = None
        self.autocast_dtype = None
        self.hdfs_client, _ = FileSystem.from_uri('hdfs://haruna/home/')
        self._initialize_model()

    def _initialize_model(self):
        # 解析配置参数
        args_parser = get_setup_args_parser(description='DINOv2 Inference')
        args_parser.add_argument('--config-file', type=str, default=self.cfg_file)
        args_parser.add_argument('--pretrained-weights', type=str, default=self.pretrained_weights)
        args = args_parser.parse_args([])  # 不解析命令行参数

        # 加载模型
        if self.model_type == 'dinov2':
            model, self.autocast_dtype = setup_and_build_model(args)
            self.cfg = setup_for_simclr(args)
        elif self.model_type == 'dinov2_mlp':
            default_setup(args)
            self.cfg = setup_for_simclr(args)
            vit_config = self.cfg.student
            vit_kwargs = dict(
                img_size=self.cfg.crops.global_crops_size,
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
            mlp_kwargs = dict(out_dim=self.out_dim)
            model = vit_giant2_with_mlp(**vit_kwargs, **mlp_kwargs)
            model.load_state_dict(
                torch.load(args.pretrained_weights, map_location='cpu')['state_dict'],
                strict=False
            )
            self.autocast_dtype = get_autocast_dtype(self.cfg)
        else:
            raise ValueError(f'Unknown model type: {self.model_type}')

        # 初始化模型和变换
        self.model = ModelWithNormalize(model).to(self.device)
        self.model.eval()
        self.transform = make_classification_eval_transform(
            resize_size=self.cfg.crops.global_crops_size,
            crop_size=self.cfg.crops.global_crops_size
        )

    def _load_image(self, image_path):
        try:
            if image_path.startswith('hdfs'):
                with self.hdfs_client.open_input_file(image_path) as f:
                    image_data = f.read()
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f'Error loading image {image_path}: {e}')

    def infer_single(self, image_path):
        """推理单张图像并返回embedding"""
        image = self._load_image(image_path)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.autocast_dtype):
            embedding = self.model(image_tensor)
        return embedding.cpu().numpy()

    def infer_batch(self, image_paths, batch_size=32):
        embeddings = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc='Processing batches'):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []

            for path in batch_paths:
                image = self._load_image(path)
                batch_images.append(self.transform(image))

            batch_tensor = torch.stack(batch_images).to(self.device)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.autocast_dtype):
                batch_embeddings = self.model(batch_tensor)
            embeddings.extend(batch_embeddings.cpu().numpy())

        return embeddings


if __name__ == '__main__':
    args = parse_args()

    encoder = DinoV2Encoder(
        model_type=args.model_type,
        config_file=args.cfg_file,
        pretrained_weights=args.pretrain_weight,
        out_dim=args.out_dim
    )

    # single image inference
    if len(args.image_paths) == 1:
        single_emb = encoder.infer_single(args.image_paths[0])
        print(f'Single image embedding shape: {single_emb.shape}')
    # multi-image inference
    else:
        batch_embs = encoder.infer_batch(args.image_paths, batch_size=16)
        print(f'Batch embeddings count: {len(batch_embs)}')
        print(f'First embedding shape: {batch_embs[0].shape}')
