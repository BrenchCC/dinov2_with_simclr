import os
import json
import faiss
import torch
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='FAISS检索系统参数')
    parser.add_argument('--embedding-path', type=str,
                        default="",
                        help='训练集embedding路径')
    parser.add_argument('--label-path', type=str,
                        default="",
                        help='训练集标签路径')
    parser.add_argument('--test-embedding-path', type=str,
                        default="",
                        help='测试集embedding路径')
    parser.add_argument('--test-labels-path', type=str,
                        default="",
                        help='测试集标签路径')
    parser.add_argument('--output-path', type=str,
                        default="",
                        help='输出结果路径')
    parser.add_argument('--index-path', type=str,
                        default="",
                        help='索引保存/加载路径')
    parser.add_argument('--index-type', type=str,
                        default="FlatL2",
                        choices=["FlatL2", "FlatIP"],
                        help='索引类型: FlatL2 (L2距离) 或 FlatIP (内积)')
    args = parser.parse_args()
    return args


class FaissIndexGPU:
    def __init__(self,
                 embedding_path: str,
                 label_path: str,
                 index_type: str = "FlatIP",
                 m: int = 32,  # HNSW连接数
                 ef_construction: int = 64,  # 构建时的精度参数
                 ef_search: int = 32  # 搜索时的精度参数
                 ):
        """
        初始化多GPU Faiss索引，加载embedding和label并构建索引

        参数:
            embedding_path: embedding.pth路径，存储shape为[N, dim]的torch.Tensor
            label_path: label_path.txt路径，每行对应一个embedding的label
            index_type: 索引类型，目前支持"FlatIP", "FlatL2"
            m: HNSW的连接数，影响索引质量和速度
            ef_construction: 构建索引时的ef参数（越大越准，速度越慢）
            ef_search: 搜索时的ef参数（越大越准，速度越慢）
        """
        self.embedding_path = embedding_path
        self.label_path = label_path
        self.index_type = index_type
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # 加载标签（新建和加载索引都需要）
        self.labels = self._load_labels()
        self.label_id_to_label = {i: self.labels[i] for i in range(len(self.labels))}

        if self.embedding_path and os.path.exists(self.embedding_path):
            self._build_new_index()
        elif self.embedding_path:
            raise FileNotFoundError(f"embedding文件不存在: {self.embedding_path}")

    def _load_labels(self) -> List[str]:
        """加载标签文件（新建和加载索引都需要）"""
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"label文件不存在: {self.label_path}")