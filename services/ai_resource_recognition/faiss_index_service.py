import os
import faiss
import torch
import numpy as np
from typing import List, Tuple, Dict
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='FAISS检索系统参数')
    parser.add_argument('--embedding-path', type=str, 
                       default="",
                       required=True,
                       help='index embedding路径')
    parser.add_argument('--label-path', type=str,
                       default="",
                       required=True,
                       help='index标签路径')
    parser.add_argument('--test-embedding-path', type=str,
                       default="",
                       required=False,
                       help='test embedding路径')
    parser.add_argument('--index-path', type=str, 
                       default="",
                       required=False,
                       help='index保存/加载路径')
    parser.add_argument('--index-type', type=str,
                       default="FlatIP",
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
        
        # 初始化属性（后续动态赋值）
        self.embeddings = None
        self.dim = None
        self.num_embeddings = None
        self.resources = self._init_gpu_resources()
        self.index = None  # 索引将在新建或加载时赋值
        
        # 若提供了embedding路径，则构建新索引
        if self.embedding_path and os.path.exists(self.embedding_path):
            self._build_new_index()
        elif self.embedding_path:
            raise FileNotFoundError(f"embedding文件不存在: {self.embedding_path}")

    def _load_labels(self) -> List[str]:
        """加载标签文件（新建和加载索引都需要）"""
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"label文件不存在: {self.label_path}")
        with open(self.label_path, "r", encoding="utf-8") as f:
            # label的格式为：xx_实体名称_实体ID xx部分可能缺失
            return [line.strip().split("/")[-3] for line in f.readlines()]  # line is filepath indeed.
        
    def _load_extra_labels(self, extra_label_path: str) -> List[str]:
        """加载标签文件（仅动态增加索引时使用）"""
        if not os.path.exists(extra_label_path):
            raise FileNotFoundError(f"label文件不存在: {extra_label_path}")
        with open(extra_label_path, "r", encoding="utf-8") as f:
            # label的格式为：xx_实体名称_实体ID xx部分可能缺失
            return [line.strip().split("/")[-3] for line in f.readlines()]  # line is filepath indeed.
        
    def _build_new_index(self):
        """新建索引流程（仅当提供embedding_path时调用）"""
        # 加载embedding并校验
        self.embeddings = self._load_embeddings()
        self.dim = self.embeddings.shape[1]
        self.num_embeddings = self.embeddings.shape[0]
        
        # 校验embedding与label数量匹配
        if len(self.embeddings) != len(self.labels):
            raise ValueError(f"embedding数量({len(self.embeddings)})与label数量({len(self.labels)})不匹配")

        d = self.embeddings.shape[1]  # 向量维度
        if self.index_type == "FlatL2":
            cpu_index = faiss.IndexFlatL2(d)  # 其他类型索引同理（如 IVF 等）
        elif self.index_type == "FlatIP":
            cpu_index = faiss.IndexFlatIP(d)
        elif self.index_type == "HNSWFlat":
            raise ValueError("Not support HNSWFlat")
        else:
            raise ValueError(f"Only support FlatL2 and FlatIP, but got {self.index_type}")

        # Multi-GPU
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True                  # 在多卡间做 shard
        co.useFloat16 = False            # 是否用半精度加速
        co.verbose = True                # 打开内部日志，方便调试

        print("*****Start build index*****")
        self.index = faiss.index_cpu_to_all_gpus(cpu_index, co)
        self.index.add(self.embeddings)
        
        print(f"*****Finish build index, size={self.num_embeddings}，dimension={self.dim}，gpus={faiss.get_num_gpus()}*****")

    def _load_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """加载embedding（仅新建索引时使用）"""
        embedding_tensor = torch.load(self.embedding_path, map_location="cpu")
        if not isinstance(embedding_tensor, torch.Tensor):
            raise TypeError("embedding.pth必须存储torch.Tensor")
        return embedding_tensor.numpy().astype(np.float32)
    
    def _load_extra_embeddings(self, extra_embedding_path: str) -> Tuple[np.ndarray, List[str]]:
        """加载 extra embedding（仅动态增加索引时使用）"""
        embedding_tensor = torch.load(extra_embedding_path, map_location="cpu")
        if not isinstance(embedding_tensor, torch.Tensor):
            raise TypeError("embedding.pth必须存储torch.Tensor")
        return embedding_tensor.numpy().astype(np.float32)

    def _init_gpu_resources(self) -> faiss.StandardGpuResources:
        """初始化GPU资源"""
        res = faiss.StandardGpuResources()
        # 可选：设置GPU内存池大小（根据需求调整）
        # res.setTempMemory(1024 * 1024 * 1024)  # 1GB临时内存
        return res

    def search(self, query_embedding: np.ndarray, k: int = 10, gt_label=None) -> Dict:
        """
        搜索与查询向量最相似的k个embedding，返回带label的结果
        
        参数:
            query_embedding: 查询向量，shape为[1, dim]或[dim]的numpy数组（float32）
            k: 召回数量
        
        返回:
            结果列表，每个元素为字典，包含"label_id"、"label"、"distance"
        """
        # 校验输入格式
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)  # 转为[1, dim]
        if query_embedding.shape[1] != self.dim:
            raise ValueError(f"查询向量维度不匹配：输入{query_embedding.shape[1]}，预期{self.dim}")
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)  # 转为float32
        
        # 执行搜索（多GPU并行）
        distances, indices = self.index.search(query_embedding, k)
        
        # 解析结果（indices和distances均为shape[1, k]）
        results = {"retrieval": [], "gt_label": gt_label}
        for i in range(k):
            label_id = indices[0][i]
            distance = distances[0][i]
            label = self.label_id_to_label.get(label_id, "未知标签")  # 映射label
            results["retrieval"].append({
                "label_id": str(label_id),
                "label": label,
                "distance": str(float(distance))
            })
        return results

    def batch_search(self, query_embeddings: np.ndarray, k: int = 10, gt_labels=None) -> List[Dict]:
        """批量搜索（支持多个查询向量）"""
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        distances, indices = self.index.search(query_embeddings, k)
        
        batch_results = []
        if gt_labels:
            assert len(gt_labels) == query_embeddings.shape[0], "number of queries and labels mismatch"
        for i in range(query_embeddings.shape[0]):
            single_results = {"retrieval": [], "gt_label": None}
            if gt_labels:
                single_results["gt_label"] = gt_labels[i]
            for j in range(k):
                label_id = indices[i][j]
                distance = distances[i][j]
                label = self.label_id_to_label.get(label_id, "未知标签")
                single_results["retrieval"].append({
                    "label_id": str(label_id),
                    "label": label,
                    "distance": str(round(float(distance), 4))
                })
            batch_results.append(single_results)
        return batch_results

    def save_index(self, save_path: str):
        """保存索引到磁盘（仅支持CPU索引，需先迁移回CPU）"""
        cpu_index = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(cpu_index, save_path)
        print(f"索引已保存到：{save_path}")

    @classmethod
    def load_index(cls, index_type: str, load_path: str, label_path: str) -> "FaissIndexGPU":
        """从磁盘加载索引（修复版）"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"索引文件不存在: {load_path}")
        
        # 1. 读取CPU索引获取元信息
        cpu_index = faiss.read_index(load_path)
        dim = cpu_index.d
        num_embeddings = cpu_index.ntotal
        
        # 2. 初始化实例（不加载embedding）
        instance = cls(
            embedding_path="",  # 不提供embedding路径，避免构建新索引
            label_path=label_path,
            index_type=index_type
        )
        
        # 3. 补充实例属性
        instance.dim = dim
        instance.num_embeddings = num_embeddings
        instance.index = faiss.index_cpu_to_all_gpus(cpu_index)
        
        print(f"索引加载完成：{num_embeddings}个向量，维度{dim}，使用{faiss.get_num_gpus()}个GPU")
        return instance


    def add_index(self, embedding_path: str, label_path: str):
        """
        增量添加新的embedding和label到现有索引
        参数:
            embedding_path: 新增embedding文件路径
            label_path: 新增label文件路径
        """
        if self.index is None:
            raise RuntimeError("索引尚未初始化，请先创建或加载索引")
        
        # 加载新的embedding和label
        new_embeddings = self._load_extra_embeddings(embedding_path)
        new_labels = self._load_extra_labels(label_path)
        
        # 验证维度匹配
        if new_embeddings.shape[1] != self.dim:
            raise ValueError(f"新embedding维度({new_embeddings.shape[1]})与现有索引维度({self.dim})不匹配")
        
        # 验证数量匹配
        if len(new_embeddings) != len(new_labels):
            raise ValueError(f"新embedding数量({len(new_embeddings)})与新label数量({len(new_labels)})不匹配")
        
        # 添加新embedding到索引
        self.index.add(new_embeddings)
        
        # 记录当前labels长度作为新label_id的起始
        start_id = len(self.labels)
        
        # 更新labels列表和label_id映射
        self.labels.extend(new_labels)
        for i, label in enumerate(new_labels):
            self.label_id_to_label[start_id + i] = label
        
        print(f"成功添加{len(new_embeddings)}个向量到索引，当前总向量数: {self.index.ntotal}")


# 示例用法
if __name__ == "__main__":
    """
    python3 dinov2/faiss_retrieval.py \
    --embedding-path /path/to/train_embeddings_step_12500.pth \
    --label-path /path/to/train_labels.txt \
    --test-embedding-path /path/to/val_embeddings_step_12500.pth \
    --index-type FlatIP
    """
    args = parse_args()
    embedding_fp = args.embedding_path
    label_fp = args.label_path
    test_embedding_path = args.test_embedding_path

    index_fp = args.index_path
    index_type = args.index_type


    if index_fp and os.path.exists(index_fp):
        print(f"Load index from saved file {index_fp}")
        faiss_index = FaissIndexGPU.load_index(
            index_type=index_type,
            load_path=index_fp,
            label_path=label_fp
        )
    else:
        print("Init index")
        faiss_index = FaissIndexGPU(
            embedding_path=embedding_fp,
            label_path=label_fp,
            index_type=index_type,
        )
        if index_fp:
            faiss_index.save_index(index_fp)

    # 随机生成查询向量
    test_embeddings = torch.load(test_embedding_path).numpy().astype(np.float32)

    BATCH_SIZE = 2

    # batch inference
    batch_queries = test_embeddings[0:BATCH_SIZE]
    batch_results = faiss_index.batch_search(batch_queries)
    print(batch_results)

    # single inference
    single_result = faiss_index.search(test_embeddings[0])
    print(single_result)
