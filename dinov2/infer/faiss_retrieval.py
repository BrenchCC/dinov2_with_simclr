import gc
import os
import re
import json
import faiss
import torch
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict
import argparse

faiss.omp_set_num_threads(8)

def parse_args():
    parser = argparse.ArgumentParser(description='FAISS检索系统参数')
    parser.add_argument('--embedding-path', type=str, 
                       default="",
                       help='训练集embedding路径')
    parser.add_argument('--label-path', type=str,
                       default="",
                       help='训练集标签路径')
    parser.add_argument('--embedding-path-dir', type=str, 
                       default="",
                       help='包含多个embedding.pth和label.txt文件的目录')
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
    parser.add_argument('--gpu-loading', type=int, 
                       default=1,
                       help='是否使用GPU加载索引。True: GPU加载，False: CPU加载')
    args = parser.parse_args()
    return args


class FaissIndexGPU:
    def __init__(self, 
                 embedding_path: str = "", 
                 label_path: str = "", 
                 index_type: str = "FlatIP", 
                 m: int = 32,  # HNSW连接数
                 ef_construction: int = 64,  # 构建时的精度参数
                 ef_search: int = 32,  # 搜索时的精度参数
                 embedding_dir: str = "",  # 新增参数：包含多个embedding和label的目录
                 gpu_loading: bool = True  # 新增参数：是否使用GPU加载索引
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
            embedding_dir: 包含多个embedding.pth和label.txt文件的目录
            gpu_loading: 是否使用GPU加载索引。True: GPU加载，False: CPU加载
        """
        self.embedding_path = embedding_path
        self.label_path = label_path
        self.index_type = index_type
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.embedding_dir = embedding_dir
        self.gpu_loading = gpu_loading
        
        # 初始化属性（后续动态赋值）
        self.embeddings = None
        self.labels = None
        self.label_id_to_label = None
        self.dim = None
        self.num_embeddings = None
        self.resources = self._init_gpu_resources() if self.gpu_loading else None
        self.index = None  # 索引将在新建或加载时赋值
        
        # 如果提供了embedding_dir，则从目录批量加载
        if self.embedding_dir and os.path.exists(self.embedding_dir):
            self._batch_load_from_directory()
        # 若提供了embedding路径，则构建新索引
        elif self.embedding_path and os.path.exists(self.embedding_path):
            self.labels = self._load_labels()
            self.label_id_to_label = {i: self.labels[i] for i in range(len(self.labels))}
            self._build_new_index()
        elif self.embedding_path:
            raise FileNotFoundError(f"embedding文件不存在: {self.embedding_path}")

    def _load_labels(self) -> List[str]:
        """加载标签文件（新建和加载索引都需要）"""
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"label文件不存在: {self.label_path}")
        with open(self.label_path, "r", encoding="utf-8") as f:
            # return [line.strip().split("/")[-3] for line in f.readlines()]  # line is filepath indeed.
            return [line.strip() for line in f.readlines()]
        
    def _batch_load_from_directory(self, batch_size=1000000):
        """从目录批量流式加载embedding和label文件，避免内存峰值过高"""
        all_files = os.listdir(self.embedding_dir)
        embedding_files = [f for f in all_files if f.endswith('.pth')]

        if not embedding_files:
            raise ValueError(f"在目录 {self.embedding_dir} 中未找到.pth文件")

        # 初始化 label 存储
        self.labels = []
        self.label_id_to_label = {}

        index_initialized = False
        dim = None
        n_total = 0

        for embedding_file in tqdm(embedding_files, desc="加载embedding文件"):
            embedding_path = os.path.join(self.embedding_dir, embedding_file)
            label_filename = re.sub(r'index_.*\.pth$', 'labels.txt', embedding_file)
            label_path = os.path.join(self.embedding_dir, label_filename)

            if not os.path.exists(label_path):
                print(f"警告: 未找到与 {embedding_file} 对应的label文件 {label_filename}")
                continue

            try:
                embedding_tensor = torch.load(embedding_path, map_location="cpu")
                if not isinstance(embedding_tensor, torch.Tensor):
                    print(f"警告: {embedding_file} 不是torch.Tensor类型")
                    continue
                # if embedding_tensor.dtype != torch.float32:
                #     embedding_tensor = embedding_tensor.float()
                # embeddings = embedding_tensor.numpy()
                embeddings = embedding_tensor.numpy().astype(np.float32, copy=False)
            except Exception as e:
                print(f"加载 {embedding_file} 时出错: {str(e)}")
                continue

            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    labels = [line.strip() for line in f.readlines()]
                    if len(labels) != embeddings.shape[0]:
                        print(f"警告: {embedding_file} 的label数量({len(labels)})与embedding数量({embeddings.shape[0]})不匹配")
                        continue
            except Exception as e:
                print(f"加载 {label_filename} 时出错: {str(e)}")
                continue

            # 初始化索引
            if not index_initialized:
                dim = embeddings.shape[1]
                if self.index_type == "FlatL2":
                    cpu_index = faiss.IndexFlatL2(dim)
                elif self.index_type == "FlatIP":
                    cpu_index = faiss.IndexFlatIP(dim)
                else:
                    raise ValueError(f"Only support FlatL2 and FlatIP, but got {self.index_type}")

                if self.gpu_loading:
                    co = faiss.GpuMultipleClonerOptions()
                    co.shard = True
                    co.useFloat16 = False
                    co.verbose = True
                    self.index = faiss.index_cpu_to_all_gpus(cpu_index, co)
                else:
                    self.index = cpu_index
                index_initialized = True

            # 分批 add
            for i in range(0, embeddings.shape[0], batch_size):
                batch_emb = np.array(embeddings[i:i+batch_size], dtype=np.float32, copy=False)
                batch_labels = labels[i:i+batch_size]

                self.index.add(batch_emb)
                self.labels.extend(batch_labels)

                # 建立 label_id → label 的映射
                for j, lab in enumerate(batch_labels):
                    self.label_id_to_label[n_total + j] = lab
                n_total += len(batch_labels)

                # 释放内存
                del batch_emb, batch_labels
                gc.collect()

            del embeddings, embedding_tensor, labels
            gc.collect()

        if not index_initialized:
            raise ValueError(f"在目录 {self.embedding_dir} 中未能加载任何有效的embedding文件")

        self.dim = dim
        self.num_embeddings = n_total
        print(f"新索引构建完成：{self.num_embeddings}个向量，维度{self.dim}，"
            f"使用{'GPU' if self.gpu_loading else 'CPU'}")
    
    def _build_index_from_embeddings(self):
        """从已加载的embeddings构建索引"""
        d = self.embeddings.shape[1]  # 向量维度
        if self.index_type == "FlatL2":
            cpu_index = faiss.IndexFlatL2(d)  # 其他类型索引同理（如 IVF 等）
        elif self.index_type == "FlatIP":
            cpu_index = faiss.IndexFlatIP(d)
        elif self.index_type == "HNSWFlat":
            raise ValueError("Not support HNSWFlat")
        else:
            raise ValueError(f"Only support FlatL2 and FlatIP, but got {self.index_type}")
    
        # 根据gpu_loading参数决定是使用CPU还是GPU索引
        if self.gpu_loading:
            # Multi-GPU
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True                  # 在多卡间做 shard
            co.useFloat16 = False            # 是否用半精度加速
            co.verbose = True                # 打开内部日志，方便调试
        
            self.index = faiss.index_cpu_to_all_gpus(cpu_index, co)
            self.index.add(self.embeddings)
            del self.embeddings
            print(">>> multi‑GPU index created")
            print(f"新索引构建完成：{self.num_embeddings}个向量，维度{self.dim}，使用{faiss.get_num_gpus()}个GPU")
        else:
            # 使用CPU索引
            self.index = cpu_index
            self.index.add(self.embeddings)
            del self.embeddings
            print(f"新索引构建完成：{self.num_embeddings}个向量，维度{self.dim}，使用CPU")

    @classmethod
    def load_index(cls, index_type: str, load_path: str, label_path: str, gpu_loading: bool = True) -> "FaissIndexGPU":
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
            index_type=index_type,
            gpu_loading=gpu_loading
        )
        
        # 3. 补充实例属性
        instance.dim = dim
        instance.num_embeddings = num_embeddings
        
        # 根据gpu_loading参数决定是使用CPU还是GPU索引
        if gpu_loading:
            instance.index = faiss.index_cpu_to_all_gpus(cpu_index)
            print(f"索引加载完成：{num_embeddings}个向量，维度{dim}，使用{faiss.get_num_gpus()}个GPU")
        else:
            instance.index = cpu_index
            print(f"索引加载完成：{num_embeddings}个向量，维度{dim}，使用CPU")
        
        # 4. 恢复HNSW参数
        if index_type == "HNSWFlat" and hasattr(instance.index, "hnsw"):
            raise ValueError("HNSWFlat is not supported in gpu env")
            # instance.index.hnsw.ef = instance.ef_search
            # instance.index.setHnswEfConstruction(instance.ef_construction)
        
        return instance

    def _build_new_index(self):
        """新建索引流程（仅当提供embedding_path时调用）"""
        # 加载embedding并校验
        self.embeddings = self._load_embeddings()
        self.dim = self.embeddings.shape[1]
        self.num_embeddings = self.embeddings.shape[0]
        
        # 校验embedding与label数量匹配
        if len(self.embeddings) != len(self.labels):
            raise ValueError(f"embedding数量({len(self.embeddings)})与label数量({len(self.labels)})不匹配")
    
        self._build_index_from_embeddings()

    def _load_embeddings(self) -> np.ndarray:
        """加载embedding（仅新建索引时使用）"""
        embedding_tensor = torch.load(self.embedding_path, map_location="cpu")
        if not isinstance(embedding_tensor, torch.Tensor):
            raise TypeError("embedding.pth必须存储torch.Tensor")
        # if embedding_tensor.dtype != torch.float32:
        #     embedding_tensor = embedding_tensor.float()
        # return embedding_tensor.numpy()
        embeddings = embedding_tensor.numpy().astype(np.float32, copy=False)
        return embeddings

    def _init_gpu_resources(self) -> faiss.StandardGpuResources:
        """初始化GPU资源"""
        res = faiss.StandardGpuResources()
        # 可选：设置GPU内存池大小（根据需求调整）
        # res.setTempMemory(1024 * 1024 * 1024)  # 1GB临时内存
        return res

    def search(self, query_embedding: np.ndarray, k: int = 1, gt_label=None) -> List[Dict]:
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

    def batch_search(self, query_embeddings: np.ndarray, k: int = 1, gt_labels=None) -> List[List[Dict]]:
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


# 示例用法
if __name__ == "__main__":
    """
    python3.9 dinov2/infer/faiss_retrieval.py \
    --embedding-path /path/to/train_embeddings_step_12500.pth \
    --label-path /path/to/train_labels.txt \
    --test-embedding-path /path/to/val_embeddings_step_12500.pth \
    --test-labels-path /path/to/val_labels.txt \
    --output-path /path/to/val_infer_ip_step12500.jsonl \
    --index-type FlatIP
    """
    args = parse_args()
    embedding_fp = args.embedding_path
    label_fp = args.label_path
    embedding_dir = args.embedding_path_dir
    test_embedding_path = args.test_embedding_path
    test_labels_fp = args.test_labels_path
    test_infer_fp = args.output_path

    index_fp = args.index_path
    index_type = args.index_type
    gpu_loading = True if args.gpu_loading else False

    if index_fp and os.path.exists(index_fp):
        print(f"Load index from saved file {index_fp}")
        faiss_index = FaissIndexGPU.load_index(
            index_type=index_type,
            load_path=index_fp,
            label_path=label_fp,
            gpu_loading=gpu_loading
        )
    else:
        print("Init index")
        # 优先使用embedding_dir
        if embedding_dir and os.path.exists(embedding_dir):
            faiss_index = FaissIndexGPU(
                index_type=index_type,
                embedding_dir=embedding_dir,
                gpu_loading=gpu_loading
            )
        else:
            faiss_index = FaissIndexGPU(
                embedding_path=embedding_fp,
                label_path=label_fp,
                index_type=index_type,
                gpu_loading=gpu_loading
            )
        if index_fp:
            faiss_index.save_index(index_fp)

    test_embeddings = torch.load(test_embedding_path)
    if test_embeddings.dtype != torch.float32:
        test_embeddings = test_embeddings.float()
    test_embeddings = test_embeddings.numpy()
    with open(test_labels_fp, "r") as f:
        test_labels = [l.strip() for l in f]

    eval_result_list = []
    BATCH_SIZE = 16  # 可调节的批量大小参数

    # 批量处理查询
    for batch_idx in tqdm(range(0, test_embeddings.shape[0], BATCH_SIZE)):
        end_idx = min(batch_idx + BATCH_SIZE, test_embeddings.shape[0])
        batch_queries = test_embeddings[batch_idx:end_idx]
        batch_labels = test_labels[batch_idx:end_idx]

        batch_results = faiss_index.batch_search(batch_queries, k=1, gt_labels=batch_labels)
        for single_result in batch_results:
            eval_result_list.append(json.dumps(single_result, ensure_ascii=False))
        
        if len(eval_result_list) % 10000 == 0:
            with open(test_infer_fp, "w") as f:
                f.write("\n".join(eval_result_list))

    with open(test_infer_fp, "w") as f:
        with open(test_infer_fp, "w") as f:
            f.write("\n".join(eval_result_list))
        # print("\n加载索引后的召回结果：")
        # for i, res in enumerate(loaded_results["retrieval"]):
        #     print(f"Top-{i+1}: label_id={res['label_id']}, label={res['label']}, distance={res['distance']:.4f}, gt={results['gt_label']}")