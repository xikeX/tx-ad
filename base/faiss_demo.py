# faiss_demo.py

import os
from pathlib import Path
import struct
import time
import numpy as np
from typing import Optional

import torch

def read_fbin(filename: str) -> np.ndarray:
    """
    读取 .fbin 格式的向量文件。
    文件格式：前8字节为 int32 的 n 和 d，接着是 n*d 个 float32 数据。
    返回 shape 为 [n, d] 的 numpy array。
    """
    with open(filename, "rb") as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
        # 根据文件后缀以不同的方式读取数据
        if filename.endswith(".fbin"):
            data = np.fromfile(f, dtype=np.float32, count=n * d)
        elif filename.endswith(".u64bin"):
            data = np.fromfile(f, dtype=np.uint64, count=n)
        # data = np.fromfile(f, dtype=np.float32, count=n * d)
        return data.reshape(n, d)

def load_emb(load_path):
    """
    从二进制文件加载 Embedding

    Args:
        load_path: 文件路径

    Returns:
        embedding: numpy array, shape [num_points, num_dimensions]
    """
    with open(load_path, 'rb') as f:
        # 读取前8字节：num_points 和 num_dimensions (两个uint32)
        header = f.read(8)
        num_points, num_dimensions = struct.unpack('II', header)
        # 读取后面的向量数据
        emb = np.fromfile(f, dtype=np.float32).reshape(num_points, num_dimensions)
    return emb


def read_u64bin(filename: str) -> np.ndarray:
    """
    读取 .u64bin 格式的 ID 文件。
    返回一维 uint64 数组。
    """
    return np.fromfile(filename, dtype=np.uint64)


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1] if emb.ndim > 1 else 1  # 向量的维度
    print(f'saving {save_path}')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)

def run_faiss_ann_search(
        dataset_vector_file_path: str = None,
        dataset_id_file_path: str = None,
        query_vector_file_path: str = None,
        result_id_file_path: str = None,
        dataset_vector: np.ndarray = None,
        dataset_id: np.ndarray = None,
        query_vector: np.ndarray = None,
        query_ann_top_k: int = 10,
        faiss_M: int = 64,
        faiss_ef_construction: int = 1280,
        query_ef_search: int = 640,
        faiss_metric_type: int = 1,  # 0: L2, 1: Inner Product
):
    """
    使用 PyTorch 实现精确最近邻搜索
    支持：超大 dataset_vector 和 超大 query_vector（分块处理）
    """
    # 1. 加载数据
    if dataset_vector is None:
        dataset_vector = read_fbin(dataset_vector_file_path)  # [N, D]
    if dataset_id is None:
        dataset_id = read_fbin(dataset_id_file_path).ravel()  # [N]
    if query_vector is None:
        query_vector = read_fbin(query_vector_file_path)  # [Q, D]

    N, D = dataset_vector.shape
    Q = query_vector.shape[0]

    # 2. 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 3. 初始化全局 top-k 结果（在 CPU 上维护）
    # 使用负无穷（L2）或负极大值（IP）作为初始 score
    if faiss_metric_type == 1:  # Inner Product
        top_scores = torch.full((Q, query_ann_top_k), -float('inf'))
    else:  # L2 距离（越小越好，我们用负距离）
        top_scores = torch.full((Q, query_ann_top_k), float('inf'))  # 注意：L2 用最小堆逻辑
    top_indices = torch.zeros((Q, query_ann_top_k), dtype=torch.long)

    # 4. Query 分块大小（根据 GPU 显存调整）
    query_chunk_size = 512  # 每次处理 512 个 query
    data_chunk_size = 100_000  # 每次处理 10 万条数据

    # 5. 遍历 query chunk
    # 记录时间
    start_time = time.time()
    for q_start in range(0, Q, query_chunk_size):
        q_end = min(q_start + query_chunk_size, Q)
        query_chunk = torch.from_numpy(query_vector[q_start:q_end]).float().to(device)  # [q_b, D]

        # 临时存储当前 query chunk 的 top-k
        local_scores = torch.full((query_chunk.size(0), query_ann_top_k), -float('inf') if faiss_metric_type == 1 else float('inf'), device=device)
        local_indices = torch.zeros((query_chunk.size(0), query_ann_top_k), dtype=torch.long, device=device)

        # 6. 遍历 dataset chunk
        for d_start in range(0, N, data_chunk_size):
            d_end = min(d_start + data_chunk_size, N)
            data_chunk = torch.from_numpy(dataset_vector[d_start:d_end]).float().to(device)  # [d_b, D]
            id_chunk = torch.from_numpy(dataset_id[d_start:d_end]).long().to(device)  # [d_b]

            # 7. 计算相似度
            if faiss_metric_type == 1:  # Inner Product
                scores = torch.matmul(query_chunk, data_chunk.t())  # [q_b, d_b]
            else:  # L2 距离（越小越好）
                diff = query_chunk.unsqueeze(1) - data_chunk.unsqueeze(0)  # [q_b, d_b, D]
                scores = -torch.sum(diff**2, dim=2)  # 负平方距离 [q_b, d_b]

            # 8. 合并到当前 query chunk 的 top-k
            if faiss_metric_type == 1:  # IP: 取最大
                merged_scores = torch.cat([local_scores, scores], dim=1)
                if len(id_chunk.shape) > 1:
                    id_chunk = id_chunk.squeeze(1)
                tmp_id_chunk = id_chunk.unsqueeze(0).expand(query_chunk.size(0), -1)
                merged_indices = torch.cat([local_indices, tmp_id_chunk], dim=1)
                topk_vals, topk_idx = torch.topk(merged_scores, k=query_ann_top_k, dim=1)
            else:  # L2: 取最小（负距离取最大）
                merged_scores = torch.cat([local_scores, scores], dim=1)
                merged_indices = torch.cat([local_indices, id_chunk.unsqueeze(0).expand(query_chunk.size(0), -1)], dim=1)
                # 取最小距离 → 负距离取最大
                topk_vals, topk_idx = torch.topk(merged_scores, k=query_ann_top_k, dim=1, largest=False)

            # 更新 local top-k
            local_scores = torch.gather(merged_scores, 1, topk_idx)
            local_indices = torch.gather(merged_indices, 1, topk_idx)

            # 清理显存
            del data_chunk, id_chunk, scores
            if device.type == 'cuda':
                torch.cuda.synchronize()

        # 9. 将当前 query chunk 的结果写回全局结果
        top_scores[q_start:q_end] = local_scores.cpu()
        top_indices[q_start:q_end] = local_indices.cpu()

        # 清理
        del query_chunk, local_scores, local_indices
        if device.type == 'cuda':
            torch.cuda.synchronize()
        # 显示进度
        end_time = time.time()
        print(f"进度: {q_start}/{Q}，耗时: {end_time - start_time:.2f},预计剩余时间: {(end_time - start_time) * (Q - q_start) / (q_end - q_start):.2f}s")

    # 10. 转为 numpy
    indices = top_indices.numpy()  # [Q, K]
    avg_score = top_scores.numpy().mean()

    # 11. 保存结果
    if result_id_file_path is not None:
        save_emb(indices.astype(np.uint64), result_id_file_path)

    return indices, avg_score

# import faiss
# def run_faiss_ann_search(
#     dataset_vector_file_path: str=None,
#     dataset_id_file_path: str=None,
#     query_vector_file_path: str=None,
#     result_id_file_path=None,
#     dataset_vector=None,
#     dataset_id=None,
#     query_vector=None,
#     query_ann_top_k: int = 10,
#     faiss_M: int = 64,
#     faiss_ef_construction: int = 1280,
#     query_ef_search: int = 640,
#     faiss_metric_type: int = 1,  # 0: L2, 1: Inner Product
# ) -> None:
#     """
#     使用 Faiss 构建 HNSW 索引并执行近似最近邻搜索，保存 Top-K 结果 ID。

#     参数:
#         dataset_vector_file_path (str): 物料向量文件路径 (.fbin)
#         dataset_id_file_path (str): 物料 ID 文件路径 (.u64bin)
#         query_vector_file_path (str): 查询向量文件路径 (.fbin)
#         result_id_file_path (str): 检索结果 ID 输出路径 (.u64bin)
#         query_ann_top_k (int): 返回每个查询的 Top-K 个最近邻
#         faiss_M (int): HNSW 中每个节点的邻居数量
#         faiss_ef_construction (int): 构建时的探索深度
#         query_ef_search (int): 查询时的探索深度
#         faiss_metric_type (int): 距离度量类型 (0=L2, 1=内积)
#     """
#     if dataset_vector is None:
#         dataset_vector = read_fbin(dataset_vector_file_path)  # [N, D]
#     if dataset_id is None:
#         dataset_id = read_fbin(dataset_id_file_path)   # [N]

#     if query_vector is None:
#         query_vector = read_fbin(query_vector_file_path)    # [Q, D]

#     assert len(dataset_id) == len(dataset_vector), "Error: 向量数量和 ID 数量不匹配！"

#     dim = dataset_vector.shape[1]
#     metric = faiss.METRIC_L2 if faiss_metric_type == 0 else faiss.METRIC_INNER_PRODUCT

#     index = faiss.IndexHNSWFlat(dim, faiss_M, metric)
#     index.hnsw.efConstruction = faiss_ef_construction
#     index.hnsw.efSearch = query_ef_search
#     index.verbose = True

#     # 包装成 IDMap 以保留原始 ID
#     index = faiss.IndexIDMap2(index)  # 包装成 IDMap
#     dataset_id = dataset_id.ravel()
#     index.add_with_ids(dataset_vector, dataset_id)

#     # ✅ 正确方式：通过 .index 访问底层 HNSW 索引
#     # index.index.hnsw.efSearch = query_ef_search

#     distances, indices = index.search(query_vector, query_ann_top_k)
#     if result_id_file_path is not None:
#         print(f"Writing Top-{query_ann_top_k} results to {result_id_file_path}")
#         save_emb(indices.astype(np.uint64),result_id_file_path)

#     return indices

# # faiss_demo.py