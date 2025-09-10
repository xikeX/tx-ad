# faiss_demo.py

import os
from pathlib import Path
import struct
import numpy as np
from typing import Optional

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

import os
from pathlib import Path
import struct
import numpy as np
from typing import Optional
import numpy as np
import hnswlib
from typing import Optional, Tuple, Union


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
) -> np.ndarray:
    """
    使用 hnswlib 构建 HNSW 索引并执行近似最近邻搜索，返回 Top-K 结果的原始 ID。

    参数:
        dataset_vector_file_path (str): 物料向量文件路径 (.fbin)
        dataset_id_file_path (str): 物料 ID 文件路径 (.u64bin)
        query_vector_file_path (str): 查询向量文件路径 (.fbin)
        result_id_file_path (str): 检索结果 ID 输出路径 (.u64bin)
        dataset_vector (np.ndarray): [N, D] 向量数组
        dataset_id (np.ndarray): [N] ID 数组
        query_vector (np.ndarray): [Q, D] 查询向量
        query_ann_top_k (int): 返回每个查询的 Top-K 个最近邻
        faiss_M (int): HNSW 中每个节点的最大连接数
        faiss_ef_construction (int): 构建时的探索深度 (ef_construction)
        query_ef_search (int): 查询时的探索深度 (ef_search)
        faiss_metric_type (int): 距离度量类型 (0=L2, 1=内积)

    返回:
        np.ndarray: [Q, K] 的原始 ID 矩阵
    """
    
    # ================== 1. 加载数据 ==================
    if dataset_vector is None:
        dataset_vector = read_fbin(dataset_vector_file_path)  # shape: [N, D]
    if dataset_id is None:
        dataset_id = read_fbin(dataset_id_file_path)          # shape: [N,] or [N,1]
    if query_vector is None:
        query_vector = read_fbin(query_vector_file_path)      # shape: [Q, D]

    dataset_vector = dataset_vector.astype('float32')
    dataset_id = dataset_id.ravel().astype('int64')  # 确保是一维
    query_vector = query_vector.astype('float32')

    assert len(dataset_vector) == len(dataset_id), "Error: 向量数量和 ID 数量不匹配！"

    dim = dataset_vector.shape[1]

    # ================== 2. 设置距离度量 ==================
    space = 'ip' if faiss_metric_type == 1 else 'l2'

    # ================== 3. 创建并初始化 hnswlib 索引 ==================
    index = hnswlib.Index(space=space, dim=dim)

    # 初始化索引结构
    index.init_index(
        max_elements=len(dataset_vector),
        ef_construction=faiss_ef_construction,
        M=faiss_M
    )

    # 设置查询时的 ef
    index.set_ef(query_ef_search)

    # 可选：启用多线程（添加数据时）
    # index.add_threads = 4  # 可根据 CPU 核心数调整

    # ================== 4. 添加向量和 ID ==================
    print(f"开始构建 HNSW 索引，共 {len(dataset_vector)} 个向量...")
    index.add_items(dataset_vector, dataset_id)
    print("索引构建完成。")

    # ================== 5. 执行搜索 ==================
    print(f"开始搜索，Top-{query_ann_top_k}...")
    labels, distances = index.knn_query(query_vector, k=query_ann_top_k)
    # labels: [Q, K] 原始 ID 矩阵
    # distances: [Q, K] 相似度矩阵

    # ================== 6. 保存结果（可选） ==================
    if result_id_file_path is not None:
        print(f"正在保存 Top-{query_ann_top_k} 结果到 {result_id_file_path}")
        save_emb(labels.astype(np.uint64), result_id_file_path)

    # ================== 7. 返回结果 ==================
    return labels  # 直接返回原始 ID，无需映射

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
#     metric = faiss.METRIC_INNER_PRODUCT

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
#         # save_emb(indices.astype(np.uint64),result_id_file_path)

#     return indices

# def run_faiss_ann_search_gpu(
#     dataset_vector_file_path: str = None,
#     dataset_id_file_path: str = None,
#     query_vector_file_path: str = None,
#     result_id_file_path: str = None,
#     dataset_vector=None,
#     dataset_id=None,
#     query_vector=None,
#     query_ann_top_k: int = 10,
#     faiss_M: int = 64,
#     faiss_ef_construction: int = 1280,
#     query_ef_search: int = 640,
#     faiss_metric_type: int = 1,
#     gpu_id: int = 0,
# ):
#     if dataset_vector is None:
#         dataset_vector = read_fbin(dataset_vector_file_path)
#     if dataset_id is None:
#         dataset_id = read_u64bin(dataset_id_file_path)  # 注意：u64bin 是 uint64
#     if query_vector is None:
#         query_vector = read_fbin(query_vector_file_path)

#     assert len(dataset_vector) == len(dataset_id), "向量和 ID 数量不匹配"

#     dim = dataset_vector.shape[1]
#     metric = faiss.METRIC_INNER_PRODUCT if faiss_metric_type == 1 else faiss.METRIC_L2

#     # CPU 索引
#     index_cpu = faiss.IndexHNSWFlat(dim, faiss_M, metric)
#     index_cpu.hnsw.efConstruction = faiss_ef_construction
#     index_cpu = faiss.IndexIDMap2(index_cpu)

#     # 转移到 GPU
#     res = faiss.StandardGpuResources()
#     index_gpu = faiss.index_cpu_to_gpu(res, gpu_id, index_cpu)

#     # 添加数据
#     index_gpu.add_with_ids(dataset_vector.astype('float32'), dataset_id.astype('int64'))

#     # 设置搜索参数
#     index_gpu.index.hnsw.efSearch = query_ef_search

#     # 搜索
#     distances, indices = index_gpu.search(query_vector.astype('float32'), query_ann_top_k)

#     # 保存
#     if result_id_file_path:
#         save_emb(indices.astype(np.uint64), result_id_file_path)

#     return indices
