# multiprocessing_preprocess.py

from io import BufferedReader
import json
import os
import numpy as np
import random
import pickle
from pathlib import Path
from typing import *
from multiprocessing import Pool, Manager
import hashlib

import torch

# 全局变量（只读）
itemnum = None
neg_sample_size = None
hard_neg_ratio = None
item_feat_dict = None
output_dir = None
data_file = None
seq_offsets = None
import numpy as np
from pathlib import Path
import pickle
import time
from datetime import datetime, timedelta
from utils import format_time,set_environment


class MMPBaseDataset():
    data_dir: Path
    data_file: BufferedReader
    mm_emb_ids: list
    mm_emb_dict: dict
    item_num: int
    user_num: int
    indexer_i_rev: dict
    indexer_u_rev: dict
    indexer: dict
    item_feature_default_value: dict
    feature_types: dict
    feat_statistics: dict
    seq_offsets:dict
    sample_index:list

    def __init__(self, data_dir, args):
        feature_file=os.environ['TEMP_PATH']+"/feature_cache.npz"
        self.data_dir = Path(data_dir)

        self.data_file = None
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

        self.max_padding_size = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer
        self.predict_feature_list = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]


         # 内存映射加载（不加载到内存）
        self.archive = np.load(feature_file, mmap_mode='r')
        self.item_ids = self.archive['item_ids']
        self.sparse_feats = self.archive['sparse_features']  # [N, S]
        self.continual_feats = self.archive['continual_features']  # [N, C]
        self.array_feats = self.archive['array_features']  # [N, A]
        self.mm_emb_feats = self.archive['mm_emb_features']  # [N, D]
        self.sparse_feature_position = {}
        self.sparse_feature_position_offset = {}
        # 构建 item_id -> index 映射（小内存）
        self.item_id_to_idx = {item_id: i for i, item_id in enumerate(self.item_ids)}
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        self._init_feat_info_2(feat_statistics=self.feat_statistics, feature_types=self.feature_types)
        self.item_feature_default_value = self.fill_missing_feat({},0,False)
        self.user_feature_default_value = self.fill_missing_feat({},0,True)
        self.data_size = int(open(os.environ['TEMP_PATH']+"/log.txt",'r').readline().strip())
        
    def _init_feat_info_2(self, feat_statistics, feature_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table

        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        """
        self.USER_SPARSE_FEAT = {
            k: feat_statistics[k] for k in feature_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feature_types['user_continual']

        self.sparse_feature_position['item_id']=0
        self.sparse_feature_position_offset['item_id'] = 0
        cnt = self.itemnum+1
        self.ITEM_SPARSE_FEAT = {}
        for index, k in enumerate(feature_types['item_sparse'],start=1):
            self.ITEM_SPARSE_FEAT[k] = feat_statistics[k]
            self.sparse_feature_position[k]=index
            self.sparse_feature_position_offset[k] = cnt
            cnt += self.ITEM_SPARSE_FEAT[k] + 1
#  '118':4,117763, '117':2,58742, '101':5,118509]
        self.ITEM_CONTINUAL_FEAT = feature_types['item_continual']
        self.USER_ARRAY_FEAT = {
            k: feat_statistics[k] for k in feature_types['user_array']}
        self.ITEM_ARRAY_FEAT = {
            k: feat_statistics[k] for k in feature_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024,
                          "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {
            k: EMB_SHAPE_DICT[k] for k in feature_types['item_emb']}  # 记录的是不同多模态特征的维度
    ## data init
    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]

        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                self.mm_emb_feats.shape[1], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    ## data process
    def _load_user_data(self, uid, data_file):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        data_file.seek(self.seq_offsets[uid])
        line = data_file.readline()
        data = json.loads(line)
        return data
    
    def fill_missing_feat(self, feat, item_id, is_user=False,use_cache=False):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        sparse_feature = []
        continual_feature = []
        array_feature = []
        if is_user:
            # user_array_feature_range
            sparse_feature.append(item_id)
            offset = 0
            offset += self.usernum + 1
            for k in self.USER_SPARSE_FEAT:
                sparse_feature.append((feat[k]+offset if k in feat else 0))
                offset += self.USER_SPARSE_FEAT[k] + 1

            for k in self.USER_CONTINUAL_FEAT:
                continual_feature.append(feat[k] if k in feat else 0)

            offset = 0
            for k in self.USER_ARRAY_FEAT:
                f = [i+offset for i in feat[k][:10]] if k in feat else []
                array_feature.extend(f+[0]*(10-len(f)))
                offset += self.USER_ARRAY_FEAT[k]+1
                
            
            return {
                "sparse_feature": sparse_feature,
                "continual_feature": continual_feature,
                "array_feature": array_feature
            }
        else:
            idx = self.item_id_to_idx.get(str(item_id))
            if use_cache:
                labels = np.zeros(len(self.predict_feature_list))
                idx = self.item_id_to_idx.get(str(item_id))
                if item_id == 0 or idx==None:
                # 返回默认值（全 0 或默认 embedding）
                    return {
                        "sparse_feature": np.zeros(self.sparse_feats.shape[1], dtype=np.int64),
                        "continual_feature": np.zeros(self.continual_feats.shape[1], dtype=np.float32),
                        "array_feature": np.zeros(self.array_feats.shape[1], dtype=np.int64),
                        "mm_emb": self.mm_emb_feats[0],  # 默认 embedding
                        "labels":labels
                    }
                # return sparse_feature, continual_feature, array_feature, mm_emb
                for index,feature_name in enumerate(self.predict_feature_list):
                    labels[index] = self.sparse_feats[idx][self.sparse_feature_position[feature_name]]
                    if labels[index]!=0:
                        labels[index]-=self.sparse_feature_position_offset[feature_name]
                    assert labels[index]>=0 and labels[index]<=self.feat_statistics[feature_name],"error"
                return {
                    "sparse_feature": self.sparse_feats[idx],  # copy 避免 view 被释放
                    "continual_feature": self.continual_feats[idx],
                    "array_feature": self.array_feats[idx],
                    "mm_emb": self.mm_emb_feats[idx],
                    "labels":labels
                }
            else:
                labels = np.zeros(len(self.predict_feature_list))
                for index,feature_name in enumerate(self.predict_feature_list):
                    labels[index] = feat.get(feature_name,0)
                sparse_feature.append(item_id)

                offset = 0
                offset += self.itemnum + 1
                for k in self.ITEM_SPARSE_FEAT:
                    sparse_feature.append((feat[k]+offset if k in feat else 0))
                    offset += self.ITEM_SPARSE_FEAT[k] + 1
                for k in self.ITEM_CONTINUAL_FEAT:
                    continual_feature.append(feat[k])

                offset = 0
                for k in self.ITEM_ARRAY_FEAT:
                    f = [i+offset for i in feat[k][:10]] if k in feat else []
                    array_feature.extend(f+[0]*(10-len(feat[k])))
                    offset += self.ITEM_ARRAY_FEAT[k]
                mm_emb = self.feature_default_value[self.mm_emb_ids[0]]
                if item_id != 0:
                    idx = self.item_id_to_idx.get(str(item_id),-1)
                    if idx !=-1:
                        mm_emb = self.mm_emb_feats[idx]
                # return sparse_feature, continual_feature, array_feature, mm_emb
                return {
                    "sparse_feature": sparse_feature,
                    "continual_feature": continual_feature,
                    "array_feature": array_feature,
                    "mm_emb": mm_emb,
                    "labels":labels
                }
    

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat


    ## function_tool
    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        # 几乎不可能命中，因为整体的随机样本的量很大
        # 因为不再使用item_feat_dict，因此在这里删除
        # while t in s or str(t) not in self.item_feat_dict:
        #     t = np.random.randint(l, r)
        return t

    def __len__(self):
        return self.data_size

    def split_index(self,probs,total_data_size=None,shuffle=True):
        split_size = len(probs)
        data_len = total_data_size if total_data_size else len(self)
        split_index = list(range(data_len))
        if shuffle:
            random.shuffle(split_index)
        res = []
        start = 0
        for prob in probs:
            data_size = int(data_len*prob)
            res.append(split_index[start:start+data_size])
            start += data_size
        return res
    

class EmbeddingDataset(torch.utils.data.Dataset):
    base_dataset:MMPBaseDataset # 基础数据集（防止深拷贝）
    sample_index:list # 样本索引
    max_padding_size:int # 最大长度
    def __init__(self, base_dataset, sample_index=[], max_padding_size=100):
        super().__init__()
        self.base_dataset = base_dataset
        self.sample_index = sample_index # 采样索引
        self.max_padding_size = max_padding_size
        self.data_file = None

    def __getitem__(self, index):
        if self.data_file is None:
            self.data_file = np.memmap(os.environ['TEMP_PATH'] + "/all_data.npy", dtype=np.int64, mode='r', shape=(len(self.base_dataset), 2))
        data = self.data_file[index]
        pos1_id, pos2_id = data[0],data[1]
        pos1_feature = self.base_dataset.fill_missing_feat({}, pos1_id, use_cache=True)
        pos1_sparse_feature = pos1_feature['sparse_feature']
        pos1_continual_feature = pos1_feature['continual_feature']
        pos1_array_feature = pos1_feature['array_feature']
        pos1_mm_emb = pos1_feature['mm_emb']
        pos1_labels = pos1_feature['labels']
        pos2_feature = self.base_dataset.fill_missing_feat({}, pos2_id, use_cache=True)
        pos2_sparse_feature = pos2_feature['sparse_feature']
        pos2_continual_feature = pos2_feature['continual_feature']
        pos2_array_feature = pos2_feature['array_feature']
        pos2_mm_emb = pos2_feature['mm_emb']
        pos2_labels = pos1_feature['labels']

        return pos1_sparse_feature, pos1_continual_feature, pos1_array_feature, pos1_mm_emb,pos1_labels, pos2_sparse_feature, pos2_continual_feature, pos2_array_feature, pos2_mm_emb, pos2_labels
    def __len__(self):
        return len(self.sample_index)
    

    
    @staticmethod
    def collate_fn(batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
        """
        return_batch = {}
        pos1_sparse_feature, pos1_continual_feature, pos1_array_feature, pos1_mm_emb,pos1_labels, pos2_sparse_feature, pos2_continual_feature, pos2_array_feature, pos2_mm_emb, pos2_labels = zip(*batch)
        return_batch['pos1'] = {
            'item_sparse_feature':torch.tensor(np.stack(pos1_sparse_feature), dtype=torch.long),
            'item_continual_feature': torch.tensor(np.stack(pos1_continual_feature), dtype=torch.float),
            'item_array_feature': torch.tensor(np.stack(pos1_array_feature), dtype=torch.long),
            'item_mm_embs': torch.tensor(np.stack(pos1_mm_emb), dtype=torch.float)
        }
        return_batch['pos1_labels'] = torch.tensor(np.stack(pos1_labels), dtype=torch.long)
        return_batch['pos2'] = {
            'item_sparse_feature': torch.tensor(np.stack(pos2_sparse_feature), dtype=torch.long),
            'item_continual_feature': torch.tensor(np.stack(pos2_continual_feature), dtype=torch.float),
            'item_array_feature': torch.tensor(np.stack(pos2_array_feature), dtype=torch.long),
            'item_mm_embs': torch.tensor(np.stack(pos2_mm_emb), dtype=torch.float)
        }
        return_batch['pos2_labels'] = torch.tensor(np.stack(pos2_labels), dtype=torch.long)
        
        return return_batch
    
def init_worker(shared_itemnum, shared_neg_size, shared_hard_ratio, shared_item_feat_dict, shared_output_dir):
    """初始化每个 worker 的全局变量"""
    global itemnum, neg_sample_size, hard_neg_ratio, item_feat_dict, output_dir, seq_offsets, data_file
    itemnum = shared_itemnum
    neg_sample_size = shared_neg_size
    hard_neg_ratio = shared_hard_ratio
    item_feat_dict = shared_item_feat_dict
    set_environment()
    output_dir = Path(shared_output_dir)
    with open(Path(os.environ["TRAIN_DATA_PATH"], 'seq_offsets.pkl'), 'rb') as f:
        seq_offsets = pickle.load(f)
    data_file = open(Path(os.environ["TRAIN_DATA_PATH"]) / "seq.jsonl", 'rb')
def load_user_data(uid):
    """
    从数据文件中加载单个用户的数据

    Args:
        uid: 用户ID(reid)

    Returns:
        data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
    """
    global seq_offsets,data_file
    data_file.seek(seq_offsets[uid])
    line = data_file.readline()
    data = json.loads(line)
    return data

def process_chunk(start_end: List[str]) -> Dict[str, List]:
    """
    处理一个 chunk 的数据，返回样本列表
    返回: {
        'users': [...],
        'input_items': [...],
        'pos_items': [...],
        'neg_items': [...],  # list of lists
        'feature_map': {id: feat}
    }
    """
    start = start_end[0]
    end = start_end[1]
    # 初始化config
    # 处理序列特征
    # 利用稀疏滑动窗口，对于
    data_pairs = []
    with open(os.environ.get("USER_CACHE_PATH")+"/train_idx.pkl", "rb") as f:
        train_idx = pickle.load(f)
    for index in range(start, end):
        index = train_idx[index]
        try:
            user_sequence = load_user_data(index)
            # 对于曝光序列，采样。
            # 对于点击序列，全连接
            all_items = []
            click_items = [] # 会出现偶发的重复采样，感觉对结果影响不大
            for record_tuple in user_sequence:
                u, i, user_feat, item_feat, action_type, timestamp = record_tuple
                if i and item_feat:
                    all_items.append(i)
                    if action_type == 1:
                        click_items.append(i)
            # 针对曝光序列，
            sample_list = [1,1,1,1,2,3,5,8,13,21,34,56,91]
            # 针对曝光序列
            for i in click_items:
                for add in sample_list:
                    if i+add < len(all_items):
                        data_pairs.append([i,all_items[i+add]])
                    else:
                        break
            # 针对点击序列两两配对
            for i in range(len(click_items)):
                for j in range(i+1,len(click_items)):
                    data_pairs.append([click_items[i],click_items[j]])
        except Exception as e:
            # 输出错误具体行
            print(f"uid {index}")
            import traceback
            traceback.print_exc()
            continue
    # 洗牌
    random.shuffle(data_pairs)
    return data_pairs

def split_file(item_size, num_chunks: int) -> List[List[str]]:
    # 创建 chunk
    handle_size = item_size // num_chunks
    start_end_pos = [(start,min(start+handle_size,item_size) ) for start in range(0, item_size, handle_size)]
    return start_end_pos



def merge_results(chunks_dir: Path, output_dir: Path):
    """
    流式合并所有 chunk 的结果，使用时间打点输出进度和预计完成时间
    """
    output_dir.mkdir(exist_ok=True)
    chunk_files = sorted(chunks_dir.glob("chunk_*.pkl"))

    if not chunk_files:
        raise ValueError("No chunk files found!")

    print(f"🔍 扫描 chunk 文件以确定总样本数... (共 {len(chunk_files)} 个文件)")
    total_samples = 0
    chunk_sizes = []
    # 第一步：扫描所有 chunk，统计总样本数
    start_scan = time.time()
    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as f:
            temp = pickle.load(f)
            n = len(temp)
            total_samples += n
            chunk_sizes.append(n)
            del temp

    scan_time = time.time() - start_scan
    print(f"📊 扫描完成，总计 {total_samples:,} 个样本，耗时 {scan_time:.2f}s")
    idx = list(range(total_samples))

    # 第二步：创建 memmap 文件（预分配）
    print("⚙️  创建内存映射文件...")
    create_start = time.time()
    data_memmap = np.memmap(os.environ['TEMP_PATH'] + "/all_data.npy", dtype=np.int64, mode='w+', shape=(total_samples,2))

    create_time = time.time() - create_start
    print(f"✅ 内存映射创建完成，耗时 {create_time:.2f}s")

    processed = 0
    start_idx = 0
    total_chunks = len(chunk_files)

    # 记录主处理开始时间
    merge_start_time = time.time()

    print("🚀 开始流式合并...")

    for i, chunk_file in enumerate(chunk_files):
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)

        end_idx = start_idx + len(chunk_data)

        # 写入对应区间
        data_memmap[start_idx:end_idx] = chunk_data[:]

        start_idx = end_idx
        processed += 1

        # 每处理 100 个 chunk 或最后一个，输出一次进度
        if processed % 100 == 0 or i == total_chunks - 1:
            elapsed = time.time() - merge_start_time
            avg_time_per_chunk = elapsed / processed
            remaining = total_chunks - processed
            eta_seconds = avg_time_per_chunk * remaining
            eta_finish = datetime.now() + timedelta(seconds=eta_seconds)

            print(
                f"📌 进度: {processed}/{total_chunks} "
                f"| 已用: {elapsed:.1f}s "
                f"| 预计剩余: {eta_seconds:.1f}s "
                f"| 预计完成: {eta_finish.strftime('%H:%M:%S')}"
            )
    # 记录数据
    with open(os.environ['TEMP_PATH']+"/log.txt", 'w',encoding='utf-8') as f:
        f.write(str(total_samples))
    # flush 所有 memmap
    print("💾 正在保存数据到磁盘...")
    data_in_memory = np.array(data_memmap)
    np.random.shuffle(data_in_memory)
    # 4. 写回 memmap
    data_memmap[:] = data_in_memory

    # 5. 同步到磁盘
    data_memmap.flush()


    # 保存 feature_map

    total_time = time.time() - merge_start_time
    print(f"✅ 合并完成！共 {total_samples:,} 个样本")
    print(f"⏱️  总耗时: {total_time:.2f}s")
    print(f"📁 输出路径: {output_dir}")

def main(
    data_file: str = "seq.jsonl",
    itemnum: int = 10000,
    neg_sample_size: int = 100,
    hard_neg_ratio: float = 0.1,
    num_workers: int = 4,
    output_dir: str = "processed_data"
):
    set_environment()
    seq_offsets = None
    with open(Path(os.environ["TRAIN_DATA_PATH"], 'seq_offsets.pkl'), 'rb') as f:
        seq_offsets = pickle.load(f)
    with open("temp.json",'w',encoding='utf-8') as f:
        json.dump(seq_offsets,f,indent=4,ensure_ascii=False)
    output_dir = Path(output_dir)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    with open(os.environ.get("USER_CACHE_PATH")+"/train_idx.pkl", "rb") as f:
        train_idx = pickle.load(f)
    # 2. 分块
    chunks = split_file(len(train_idx), num_workers)

    # 3. 多进程处理
    args = (itemnum, neg_sample_size, hard_neg_ratio, item_feat_dict, str(chunks_dir))
    # init_worker(*args)
    # process_chunk(chunks[0])
    with Pool(processes=num_workers, initializer=init_worker, initargs=args) as pool:
        tasks = [(chunk,) for chunk in chunks]
        results = pool.starmap(process_chunk, tasks)

        # 4. 保存每个 chunk 的结果
        for i, result in enumerate(results):
            chunk_file = chunks_dir / f"chunk_{i}.pkl"
            with open(chunk_file, 'wb') as f:
                pickle.dump(result, f)

    # 5. 合并
    merge_results(chunks_dir, output_dir)

    print("🎉 预处理完成！")

if __name__ == "__main__":
    main(num_workers=8)  # 设置你的 CPU 核心数