# my_dataset_v1_aug.py
"""
现在的速度瓶颈：
1. 在于所有的数据都在内存里，所以开多进程需要考虑内存的问题，导致CPU的利用率不高
改进：
1. 预处理所有的负样本feature_map,在需要的时候在读取。
2. 现在所有的feature_map 都存在了item_feat_dict中。
3.
"""
import argparse
from io import BufferedReader
import json
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
from tqdm import tqdm
import functools

def get_position_from_numpy(row,pos):
    return row[pos]

get_position_function_map = {
    0:np.vectorize(functools.partial(get_position_from_numpy,pos=0)),
    1:np.vectorize(functools.partial(get_position_from_numpy,pos=1)),
    2:np.vectorize(functools.partial(get_position_from_numpy,pos=2)),
    3:np.vectorize(functools.partial(get_position_from_numpy,pos=3))
}

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


         # 内存映射加载（不加载到内存）
        self.archive = np.load(feature_file, mmap_mode='r')
        self.item_ids = self.archive['item_ids']
        self.sparse_feats = self.archive['sparse_features']  # [N, S]
        self.continual_feats = self.archive['continual_features']  # [N, C]
        self.array_feats = self.archive['array_features']  # [N, A]
        self.mm_emb_feats = self.archive['mm_emb_features']  # [N, D]
        # 构建 item_id -> index 映射（小内存）
        self.item_id_to_idx = {item_id: i for i, item_id in enumerate(self.item_ids)}
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        self._init_feat_info_2(feat_statistics=self.feat_statistics, feature_types=self.feature_types)
        self.item_feature_default_value = self.fill_missing_feat({},0,False)
        self.user_feature_default_value = self.fill_missing_feat({},0,True)
        
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
        self.ITEM_SPARSE_FEAT = {
            k: feat_statistics[k] for k in feature_types['item_sparse']}
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
            # return sparse_feature, continual_feature, array_feature, mm_emb
            return {
                "sparse_feature": sparse_feature,
                "continual_feature": continual_feature,
                "array_feature": array_feature
            }
        else:
            if use_cache:
                idx = self.item_id_to_idx.get(str(item_id))
                if item_id == 0 or idx==None:
                # 返回默认值（全 0 或默认 embedding）
                    return {
                        "sparse_feature": np.zeros(self.sparse_feats.shape[1], dtype=np.int64),
                        "continual_feature": np.zeros(self.continual_feats.shape[1], dtype=np.float32),
                        "array_feature": np.zeros(self.array_feats.shape[1], dtype=np.int64),
                        "mm_emb": self.mm_emb_feats[0]  # 默认 embedding
                    }
                return {
                    "sparse_feature": self.sparse_feats[idx],  # copy 避免 view 被释放
                    "continual_feature": self.continual_feats[idx],
                    "array_feature": self.array_feats[idx],
                    "mm_emb": self.mm_emb_feats[idx]
                }
            else:
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
                    "mm_emb": mm_emb
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
        return len(self.seq_offsets)

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

def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            file_path = Path(mm_path, f'emb_{feat_id}.pkl')
            if os.path.exists(file_path):
                with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                    emb_dict = pickle.load(f)
            else:
                folder = Path(mm_path, f'emb_{feat_id}_32')
                files = os.listdir(folder)
                for file in files:
                    path = Path(folder, file)
                    with open(path,'r',encoding='utf-8') as f:
                        for line in f:
                            line = json.loads(line)
                            if 'emb' in line:
                                emb_dict[line['anonymous_cid']] = np.array(line['emb'])
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict


# preprocess.py
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm

def process_single_item(args):
    """
    单个 item 的特征处理函数
    args: (item_id, feat_dict, config)
    """
    idx, item_id, feat_dict, config = args

    sparse_feature = []
    continual_feature = []
    array_feature = []
    mm_emb = config['feature_default_value'][config['mm_emb_ids'][0]]  # 默认值

    # 1. Item ID
    sparse_feature.append(item_id)

    # 2. Sparse Features
    offset = config['itemnum'] + 1
    for k, vocab_size in config['ITEM_SPARSE_FEAT'].items():
        val = feat_dict.get(k, 0)
        sparse_feature.append(val + offset if val else 0)
        offset += vocab_size + 1
#  '118':4,117763, '117':2,58742, '101':5,118509]
    # 3. Continual Features
    for k in config['ITEM_CONTINUAL_FEAT']:
        val = feat_dict.get(k, 0.0)
        continual_feature.append(float(val))

    # 4. Array Features (multi-hot, max 10)
    arr_offset = 0
    for k, vocab_size in config['ITEM_ARRAY_FEAT'].items():
        vals = feat_dict.get(k, [])[:10]
        padded = [v + arr_offset for v in vals] + [0] * (10 - len(vals))
        array_feature.extend(padded)
        arr_offset += vocab_size + 1

    # 5. MM Embedding (e.g., image embedding)
    if item_id != 0 and config['indexer_i_rev'].get(item_id):
        real_id = config['indexer_i_rev'][item_id]
        for feat_id in config['feature_types']['item_emb']:
            emb_dict = config['mm_emb_dict'][feat_id]
            if real_id in emb_dict and isinstance(emb_dict[real_id], np.ndarray):
                mm_emb = emb_dict[real_id].astype(np.float32)
                break

    return idx, item_id, np.array(sparse_feature), np.array(continual_feature), np.array(array_feature), mm_emb


def build_feature_file(item_feat_dict, output_path, num_workers=8, config=None):
    """
    多线程预处理所有 item 特征，保存为 .npz 文件
    """
    print("Starting feature preprocessing...")

    # 准备参数
    items = list(item_feat_dict.keys())
    args_list = [(idx,item_id, item_feat_dict[item_id], config) for idx,item_id in enumerate(items)]

    # 多线程处理
    sparse_list = [None] * len(items)
    continual_list = [None] * len(items)
    array_list = [None] * len(items)
    mm_emb_list = [None] * len(items)
    start_time = round(time.time())
    cnt = 0 
    max_size = config['itemnum']
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        results = list(executor.map(lambda args: process_single_item(*args), args_list))

        for future in as_completed(futures):
            idx, item_id, s, c, a, m = future.result()
            sparse_list[idx] = s
            continual_list[idx] = c
            array_list[idx] = a
            mm_emb_list[idx] = m
            cnt+=1
            if cnt%100000==0:
                end_time = round(time.time())
                use_time = format_time(end_time - start_time)
                remain_time = format_time((end_time - start_time) / (cnt) * max_size)
                print(f"prepare datat [{use_time}/{remain_time}] [{cnt}/{max_size}]")
    # 转为 numpy 数组（自动补零对齐）
    max_sparse_len = max(len(x) for x in sparse_list)
    max_array_len = max(len(x) for x in array_list)

    sparse_array = np.zeros((len(items), max_sparse_len), dtype=np.int64)
    for i, s in enumerate(sparse_list):
        sparse_array[i, :len(s)] = s

    continual_array = np.array(continual_list, dtype=np.float32)  # shape: [N, C]

    array_array = np.zeros((len(items), max_array_len), dtype=np.int64)
    for i, a in enumerate(array_list):
        array_array[i, :len(a)] = a

    mm_emb_array = np.stack(mm_emb_list, axis=0)  # shape: [N, D]

    # 保存
    np.savez_compressed(output_path,
                        item_ids=np.array(items),
                        sparse_features=sparse_array,
                        continual_features=continual_array,
                        array_features=array_array,
                        mm_emb_features=mm_emb_array)
    print(f"Feature file saved to {output_path}")
    
    
def main(args):

    def init_feat_info(featuere_statistics, feature_types):
        """
        将特征统计信息按类型分组，生成配置字典（config），避免使用 self

        Args:
            feat_statistics: dict, 特征ID -> 特征数量（如 {'61': 1000, '62': 500}）
            feature_types: dict, 特征类型分组（如 {'item_sparse': ['61','62'], ...}）

        Returns:
            config: dict, 包含所有特征结构信息
        """
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}

        config = {}

        # 用户稀疏特征：特征ID -> 类别数
        config['USER_SPARSE_FEAT'] = {
            k: featuere_statistics[k] for k in feature_types.get('user_sparse', [])
        }

        # 用户连续值特征：只记录特征名（无类别数）
        config['USER_CONTINUAL_FEAT'] = feature_types.get('user_continual', [])

        # 物品稀疏特征
        config['ITEM_SPARSE_FEAT'] = {
            k: featuere_statistics[k] for k in feature_types.get('item_sparse', [])
        }

        # 物品连续值特征
        config['ITEM_CONTINUAL_FEAT'] = feature_types.get('item_continual', [])

        # 用户 array 特征（multi-hot）
        config['USER_ARRAY_FEAT'] = {
            k: featuere_statistics[k] for k in feature_types.get('user_array', [])
        }

        # 物品 array 特征
        config['ITEM_ARRAY_FEAT'] = {
            k: featuere_statistics[k] for k in feature_types.get('item_array', [])
        }

        # 物品 embedding 特征维度（多模态）
        config['ITEM_EMB_FEAT'] = {
            k: EMB_SHAPE_DICT[k] for k in feature_types.get('item_emb', []) if k in EMB_SHAPE_DICT
        }

        # 可选：保存原始输入，便于调试
        config['feat_statistics'] = featuere_statistics
        config['feature_types'] = feature_types
        return config

    def _init_feat_info_2(indexer,mm_emb_ids,mm_emb_dict):
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
        feat_types['item_emb'] = mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics
    data_dir = Path(os.environ['TRAIN_DATA_PATH'])
    indexer=None
    mm_emb_ids = args.mm_emb_id
    
    item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
    with open(data_dir / 'indexer.pkl', 'rb') as ff:
        indexer = pickle.load(ff)
    mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), mm_emb_ids)
    feature_default_value , feature_types, featuere_statistics = _init_feat_info_2(indexer,mm_emb_ids,mm_emb_dict)
    config = init_feat_info(featuere_statistics,feature_types)
    config['itemnum'] = len(indexer['i'])
    config['usermnum'] = len(indexer['u'])
    config['indexer_i_rev'] = {v: k for k, v in indexer['i'].items()}
    config['mm_emb_ids'] = args.mm_emb_id
    config['mm_emb_dict'] = mm_emb_dict
    config['feature_default_value']=feature_default_value  # 默认值
    build_feature_file(item_feat_dict,os.environ['TEMP_PATH']+"/feature_cache.npz",num_workers=args.num_worker,config=config)

from utils import format_time, set_environment
set_environment()
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_worker', type=int, default=1, help='序列最大长度')
    parser.add_argument('--mm_emb_id', nargs='+', type=str, default=['81'], choices=[str(s) for s in range(81, 87)], help='多模态嵌入特征 ID 列表')

    # 解析参数
    args,unkown = parser.parse_known_args()

    return args
if __name__ == "__main__":
    # 先处理一下数据
    args =get_args()
    main(args)
    

