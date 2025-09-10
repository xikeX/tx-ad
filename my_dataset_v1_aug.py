# my_dataset_v1_aug.py
from io import BufferedReader
import json
import os
from pathlib import Path
import pickle
import random

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

class BaseDataset():
    data_dir: Path
    data_file: BufferedReader
    mm_emb_ids: list
    item_feat_dict: dict
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
        self.data_dir = Path(data_dir)

        self.data_file = None
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

        self.max_padding_size = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        self._init_feat_info_2(feat_statistics=self.feat_statistics, feature_types=self.feature_types)
        self.item_feature_default_value = self.fill_missing_feat({},0,False)
        self.user_feature_default_value = self.fill_missing_feat({},0,True)
        print("init_finish")
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
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
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
        mm_emb = None
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
            for feat_id in self.feature_types['item_emb']:
                if item_id != 0:
                    if self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                        if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                            mm_emb = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]
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
    def _random_neq(self, l, r, s,feat=None):
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
        if feat:
            cnt=100
            while cnt>0 and (t in s or str(t) not in self.item_feat_dict or self.item_feat_dict[str(t)].get('112',0)!=feat):
                t = np.random.randint(l, r)
                cnt-=1
                
        else:
            while t in s or str(t) not in self.item_feat_dict:
                t = np.random.randint(l, r)
            
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

class TrainDataset(torch.utils.data.Dataset):
    base_dataset:BaseDataset # 基础数据集（防止深拷贝）
    sample_index:list # 样本索引
    max_padding_size:int # 最大长度
    def __init__(self, base_dataset, sample_index=[], neg_sample_size = 1, max_padding_size=100):
        super().__init__()
        self.base_dataset = base_dataset
        self.sample_index = sample_index # 采样索引
        self.max_padding_size = max_padding_size
        self.neg_sample_size = neg_sample_size
        self.data_file = None

    def __getitem__(self, index):
        if self.data_file is None:
            self.data_file = open(self.base_dataset.data_dir / "seq.jsonl", 'rb')
        uid = self.sample_index[index]
        user_sequence = self.base_dataset._load_user_data(uid,self.data_file)

        return self.build_example(user_sequence)

    def build_example(self, user_sequence):
        # 构建训练样本
        ext_user_sequence = []
        explore_unlick_sample_index = []
        cur_u, cur_user_feat, cur_action_type = None,None,None
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            if u and user_feat:
                # ext_user_sequence.insert(0, (u, user_feat, 2, action_type)) # 插入第一位
                cur_u, cur_user_feat, cur_action_type = u, user_feat, action_type
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))
                if action_type==0:
                    explore_unlick_sample_index.append(len(ext_user_sequence)-1)
        # if len(explore_unlick_sample_index)>2:
        #     while random.random()<0.2:
        #         a = random.choice(explore_unlick_sample_index)
        #         b = random.choice(explore_unlick_sample_index)
        #         ext_user_sequence[a], ext_user_sequence[b] = ext_user_sequence[b], ext_user_sequence[a]
        if cur_u!=None: # 插入用户
            ext_user_sequence.insert(0, (cur_u, cur_user_feat, 2, cur_action_type))
        
        token_type = np.zeros([self.max_padding_size + 1], dtype=np.int32) # 是用户还是商品 0/1/2
        next_token_type = np.zeros([self.max_padding_size + 1], dtype=np.int32)
        next_action_type = np.zeros([self.max_padding_size + 1], dtype=np.int32)
        if cur_user_feat is None:
            cur_user_feat = {}
            cur_u = 0
        user_feat = self.base_dataset.fill_missing_feat(cur_user_feat, cur_u, True)
        user_feat_sparse = user_feat['sparse_feature']
        user_feat_continual = user_feat['continual_feature']
        user_feat_array = user_feat['array_feature']

        seq_feat_sparse = np.full([self.max_padding_size + 1,len(self.base_dataset.item_feature_default_value["sparse_feature"])],
                                  self.base_dataset.item_feature_default_value["sparse_feature"], dtype=np.int32)
        seq_feat_continual = np.full([self.max_padding_size + 1,len(self.base_dataset.item_feature_default_value["continual_feature"])]
                                      ,self.base_dataset.item_feature_default_value["continual_feature"], dtype=np.float32)
        seq_feat_array = np.full([self.max_padding_size + 1,len(self.base_dataset.item_feature_default_value["array_feature"])],
                                  self.base_dataset.item_feature_default_value["array_feature"], dtype=np.int32)
        seq_feat_mm_embs = np.full([self.max_padding_size + 1,len(self.base_dataset.item_feature_default_value["mm_emb"])],
                                  self.base_dataset.item_feature_default_value["mm_emb"], dtype=np.float32)
    
        pos_feat_sparse = np.full([self.max_padding_size + 1,len(self.base_dataset.item_feature_default_value["sparse_feature"])],
                                  self.base_dataset.item_feature_default_value["sparse_feature"], dtype=np.int32)
        pos_feat_continual = np.full([self.max_padding_size + 1,len(self.base_dataset.item_feature_default_value["continual_feature"])]
                                      ,self.base_dataset.item_feature_default_value["continual_feature"], dtype=np.float32)
        pos_feat_array = np.full([self.max_padding_size + 1,len(self.base_dataset.item_feature_default_value["array_feature"])],
                                  self.base_dataset.item_feature_default_value["array_feature"], dtype=np.int32)
        pos_feat_mm_embs = np.full([self.max_padding_size + 1,len(self.base_dataset.item_feature_default_value["mm_emb"])],
                                  self.base_dataset.item_feature_default_value["mm_emb"], dtype=np.float32)
    
        neg_feat_sparse = np.full([(self.max_padding_size + 1) * self.neg_sample_size,len(self.base_dataset.item_feature_default_value["sparse_feature"])],
                                  self.base_dataset.item_feature_default_value["sparse_feature"], dtype=np.int32)
        neg_feat_continual = np.full([(self.max_padding_size + 1) * self.neg_sample_size,len(self.base_dataset.item_feature_default_value["continual_feature"])],
                                  self.base_dataset.item_feature_default_value["continual_feature"], dtype=np.int32)
        neg_feat_array = np.full([(self.max_padding_size + 1) * self.neg_sample_size,len(self.base_dataset.item_feature_default_value["array_feature"])],
                            self.base_dataset.item_feature_default_value["array_feature"], dtype=np.int32)
        neg_feat_mm_embs = np.full([(self.max_padding_size + 1) * self.neg_sample_size,len(self.base_dataset.item_feature_default_value["mm_emb"])],
                                  self.base_dataset.item_feature_default_value["mm_emb"], dtype=np.float32)


        idx = self.max_padding_size

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # 预处理label，类似于大模型向右填充
        next_i, next_feat, next_type, next_act_type = ext_user_sequence[-1]
        next_feat = self.base_dataset.fill_missing_feat(next_feat, next_i)
        tag = None
        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, dict_feat, type_, act_type = record_tuple
            feat = self.base_dataset.fill_missing_feat(dict_feat, i)
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            if  type_==1:
                seq_feat_sparse[idx], seq_feat_continual[idx], seq_feat_array[idx], seq_feat_mm_embs[idx] = \
                    feat['sparse_feature'], feat['continual_feature'], feat['array_feature'], feat['mm_emb']
            else:
                ...
            if next_type == 1 and next_i != 0:
                pos_feat_sparse[idx], pos_feat_continual[idx], pos_feat_array[idx], pos_feat_mm_embs[idx] = \
                    next_feat['sparse_feature'], next_feat['continual_feature'], next_feat['array_feature'], next_feat['mm_emb']
                # 总感觉这样采样不合适，自回归模型基本没有这样采样的方式
                for j in range( self.neg_sample_size):
                    neg_id = self.base_dataset._random_neq(1, self.base_dataset.itemnum + 1, ts,dict_feat.get('112',None))
                    tag = tag or ts,dict_feat.get('112',None)
                    feature_map = self.base_dataset.fill_missing_feat({}, neg_id,use_cache=True)
                    position = idx + self.max_padding_size * j
                    neg_feat_sparse[position],neg_feat_continual[position],neg_feat_array[position],neg_feat_mm_embs[position]=\
                        feature_map['sparse_feature'], feature_map['continual_feature'], feature_map['array_feature'], feature_map['mm_emb']
            next_i, next_feat, next_type, next_act_type, next_feat = \
                i, feat, type_, act_type, feat
            idx -= 1
            if idx == -1:
                break
        while idx !=-1:
            for j in range( self.neg_sample_size):
                neg_id = self.base_dataset._random_neq(1, self.base_dataset.itemnum + 1, ts,tag)
                feature_map = self.base_dataset.fill_missing_feat({}, neg_id,use_cache=True)
                position = idx + self.max_padding_size * j
                neg_feat_sparse[position],neg_feat_continual[position],neg_feat_array[position],neg_feat_mm_embs[position]=\
                    feature_map['sparse_feature'], feature_map['continual_feature'], feature_map['array_feature'], feature_map['mm_emb']
            idx-=1
        
        # seq_feat = np.where(seq_feat == None, self.base_dataset.item_feature_default_value, seq_feat)
        # pos_feat = np.where(pos_feat == None, self.base_dataset.item_feature_default_value, pos_feat)
        # neg_feat = np.where(neg_feat == None, self.base_dataset.item_feature_default_value, neg_feat)

        return token_type, next_token_type, next_action_type, seq_feat_sparse, seq_feat_continual, seq_feat_array, seq_feat_mm_embs, \
                pos_feat_sparse, pos_feat_continual, pos_feat_array, pos_feat_mm_embs,\
                neg_feat_sparse, neg_feat_continual, neg_feat_array, neg_feat_mm_embs,\
                user_feat_sparse, user_feat_continual, user_feat_array

    
    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
        """
        token_type, next_token_type, next_action_type, seq_feat_sparse, seq_feat_continual, seq_feat_array, seq_feat_mm_embs, \
                pos_feat_sparse, pos_feat_continual, pos_feat_array, pos_feat_mm_embs,\
                neg_feat_sparse, neg_feat_continual, neg_feat_array, neg_feat_mm_embs,\
                user_feat_sparse, user_feat_continual, user_feat_array = zip(*batch)
        return_batch = {}
        return_batch['token_type'] = torch.from_numpy(np.array(token_type))
        return_batch['next_token_type'] = torch.from_numpy(np.array(next_token_type))
        return_batch['next_action_type'] = torch.from_numpy(np.array(next_action_type))
        return_batch['seq_feat'] = {
            "item_sparse_feature": torch.tensor(np.stack(seq_feat_sparse,axis=0)),
            "item_continual_feature": torch.tensor(np.stack(seq_feat_continual,axis=0)),
            "item_array_feature": torch.tensor(np.stack(seq_feat_array,axis=0)),
            "item_mm_embs": torch.tensor(np.stack(seq_feat_mm_embs,axis=0)),
        }
        return_batch['pos_feat'] = {
            "item_sparse_feature": torch.tensor(np.stack(pos_feat_sparse,axis=0)),
            "item_continual_feature": torch.tensor(np.stack(pos_feat_continual,axis=0)),
            "item_array_feature": torch.tensor(np.stack(pos_feat_array,axis=0)),
            "item_mm_embs": torch.tensor(np.stack(pos_feat_mm_embs,axis=0)),
        }
        return_batch['neg_feat'] = {
            "item_sparse_feature": torch.tensor(np.stack(neg_feat_sparse,axis=0)),
            "item_continual_feature": torch.tensor(np.stack(neg_feat_continual,axis=0)),
            "item_array_feature": torch.tensor(np.stack(neg_feat_array,axis=0)),
            "item_mm_embs": torch.tensor(np.stack(neg_feat_mm_embs,axis=0)),
        }
        return_batch['user_feat'] = {
            "user_sparse_feature": torch.tensor(user_feat_sparse),
            "user_continual_feature": torch.tensor(user_feat_continual),
            "user_array_feature": torch.tensor(user_feat_array)
        }
        return return_batch



    def __len__(self):
        return len(self.sample_index)
    

class ValidDataset(torch.utils.data.Dataset):
    base_dataset:BaseDataset # 基础数据集（防止深拷贝）
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
            self.data_file = open(self.base_dataset.data_dir / "seq.jsonl", 'rb')
        uid = self.sample_index[index]
        user_sequence = self.base_dataset._load_user_data(uid,self.data_file)

        return self.build_example(user_sequence)
    def __len__(self):
        return len(self.sample_index)
    
    def build_example(self, user_sequence):
        user_sparse_feature, user_continual_feature, user_array_feature = None,None,None
        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.base_dataset.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self.base_dataset._process_cold_start_feat(user_feat)
                feat = self.base_dataset.fill_missing_feat(user_feat, u,is_user=True)
                user_sparse_feature, user_continual_feature, user_array_feature = feat['sparse_feature'], feat['continual_feature'], feat['array_feature']
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.base_dataset.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self.base_dataset._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1,action_type))
        if user_sparse_feature is None:
            user_sparse_feature, user_continual_feature, user_array_feature = self.base_dataset.user_feature_default_value['sparse_feature'], self.base_dataset.user_feature_default_value['continual_feature'], self.base_dataset.user_feature_default_value['array_feature']
        last_index_action_type_eq_1 = None
        for index,item in enumerate(ext_user_sequence):
            if item[-1]==1:
                last_index_action_type_eq_1=index
        if last_index_action_type_eq_1:
            ext_user_sequence = ext_user_sequence[:last_index_action_type_eq_1+1]
        token_type = np.zeros([self.max_padding_size + 1], dtype=np.int32)
        seq_feat_sparse = np.full([self.max_padding_size + 1,len(self.base_dataset.item_feature_default_value["sparse_feature"])],
                                  self.base_dataset.item_feature_default_value["sparse_feature"], dtype=np.int32)
        seq_feat_continual = np.full([self.max_padding_size + 1,len(self.base_dataset.item_feature_default_value["continual_feature"])]
                                      ,self.base_dataset.item_feature_default_value["continual_feature"], dtype=np.float32)
        seq_feat_array = np.full([self.max_padding_size + 1,len(self.base_dataset.item_feature_default_value["array_feature"])],
                                  self.base_dataset.item_feature_default_value["array_feature"], dtype=np.int32)
        seq_feat_mm_embs = np.full([self.max_padding_size + 1,len(self.base_dataset.item_feature_default_value["mm_emb"])],
                                  self.base_dataset.item_feature_default_value["mm_emb"], dtype=np.float32)
        next_action_type = np.zeros([self.max_padding_size + 1], dtype=np.int32)

        idx = self.max_padding_size

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])
        _next_action_type=1
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_,action_type = record_tuple
            feat = self.base_dataset.fill_missing_feat(feat, i)
            token_type[idx] = type_
            if type_ != 2:
                seq_feat_sparse[idx], seq_feat_continual[idx], seq_feat_array[idx], seq_feat_mm_embs[idx] = \
                    feat["sparse_feature"], feat["continual_feature"], feat["array_feature"], feat["mm_emb"]
            if _next_action_type:
                next_action_type[idx]=_next_action_type
            _next_action_type=action_type
            idx -= 1
            if idx == -1:
                break
        label = ext_user_sequence[-1][0]
        return token_type,next_action_type, seq_feat_sparse,seq_feat_continual,seq_feat_array, seq_feat_mm_embs, user_sparse_feature, user_continual_feature, user_array_feature, label
    
    
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
        token_type,next_action_type, seq_feat_sparse,seq_feat_continual,seq_feat_array, seq_feat_mm_embs, user_sparse_feature, user_continual_feature, user_array_feature, label = zip(*batch)
        return_batch['token_type'] = torch.from_numpy(np.array(token_type))
        return_batch['next_action_type']=torch.from_numpy(np.array(next_action_type))
        return_batch['seq_feat'] = {
            "item_sparse_feature": torch.from_numpy(np.stack(seq_feat_sparse, axis=0)),
            "item_continual_feature": torch.from_numpy(np.stack(seq_feat_continual, axis=0)),
            "item_array_feature": torch.from_numpy(np.stack(seq_feat_array, axis=0)),
            "item_mm_embs": torch.from_numpy(np.stack(seq_feat_mm_embs, axis=0)),
        }
        return_batch['user_feat'] = {
            "user_sparse_feature": torch.tensor(user_sparse_feature),
            "user_continual_feature": torch.tensor(user_continual_feature),
            "user_array_feature": torch.tensor(user_array_feature),
        }
        return_batch['label'] = np.array(label)
        return return_batch
    

class EmbeddingDataset(torch.utils.data.IterableDataset):
    """可以改为预处理的模式，但是我比较懒，所以先用这个模型，先跑起来看效果"""
    base_dataset:BaseDataset # 基础数据集（防止深拷贝）
    sample_index:list # 样本索引
    max_padding_size:int # 最大长度
    feature_map:dict

    def __init__(self, base_dataset, sample_index=[], max_padding_size=100, neg_sample_size = 10, max_cache_size=100000):
        super().__init__()
        self.base_dataset = base_dataset
        self.sample_index = sample_index # 采样索引
        self.orginal_data_size = len(self.sample_index)
        self.original_data_index = 0 # 索引指针用来统计还有多少索引没有被使用
        self.max_padding_size = max_padding_size
        self.feature_map = {}
        self.cache = []
        self.cache_index = 0
        self.max_cache_size = max_cache_size
        self.neg_sample_size = neg_sample_size
        self.data_file = None

    def __iter__(self):
        while True:
            if self.cache_index < len(self.cache):
                res = self.cache[self.cache_index]
            
            else:
                self.build_example_cache()
                if len(self.cache)==0:
                    self.original_data_index=0
                    break
                res = self.cache[self.cache_index]
            self.cache_index += 1
            u, pos1, pos2, neg1_list = res
            user_feat = self.feature_map[u]
            pos1_feat = self.feature_map[pos1]
            pos2_feat = self.feature_map[pos2]
            neg1_feat_list = []
            for neg1 in neg1_list:
                neg1_feat_list.append(self.feature_map[neg1])
            yield  u, pos1, pos2, neg1_list, user_feat, pos1_feat, pos2_feat, neg1_feat_list
    
    def build_example_cache(self):
        self.cache = []
        self.cache_index = 0
        self.feature_map = {}
        print("building new cache")
        while self.original_data_index < self.orginal_data_size and len(self.cache)<self.max_cache_size:
            data_pairs = []
            click_item_sequence = []
            explore_item_sequence = []
            uid = self.sample_index[self.original_data_index]
            self.original_data_index+=1
            if self.data_file is None:
                self.data_file = open(self.base_dataset.data_dir / "seq.jsonl", 'rb')
            user_sequence = self.base_dataset._load_user_data(uid, self.data_file)
            ts = set()
            u=None
            for record_tuple in user_sequence:
                u, i, user_feat, item_feat, action_type, timestamp = record_tuple
                if u and user_feat: # 插入第一位
                    cur_u=u
                    if u not in self.feature_map:
                        self.feature_map[u] = self.base_dataset.fill_missing_feat(user_feat,u)
                if i and item_feat:
                    if action_type == 1:
                        click_item_sequence.append(i)
                        if i not in self.feature_map:
                            self.feature_map[i] = self.base_dataset.fill_missing_feat(item_feat,i)
                        ts.add(i)
                    else:
                        explore_item_sequence.append(i)
                        ts.add(i)
            if not cur_u:
                continue
            for cur,item_index in enumerate(click_item_sequence[:-1]):
                pos1 = item_index
                pos2 = click_item_sequence[cur+1]
                neg1_list = []
                for _ in range(self.neg_sample_size):
                    neg1 = self.base_dataset._random_neq(1, self.base_dataset.itemnum + 1, ts) # 随机采集负样本
                    if random.random() < 0.01:
                        # 难样本
                        neg1 = random.choice(explore_item_sequence)
                    if neg1 not in self.feature_map:
                        self.feature_map[neg1] = self.base_dataset.fill_missing_feat(self.base_dataset.item_feat_dict[str(neg1)],neg1)
                    neg1_list.append(neg1)
                data_pairs.append((cur_u, pos1, pos2, neg1_list))
            self.cache.extend(data_pairs)
            # print(f"process {len(self.cache)}/{self.max_cache_size}")
        random.shuffle(self.cache)

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
        user, input_item, pos1_item, neg1_item_list, user_featrue, input_item_feature, pos1_item_feature, neg1_item_feature_list = zip(*batch)
        # seq = torch.from_numpy(np.array(seq))
        # token_type = torch.from_numpy(np.array(token_type))
        # seq_feat = list(seq_feat)
        user = torch.tensor(user).unsqueeze(0)
        input_item = torch.tensor(input_item).unsqueeze(0)
        pos1_item = torch.tensor(pos1_item).unsqueeze(0)
        neg1_item_list = torch.tensor(neg1_item_list)
        user_featrue = [list(user_featrue)]
        input_item_feature = [list(input_item_feature)]
        pos1_item_feature = [list(pos1_item_feature)]
        neg1_item_feature_list = list(neg1_item_feature_list)
        #
        return user, input_item, pos1_item, neg1_item_list, user_featrue, input_item_feature, pos1_item_feature, neg1_item_feature_list