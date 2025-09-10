# infer_class_v1.py
import argparse
import os
# os.environ['TRAIN_LOG_PATH'] = './log'
# os.environ['TRAIN_TF_EVENTS_PATH'] = './log'
# os.environ['TRAIN_DATA_PATH'] = '/HOME/hitsz_mszhang/hitsz_mszhang_1/RL_ERC/tecent_recommondation/TencentGR_1k'
# os.environ['TRAIN_CKPT_PATH'] = './checkpoint'
# os.environ['EVAL_RESULT_PATH'] = './eval_result'
# os.environ['DEBUG_MODE'] = 'True'
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import os
from pathlib import Path
import struct
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MyDataset, MyTestDataset, save_emb
from faiss_demo_gpu import run_faiss_ann_search
from torch.utils.tensorboard import SummaryWriter

import pickle

from my_dataset_preprocess import load_mm_emb
from utils import format_time
# from train_embedding_dataset import TrainEmbeddingDataset
# 本地infer算法
def final_score(hit_rate,ndcg):
    return 1/3 * hit_rate + 2/3 * ndcg

def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))



class Infer:
    def __init__(self,args, model, eval_dataset, candidate_path,name,query_ann_top_k=10):
        self.model:torch.nn.Module = model
        self.eval_dataset:MyTestDataset = eval_dataset
        self.eval_dataloader = DataLoader(
           self.eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=3 if args.num_worker != 0 else False,
            prefetch_factor=5 if args.num_worker != 0 else None,
            collate_fn=self.eval_dataset.collate_fn
        )
        if isinstance(self.eval_dataset, torch.utils.data.Dataset):
            self.total_data_size = len(self.eval_dataset)
        else:
            self.total_data_size = self.eval_dataset.base_dataset.total_data_size
        self.total_batch = self.total_data_size // args.batch_size
        # 再eval_dataset 中带有label_item,同时label也应该在condidate中
        self.candidate_path = candidate_path
        self.device = args.device
        self.name = name
        self.query_ann_top_k = query_ann_top_k
        self.mm_emb_ids=args.mm_emb_id

    def infer(self):
        self.model.eval()
        all_embs = []
        # user_list = []
        labels = []
        # 预测下一个query的embeding
        start_time = round(time.time())
        cnt = 0
        max_size = len(self.eval_dataloader)
        
        for step, batch in enumerate(self.eval_dataloader):
            # with open("t2.pkl",'wb') as f:
            #     pickle.dump(batch,f) 
            token_type, seq_feat, user_feat, label = \
                batch['token_type'], batch['seq_feat'], batch['user_feat'], batch['label']
            for m in [seq_feat, user_feat, user_feat]:
                for k in m:
                    m[k] = m[k].to(self.device)
            next_action_type = batch['next_action_type'].to(self.device)
            token_type = token_type.to(self.device)
            with torch.no_grad():
                logits = self.model.predict(seq_feat, user_feat, token_type,next_action_type)
            all_embs.append(logits.detach().cpu().numpy().astype(np.float32))
            # user_list.extend(user_id)
            labels.extend([i for i in label])
            cnt+=1
            if cnt%1000==0:
                end_time = round(time.time())
                use_time = format_time(end_time - start_time)
                remain_time = format_time((end_time - start_time) / (cnt) * max_size)
                print(f"infer {self.name} featrue [{use_time}/{remain_time}] [{cnt}/{max_size}]")
        query_vector = np.concatenate(all_embs, axis=0)
        save_emb(query_vector, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))

        # 构建找回池query的向量，返回query词的物品id
        print("进入 get_candidate_emb")
        item_ids, dataset_vector,dataset_id = self.get_candidate_emb()
        # ann_cmd = (
        #     str(Path("/workspace", "faiss-based-ann", "faiss_demo"))
        #     + " --dataset_vector_file_path="
        #     + str(Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin"))
        #     + " --dataset_id_file_path="
        #     + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin"))
        #     + " --query_vector_file_path="
        #     + str(Path(os.environ.get("EVAL_RESULT_PATH"), "query.fbin"))
        #     + " --result_id_file_path="
        #     + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
        #     + " --query_ann_top_k=10 --faiss_M=64 --faiss_ef_construction=1280 --query_ef_search=640 --faiss_metric_type=0"
        # )
        print(f"进入run_faiss_ann_search")
        top10s_retrieved,avg_distance = run_faiss_ann_search(
            dataset_vector = dataset_vector,
            dataset_id = dataset_id,
            query_vector = query_vector,
            query_ann_top_k = self.query_ann_top_k
        )
        # 为什么要这样读写文件？？？
        
        # ANN 检索
        # top10s_untrimmed = []
        # for top10 in tqdm(top10s_retrieved):
        #     for item in top10:
        #         top10s_untrimmed.append(item)
        # top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]
        # 根据labels和ids计算hit_rate和ndcg
        hit_rate = sum([labels[i] in top10s_retrieved[i] for i in range(len(labels))])/len(labels)
        return hit_rate, avg_distance
    def get_candidate_emb(self):
        """
        生产候选库item的id和embedding

        Args:
            indexer: 索引字典
            feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
            feature_default_value: 特征缺省值
            mm_emb_dict: 多模态特征字典
            model: 模型
        Returns:
            retrieve_id2creative_id: 索引id->creative_id的dict
        """
        indexer = self.eval_dataset.base_dataset.indexer
        feat_types = self.eval_dataset.base_dataset.feature_types
        feat_default_value = self.eval_dataset.base_dataset.feature_default_value
        mm_emb_dict = load_mm_emb(Path(os.environ['TRAIN_DATA_PATH'], "creative_emb"), self.mm_emb_ids)
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        candidate_path = Path(self.candidate_path)
        item_ids, creative_ids, retrieval_ids, features = [], [], [], []
        retrieve_id2creative_id = {}
        temp_file = os.environ.get("TEMP_PATH", "./temp")
        data_file = Path(temp_file, f"infer_data_{self.name}.pkl")
        if  os.path.exists(data_file):
            print(f"Load infer data from {data_file}")
            with open(data_file, "rb") as f:
                item_ids, creative_ids, retrieval_ids, features = pickle.load(f)
        else:
            # 感觉可以开多线程处理
            print("开始处理数据")
            with open(candidate_path, 'r') as f:
                condidate_map = json.load(f)
                max_size = len(condidate_map)
                start_time = time.time()
                cnt=0
                for item_id in condidate_map:
                    item_id = int(item_id)
                    # 读取item特征，并补充缺失值
                    feature = condidate_map[str(item_id)]
                    if os.environ.get("DEBUG_MODE","")=="True":
                        if item_id in self.eval_dataset.base_dataset.indexer_i_rev:
                            creative_id = self.eval_dataset.base_dataset.indexer_i_rev[item_id] 
                        else: 
                            creative_id = 0
                    else:
                        creative_id = self.eval_dataset.base_dataset.indexer_i_rev[item_id]
                    # 1，冷启动特征，相当于噪声，仅使用训练的特征
                    # 2. 补充缺失值，在fill_missing_feat 中已经可以处理了。
                    missing_fields = set(
                        feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
                    ) - set(feature.keys())
                    feature = self.process_cold_start_feat(feature)
                    for feat_id in missing_fields:
                        feature[feat_id] = feat_default_value[feat_id]
                    for feat_id in feat_types['item_emb']:
                        if creative_id in mm_emb_dict[feat_id]:
                            feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                        else:
                            feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)
                    item_ids.append(item_id)
                    creative_ids.append(creative_id)
                    features.append(self.eval_dataset.base_dataset.fill_missing_feat(feature, item_id, False))
                cnt+=1
                if cnt%1000==0:
                    print(cnt)
                    end_time = round(time.time())
                    use_time = format_time(end_time - start_time)
                    remain_time = format_time((end_time - start_time) / (cnt) * max_size)
                    print(f"create {self.name} featrue [{use_time}/{remain_time}] [{cnt}/{max_size}]")
                # pickle.dump(creative_ids, open("data_v1.pkl",'wb'))
            print("处理完成")
            with open(data_file, "wb") as f:  # ← 注意是 "wb"，不是 "w"
                pickle.dump([item_ids, creative_ids, retrieval_ids, features], f)
        # 写到临时文件中等下一次直接读取
        print("处理结束")

        dataset_vector,dataset_id = self.model.save_item_emb(item_ids, item_ids, features, os.environ.get('EVAL_RESULT_PATH'))
        with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
            json.dump(retrieve_id2creative_id, f)
        print("写入成功")
        return item_ids, dataset_vector,dataset_id

    def process_cold_start_feat(self, feat):
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
