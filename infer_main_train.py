# main.py
from datetime import datetime
import os
import pickle
import random

from my_dataset_v1_aug import BaseDataset, ValidDataset
if os.environ.get("DEBUG_MODE","") == 'True':
    print("in debug model")
    os.environ['TRAIN_LOG_PATH']='./log_path'
    os.environ['TRAIN_TF_EVENTS_PATH']='./log_path'
    os.environ['TRAIN_DATA_PATH']='./TencentGR_1k'
    os.environ['TRAIN_CKPT_PATH']='./checkpoint'
    os.environ['USER_CACHE_PATH']='./user_cache_file'
    os.makedirs(os.environ['USER_CACHE_PATH'], exist_ok = True)
    os.environ['DEBUG_MODE'] = 'True'
    # 设置单GPU
# 获取当前目录（'.'）的绝对路径
current_abs_path = os.path.abspath('./temp')
os.environ['TEMP_PATH'] = current_abs_path
os.makedirs(os.environ['TEMP_PATH'], exist_ok = True)
current_abs_path = os.path.abspath('./eval_result')
os.environ['EVAL_RESULT_PATH'] = current_abs_path
os.makedirs(os.environ['EVAL_RESULT_PATH'], exist_ok = True)
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from infer_class_v1 import Infer
from baseline_model_v7_01 import BaselineModel



def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--hidden_units', type=int, default=32, help='隐藏层维度')
    parser.add_argument('--num_blocks', type=int, default=1, help='Transformer 块数')
    parser.add_argument('--num_heads', type=int, default=1, help='多头注意力头数')
    # Baseline Model construction
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default="v3", type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--train_name',type=str, default="v7-hstu", help='训练名称')
    parser.add_argument('--num_worker',type=int, default=0, help='训练线程')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args
def set_seed(seed=42):
    """设置随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    
def main():
    set_seed(42)

    # 环境变量检查
    required_env_vars = ['TRAIN_LOG_PATH', 'TRAIN_TF_EVENTS_PATH', 'TRAIN_DATA_PATH', 'USER_CACHE_PATH']
    for var in required_env_vars:
        if not os.environ.get(var):
            raise EnvironmentError(f"缺少必需的环境变量: {var}")

    # 创建日志与事件目录
    Path(os.environ['TRAIN_LOG_PATH']).mkdir(parents=True, exist_ok=True)
    Path(os.environ['TRAIN_TF_EVENTS_PATH']).mkdir(parents=True, exist_ok=True)
    if os.environ.get('DEBUG_MODE',"")=="True":
        writer = SummaryWriter(Path(os.environ['TRAIN_TF_EVENTS_PATH'])/datetime.now().strftime('%H-%M-%S'))
    else:
        writer = SummaryWriter(os.environ['TRAIN_TF_EVENTS_PATH'])

    args = get_args()

    # 数据加载
    data_path = os.environ['TRAIN_DATA_PATH']
    base_dataset = BaseDataset(data_path, args)
    # 构建嵌入训练与验证数据集
    usernum, itemnum = base_dataset.usernum, base_dataset.itemnum
    feat_stats, feat_types = base_dataset.feat_statistics, base_dataset.feature_types
    with open(os.environ.get("USER_CACHE_PATH")+"/train_idx.pkl", "rb") as f:
        valid_idx = pickle.load(f)
    # ========== 推理 ==========
    test_dataset = ValidDataset(base_dataset, sample_index=valid_idx)  # 可替换为独立测试集

    # v2/global_step_3/model.pt

    usernum, itemnum = base_dataset.usernum, base_dataset.itemnum
    feat_statistics, feat_types = base_dataset.feat_statistics, base_dataset.feature_types

    downstream_model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    save_dir = Path(os.environ.get('USER_CACHE_PATH')) / args.train_name
    print(f"{save_dir=}")
    # 删除旧的模型文件
    if os.path.exists(save_dir) and os.path.isdir(save_dir) and args.train_name:
        print(f"folder: {os.listdir(save_dir)}")
        # if len(os.listdir(save_dir))!=0:
        #     for i in os.listdir(save_dir):
        #         if i !='model.pt':
        #             args.state_dict_path = save_dir/i/"model.pt"
        #             break
        args.state_dict_path = Path(os.environ.get('USER_CACHE_PATH')) /args.train_name /"model.pt"
        print(f"new_path {args.state_dict_path}!!!")
    else:
        args.state_dict_path =Path(os.environ.get('USER_CACHE_PATH'))/args.state_dict_path
    if args.state_dict_path is not None:
        try:
            print(f"{args.state_dict_path=}")
            downstream_model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            downstream_model = downstream_model.to(args.device).float()
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')


    print("评估开始")
    eval_candidate_path = os.path.join(os.environ['TRAIN_DATA_PATH'], 'item_feat_dict.json')
    infer = Infer(args, downstream_model, eval_dataset=test_dataset, candidate_path=eval_candidate_path,name='final_global_train',query_ann_top_k=10)
    hitrate_eval = infer.infer()
    print("✅ 评估结果")
    print("eval:", hitrate_eval)

if __name__ == "__main__":
    main()

# infer_main.py