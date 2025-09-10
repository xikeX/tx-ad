# main.py
import os
import pickle
if os.environ.get("DEBUG_MODE","") == 'True':
    print("in debug model")
    os.environ['TRAIN_LOG_PATH']='./log_path'
    os.environ['TRAIN_TF_EVENTS_PATH']='./log_path'
    os.environ['TRAIN_DATA_PATH']='../TencentGR_1k'
    os.environ['TRAIN_CKPT_PATH']='./checkpoint'
    os.environ['USER_CACHE_PATH']='../user_cache_file'
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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyTestDataset
from infer import Infer
from model import BaselineModel



def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default="global_step_2/model.pt", type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args
def main():
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyTestDataset(data_path, args)
    with open(os.environ.get("USER_CACHE_PATH")+"/valid_idx.pkl", "rb") as f:
        valid_idx = pickle.load(f)
    dataset.sample_index = valid_idx
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    if args.state_dict_path is not None:
        try:
            print(os.environ.get("USER_CACHE_PATH")+"/" + args.state_dict_path)
            model.load_state_dict(torch.load(os.environ.get("USER_CACHE_PATH")+"/" + args.state_dict_path, map_location=torch.device(args.device)))
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')



    print("Start Evaluation")
    infer = Infer(args,model,eval_dataset=dataset,candidate_path=os.path.join(os.environ.get('USER_CACHE_PATH'),'item_feat_dict_eval.json'))
    hit_rate = infer.infer()
    print(f"{hit_rate=}")

if __name__ == "__main__":
    main()

# infer_main.py