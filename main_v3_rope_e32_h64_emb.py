# 标准库
import argparse
import gc
import json
import os
import pickle
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# 第三方库
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

# 本地自定义模块
from baseline_model_v3_infonce_e32_h64_emb import BaselineModel as DownstreamModel
from infer_class_v1 import Infer
# from my_dataset_preprocess import MMPBaseDataset as BaseDataset
from embeding_dataset import MMPBaseDataset as BaseDataset
from embeding_dataset import EmbeddingDataset
from utils import analyze_top_level_layers, set_environment, format_time

set_environment()


def get_args():
    parser = argparse.ArgumentParser()

    # ================== 数据相关参数 ==================
    parser.add_argument('--train_data_size', type=int, default=None, help='训练数据大小（-1 表示全量）')
    parser.add_argument('--batch_size', type=int, default=32, help='训练/验证批大小')
    parser.add_argument('--maxlen', type=int, default=101, help='序列最大长度')
    parser.add_argument('--num_worker', type=int, default=0, help='序列最大长度')
    parser.add_argument('--train_name', type=str, default="v3-embed", help='训练名称')

    # ================== 模型结构参数 ==================
    parser.add_argument('--hidden_units', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--num_blocks', type=int, default=12, help='Transformer 块数')
    parser.add_argument('--num_heads', type=int, default=4, help='多头注意力头数')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout 比例')
    parser.add_argument('--l2_emb', type=float, default=0.0, help='嵌入层 L2 正则强度')
    parser.add_argument('--norm_first', action='store_true', help='是否在 Transformer 中先归一化（Pre-LN）')
    parser.add_argument('--mm_emb_id', nargs='+', type=str, default=['81'], choices=[str(s) for s in range(81, 87)], help='多模态嵌入特征 ID 列表')
    
    # ================== 训练优化参数 ==================
    parser.add_argument('--num_epochs', type=int, default=5, help='训练总轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='下游任务学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减（AdamW 等优化器使用）')

    # ================== 运行环境 ==================
    parser.add_argument('--device', type=str, default='', help='运行设备: cpu 或 cuda')
    parser.add_argument('--reflesh_cache', action='store_true', help='刷新运行时的cache')

    parser.add_argument('--state_dict_path', type=str, default=None, help='预训练权重路径')
    # 解析参数
    args = parser.parse_args()
    args.norm_first=True
    # 自动设置 device
    if args.device == '':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args


import torch.nn as nn

def initialize_model_weights(model: nn.Module):
    """递归初始化模型权重，使用 Kaiming 初始化（适用于 ReLU/GELU 激活函数）"""

    def init_weights(m):
        if isinstance(m, nn.Linear):
            # Kaiming 正态分布初始化，适用于 ReLU/GELU
            nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.Embedding):
            # Embedding 层通常仍用小正态分布
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        elif isinstance(m, nn.Conv1d):
            # Conv1d 同样使用 Kaiming 初始化
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            # 对其他层，如果有 weight 且未特别处理，尝试 Kaiming
            if hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):
                if m.weight.ndimension() >= 2:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # bias 一般置零
                if hasattr(m, 'bias') and isinstance(m.bias, nn.Parameter):
                    nn.init.zeros_(m.bias)

    model.apply(init_weights)


def evalution(model, valid_loader, global_step, args, writer):
    model.eval()
    record = defaultdict(float)
    val_batches = 0
    with torch.no_grad():
        for batch in valid_loader:
            for key_1 in batch:
                if isinstance(batch[key_1],dict):
                    for key in batch[key_1]:
                        batch[key_1][key] = batch[key_1][key].to(args.device)
                else:
                    batch[key_1] = batch[key_1].to(args.device)
            # 前向传播
            output = model(**batch)
            if hasattr(model, 'eval_record'):
                for record_key in model.eval_record:
                    record[record_key] += output[record_key]
            val_batches += 1

    for key in record:
        record[key] /= val_batches
        writer.add_scalar(f'Eval_Loss/{key}', record[key], global_step)
    
    return record['total_loss']


def print_train_log(writer, epoch, step, global_step, model, optimizer, train_loader, output, start_time, total_loss, epoch_size):
    if global_step % 100 == 0 or os.environ.get('DEBUG_MODE', "") == "True":
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], global_step)
        if hasattr(model, 'train_record'):
            for record_key in model.train_record:
                writer.add_scalar(f'Train/{record_key}', output[record_key], global_step)
    if global_step % 100 == 0 and step != 0:
        end_time = round(time.time())
        use_time = format_time(end_time - start_time)
        remain_time = format_time((end_time - start_time) / step * epoch_size)
        msg = f"[{use_time}/{remain_time}]"
        msg += f"[{step}/{len(train_loader)}]"
        msg += f"global_step:{global_step} "
        msg += f"epoch{epoch} "
        msg += f"total_loss:{total_loss.item():.5f} "
        msg += f"lr:{optimizer.param_groups[0]['lr']:0.5f}"
        if hasattr(model, 'train_record'):
            for record_key in model.train_record:
                msg += f" {record_key}:{output[record_key]:0.5f}"
        print(msg + '\n')


def train_model(args, model, train_loader, scaler, optimizer, scheduler, writer, epoch, global_step):
    model.train()
    total_loss_epoch = 0.0
    start_time = round(time.time())
    for step, batch in enumerate(train_loader):
        # 解包数据
        for key_1 in batch:
            if isinstance(batch[key_1],dict):
                for key in batch[key_1]:
                    batch[key_1][key] = batch[key_1][key].to(args.device)
            else:
                batch[key_1] = batch[key_1].to(args.device)
                
                    
        if scaler:
            with torch.amp.autocast('cuda'):
                output = model(**batch)
        else:
            # 前向传播
            output = model(**batch)

        total_loss = output['total_loss']
        if args.l2_emb > 0:
            l2_reg = 0.0
            for param in model.item_emb.parameters():
                l2_reg += torch.norm(param)
            total_loss += args.l2_emb * l2_reg

        if scaler:
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            # 更新参数
            # 输出所有层的梯度到writer
            # for name, param in model.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         writer.add_scalar(name + '_grad', param.grad.norm(), global_step)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            total_loss.backward()
            # ✅ 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if scheduler:
            scheduler.step()
        print_train_log(writer=writer, epoch=epoch, step=step, global_step=global_step, model=model, optimizer=optimizer, train_loader=train_loader, output=output, start_time=start_time,
                        total_loss=total_loss, epoch_size=len(train_loader))
        total_loss_epoch += total_loss.item()
        global_step += 1
    avg_train_loss = total_loss_epoch / len(train_loader)
    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f}")
    return global_step


def train_downstream_model(model, train_dataset, valid_dataset, args, writer, test_dataset=None, test_dataset_2=None):

    print("开始下游任务训练...")

    # ✅ 添加：学习率调度器（可选：ReduceLROnPlateau）
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=getattr(args, 'weight_decay', 0.0))
    total_steps = args.num_epochs * len(train_dataset)//args.batch_size
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps * 0.01,  # 例如：1000 步 warm-up
        num_training_steps=total_steps  # 总训练步数
    )
    if os.environ.get('DEBUG_MODE', "") == "":
        model = torch.compile(model)
    scaler = torch.amp.GradScaler("cuda") if os.environ.get("DEBUG_MODE", "") == "True" else None

    best_eval_loss = -float('inf')  # ✅ 记录最佳验证损失
    global_step = 0
    # model.special_embedding_apply()
    for epoch in range(1, args.num_epochs + 1):
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True,
            num_workers=args.num_worker,
            prefetch_factor=5 if args.num_worker != 0 else None,
        )
        # ========== 训练阶段 ==========
        global_step = train_model(args, model, train_loader, scaler, optimizer, scheduler, writer, epoch, global_step)
        train_loader=None
        gc.collect()
        # ========== 验证阶段 ==========
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=valid_dataset.collate_fn,
            pin_memory=True,
            num_workers=args.num_worker,
            prefetch_factor=5 if args.num_worker != 0 else None,
        )
        eval_loss = evalution(model, valid_loader, global_step, args, writer)
        valid_loader=None
        gc.collect()
        # ========== 评估阶段 ==========

        if eval_loss > best_eval_loss:
            best_eval_loss = eval_loss
            save_dir = Path(os.environ.get('USER_CACHE_PATH')) / args.train_name
            # 删除旧的模型文件
            if os.path.exists(save_dir) and len(os.listdir(save_dir)) != 0:
                for folder in os.listdir(save_dir):
                    folder = os.path.join(save_dir, folder)
                    if os.path.isdir(folder):
                        shutil.rmtree(folder)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "model.pt")
            print(f"✅ 最佳模型已保存至: {save_dir / 'model.pt'}")

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH')) / f"global_step{global_step}.hitrate={best_eval_loss:.4f}"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")
        print(f"✅ 模型已保存至: {save_dir / 'model.pt'}")

        # if test_dataset_2 and os.environ.get("DEBUG_MODE", "") == "True":
        gc.collect()
    return model


def pre_define():
    # 创建日志与事件目录
    Path(os.environ['TRAIN_LOG_PATH']).mkdir(parents=True, exist_ok=True)
    Path(os.environ['TRAIN_TF_EVENTS_PATH']).mkdir(parents=True, exist_ok=True)
    if os.environ.get('DEBUG_MODE', "") == "True":
        writer = SummaryWriter(Path(os.environ['TRAIN_TF_EVENTS_PATH']) / datetime.now().strftime('%H-%M-%S'))
    else:
        writer = SummaryWriter(os.environ['TRAIN_TF_EVENTS_PATH'])
    return writer


def create_train_dataset(args):
    base_dataset = BaseDataset(os.environ['TRAIN_DATA_PATH'], args)
    train_idx, valid_idx = base_dataset.split_index([0.9, 0.1], args.train_data_size)
    # if args.train_data_size == -1:
    #     args.train_data_size = None
    # if args.reflesh_cache or not os.path.exists(os.environ.get("USER_CACHE_PATH") + "/train_idx.pkl"):
    #     with open(os.environ.get("USER_CACHE_PATH") + "/train_idx.pkl", "wb") as f:
    #         pickle.dump(train_idx, f)
    # if args.reflesh_cache or not os.path.exists(os.environ.get("USER_CACHE_PATH") + "/valid_idx.pkl"):
    #     with open(os.environ.get("USER_CACHE_PATH") + "/valid_idx.pkl", "wb") as f:
    #         pickle.dump(valid_idx, f)
    train_dataset = EmbeddingDataset(base_dataset, sample_index=train_idx)  # 全量训练
    valid_dataset = EmbeddingDataset(base_dataset, sample_index=valid_idx)
    return base_dataset, train_dataset, valid_dataset




def main():
    writer = pre_define()
    args = get_args()

    # 数据加载
    base_dataset, train_dataset, valid_dataset = create_train_dataset(args)

    #模型加载
    downstream_model = DownstreamModel(base_dataset.usernum, base_dataset.itemnum, base_dataset.feat_statistics, base_dataset.feature_types, args).to(args.device)
    initialize_model_weights(downstream_model)
    if args.state_dict_path:
        try:
            downstream_model.load_state_dict(torch.load(args.state_dict_path, map_location=args.device))
            print(f"✅ 已加载预训练权重: {args.state_dict_path}")
        except Exception as e:
            print(f"⚠️ 权重加载失败: {e}")

    # 模型训练
    analyze_top_level_layers(downstream_model)
    train_downstream_model(downstream_model, train_dataset, valid_dataset, args, writer)

    # 加载最佳模型权重
    try:
        path = Path(os.environ.get('USER_CACHE_PATH')) / args.train_name
        path = path / "model.pt"
        downstream_model.load_state_dict(torch.load(path))
        print(f"✅ 加载最佳模型权重: {path}")
    except Exception as e:
        print(f"⚠️ 权重加载失败: {e}")

    # 模型评估
    # print("✅ 推理开始")
    # eval_candidate_path = os.path.join(os.environ['TRAIN_DATA_PATH'], 'item_feat_dict.json')
    # infer = Infer(args, downstream_model, eval_dataset=valid_dataset_valid, candidate_path=eval_candidate_path, name='global_test', query_ann_top_k=10)
    # hitrate_eval, avg_distance = infer.infer()
    # print(f"✅ 验证集评估结果: {hitrate_eval=} {avg_distance=}", )
    # train_candidate_path = os.path.join(os.environ['TRAIN_DATA_PATH'], 'item_feat_dict.json')
    # infer = Infer(args, downstream_model, eval_dataset=train_dataset_valid, candidate_path=train_candidate_path, name='global_train', query_ann_top_k=10)
    # hitrate_train, avg_distance = infer.infer()
    # print(f"✅ 训练集评估结果: {hitrate_train=} {avg_distance=}", )
    # 清理资源
    writer.close()


if __name__ == '__main__':
    main()