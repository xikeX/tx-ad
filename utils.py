import os
import random

import torch
import numpy as np

def set_environment():
    """
    设置训练所需的环境变量和目录。
    在 DEBUG_MODE 下使用本地路径，否则依赖外部环境变量配置。
    """
    # 检查是否处于调试模式
    debug_mode = os.environ.get("DEBUG_MODE", "").lower() == 'true'
    os.environ['DEBUG_MODE'] = 'True' if debug_mode else 'False'

    if debug_mode:
        print("Running in debug mode")

        # 定义调试模式下的路径映射
        debug_paths = {
            'TRAIN_LOG_PATH': './log_path',
            'TRAIN_TF_EVENTS_PATH': './log_path',
            'TRAIN_DATA_PATH': './TencentGR_1k',
            'TRAIN_CKPT_PATH': './checkpoint',
            'USER_CACHE_PATH': './user_cache_file',
            'TEMP_PATH': './temp',
            'EVAL_RESULT_PATH': './eval_result'
        }

        # 创建目录并设置环境变量
        for env_key, path in debug_paths.items():
            abs_path = os.path.abspath(path)
            os.makedirs(abs_path, exist_ok=True)
            os.environ[env_key] = abs_path

    else:
        # 生产模式：确保必要环境变量已设置，否则使用默认值或抛出错误
        default_paths = {
            'TEMP_PATH': './temp',
            'EVAL_RESULT_PATH': './eval_result'
        }
        for env_key, default_path in default_paths.items():
            if not os.environ.get(env_key):
                abs_path = os.path.abspath(default_path)
                os.makedirs(abs_path, exist_ok=True)
                os.environ[env_key] = abs_path
    set_seed()

def set_seed(seed=42):
    """设置随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.set_num_threads(1)
    
def format_time(seconds):
    """将秒转换为 HH:MM:SS 格式"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

import torch

def analyze_top_level_layers(model, total_size_gb=3.1):
    """
    统计模型第一级子模块的参数量和内存占比

    Args:
        model (torch.nn.Module): 模型
        total_size_gb (float): 模型总大小 (GB)
    """
    print(f"{'Layer Name':<30} {'Size (GB)':<12} {'Percentage':<12}")
    print("-" * 50)

    total_computed = 0.0
    layer_stats = []

    # 遍历第一级子模块
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        size_gb = num_params * 4 / (1024**3)  # float32: 4 bytes
        total_computed += size_gb

        layer_stats.append([name, size_gb, size_gb])
    for item in layer_stats:
        item[2]/=total_computed
    # 按大小降序排列
    layer_stats.sort(key=lambda x: x[1], reverse=True)

    # 打印每一层
    for name, size_gb, percentage in layer_stats:
        print(f"{name:<30} {size_gb:<12.4f} {percentage:<12.4f}%")

    # 打印总计
    print("-" * 50)
    print(f"{'TOTAL (computed)':<30} {total_computed:<12.4f} {100.0:<12.4f}%")
    print(f"{'TOTAL (given)':<30} {total_size_gb:<12.4f} {100.0:<12.4f}%")

    if abs(total_computed - total_size_gb) > 0.1:
        print(f"[⚠️] 注意：计算值 ({total_computed:.2f}GB) 与给定值 ({total_size_gb}GB) 差异较大")