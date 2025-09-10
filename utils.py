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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    
def format_time(seconds):
    """将秒转换为 HH:MM:SS 格式"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"