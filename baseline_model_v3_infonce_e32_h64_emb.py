# baseline_model_v1.py
from pathlib import Path
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from dataset import save_emb
from utils import format_time

class ClassificationHead(nn.Module):
    """
    一个通用的分类头，用于接在 item embedding 或表示层之后。
    
    参数:
        input_dim (int): 输入特征的维度，例如 self.attention_hidden_units
        num_classes (int): 分类任务的类别总数
        hidden_units (list): 各隐藏层的神经元数量，默认为 [input_dim]（即一层）
        dropout_rate (float): Dropout 概率，默认 0.1
        activation (str): 激活函数类型，支持 'relu', 'gelu', 'tanh'，默认 'relu'
        use_layer_norm (bool): 是否在每个线性层后加 LayerNorm，默认 False
    """
    def __init__(self, input_dim, num_classes, hidden_units=None, dropout_rate=0.1, activation='relu', use_layer_norm=False):
        super(ClassificationHead, self).__init__()
        
        # 默认使用一个与输入维度相同的隐藏层
        if hidden_units is None:
            hidden_units = [input_dim]
        
        layers = []
        in_features = input_dim
        
        for units in hidden_units:
            layers.append(nn.Linear(in_features, units))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(units))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            
            layers.append(nn.Dropout(dropout_rate))
            in_features = units
        
        # 输出层
        layers.append(nn.Linear(in_features, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        输入:
            x: shape [batch_size, input_dim]
        输出:
            logits: shape [batch_size, num_classes]
        """
        return self.classifier(x)

def info_nce_loss(arch_emb, pos_emb, temperature=0.1):
    device = arch_emb.device
    B, D = arch_emb.shape

    # 归一化
    arch_emb = F.normalize(arch_emb, dim=-1)
    pos_emb = F.normalize(pos_emb, dim=-1)

    # 拼接 [2B, D]
    z = torch.cat([arch_emb, pos_emb], dim=0)  # [2B, D]

    # 计算相似度矩阵 [2B, 2B]
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)  # [2B, 2B]
    sim /= temperature

    # ========== 提取监控指标（复用 sim，无额外计算）==========
    with torch.no_grad():
        # 正样本相似度：arch_emb[i] vs pos_emb[i] → sim[i, i+B]
        pos_sim = sim[range(B), range(B, 2*B)].mean()

        # 负样本相似度：所有非正样本对
        # 创建 mask：排除 (i, i+B) 和 (i+B, i)
        neg_mask = torch.ones(2*B, 2*B, dtype=torch.bool, device=device)
        neg_mask[range(B), range(B, 2*B)] = False  # arch[i] vs pos[i]
        neg_mask[range(B, 2*B), range(B)] = False  # pos[i] vs arch[i]
        # 可选：也可以排除自相似（i,i），但通常很小
        # neg_mask[range(2*B), range(2*B)] = False

        neg_sim = sim[neg_mask].mean()
    # ====================================================

    # ========== 计算损失 ==========
    # logits: [B, 2B], 以 arch_emb 为 anchor
    logits = sim[:B]  # [B, 2B]

    # label: 正样本是 pos_emb[i]，索引为 i + B
    labels = torch.arange(B, 2*B, device=device)  # [B]

    loss = F.cross_entropy(logits, labels)
    # ==============================

    return loss, pos_sim, neg_sim
class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads,
                   self.head_dim).transpose(1, 2)
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)
        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(
                    1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(
                attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)

        return output, None


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        # 方案一：标准 SwiGLU（推荐：扩展中间维度）
        expansion_factor = 4  # 或 3.5, 2/3 等，常见为 4
        self.w1 = nn.Linear(hidden_units, hidden_units * expansion_factor)  # 可学习参数 V
        self.w2 = nn.Linear(hidden_units, hidden_units * expansion_factor)  # 可学习参数 W
        self.w3 = nn.Linear(hidden_units * expansion_factor, hidden_units)  # 输出投影

        self.dropout = nn.Dropout(p=dropout_rate)
        self.activation = nn.SiLU()  # SiLU 是 Swish: x * sigmoid(x)

    def forward(self, inputs):
        # inputs: [B, seq_len, hidden_units]
        gate = self.w1(inputs)        # [B, seq_len, expansion * hidden]
        x = self.w2(inputs)           # [B, seq_len, expansion * hidden]
        fused = self.activation(gate) * x  # SwiGLU: swish(gate) * x
        outputs = self.w3(fused)      # [B, seq_len, hidden_units]
        outputs = self.dropout(outputs)
        return outputs

class SwiGLU(nn.Module):
    """SwiGLU 激活函数实现：(x * sigmoid(x)) * gate"""
    def __init__(self, hidden_units, expansion_factor=4):
        super(SwiGLU, self).__init__()
        self.w1 = nn.Linear(hidden_units, hidden_units * expansion_factor)  # gate
        self.w2 = nn.Linear(hidden_units, hidden_units * expansion_factor)  # value
        self.w3 = nn.Linear(hidden_units * expansion_factor, hidden_units)  # output
        self.activation = nn.SiLU()

    def forward(self, x):
        gate = self.w1(x)
        value = self.w2(x)
        fused = self.activation(gate) * value
        return self.w3(fused)


class FastMoE(nn.Module):
    def __init__(self, hidden_units, num_experts=4, top_k=2, dropout_rate=0.1, expansion_factor=4):
        super(FastMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 创建多个专家（每个专家是一个 SwiGLU FFN）
        self.experts = nn.ModuleList([
            SwiGLU(hidden_units, expansion_factor) for _ in range(num_experts)
        ])

        # 门控网络：输出每个 token 对应的专家权重
        self.gate = nn.Linear(hidden_units, num_experts)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        # inputs: [B, seq_len, hidden_units]
        B, T, H = inputs.shape
        x = inputs.view(-1, H)  # [B*T, H]

        # 计算门控权重：[B*T, num_experts]
        gate_logits = self.gate(x)  # [B*T, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)

        # 取 top-k 专家
        top_k_weights, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)  # [B*T, top_k]
        top_k_weights = F.softmax(top_k_weights, dim=-1)  # renormalize

        # 初始化输出
        output = torch.zeros_like(x)

        # 对每个专家分别处理
        for i in range(self.top_k):
            expert_weights = top_k_weights[:, i]  # [B*T]
            expert_idx = top_k_indices[:, i]      # [B*T]

            # 对每个专家，收集其对应的 token
            for e in range(self.num_experts):
                # mask 表示当前专家 e 是否在 top-k 中
                mask = (expert_idx == e)
                if mask.sum() == 0:
                    continue

                # 获取该专家处理的 token
                expert_inputs = x[mask]
                expert_output = self.experts[e](expert_inputs)  # [num_masked, H]

                # 加权并累加到输出
                output[mask] += expert_output * expert_weights[mask].unsqueeze(-1)

        # 残差连接（可选）+ Dropout
        output = self.dropout(output)

        # 恢复形状
        output = output.view(B, T, H)
        return output
    

class BaselineModel(torch.nn.Module):
    """
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        args: 全局参数

    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  #
        super(BaselineModel, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        self.embedding_hidden_units = 64
        self.attention_hidden_units = 256
        self._init_feat_info(feat_statistics, feat_types)
        self.device = args.device
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch L1/L2 regularization in PyTorch L1/L2 regularization in PyTorch          
        self.item_sparse_embedding_size = 0
        self.item_sparse_embedding_size += self.item_num + 1
        for k in self.ITEM_SPARSE_FEAT:
            self.item_sparse_embedding_size += self.ITEM_SPARSE_FEAT[k] + 1

        self.user_sparse_embedding_size = 0
        self.user_sparse_embedding_size += self.user_num + 1
        for k in self.USER_SPARSE_FEAT:
            self.user_sparse_embedding_size += self.USER_SPARSE_FEAT[k] + 1

        self.item_array_embedding_size = 0
        for k in self.ITEM_ARRAY_FEAT:
            self.item_array_embedding_size += self.ITEM_ARRAY_FEAT[k] * 1

        self.user_array_embedding_size = 0
        for k in self.USER_ARRAY_FEAT:
            self.user_array_embedding_size += self.USER_ARRAY_FEAT[k] + 1

        self.item_sparse_emb = torch.nn.Embedding(
            self.item_sparse_embedding_size, self.embedding_hidden_units)
        self.user_sparse_emb = torch.nn.Embedding(
            self.user_sparse_embedding_size, self.embedding_hidden_units)
        self.user_array_emb = torch.nn.Embedding(
            self.user_array_embedding_size, self.embedding_hidden_units)
        self.item_array_emb = torch.nn.Embedding(
            self.item_array_embedding_size, self.embedding_hidden_units)

        # self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        # self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        # self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        # for k in self.USER_SPARSE_FEAT:
        #     self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        # for k in self.ITEM_SPARSE_FEAT:
        #     self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        # for k in self.ITEM_ARRAY_FEAT:
        #     self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        # for k in self.USER_ARRAY_FEAT:
        #     self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        mm_input_len = 0
        for k in self.ITEM_EMB_FEAT:
            mm_input_len += self.ITEM_EMB_FEAT[k]
        self.emb_transform = torch.nn.Linear(mm_input_len, self.embedding_hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()


        itemdim = (
            self.embedding_hidden_units * (len(self.ITEM_SPARSE_FEAT) +
                                 1 + len(self.ITEM_ARRAY_FEAT))
            + len(self.ITEM_CONTINUAL_FEAT)
            + self.embedding_hidden_units * len(self.ITEM_EMB_FEAT)
        )

        self.itemdnn = torch.nn.Linear(itemdim, self.attention_hidden_units)
        # 训练一些特殊的值
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
        self.classifiers = nn.ModuleDict()
        for key in self.predict_feature_list:
            self.classifiers[key] = ClassificationHead(self.attention_hidden_units,feat_statistics[key]+1)
        
        self.train_record = ['total_loss', 'infonce_loss', 'pos_sim', 'neg_sim'] +[f'cls_{i}' for i in self.predict_feature_list]
        self.eval_record = self.train_record

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table

        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        """
        self.USER_SPARSE_FEAT = {
            k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {
            k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {
            k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {
            k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024,
                          "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {
            k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度
    def special_embedding_apply(self):
        # 初始化模型权重
        # 所有偏置值为0
        # 所有权重设置为全1矩阵
        def set_seed(seed):
            """设置随机种子，确保实验可复现"""
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # 可选：设置 Python 和 NumPy 种子
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        set_seed(42)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            # elif isinstance(m, nn.Embedding):
            #     # 使用小正态分布初始化嵌入层
            #     nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif isinstance(m, nn.Conv1d):
                # 对 Conv1d (等价于 Linear) 使用 Xavier/Glorot 初始化
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                if hasattr(m, 'weight.data'):
                    nn.init.xavier_normal_(m.weight.data)

        
        self.apply(init_weights)
        sparse_weights =[]
        sparse_weights.append(torch.arange(self.item_num + 1).unsqueeze(1).expand(-1, 32).float())
        for k in self.ITEM_SPARSE_FEAT:
            sparse_weights.append(torch.arange(self.ITEM_SPARSE_FEAT[k] + 1).unsqueeze(1).expand(-1, 32).float())
        self.item_sparse_emb.weight.data = torch.cat(sparse_weights, dim=0).to(self.dev)

        sparse_weights = []
        sparse_weights.append(torch.arange(self.user_num + 1).unsqueeze(1).expand(-1, 32).float())
        for k in self.USER_SPARSE_FEAT:
            sparse_weights.append(torch.arange(self.USER_SPARSE_FEAT[k] + 1).unsqueeze(1).expand(-1, 32).float())
        self.user_sparse_emb.weight.data = torch.cat(sparse_weights, dim=0).to(self.dev)

        # array_weights = []
        # for k in self.ITEM_ARRAY_FEAT:
        #     array_weights.append(torch.arange(self.ITEM_ARRAY_FEAT[k] + 1).unsqueeze(1).expand(-1, 32).float())
        # self.item_array_emb.weight.data = torch.cat(array_weights, dim=0)

        array_weights = []
        for k in self.USER_ARRAY_FEAT:
            array_weights.append(torch.arange(self.USER_ARRAY_FEAT[k] + 1).unsqueeze(1).expand(-1, 32).float())
        self.user_array_emb.weight.data = torch.cat(array_weights, dim=0).to(self.dev)
        self.emb_transform.weight.data = torch.eye(32).to(self.dev).to(self.dev)

        self.userdnn.weight.data = torch.ones(self.userdnn.weight.data.shape[0],self.userdnn.weight.data.shape[1]).float().to(self.dev)
        self.itemdnn.weight.data = torch.ones(self.itemdnn.weight.data.shape[0],self.itemdnn.weight.data.shape[1]).float().to(self.dev)
        self.userdnn.bias.data = torch.zeros(self.userdnn.bias.data.shape[0]).float().to(self.dev)
        self.itemdnn.bias.data = torch.zeros(self.itemdnn.bias.data.shape[0]).float().to(self.dev)
        self.pos_emb.weight.data = torch.ones(self.pos_emb.weight.data.shape[0],self.pos_emb.weight.data.shape[1]).float().to(self.dev)
        
    
    def feat2emb(self,
            user_sparse_feature=None,
            user_continual_feature=None,
            user_array_feature=None,
            item_sparse_feature=None,
            item_continual_feature=None,
            item_array_feature=None,
            item_mm_embs=None,
            mask=None,
            include_user=False
        ):
        """
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征，在两种情况下不打开：1) 训练时在转换正负样本的特征时（因为正负样本都是item）;2) 生成候选库item embedding时。
        Returns:
            seqs_emb: 序列特征的Embedding
        """
        # item 
        # item_array_embs = self.item_array_emb(item_array_feature)

        # item_array_embs = torch.stack([
        #     item_array_embs[:, :, start:start+10, :].sum(dim=-2)
        #     for start in range(0,item_array_embs.shape[-2],10)
        # ], dim=-1) # 最后一维拼接 [batch_size, seq_len, sparse_num*hidden_units]
        
        item_sparse_embs = self.item_sparse_emb(item_sparse_feature).reshape(item_sparse_feature.shape[0], -1) # [batch_size, sparse_num*hidden_units]
        # 拼接过dnn输出
        item_mm_embs = self.emb_transform(item_mm_embs)
        all_item_emb = torch.cat([item_sparse_embs, item_continual_feature, item_mm_embs], dim=-1)
        all_embedding = torch.nn.functional.leaky_relu(self.itemdnn(all_item_emb))

        if include_user:
            user_array_embs = self.user_array_emb(user_array_feature) # [batch_size,]
            user_array_embs = [user_array_embs[:, start:start+10, :].sum(dim=-2)
                for start in range(0,user_array_embs.shape[-2],10)]
            user_array_embs = torch.cat(user_array_embs, dim=-1)
            user_sparse_embs = self.user_sparse_emb(user_sparse_feature).reshape(user_sparse_feature.shape[0],-1) #[batch_size, seq_len, num_sparse_feat *hidden_units]
            all_user_embs = torch.cat([user_sparse_embs, user_array_embs, user_continual_feature], dim=-1)
            all_user_embs = torch.nn.functional.leaky_relu(self.userdnn(all_user_embs))
            # insert all_user_emb to all_embedding
            user_mask_expanded = (mask == 2).unsqueeze(-1).repeat(1,1,all_embedding.shape[-1])

            # torch.nonzero(user_mask[0])
            all_embedding = torch.where(user_mask_expanded, all_user_embs.unsqueeze(1), all_embedding)

        all_embedding = F.normalize(all_embedding, dim=-1)
        return all_embedding

   
    def forward(
            self, pos1, pos2,pos1_labels, pos2_labels
        ):
        """
        训练时调用，计算正负样本的logits

        Args:
        Returns:

        """

        # 只构建对比学习样本
        pos1_emb = self.feat2emb(**pos1)
        pos2_emb = self.feat2emb(**pos2)
        infonce_loss, pos_sim, neg_sim = info_nce_loss(pos1_emb, pos2_emb)
        classify_loss = {}
        # 分类损失计算，用pos1_emb,pos2_emb 作为输入
        for i, feat_key in enumerate(self.predict_feature_list):
            classifier = self.classifiers[feat_key]  # ✅ 静态访问，支持编译

            label_1 = pos1_labels[:, i]  # [B]
            label_2 = pos2_labels[:, i]  # [B]

            # 计算 logits
            logits_1 = classifier(pos1_emb)
            logits_2 = classifier(pos2_emb)

            # 计算损失，忽略 -1
            loss_1 = nn.functional.cross_entropy(logits_1, label_1, ignore_index=0, reduction='mean')
            loss_2 = nn.functional.cross_entropy(logits_2, label_2, ignore_index=0, reduction='mean')

            total_cls_loss = (loss_1 + loss_2) / 2.0
            classify_loss[f'cls_{feat_key}'] = total_cls_loss
                # Step 5: 混合损失
        lambda_co = 1.0
        lambda_cls = 0.01
        cls_weights = {
            'cls_100': 5.0,
            'cls_117': 1.0,
            'cls_111': 0.3,
            'cls_118': 0.7,
            'cls_101': 1.7,
            'cls_102': 0.3,
            'cls_119': 0.5,
            'cls_120': 1.0,
            'cls_114': 2.2,
            'cls_112': 2.9,
            'cls_121': 0.3,
            'cls_115': 4.3,
            'cls_122': 0.3,
            'cls_116': 2.2,
        }
        total_loss = lambda_co * infonce_loss
        for loss in classify_loss.values():
            total_loss += lambda_cls * loss
        # for key in classify_loss:
        #     total_loss +=classify_loss[key]*cls_weights[key]
        return {
            "total_loss": total_loss,
            "infonce_loss":infonce_loss,
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
            **classify_loss
        }


    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding，用于检索

        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []
        start_time = time.time()
        max_size = len(item_ids)
        cnt = 0
        with torch.no_grad():
            for start_idx in range(0, max_size, batch_size):
                end_idx = min(start_idx + batch_size, max_size)

                # item_seq = torch.tensor(
                #     item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
                batch_feat = {
                    "item_sparse_feature":[],
                    "item_array_feature":[],
                    "item_continual_feature":[],
                    "item_mm_embs":[]
                }
                for i in range(start_idx, end_idx):
                    batch_feat["item_sparse_feature"].append(feat_dict[i]['sparse_feature'])
                    batch_feat["item_array_feature"].append(feat_dict[i]['array_feature'])
                    batch_feat["item_continual_feature"].append(feat_dict[i]['continual_feature'])
                    batch_feat["item_mm_embs"].append(feat_dict[i]['mm_emb'])

                for key in batch_feat:
                    if key not in ["item_mm_embs","item_continual_feature"]:
                        batch_feat[key] = torch.tensor(np.stack(batch_feat[key],axis=0),).unsqueeze(0).to(self.device)
                batch_feat["item_continual_feature"] = torch.tensor(np.stack(batch_feat["item_continual_feature"],axis=0),dtype=torch.float32).unsqueeze(0).to(self.device)
                
                batch_feat["item_mm_embs"] = torch.tensor(np.stack(batch_feat["item_mm_embs"]),dtype=torch.float32).unsqueeze(0).to(self.device)

                batch_emb = self.feat2emb(
                    **batch_feat, include_user=False).squeeze(0)
                # pickle.dump(batch_emb, open("data_v1.pkl",'wb'))

                all_embs.append(
                    batch_emb.detach().cpu().numpy().astype(np.float32))
                cnt+=1
                if cnt % 1000 == 0:
                    end_time = round(time.time())
                    use_time = format_time(end_time - start_time)
                    remain_time = format_time((end_time - start_time) / (cnt*batch_size) * max_size)
                    print(f"embedding[{use_time}/{remain_time}] [{cnt*batch_size}/{max_size}]")
            # 合并所有批次的结果并保存
            final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
            final_embs = np.concatenate(all_embs, axis=0)
            # save_emb(final_embs, Path(save_path, 'embedding.fbin'))
            # save_emb(final_ids, Path(save_path, 'id.u64bin'))
        return final_embs, final_ids