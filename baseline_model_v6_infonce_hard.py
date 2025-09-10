# baseline_model_v1.py
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from dataset import save_emb


def info_nce_loss(anchor, positive, anchor_mask=None, negatives=None,weight=None, temperature=0.07, use_inbatch_negatives=True):
    """
    InfoNCE Loss 支持：
    - 显式负样本 (negatives)
    - batch 内负样本 (可选)
    
    Args:
        anchor: (N, D)
        positive: (N, D)
        negatives: (N, K, D) 或 None
        temperature: float
        use_inbatch_negatives: bool, 是否使用 batch 内其他样本作为负样本
    
    Returns:
        loss: scalar
    """
    device = anchor.device
    batch_size = anchor.size(0)

    # L2 归一化

    # ========== 构建 logits ==========
    # 正样本相似度: (N, 1)
    pos_sim = torch.sum(anchor * positive, dim=1, keepdim=True)  # (N, 1)

    # 显式负样本相似度: (N, K)
    neg_sim_list = []

    if negatives is not None:
        explicit_neg_sim = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2))  # (N, 1, K)
        explicit_neg_sim = explicit_neg_sim.squeeze(1)  # (N, K)
        neg_sim_list.append(explicit_neg_sim)

    # batch 内负样本: anchor 与所有 positive 的相似度，去掉对角线（自己）
    if use_inbatch_negatives:
        # 计算 anchor 与所有 positive 的相似度（包括自己）
        inbatch_sim = torch.matmul(anchor, positive.T)  # (N, N)
        # 除了本样本之外所有的其他样本作为负样本
        # 创建 mask，去掉对角线（正样本）
        # anchor_mask = [1,1,1,2,2,3,3,3]
        # 变
        # [T,T,T,F,F,F,F,F]
        # [T,T,T,F,F,F,F,F]
        # [T,T,T,F,F,F,F,F]
        # [F,F,F,T,T,F,F,F]
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        if anchor_mask is not None:
            batch_mask = (anchor_mask.unsqueeze(0) == anchor_mask.unsqueeze(1)) 
            mask = mask & batch_mask
        inbatch_neg_sim = inbatch_sim.masked_select(~mask).view(batch_size, -1)  # (N, N-1)
        neg_sim_list.append(inbatch_neg_sim)

    # 拼接所有负样本相似度
    if neg_sim_list:
        neg_sim = torch.cat(neg_sim_list, dim=1)  # (N, K + N - 1)
    else:
        raise ValueError("At least one type of negative samples must be provided.")

    # 拼接正 + 负
    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature  # (N, 1 + K + N - 1)

    # 标签：正样本在第 0 位
    labels = torch.zeros(batch_size, dtype=torch.long).to(device)

    loss = F.cross_entropy(logits, labels, reduction='none')
    if weight is not None:
        loss = loss*weight
    return loss.mean(), pos_sim.mean(), neg_sim.mean()




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


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(
            self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        # as Conv1D requires (N, C, Length)
        outputs = outputs.transpose(-1, -2)
        return outputs


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
        self.train_record = [
            "total_loss",
            "loss_local",
            "loss_global",
            "pos_sim",
            "local_neg_sim",
            "global_neg_sim"
        ]
        self.eval_record = self.train_record
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        self.hidden_units = args.hidden_units
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
            self.item_sparse_embedding_size, args.hidden_units)
        self.user_sparse_emb = torch.nn.Embedding(
            self.user_sparse_embedding_size, args.hidden_units)
        self.user_array_emb = torch.nn.Embedding(
            self.user_array_embedding_size, args.hidden_units)
        self.item_array_emb = torch.nn.Embedding(
            self.item_array_embedding_size, args.hidden_units)
        self.pos_emb = torch.nn.Embedding(
            2 * args.maxlen + 1, args.hidden_units, padding_idx=0)

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
        self.emb_transform = torch.nn.Linear(mm_input_len, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()


        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        itemdim = (
            args.hidden_units * (len(self.ITEM_SPARSE_FEAT) +
                                 1 + len(self.ITEM_ARRAY_FEAT))
            + len(self.ITEM_CONTINUAL_FEAT)
            + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )

        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(
                args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )  # 优化：用FlashAttention替代标准Attention
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(
                args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

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
        
    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)

        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_array_len = 0
            max_seq_len = 0

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(
                    len(item_data) for item_data in seq_data))

            batch_data = np.zeros(
                (batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]

            return torch.from_numpy(batch_data).to(self.dev)
        else:
            # 如果特征是Sparse类型，直接转换为tensor
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data

            return torch.from_numpy(batch_data).to(self.dev)

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
        
        item_sparse_embs = self.item_sparse_emb(item_sparse_feature).reshape(item_sparse_feature.shape[0], item_sparse_feature.shape[1], -1) # [batch_size, seq_len, sparse_num*hidden_units]
        # 拼接过dnn输出
        item_mm_embs = self.emb_transform(item_mm_embs)
        all_item_emb = torch.cat([item_sparse_embs, item_continual_feature, item_mm_embs], dim=-1)
        all_embedding = torch.relu(self.itemdnn(all_item_emb))

        if include_user:
            user_array_embs = self.user_array_emb(user_array_feature) # [batch_size,]
            user_array_embs = [user_array_embs[:, start:start+10, :].sum(dim=-2)
                for start in range(0,user_array_embs.shape[-2],10)]
            user_array_embs = torch.cat(user_array_embs, dim=-1)
            user_sparse_embs = self.user_sparse_emb(user_sparse_feature).reshape(user_sparse_feature.shape[0],-1) #[batch_size, seq_len, num_sparse_feat *hidden_units]
            all_user_embs = torch.cat([user_sparse_embs, user_array_embs, user_continual_feature], dim=-1)
            all_user_embs = torch.relu(self.userdnn(all_user_embs))
            # insert all_user_emb to all_embedding
            user_mask_expanded = (mask == 2).unsqueeze(-1).repeat(1,1,all_embedding.shape[-1])

            # torch.nonzero(user_mask[0])
            all_embedding = torch.where(user_mask_expanded, all_user_embs.unsqueeze(1), all_embedding)

        all_embedding = F.normalize(all_embedding, dim=-1)
        return all_embedding

    def log2feats(self, seq_feature, user_feature, mask):
        """
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典

        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size = mask.shape[0]
        maxlen = mask.shape[1]
        seqs = self.feat2emb(**seq_feature,
                             **user_feature,
                             mask=mask, include_user=True)



        seqs *= self.hidden_units**0.5
        poss = torch.arange(
            1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= mask != 0
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)
        maxlen = seqs.shape[1]
        ones_matrix = torch.ones(
            (maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(
            0) & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = seqs
                x = self.attention_layernorms[i](x)
                mha_outputs, _ = self.attention_layers[i](
                    x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + \
                    self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](
                    seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](
                    seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)
        log_feats = F.normalize(log_feats, dim=-1)
        return log_feats

    def forward(
            self, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature, user_feature
        ):
        """
        训练时调用，计算正负样本的logits

        Args:
            user_item: 用户序列ID
            pos_seqs: 正样本序列ID
            neg_seqs: 负样本序列ID
            mask: token类型掩码，1表示item token，2表示user token
            next_mask: 下一个token类型掩码，1表示item token，2表示user token
            next_action_type: 下一个token动作类型，0表示曝光，1表示点击
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            pos_feature: 正样本特征list，每个元素为当前时刻的特征字典
            neg_feature: 负样本特征list，每个元素为当前时刻的特征字典

        Returns:
            pos_logits: 正样本logits，形状为 [batch_size, maxlen]
            neg_logits: 负样本logits，形状为 [batch_size, maxlen]
        """

        log_feats = self.log2feats(seq_feature,user_feature,mask)
        loss_mask = (next_mask == 1).to(self.dev)
        # 正样本，负样本拼接
        # seq_len = mask.shape[1]
        # input = {}
        # for key in pos_feature:
        #     input[key]=torch.cat([pos_feature[key],neg_feature[key]],dim=1)
        pos_embs = self.feat2emb(**pos_feature, include_user=False)
        neg_embs = self.feat2emb(**neg_feature, include_user=False)
        neg_embs = neg_embs.reshape(pos_embs.shape[0],pos_embs.shape[1],-1,pos_embs.shape[-1])
        # embds = self.feat2emb(**input, include_user=False)
        # pos_embs = embds[:, :seq_len, :]  # (B, L_pos, D)
        # neg_embs = embds[:, seq_len:, :]  # (B, L_neg, D)


        loss, loss_local, loss_global, pos_sim, local_neg_sim, global_neg_sim = self.loss_dual_path(log_feats, pos_embs, neg_embs, loss_mask)

        
        # 自己和自己的相似度
        # pos_embs_2 = self.feat2emb(**pos_feature, include_user=False)
        # self_sce_loss,sce_pos_sim, sce_neg_sim = self.loss(pos_embs, pos_embs_2, neg_embs=None,loss_mask=loss_mask, need_anchor_mask=False)
        return {
            "total_loss": loss,
            "loss_local":loss_local,
            "loss_global": loss_global,
            "pos_sim": pos_sim,
            "local_neg_sim":local_neg_sim,
            "global_neg_sim":global_neg_sim,
        }

    def loss_dual_path(self, log_feats, pos_embs, hard_neg_embs, loss_mask, alpha=1.0, beta=1.0, num_simple_negs=4096):
        device = pos_embs.device
        valid_mask = loss_mask.bool()
        V = valid_mask.sum().item()

        if V == 0:
            dummy = torch.tensor(0.0, device=device, requires_grad=True)
            return dummy, 0.0, 0.0, 0.0, 0.0, 0.0

        query = log_feats[valid_mask]        # [V, D]
        pos = pos_embs[valid_mask]           # [V, D]
        own_negs = hard_neg_embs[valid_mask] # [V, K, D]
        D, K = query.size(-1), own_negs.size(1)

        # =====================
        # Loss 1: Local Path
        # =====================
        pos_logits_local = (query * pos).sum(-1, keepdim=True)
        neg_logits_local = torch.einsum('vd,vkd->vk', query, own_negs)
        logits_local = torch.cat([pos_logits_local, neg_logits_local], dim=1)
        logits_local /= 0.07
        loss_local = F.cross_entropy(logits_local, torch.zeros(V, device=device, dtype=torch.long))

        # =====================
        # Loss 2: Global Path - 高效随机采样（允许极小概率采到自身，忽略）
        # =====================
        all_hard_negs = own_negs.view(-1, D)  # [V*K, D]
        N_total = all_hard_negs.size(0)
        n_sample = min(num_simple_negs, N_total)

        # 向量化采样：每个 query 独立采样 n_sample 个负样本
        sampled_indices = torch.randint(0, N_total, (V, n_sample), device=device)  # [V, n_sample]
        simple_negs = all_hard_negs[sampled_indices]  # [V, n_sample, D]

        neg_logits_global = torch.einsum('vd,vnd->vn', query, simple_negs)
        pos_logits_global = pos_logits_local

        logits_global = torch.cat([pos_logits_global, neg_logits_global], dim=1)
        logits_global /= 0.07
        loss_global = F.cross_entropy(logits_global, torch.zeros(V, device=device, dtype=torch.long))

        # =====================
        # Combine
        # =====================
        loss = alpha * loss_local + beta * loss_global

        pos_sim = pos_logits_local.mean().item()
        local_neg_sim = neg_logits_local.mean().item()
        global_neg_sim = neg_logits_global.mean().item()

        return loss, loss_local.item(), loss_global.item(), pos_sim, local_neg_sim, global_neg_sim

    def predict(self, seq_feature, user_feat, mask):
        """
        计算用户序列的表征
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            mask: token类型掩码，1表示item token，2表示user token
        Returns:
            final_feat: 用户序列的表征，形状为 [batch_size, hidden_units]
        """
        log_feats = self.log2feats(seq_feature,user_feat,mask=mask)

        final_feat = log_feats[:, -1, :]

        return final_feat

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
        with torch.no_grad():
            for start_idx in range(0, len(item_ids), batch_size):
                end_idx = min(start_idx + batch_size, len(item_ids))

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
                # item_sparse_feature=None,
                # item_continual_feature=None,
                # item_array_feature=None,
                # item_mm_embs=None,
                for key in batch_feat:
                    if key !="item_mm_embs":
                        batch_feat[key] = torch.tensor(batch_feat[key]).unsqueeze(0).to(self.device)
                batch_feat["item_mm_embs"] = torch.tensor(batch_feat["item_mm_embs"],dtype=torch.float32).unsqueeze(0).to(self.device)

                batch_emb = self.feat2emb(
                    **batch_feat, include_user=False).squeeze(0)
                # pickle.dump(batch_emb, open("data_v1.pkl",'wb'))

                all_embs.append(
                    batch_emb.detach().cpu().numpy().astype(np.float32))

            # 合并所有批次的结果并保存
            final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
            final_embs = np.concatenate(all_embs, axis=0)
            # save_emb(final_embs, Path(save_path, 'embedding.fbin'))
            # save_emb(final_ids, Path(save_path, 'id.u64bin'))
        return final_embs, final_ids