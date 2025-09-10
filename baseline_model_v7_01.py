# baseline_model_v7_hstu.py
"""
删除了位置编码，仅使用在HSTU中的相对位置编码。
"""
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
# baseline_model_v1.py
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from dataset import save_emb

from hstu import GenRec

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
        self.train_record = ['total_loss', 'pos_sim', 'neg_sim']
        self.eval_record = ['total_loss', 'pos_sim', 'neg_sim']
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        self.hidden_units = args.hidden_units
        self._init_feat_info(feat_statistics, feat_types)
        self.device = args.device
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch 
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
        mm_input_len = 0
        for k in self.ITEM_EMB_FEAT:
            mm_input_len += self.ITEM_EMB_FEAT[k]
    
        self.emb_transform = torch.nn.Linear(mm_input_len, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.hstu_layers = GenRec(args.hidden_units, args.num_heads, args.num_blocks, args.dropout_rate)


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
        batch_size,seq_len = item_sparse_feature.shape[0],item_sparse_feature.shape[1]
        item_sparse_embs = self.item_sparse_emb(item_sparse_feature)# [batch_size, seq_len, sparse_num*hidden_units]
        # 拼接过dnn输出
        item_mm_embs = self.emb_transform(item_mm_embs)
        all_item_emb = torch.cat([item_sparse_embs, item_mm_embs.unsqueeze(-2)], dim=-2)
        all_item_emb = all_item_emb.reshape(batch_size,seq_len,-1)
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


        all_embedding = F.normalize(all_embedding,dim=-1)
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
        # poss = torch.arange(
        #     1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        # poss *= mask != 0
        # seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)
        maxlen = seqs.shape[1]
        ones_matrix = torch.ones(
            (maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(
            0) & attention_mask_pad.unsqueeze(1)
        self.hstu_layers(seqs,attention_mask)
        log_feats = self.last_layernorm(seqs)
        log_feats = F.normalize(log_feats,dim=-1)
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
        pos_embs = self.feat2emb(**pos_feature, include_user=False)
        neg_embs = self.feat2emb(**neg_feature, include_user=False)



        loss, pos_sim, neg_sim = self.loss(log_feats, pos_embs, neg_embs, loss_mask)

        return {
            "total_loss": loss,
            "pos_sim": pos_sim,
            "neg_sim": neg_sim
        }

    def loss(self, log_feats, pos_embs, neg_embs, loss_mask, temperature=0.07):
        device = log_feats.device
        B, T, D = log_feats.shape
        N = neg_embs.shape[1] // T  # 每个位置的负样本数

        # Reshape neg_embs: [B, N*T, D] -> [B, T, N, D]
        neg_embs = neg_embs.view(B, T, N, D)  # 更高效 than unsqueeze + reshape

        # 提取有效位置 mask
        valid_mask = loss_mask.bool()  # [B, T]
        if not valid_mask.any():
            dummy = torch.tensor(0.0, device=device, requires_grad=True)
            return dummy, 0.0, 0.0

        # ================================
        # Step 1: 提取 valid 的 query, pos, negs
        # ================================
        # 注意：neg_embs 需要按位置提取，不能直接索引
        # 我们先计算所有位置的 logits，再 mask + gather

        # 计算所有位置的正样本相似度: [B, T]
        pos_logits = torch.einsum('btd,btd->bt', log_feats, pos_embs)  # [B, T]

        # 计算负样本相似度: [B, T] x [B, N, T, D] -> [B, N, T]
        # 使用 einsum: 'btd,bnTd->bnT'
        neg_logits = torch.einsum('btd,btnd->btn', log_feats, neg_embs)  # [B, T, N]
        # 展平并筛选
        pos_logits_valid = pos_logits[valid_mask]        # [V]
        neg_logits_valid = neg_logits[valid_mask]        # [V, N]


        # 构造 InfoNCE logits: [V, 1 + N]
        logits = torch.cat([pos_logits_valid.unsqueeze(1), neg_logits_valid], dim=1)  # [V, 1+N]
        logits /= temperature
        labels = torch.zeros(logits.size(0), device=device, dtype=torch.long)  # 正样本位置 = 0

        loss = F.cross_entropy(logits, labels)

        # ================================
        # 监控指标
        # ================================
        pos_sim = pos_logits_valid.mean().item()
        neg_sim = neg_logits_valid.mean().item()

        return loss, pos_sim, neg_sim

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

                for key in batch_feat:
                    if key !="item_mm_embs":
                        batch_feat[key] = torch.tensor(np.stack(batch_feat[key])).unsqueeze(0).to(self.device)
                batch_feat["item_mm_embs"] = torch.tensor(np.stack(batch_feat["item_mm_embs"]),dtype=torch.float32).unsqueeze(0).to(self.device)

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