import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import reduce
from operator import mul
from torch import Tensor

from .utils import *


class X_Agent(nn.Module):
    def __init__(
        self,
        num_layers: int,
        patch_size: int = 16,  # 16 for ViT-B/16 | 14 for ViT-L/14@336px
        agent_length: int = 10,
        embed_dims: int = 768,  # 768 for ViT-B/16 | 1024 for ViT-L/14@336px
        query_dims: int = 256,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        text_dim: int = 512,  # 512 for ViT-B/16 | 768 for ViT-L/14@336px
        num_heads: int = 8,  # 8 for ViT-B/16 | 16 for ViT-L/14@336px
        ot: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.agent_length = agent_length
        self.query_dims = query_dims
        self.use_softmax = use_softmax
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.text_dim = text_dim
        self.num_heads = num_heads
        self.create_model()
        self.sinkhornkeops = None
        if ot:
            print("\033[1;31m use optimal transportation in agent attention. \033[0m")
            eps = 1e-3
            max_iter = 100
            self.sinkhornkeops = SinkhornDistance(eps=eps, max_iter=max_iter, cost=dotmat)
        self.use_sigmoid = True
        
    def create_model(self):
        # val = math.sqrt(
        #     6.0
        #     / float(
        #         3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
        #     )
        # )
        # self.transform = nn.Linear(self.embed_dims, self.query_dims)
        # self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
        # self.agent = nn.Parameter(torch.empty([self.num_layers, self.agent_length, self.embed_dims]))  # TODO: for froward
        # nn.init.uniform_(self.agent.data, -val, val)
        self.agent_scale = nn.Parameter(torch.tensor(self.scale_init))
        # self.metric_scale = nn.Parameter(torch.tensor(self.scale_init))
        # self.agent_proj_1 = nn.Linear(self.embed_dims, self.embed_dims)  # TODO: for agent_attention
        # nn.init.kaiming_uniform_(self.agent_proj_1.weight, a=math.sqrt(5))  # TODO: for agent_attention
        # self.agent_proj_2 = nn.Linear(self.embed_dims, self.embed_dims)  # TODO: for agent_attention
        # nn.init.kaiming_uniform_(self.agent_proj_2.weight, a=math.sqrt(5))  # TODO: for agent_attention
        self.mask_pooling = AgentMaskPooling(self.embed_dims)
        self.gamma1 = nn.Parameter(torch.zeros(1, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.gamma2 = nn.Parameter(torch.zeros(1, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.gamma_init = 0.1
        self.mlp_text = nn.Sequential(
            nn.Linear(self.text_dim, self.embed_dims),
            nn.ReLU(),
        )
        # self.agent_attn = ResidualAgentAttentionBlock(self.embed_dims, self.num_heads)  # TODO：for agent_attention_v2
        self.agent_attn = ResidualAgentAttentionWithDiffBlock(self.embed_dims, int(self.num_heads//2))  # TODO：for agent_attention_v2

    # def return_auto(self, feats):
    #     if self.link_token_to_query:
    #         tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)  # [agent_length, query_dims, num_layers]
    #         tokens = torch.cat(
    #             [
    #                 F.max_pool1d(tokens, kernel_size=self.num_layers),
    #                 F.avg_pool1d(tokens, kernel_size=self.num_layers),  # 挑选层中最值
    #                 tokens[:, :, -1].unsqueeze(-1),
    #             ],
    #             dim=-1,
    #         )  # [agent_length, query_dims, 3]
    #         querys = self.merge(tokens.flatten(-2, -1))  # [agent_length, query_dims]
    #         return feats, querys
    #     else:
    #         return feats
    
    def get_tokens(self, layer: int) -> Tensor:
        """返回agent"""
        if layer == -1:
            return self.agent
        else:
            return self.agent[layer]
    
    def proj_text(self, text_feats: Tensor):
        text_feats = text_feats.mean(dim=-2)  # [b, cls, text_dim]
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_emb = self.mlp_text(text_feats)  # [b, cls, embed_dims]
        return text_emb

    def agent_mask(self, agent: Tensor, feats: Tensor, text: Tensor, scale: float=None) -> Tensor:
        """"
        agent: shape: [bs, agent_length, embed_dims]
        feats: shape: [ h*w, batch, embed_dims]
        text: shape: [bs, cls, embed_dims]
        """
        _, bs, embed_dims = feats.shape
        scale = scale or embed_dims ** -0.5
        # mask text
        masked_text = torch.einsum('bad,bcd->bca', agent, text) * scale  # [bs, cls, agent_length]
        masked_text = self.mask_pooling(text, masked_text)  # [bs, agent_length, embed_dims]
        # mask visual
        masked_vis = torch.einsum('bnd,bad->bna', feats.permute(1, 0, 2), agent) * scale  # [bs, hw, agent_length]
        masked_vis = self.mask_pooling(feats.permute(1, 0, 2), masked_vis)  # [bs, agent_length, embed_dims]
        gamma_1 = torch.exp(self.gamma1).type_as(agent)
        gamma_2 = torch.exp(self.gamma2).type_as(agent)
        gamma = gamma_1 - gamma_2 + self.gamma_init
        agent = agent + gamma * masked_text + masked_vis
        return agent

    def agent_attention(self, agent: Tensor, feats: Tensor, text: Tensor, scale: float=None) -> Tensor:
        """"
        agent: shape: [bs, agent_length, embed_dims]
        feats: shape: [ h*w, batch, embed_dims]
        text: shape: [bs, cls, embed_dims]
        """
        _, bs, embed_dims = feats.shape
        scale = scale or embed_dims ** -0.5
        debug = False
        if self.sinkhornkeops is not None and debug:
            attn, _, _, _ = self.sinkhornkeops(F.normalize(agent, p=2, dim=-1), F.normalize(text, p=2, dim=-1))  # shape: [bs, a, embed_dims]和[bs, cls, embed_dims] -> [bs, a, cls]
            attn = attn * scale
            if self.use_softmax:
                attn = F.softmax(attn, dim=-1)
            v = torch.einsum('bac,bcd->bad', attn, text)
            v = self.agent_proj_1(v)
            attn, _, _, _ = self.sinkhornkeops(F.normalize(feats.permute(1, 0, 2), p=2, dim=-1), F.normalize(agent, p=2, dim=-1))  # shape: [bs, hw, embed_dims]和[bs, a, embed_dims] -> [bs, hw, a]
            attn = attn * scale
            if self.use_softmax:
                attn = F.softmax(attn, dim=-1)
            delta_feat = self.agent_proj_2(delta_feat)
            delta_feat = torch.einsum('bna,bad->bnd', attn, v)
        else:
            # 首先计算agent和text间的交叉注意力
            attn = torch.einsum('bad,bcd->bac', agent, text) * scale
            if self.use_softmax:
                attn = attn.softmax(dim=-1)
            v = torch.einsum('bac,bcd->bad', attn, text)
            v = self.agent_proj_1(v)
            # 然后计算agent和feats间的交叉注意力
            attn = torch.einsum('bnd,bad->bna', feats.permute(1, 0, 2), agent) * scale
            if self.use_softmax:
                attn = attn.softmax(dim=-1)
            delta_feat = torch.einsum('bna,bad->bnd', attn, v)
            delta_feat = self.agent_proj_2(delta_feat)
        return delta_feat.permute(1, 0, 2),  attn.detach() # shape: [h*w, batch, embed_dims], [bs, hw, a]
    
    def agent_attention_v2(self, agent: Tensor, feats: Tensor, text: Tensor, scale: float=None) -> Tensor:
        """"
        agent: shape: [bs, agent_length, embed_dims]
        feats: shape: [ h*w, batch, embed_dims]
        text: shape: [bs, cls, embed_dims]
        """
        return self.agent_attn(agent.permute(1, 0, 2), text.permute(1, 0, 2), feats), None
    
    def calculate_metrics(self, attn: Tensor, K: int, h: int, w: int, sigma=1.0) -> Tensor:
        """
        根据patch和agent之间的最优传输矩阵，衡量已知类别与未知类别中心的距离 
        TODO：显存占用过多，需要优化，自注意力计算过程中attn_mask支持bool和float，本质问题在于float占位很大，需要pytorch支持sparsetensor，
        由于最终求得的distance是仅调整seen部分，即非topK部分，而其他部分均保持不变，因此一种可替代且高效显存占用几乎忽略的方式是直接作用于自注意力的q
        """
        # 步骤1：生成二维高斯空间权重矩阵（以中心为均值，TODO：自适应中心）
        x_coords = torch.linspace(-(h-1)/2, (h-1)/2, h, device=attn.device, dtype=torch.int16)
        y_coords = torch.linspace(-(w-1)/2, (w-1)/2, w, device=attn.device, dtype=torch.int16)
        x, y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        spatial_weights = torch.exp(-(x**2 + y**2)/(2*sigma**2)).flatten().half().detach()  # shape: [hw, ]
        b, hw, q = attn.shape
        with torch.no_grad(), torch.cuda.amp.autocast():
            # 步骤二：计算sigmoid和topK
            sigmoid_x = torch.sigmoid(attn).half()  # [b, hw, q]
            topk_vals, topk_idx = torch.topk(sigmoid_x, k=K, dim=1, sorted=False)
            # 步骤三：计算topK的距离加权中心
            topK_weights = spatial_weights[topk_idx.view(-1)].view(b, K, q)
            weight_sum = topK_weights.sum(dim=1, keepdim=True) + 1e-6
            topK_center = (topk_vals * topK_weights) / weight_sum  # [b, K, q]
            topK_center = topK_center.mean(dim=1, keepdim=True).half()  # 转回half精度 [b, 1, q]
            # 步骤四：计算非topK与topK中心的距离
            mask = torch.zeros_like(attn, dtype=torch.bool).scatter(1, topk_idx, True)
            val_diff = (sigmoid_x - topK_center).pow(2)
            pos_diff = (spatial_weights.view(1, hw, 1) - topK_weights.mean(dim=1, keepdim=True)).pow(2)
            distence = torch.sqrt(torch.addcmul(torch.square(val_diff), pos_diff, pos_diff))
            distence.mul_((~mask).to(distence.dtype))
            distence = torch.matmul(distence, distence.transpose(1,2).contiguous())
            distence = distence.mean(dim=0).half().detach()  # [hw, hw]
            distence = F.pad(distence, (0, 1, 0, 1), mode='constant', value=0)  # 填充[CLS] [hw+1, hw+1]
        return distence

    def forward(
        self, m: nn.Module, input: Tensor, output:Tensor, text_feats: Tensor, layer: int, h: float=None, w: float=None, batch_first=False, has_cls_token=True
    ) -> Tensor:
        """采取可学习的agent"""
        text_feat = text_feats[0]  # [cls, P, text_dim] P为模板个数
        if not m.training:  # 测试模式
            text_feat = text_feats[1]
        assert has_cls_token, "Agent need cls token."
        if batch_first:
            output = output.permute(1, 0, 2)  # shape: [1+h*w, batch, embed_dims]
        if has_cls_token:
            cls_token, output = torch.tensor_split(output, [1], dim=0)
        agent = self.get_tokens(layer)  # shape: [agent_length, embed_dims]
        # Agent
        _, bs, _ = output.shape
        agent = agent.expand(bs, -1, -1)
        text_feat = text_feat.expand(bs, -1, -1, -1)  # [bs, cls, P, text_dim]
        text_emb = self.proj_text(text_feat)  # [bs, cls, embed_dims]
        agent_mask = self.agent_mask(agent, output, text_emb)
        delta_feat, _ = self.agent_attention_v2(agent_mask, output, text_emb)  # shape: [h*w, batch, embed_dims]
        output = output + delta_feat * self.agent_scale
        # distance = self.calculate_metrics(attn, int(h*w/2), h, w)  # shape: [batch, hw, hw]
        if has_cls_token:
            output = torch.cat([cls_token, output], dim=0)  # shape: [1+h*w, batch, embed_dims]
        if batch_first:
            output = output.permute(1, 0, 2)
        return output
    
    @torch.no_grad()
    def agent_selection(self, m: nn.Module, input: Tensor, text_emb: Tensor, h: float, w: float, K: int=10, Q: int=4, batch_first=False, has_cls_token=True) -> Tensor:
        if batch_first:
            input = input.permute(1, 0, 2)  # shape: [1+h*w, batch, embed_dims]
        if has_cls_token:
            cls_token, input = torch.tensor_split(input, [1], dim=0)
        if m.attn.in_proj_weight is not None:
            y = F.linear(input, m.attn.in_proj_weight, m.attn.in_proj_bias)
        else:
            proj = torch.cat((m.attn.q_proj_weight, m.attn.k_proj_weight, m.attn.v_proj_weight), dim=0)
            y = F.linear(input, proj, m.attn.in_proj_bias)
        L, N, D = y.shape  # [h*w, batch, 3*embed_dims]
        y = y.reshape(L, N, 3, D // 3).permute(2, 1, 0, 3).reshape(3 * N, L, D // 3)    
        q, k, v = y.tensor_split(3, dim=0)  # [batch, h*w, embed_dims]
        metric_feature = k  # 采取K作为相似度衡量
        metric_feature_normalized = F.normalize(metric_feature, p=2, dim=-1)  # shape: [batch, h*w, embed_dims]
        text_emb_normalized = F.normalize(text_emb, p=2, dim=-1)            # shape: [batch, cls, embed_dims]
        affinity = torch.einsum('bld,bcd->blc', metric_feature_normalized, text_emb_normalized)  # [batch, h*w, cls] TODO：使用OT
        if self.sinkhornkeops is not None and m.training:
            affinity_plan, _, _, _ = self.sinkhornkeops(metric_feature_normalized, text_emb_normalized)  # shape: [batch, h*w, embed_dims]和[bs, cls, embed_dims] -> [batch, h*w, cls]
            affinity = affinity * affinity_plan
        # if self.use_softmax:
        #     affinity = F.softmax(affinity, dim=-1)
        if self.use_sigmoid:
            affinity = torch.sigmoid(affinity)
        cls_affinity = affinity.mean(dim=1)  # [batch, cls]
        K = min(text_emb.size(1), K)
        _, topk_cls = torch.topk(cls_affinity, K, dim=-1, sorted=False, largest=True)  # [batch, K]
        selected_affinity = torch.gather(affinity, dim=-1, index=topk_cls.unsqueeze(1).expand(-1, L, K))  # [batch, hw, K]
        _, indices = torch.topk(selected_affinity, Q, dim=1, largest=True)  # [batch, Q, K] TODO：筛选高于阈值部分的token
        candidates = indices.reshape(N, -1)  # [batch, Q*K] TODO：该索引是相对索引还是全局索引？
        if getattr(self, "__VISUALIZATION__", False):
            setattr(self, "__data__", dict(
                candidates=candidates,
            ))
        agent = torch.gather(v, dim=1, index=candidates.unsqueeze(-1).expand(-1, -1, metric_feature.size(2)))  # [batch, Q*K, embed_dims]  由于agent中可能存在重叠位置，需要在agent计算中引入mask TODO：从q,k,v,input中选择
        mask = torch.zeros((N, K*Q), dtype=torch.bool, device=affinity.device)  # [batch, Q*K]
        for i in range(N):
            counts = torch.bincount(candidates[i], minlength=L)
            mask[i] = (counts[candidates[i]] > 1)
        agent = torch.where(mask.unsqueeze(-1).expand(-1, -1, agent.size(2)), torch.tensor(float('0.0'), device=agent.device), agent)
        if has_cls_token:
            input = torch.cat([cls_token, input], dim=0)  # shape: [1+h*w, batch, embed_dims]
        if batch_first:
            input = input.permute(1, 0, 2)
        return agent

    def forward_v2(
        self, m: nn.Module, input, output:Tensor, text_feats: Tensor, layer: int, h: float=None, w: float=None, batch_first=False, has_cls_token=True
    ) -> Tensor:
        """采取与文本特征注意力计算agent"""
        assert has_cls_token, "Agent need cls token."
        text_feat = text_feats[0]  # [cls, P, text_dim] P为模板个数
        if not m.training:  # 测试模式
            text_feat = text_feats[1]
        if batch_first:
            output = output.permute(1, 0, 2)  # shape: [1+h*w, batch, embed_dims]
        if has_cls_token:
            cls_token, output = torch.tensor_split(output, [1], dim=0)
        _, bs, _ = output.shape
        text_feat = text_feat.expand(bs, -1, -1, -1)  # [bs, cls, P, text_dim]
        text_emb = self.proj_text(text_feat)  # [bs, cls, embed_dims]
        agent = self.agent_selection(m, input[0], text_emb, h, w, K=5, Q=80, batch_first=batch_first, has_cls_token=has_cls_token)  # shape: [batch, agent_length, embed_dims]
        agent_mask = self.agent_mask(agent, output, text_emb)
        delta_feat, _ = self.agent_attention_v2(agent_mask, output, text_emb)  # shape: [h*w, batch, embed_dims]
        output = output + delta_feat * self.agent_scale
        # distance = self.calculate_metrics(attn, int(h*w/2), h, w)  # shape: [batch, hw, hw]
        if has_cls_token:
            output = torch.cat([cls_token, output], dim=0)  # shape: [1+h*w, batch, embed_dims]
        if batch_first:
            output = output.permute(1, 0, 2)
        return output


class LoRAX_Agent(X_Agent):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()
        del self.agent
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )
        self.agent_a = nn.Parameter(torch.empty([self.num_layers, self.agent_length, self.lora_dim]))
        self.agent_b = nn.Parameter(torch.empty([self.num_layers, self.lora_dim, self.embed_dims]))
        nn.init.uniform_(self.agent_a.data, -val, val)
        nn.init.uniform_(self.agent_b.data, -val, val)

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            return self.agent_a @ self.agent_b
        else:
            return self.agent_a[layer] @ self.agent_b[layer]