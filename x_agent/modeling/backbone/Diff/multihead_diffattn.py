import math
import torch
import torch.nn.functional as F
from torch import nn

from .kernel.rotary import apply_rotary_emb
# from flash_attn import flash_attn_func
try:
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    # print("No fused RMSNorm")
    from .rms_norm import RMSNorm


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


# class MultiheadDiffAttn(nn.Module):
#     def __init__(
#         self,
#         embed_dim,
#         depth, # current layer index
#         num_heads,
#         num_kv_heads=None,
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
        
#         # arg num_heads set to half of baseline Transformer's num_heads
#         # for e.g., to compare with a baseline Transformer with 16 heads, pass in num_heads=8 for DIFF Transformer
#         self.num_heads = num_heads
        
#         # arg num_kv_heads set to half of baseline Transformer's num_kv_heads if use GQA
#         # for e.g., to compare with a baseline Transformer with 16 heads and 8 kv_heads, 
#         # pass in num_heads=8, num_kv_heads=4 for DIFF Transformer
#         # if use MHA, pass in num_kv_heads=None
#         self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
#         self.n_rep = self.num_heads // self.num_kv_heads
        
#         self.head_dim = embed_dim // num_heads // 2
#         self.scaling = self.head_dim ** -0.5
        
#         self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
#         self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

#         # depth means current layer index
#         self.lambda_init = lambda_init_fn(depth)
#         self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
#         self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
#         self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
#         self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

#         self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
#     def forward(
#         self,
#         x,
#         rel_pos,
#         attn_mask=None,
#     ):
#         bsz, tgt_len, embed_dim = x.size()
#         src_len = tgt_len

#         q = self.q_proj(x)
#         k = self.k_proj(x)
#         v = self.v_proj(x)

#         q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
#         k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
#         v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

#         q = apply_rotary_emb(q, *rel_pos, interleaved=True)
#         k = apply_rotary_emb(k, *rel_pos, interleaved=True)

#         offset = src_len - tgt_len
#         q = q.transpose(1, 2)
#         k = repeat_kv(k.transpose(1, 2), self.n_rep)
#         v = repeat_kv(v.transpose(1, 2), self.n_rep)
#         q *= self.scaling
#         attn_weights = torch.matmul(q, k.transpose(-1, -2))
#         if attn_mask is None:
#             attn_mask = torch.triu(
#                 torch.zeros([tgt_len, src_len])
#                 .float()
#                 .fill_(float("-inf"))
#                 .type_as(attn_weights),
#                 1 + offset,
#             )
#         attn_weights = torch.nan_to_num(attn_weights)
#         attn_weights += attn_mask   
#         attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
#             attn_weights
#         )

#         lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
#         lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
#         lambda_full = lambda_1 - lambda_2 + self.lambda_init
#         attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
#         attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
#         attn = torch.matmul(attn_weights, v)
#         attn = self.subln(attn)
#         attn = attn * (1 - self.lambda_init)
#         attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

#         attn = self.out_proj(attn)
#         return attn


class MultiheadDiffAttn(nn.Module):
    """交叉注意力旋转位置编码"""
    def __init__(
        self,
        embed_dim,
        depth, # current layer index
        num_heads,
        num_kv_heads=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # arg num_heads set to half of baseline Transformer's num_heads
        # for e.g., to compare with a baseline Transformer with 16 heads, pass in num_heads=8 for DIFF Transformer
        self.num_heads = num_heads
        
        # arg num_kv_heads set to half of baseline Transformer's num_kv_heads if use GQA
        # for e.g., to compare with a baseline Transformer with 16 heads and 8 kv_heads, 
        # pass in num_heads=8, num_kv_heads=4 for DIFF Transformer
        # if use MHA, pass in num_kv_heads=None
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def get_cross_rotary_embeddings(
        self,
        q_seq_len: int, 
        k_seq_len: int, 
        head_dim: int, 
        device: torch.device,
        offset: int = 0
    ):
        """
        生成交叉注意力的旋转位置编码（支持不同序列长度）
        Args:
            q_seq_len (int): Query序列长度
            k_seq_len (int): Key序列长度
            head_dim (int): 注意力头维度（必须为偶数）
            device (torch.device): 设备
            offset (int): Key序列的位置索引偏移量（默认从0开始）
        Returns:
            (q_cos, q_sin): Query的位置编码，形状 [q_seq_len, head_dim//2]
            (k_cos, k_sin): Key的位置编码，形状 [k_seq_len, head_dim//2]
        """
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, but got {head_dim}")
        q_position = torch.arange(q_seq_len, device=device, dtype=torch.float32)
        k_position = torch.arange(offset, offset + k_seq_len, device=device, dtype=torch.float32)
        theta = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
        inv_freq = 1.0 / (10000 ** theta + 1e-8)
        q_sinusoid = torch.einsum("i,j->ij", q_position, inv_freq)
        k_sinusoid = torch.einsum("i,j->ij", k_position, inv_freq)
        q_cos, q_sin = torch.cos(q_sinusoid), torch.sin(q_sinusoid)
        k_cos, k_sin = torch.cos(k_sinusoid), torch.sin(k_sinusoid)
        return (q_cos, q_sin), (k_cos, k_sin)

    def forward(
        self,
        q,
        k,
        v,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = q.size()
        _, src_len, _ = k.size()
        _, src_len, _ = v.size()
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
        q_rel_pos, k_rel_pos = self.get_cross_rotary_embeddings(tgt_len, src_len, self.head_dim, q.device)
        q = apply_rotary_emb(q, *q_rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *k_rel_pos, interleaved=True)
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_weights += attn_mask   
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        attn = self.out_proj(attn)
        return attn


# class MultiheadDiffAttn(nn.Module):
#     """无旋转位置编码"""
#     def __init__(
#         self,
#         embed_dim,
#         depth, # current layer index
#         num_heads,
#         num_kv_heads=None,
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
        
#         # arg num_heads set to half of baseline Transformer's num_heads
#         # for e.g., to compare with a baseline Transformer with 16 heads, pass in num_heads=8 for DIFF Transformer
#         self.num_heads = num_heads
        
#         # arg num_kv_heads set to half of baseline Transformer's num_kv_heads if use GQA
#         # for e.g., to compare with a baseline Transformer with 16 heads and 8 kv_heads, 
#         # pass in num_heads=8, num_kv_heads=4 for DIFF Transformer
#         # if use MHA, pass in num_kv_heads=None
#         self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
#         self.n_rep = self.num_heads // self.num_kv_heads
        
#         self.head_dim = embed_dim // num_heads // 2
#         self.scaling = self.head_dim ** -0.5
        
#         self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
#         self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
#         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

#         # depth means current layer index
#         self.lambda_init = lambda_init_fn(depth)
#         self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
#         self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
#         self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
#         self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

#         self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

#     def forward(
#         self,
#         q,
#         k,
#         v,
#         attn_mask=None,
#     ):
#         bsz, tgt_len, embed_dim = q.size()
#         _, src_len, _ = k.size()
#         _, src_len, _ = v.size()

#         q = self.q_proj(q)
#         k = self.k_proj(k)
#         v = self.v_proj(v)

#         q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
#         k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
#         v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

#         q = q.transpose(1, 2)
#         k = repeat_kv(k.transpose(1, 2), self.n_rep)
#         v = repeat_kv(v.transpose(1, 2), self.n_rep)
#         q *= self.scaling
#         attn_weights = torch.matmul(q, k.transpose(-1, -2)) 
#         attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
#             attn_weights
#         )
#         lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
#         lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
#         lambda_full = lambda_1 - lambda_2 + self.lambda_init
#         attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
#         attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
#         attn = torch.matmul(attn_weights, v)
#         attn = self.subln(attn)
#         attn = attn * (1 - self.lambda_init)
#         attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
#         attn = self.out_proj(attn)
#         return attn