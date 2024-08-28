from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einops import rearrange, repeat, einsum
from einops.layers.torch import Rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def l2norm(t):
    return F.normalize(t, dim = -1)

def softclamp(t, value = 50.):
    return (t / value).tanh() * value

# flex attention mask construction
# https://pytorch.org/blog/flexattention/

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def modality(offset, length):

    def mask_fn(b, h, q_idx, kv_idx):
        return q_idx >= offset & kv_idx <= (offset + length)

    return mask_fn

def transfusion_mask(modalities: list[tuple[int, int]]):

    def mask_mod(*args):
        is_causal = causal(*args)

        modality_mask_mods = [modality(*modality_coors_info) for modality_coors_info in modalities]

        is_modality = any([fn(*args) for fn in modality_mask_mods])

        return is_causal | is_modality

    return mask_mod

def naive_attn_mask(
    modalities: list[tuple[int, int]]
):

    offsets, length = torch.tensor(modalities).unbind(dim = -1)

    def create_mask(seq_len):
        seq = torch.arange(seq_len)

        is_causal = einx.greater_equal('i, j -> i j', seq, seq)

        is_modality = (
            einx.greater_equal('i, modality -> modality i 1', seq, offsets) &
            einx.less_equal('j, modality -> modality 1 j', seq, offsets + length)
        )

        return is_causal | is_modality

    return create_mask

# attention

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return l2norm(x) * self.scale * (self.gamma + 1) # use unit offset from Ohad Rubin

class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return F.gelu(gates) * x

def FeedForward(
    dim,
    expansion_factor = 4.,
    dropout = 0.
):
    dim_inner = int(dim * expansion_factor * 2 / 3)
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
    )

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        softcap_value = 50.,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        )

        self.softcap_value = softcap_value

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False)
        )

    def forward(
        self,
        x,
        attn_mask = None
    ):
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q = q * self.scale
        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        sim = softclamp(sim, self.softcap_value)

        if exists(attn_mask):
            mask_value = -torch.finfo(sim.dtype).max
            sim = einx.where('b i j, b h i j, -> b h i j', attn_mask, sim, mask_value)

        attn = sim.softmax(dim = -1)

        attn = self.dropout(attn)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        return self.to_out(out)

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_expansion_factor = 4,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict()
    ):
        super().__init__()

        layers = ModuleList([])

        for _ in range(depth):
            layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout, **attn_kwargs),
                FeedForward(dim = dim, expansion_factor = ff_expansion_factor, **ff_kwargs)
            ]))

        self.layers = layers
        self.norm = RMSNorm(dim)

    def forward(
        self,
        x,
        attn_mask = None
    ):

        for attn, ff in self.layers:
            x = attn(x, attn_mask = attn_mask) + x
            x = ff(x) + x

        return self.norm(x)

# classes

class Transfusion(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

    def forward(
        self,
        x
    ):
        return x
