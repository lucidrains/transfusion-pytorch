from __future__ import annotations

"""
global ein notation

b - batch
m - modalities
n - sequence
d - dimension
l - logits (text)
i, j - sequence (row, col)
"""

from functools import partial
from collections import namedtuple

import torch
from torch import nn, Tensor, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.nn.utils.rnn import pad_sequence

import einx
from einops import rearrange, repeat, reduce, einsum, pack
from einops.layers.torch import Rearrange

from transfusion_pytorch.tensor_typing import Float, Int, Bool

pad_sequence = partial(pad_sequence, batch_first = True)

# constants

RawModalityInfo = list[list[tuple[int, int]]]

LossBreakdown = namedtuple('LossBreakdown', ['total', 'text', 'diffusion'])

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

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
        return q_idx >= offset & kv_idx < (offset + length)

    return mask_fn

def transfusion_attn_mask(modalities: list[tuple[int, int]]):

    def mask_mod(*args):
        is_causal = causal(*args)

        modality_mask_mods = [modality(*modality_coors_info) for modality_coors_info in modalities]

        is_modality = any([fn(*args) for fn in modality_mask_mods])

        return is_causal | is_modality

    return mask_mod

# functions for managing modality token mask

def modalities_to_tensor(
    modalities: RawModalityInfo,
    pad_value = 0
) -> Int['b m 2']:

    modalities: list[Tensor] = [tensor(modality) for modality in modalities]
    modalities = pad_sequence(modalities, padding_value = pad_value)
    return modalities

def modalities_tensor_to_is_modality_mask(
    seq_len: int,
    modalities: RawModalityInfo | Int['b m 2'],
) -> Bool['b m n']:

    if isinstance(modalities, list):
        modalities = modalities_to_tensor(modalities)

    left, right = modalities.cumsum(dim = -1).unbind(dim = -1)

    seq = torch.arange(seq_len, device = modalities.device)

    is_modality = (
        einx.greater_equal('i, b m -> b m i', seq, left) &
        einx.less('j, b m -> b m j', seq, right)
    )

    return is_modality

def naive_attn_mask(
    seq_len: int,
    modalities: RawModalityInfo | Int['b m 2'],
    device = None
) -> Bool['b i j']:

    if isinstance(modalities, list):
        modalities = modalities_to_tensor(modalities)

    offsets, length = modalities.unbind(dim = -1)

    seq = torch.arange(seq_len, device = device)

    is_causal = einx.greater_equal('i, j -> i j', seq, seq)

    is_modality = (
        einx.greater_equal('i, b m -> b m i 1', seq, offsets) &
        einx.less('j, b m -> b m 1 j', seq, offsets + length)
    )

    return is_causal | is_modality.any(dim = 1)

# random fourier embedding

class RandomFourierEmbed(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        self.dim = dim
        self.register_buffer('weights', torch.randn(dim // 2))

    def forward(
        self,
        times: Float['b n']
    ) -> Float['b n {self.dim + 1}']:

        freqs = einx.multiply('... i, j -> ... i j', times, self.weights) * 2 * torch.pi
        fourier_embed, _ = pack((times, freqs.sin(), freqs.cos()), 'b n *')
        return fourier_embed

# adaptive layernorm and ada-ln zero rolled into one wrapper
# from DiT paper and sota for time conditioning for now

class AdaptiveWrapper(Module):
    def __init__(
        self,
        fn: Module,
        dim,
        dim_cond,
        ada_ln_zero_init_bias = -2
    ):
        super().__init__()
        self.fn = fn
        self.dim = dim
        self.dim_cond = dim_cond

        self.layernorm = nn.LayerNorm(dim, elementwise_affine = False)

        # text will be subjected to normal layernorm bias
        # and for output will use layerscale

        self.layernorm_gamma = nn.Parameter(torch.zeros(dim))
        self.layerscale = nn.Parameter(torch.ones(dim))

        # modalities will get the adaptive layernorm + ada-ln zero

        self.to_film = nn.Linear(dim_cond, dim * 2)
        self.to_ada_ln_zero = nn.Linear(dim_cond, dim)

        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_ada_ln_zero.weight)
        nn.init.constant_(self.to_ada_ln_zero.bias, ada_ln_zero_init_bias)

    def forward(
        self,
        x: Float['b n {self.dim}'],
        cond: Float['b {self.dim_cond}'] | Float['b n {self.dim_cond}'],
        is_any_modality: Bool['b n'],
        **kwargs
    ):
        is_any_modality = rearrange(is_any_modality, '... -> ... 1')

        if cond.ndim == 2:
            cond = rearrange(cond, 'b d -> b 1 d')

        x = self.layernorm(x)

        gamma, beta = self.to_film(cond).chunk(2, dim = -1)

        text_tokens = x * (self.layernorm_gamma + 1.)
        modality_tokens = x * (gamma + 1.) + beta

        x = torch.where(is_any_modality, modality_tokens, text_tokens)

        # attention or feedforwards

        out = self.fn(x, **kwargs)

        # take care of conditioning output separately for text vs modality

        text_out = out * self.layerscale
        modalities_out = out * self.to_ada_ln_zero(cond).sigmoid()

        return torch.where(is_any_modality, modalities_out, text_out)

# attention

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return l2norm(x) * self.scale * (self.gamma + 1.) # use unit offset from Ohad Rubin

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
        self.dim = dim

        self.to_time_cond = nn.Sequential(
            RandomFourierEmbed(dim),
            nn.Linear(dim + 1, dim * 4),
            nn.SiLU()
        )

        layers = ModuleList([])

        for _ in range(depth):
            attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout, **attn_kwargs)

            ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor, **ff_kwargs)

            attn = AdaptiveWrapper(attn, dim = dim, dim_cond = dim * 4)
            ff = AdaptiveWrapper(ff, dim = dim, dim_cond = dim * 4)

            layers.append(ModuleList([attn, ff]))

        self.layers = layers
        self.norm = RMSNorm(dim)

    def forward(
        self,
        x,
        times: Float[''] | Float['b'] | Float['b n'],
        attn_mask: Bool['b i j'] | None = None,
        modalities: RawModalityInfo | Int['b n 2'] | None = None,
        is_any_modality: Bool['b n'] | None = None
    ):
        seq_len, device = x.shape[-2], x.device
        assert exists(attn_mask) ^ exists(modalities)

        # handle time

        if times.ndim == 0:
            times = repeat(times, ' -> b', b = batch)

        cond = self.to_time_cond(times)

        # create the specialized mask needed for autoregressive text + bidirectional diffusion attention

        if exists(modalities):
            attn_mask = naive_attn_mask(seq_len, modalities, device = device)

        if not exists(is_any_modality):
            assert exists(modalities)
            is_any_modality = modalities_tensor_to_is_modality_mask(seq_len, modalities).any(dim = 1)

        adaptive_kwargs = dict(cond = cond, is_any_modality = is_any_modality)

        # transformer layers as usual, using mask from above

        for attn, ff in self.layers:
            x = attn(x, attn_mask = attn_mask, **adaptive_kwargs) + x
            x = ff(x, **adaptive_kwargs) + x

        return self.norm(x)

# classes

class Transfusion(Module):
    def __init__(
        self,
        *,
        num_text_tokens,
        transformer: dict | Transformer,
        ignore_index = -1,
        diffusion_loss_weight = 1.
    ):
        super().__init__()

        # transformer

        if isinstance(transformer, dict):
            transformer = Transformer(**transformer)

        self.transformer = transformer
        dim = transformer.dim

        # embeddings and un-embeddings

        self.text_embed = nn.Embedding(num_text_tokens, dim)

        self.to_text_logits = nn.Linear(dim, num_text_tokens, bias = False)

        self.to_pred_flow = nn.Linear(dim, dim, bias = False)

        # loss related

        self.ignore_index = ignore_index
        self.diffusion_loss_weight = diffusion_loss_weight

    def forward(
        self,
        text: Int['b n'],
        modality_tokens: Float['b n d'],
        modalities: RawModalityInfo,
        times: Int['b'] | None = None,
        return_loss = True

    ) -> Float['b n l'] | Float['']:

        # if returning loss, split text for next token prediction

        if return_loss:
            text, text_labels = text[:, :-1], text[:, 1:]

        # derive is_modality mask for diffusion on the right tokens + diffusion loss

        batch, seq_len, device = *text.shape, text.device

        assert len(modalities) == batch

        modalities_tensor = modalities_to_tensor(modalities)

        num_modalities = modalities_tensor.shape[-2]

        is_modalities = modalities_tensor_to_is_modality_mask(seq_len, modalities_tensor)

        is_any_modality = reduce(is_modalities, 'b m n -> b n', 'any')

        # embed text

        text_tokens = self.text_embed(text)

        # noise the modality tokens

        if not exists(times):
            times = torch.rand((batch, num_modalities), device = device)

        if return_loss:
            times = einsum(is_modalities.float(), times, 'b m n, b m -> b n')
            padded_times = rearrange(times, 'b n -> b n 1')
            noise = torch.randn_like(modality_tokens)

            modality_tokens = modality_tokens * padded_times + noise * (1. - padded_times)

            # the flow is the (data - noise)

            flow = modality_tokens - noise

        # intersperse the modalities with the text for the joint transformer + diffusion system

        tokens = einx.where('b n, b n d, b n d', is_any_modality, modality_tokens, text_tokens)

        # attention

        embed = self.transformer(
            tokens,
            times = times,
            modalities = modalities_tensor
        )

        # text unembedding

        text_logits = self.to_text_logits(embed)

        if not return_loss:
            return text_logits

        # text autoregressive loss

        text_loss = F.cross_entropy(
            rearrange(text_logits, 'b n l -> b l n'),
            text_labels,
            ignore_index = self.ignore_index,
            reduction = 'none'
        )

        text_loss = text_loss[~is_any_modality].mean()

        # diffusion loss

        pred_flow = self.to_pred_flow(embed)

        diffusion_loss = F.mse_loss(
            pred_flow,
            flow,
            reduction = 'none'
        )

        diffusion_loss = diffusion_loss[is_any_modality].mean()

        # only the token positions that are not modalities have autoregressive loss

        total_loss = (
            text_loss +
            diffusion_loss * self.diffusion_loss_weight
        )

        return total_loss, LossBreakdown(total_loss, text_loss, diffusion_loss)
