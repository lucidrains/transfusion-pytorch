from __future__ import annotations

"""
global ein notation

b - batch
t - one modality type
m - separate modality instance
n - sequence
d - dimension
l - logits (text)
i, j - sequence (row, col)
"""

import math
from functools import partial
from typing import NamedTuple, Callable, Literal

import torch
from torch import nn, Tensor, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear
from torch.nn.utils.rnn import pad_sequence
from torch.utils._pytree import tree_map

import einx
from einops import rearrange, repeat, reduce, einsum, pack
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from beartype import beartype
from beartype.door import is_bearable
from tqdm import tqdm

pad_sequence = partial(pad_sequence, batch_first = True)

# tensor typing

import jaxtyping

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]

Float = TorchTyping(jaxtyping.Float)
Int   = TorchTyping(jaxtyping.Int)
Bool  = TorchTyping(jaxtyping.Bool)

# maybe flex attention

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)

except ImportError:
    flex_attention = None

# constants

ModalitySample = list[Int['_'] | Float['_ _'] | tuple[int, Float['_ _']]]

ModalityTokenTransform = str | Callable | None

RawModalityPositions = list[list[tuple[int, int]]]

class LossBreakdown(NamedTuple):
    total: Float['']
    text: Float['']
    flow: list[Float['']]

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def first(it):
    return it[0]

def prepend(arr, el):
    arr.insert(0, el)

def join(arr, delimiter = ''):
    return delimiter.join(arr)

def divisible_by(num, den):
    return (num % den) == 0

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def tree_map_tensor(sample, fn: Callable):
    return tree_map(lambda t: t if not torch.is_tensor(t) else fn(t), sample)

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

# default function for constituting modality shape from string

def default_to_modality_shape_fn(maybe_shape_str) -> tuple[int, ...]:
    return tuple([*map(int, maybe_shape_str.split(','))])

# default function for translating modality length to times (noise level, where 0 is highest noise)

def random_modality_length_to_time_fn(modality_length: Int['b m']) -> Float['b m']:
    return torch.rand_like(modality_length.float())

def default_modality_length_to_time_fn(modality_length: Int['b m']) -> Float['b m']:
    total_modalities, device = modality_length.shape[-1], modality_length.device

    num_modalities = (modality_length > 0).sum(dim = -1).float()
    rand_num_modalities = torch.floor(torch.rand_like(num_modalities) * num_modalities)
    seq = torch.arange(total_modalities, device = device)

    prev_decoded_modality = einx.less_equal('n, b -> b n', seq, rand_num_modalities)
    curr_modality_rand_time = torch.rand_like(num_modalities)

    # in paper, they fix previous decoded modalities to 500 / 1000 steps for discrete ddpm, here using flow matching with times 0 - 1 so corresponds to 0.5
    return torch.where(prev_decoded_modality, 0.5, curr_modality_rand_time)

# pretty print

def print_modality_sample(
    modality_sample: ModalitySample
):
    output = []

    for sample in modality_sample:
        if isinstance(sample, tuple):
            modality_type, sample = sample
            output.append((f'modality:{modality_type}', sample.shape))
        elif sample.dtype in (torch.int, torch.long):
            output.append(('text', sample.shape))
        else:
            output.append(('modality', sample.shape))

    print(output)

# character based tokenizer

def char_tokenize(
    text: str,
    device = None,
    offset = 0
) -> Tensor:
    return tensor([*map(ord, text)], device = device) + offset

def decode_chars(
    t: Tensor,
    offset = 0,
) -> str:
    byte_list = (t - offset).clamp(min = 0, max = 127).tolist()
    return ''.join([*map(chr, byte_list)])

def get_tokens_since_rightmost_id(
    t: Tensor,
    rightmost_id: int
) -> Tensor:
    """
    ex. [9] [2] [8] [4] [7]
    2 would return [8] [4] [7]
    """

    mask = t == rightmost_id

    if not mask.any():
        return t[0:0] # return empty tensor if no id found

    reverse_cumsum = mask.flip(dims = (0,)).cumsum(dim = 0).flip(dims = (0,))
    after_right_mask = reverse_cumsum == 0
    return t[after_right_mask]

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
        return (q_idx >= offset) & (kv_idx < (offset + length))

    return mask_fn

def transfusion_attn_mask(modalities: Int['b m 3']):
    modalities = modalities.long()

    def mask_mod(b, h, q_idx, kv_idx):
        mask = causal(b, h, q_idx, kv_idx)

        modality_batch = modalities[b]

        for _, offset, length in modality_batch:
            mask = mask | modality(offset, length)(b, h, q_idx, kv_idx)

        return mask

    return mask_mod

def softcap_score_mod(softcap):
    def inner(score, b, h, q_idx, kv_idx):
        score = score / softcap
        score = torch.tanh(score)
        score = score * softcap
        return score
    return inner

# converting a raw list of modality offsets and lengths to tensor

def modality_positions_to_tensor(
    modalities: RawModalityPositions,
    pad_value = 0,
    device = None
) -> Int['b m 2'] | Int['b m 3']:

    modalities: list[Tensor] = [tensor(modality, device = device) for modality in modalities]
    modalities = pad_sequence(modalities, padding_value = pad_value)

    if modalities.ndim == 2:
        modalities = modalities.reshape(*modalities.shape, 3)

    return modalities

# sanitizing modalities tensor, making sure it is ordered

def order_modality_positions_by_seq_offset(
    modalities: Int['b m 3']
) -> tuple[Int['b m 3'], Int['b m']]:

    type, offsets, lengths = modalities.unbind(dim = -1)

    no_modality_mask = lengths <= 0 # there may be uneven number of modalities per batch sample
    offsets_to_sort = offsets.masked_fill(no_modality_mask, 1e10)
    _, sorted_indices = offsets_to_sort.sort(dim = -1)

    # sort by ascending offset

    modalities = einx.get_at('b [mi] ..., b mo -> b mo ...', modalities, sorted_indices)
    return modalities, sorted_indices

# deriving relative positions from modality positions
# ex. given a sequence of 10 with an image at offset 3 with length 4 - [t] [t] [t] [i] [i] [i] [i] [t] [t] [t]
# relative positions for rotary will be [0] [1] [2] [3] [3] [3] [3] [4] [5] [6]
# rationale is that each modality will need the same position so there is no distance when conducting bidirectional attention, but should still have a relative distance to other text tokens and modalities

def derive_rotary_positions_from_modality_positions(
    seq_len: int,
    modalities: Int['b m 3']
) -> Int['b n']:

    device = modalities.device

    modality_mask = modality_positions_to_is_modality_mask(seq_len, modalities, offset = torch.tensor([1, -1]))
    is_any_modality = reduce(modality_mask, 'b t m n -> b n', 'any')

    return torch.arange(seq_len, device = device) - is_any_modality.cumsum(dim = -1)

# modality tokens are given as list of tensors, can be then be embedded into the modality tokens for attending alongside text tokens

def embed_modality_tokens(
    seq_len: int,
    dim: int,
    modality_tokens: list[list[Float['_ d']]],
    modalities: Int['b m 3'],
    modality_id: int
) -> Float['b n d']:

    batch, device = modalities.shape[0], modalities.device

    output = torch.zeros((batch, seq_len, dim), device = device)

    for batch_ind, (one_modality, one_modality_token) in enumerate(zip(modalities, modality_tokens)):
        for (type, offset, length), batch_modality_token in zip(one_modality, one_modality_token):

            if modality_id != type or length <= 0:
                continue

            modality_shape = batch_modality_token.shape

            assert length == modality_shape[0], f'received a modality of shape {modality_shape} but sequence length in modalities info is {length}'
            assert dim == modality_shape[1], f'received modality [{modality_id}] with shape {modality_shape} but expected dimension of {dim}'

            output[batch_ind, offset:(offset + length)] = batch_modality_token

    return output

# functions for managing modality token mask

def modality_positions_to_is_modality_mask(
    seq_len: int,
    modalities: Int['b m 3'],
    offset: Int['2'] | None = None,
    device = None,
    num_modalities = 1
) -> Bool['b t m n']:

    device = modalities.device

    if exists(offset):
        offset = F.pad(offset, (1, 0))
        modalities = modalities + offset.to(modalities)

    seq = torch.arange(seq_len, device = device)
    type_seq = torch.arange(num_modalities, device = device)

    modality_types = modalities[..., 0]

    left, right = modalities[..., 1:].cumsum(dim = -1).unbind(dim = -1)

    is_instance_for_type = einx.equal('b m, t -> b t m', modality_types, type_seq)

    is_modality_along_seq = (
        einx.greater_equal('i, b m -> b m i', seq, left) &
        einx.less('j, b m -> b m j', seq, right)
    )

    return einx.logical_and('b t m, b m n -> b t m n', is_instance_for_type, is_modality_along_seq)

def naive_attn_mask(
    seq_len: int,
    modalities: Int['b m 3'],
    device = None
) -> Bool['b i j']:

    _, offsets, length = modalities.unbind(dim = -1)

    seq = torch.arange(seq_len, device = device)

    is_causal = einx.greater_equal('i, j -> i j', seq, seq)

    is_modality = (
        einx.greater_equal('i, b m -> b m i 1', seq, offsets) &
        einx.less('j, b m -> b m 1 j', seq, offsets + length)
    )

    return is_causal | is_modality.any(dim = 1)

# sampling related functions

# min_p for text
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

from torchdiffeq import odeint

# random fourier embedding

class RandomFourierEmbed(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        self.dim = dim
        self.register_buffer('weights', torch.randn(dim // 2))

    def forward(
        self,
        times: Float['b n'] | Float['b']
    ) -> Float['b n {self.dim + 1}']:

        if times.ndim == 1:
            times = rearrange(times, 'b -> b 1')

        freqs = einx.multiply('... i, j -> ... i j', times, self.weights) * 2 * torch.pi
        fourier_embed, _ = pack((times, freqs.sin(), freqs.cos()), 'b n *')
        return fourier_embed

# adaptive layernorm and ada-ln zero rolled into one wrapper
# from DiT paper and sota for time conditioning for now

class AdaptiveWrapper(Module):
    @beartype
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
        self.layerscale = nn.Parameter(torch.zeros(dim))

        # modalities will get the adaptive layernorm + ada-ln zero

        self.to_film = Linear(dim_cond, dim * 2)
        self.to_ada_ln_zero = Linear(dim_cond, dim)

        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_ada_ln_zero.weight)
        nn.init.constant_(self.to_ada_ln_zero.bias, ada_ln_zero_init_bias)

    def forward_text(
        self,
        x: Float['b n {self.dim}'],
        **kwargs
    ):

        x = self.layernorm(x)

        x = x * (self.layernorm_gamma + 1.)

        out = self.fn(x, **kwargs)

        multiple_returns = isinstance(out, tuple)

        if multiple_returns:
            out, *rest = out

        out = out * (self.layerscale + 1.)

        if multiple_returns:
            out = (out, *rest)

        return out

    def forward_modality(
        self,
        x: Float['b n {self.dim}'],
        cond: Float['b {self.dim_cond}'] | Float['b n {self.dim_cond}'],
        **kwargs
    ):
        x = self.layernorm(x)

        gamma, beta = self.to_film(cond).chunk(2, dim = -1)

        modality_tokens = x * (gamma + 1.) + beta

        # attention or feedforwards

        out = self.fn(x, **kwargs)

        multiple_returns = isinstance(out, tuple)

        if multiple_returns:
            out, *rest = out

        # take care of conditioning output separately for text vs modality

        modalities_out = out * self.to_ada_ln_zero(cond).sigmoid()

        # take care of function returning cache

        if not multiple_returns:
            return modalities_out

        return (modalities_out, *rest)

        return modalities_out

    def forward(
        self,
        x: Float['b n {self.dim}'],
        cond: Float['b {self.dim_cond}'] | Float['b n {self.dim_cond}'] | None = None,
        is_any_modality: bool | Bool['b n'] | None = None,
        modality_only = False,
        **kwargs
    ):
        if exists(cond) and cond.ndim == 2:
            cond = rearrange(cond, 'b d -> b 1 d')

        if modality_only:
            return self.forward_modality(x, cond = cond, **kwargs)

        assert not (exists(cond) ^ exists(is_any_modality))

        has_modality = exists(is_any_modality)

        if not has_modality:
            return self.forward_text(x, **kwargs)

        if isinstance(is_any_modality, bool):
            is_any_modality = torch.full((x.shape[:-1]), is_any_modality, device = x.device, dtype = torch.bool)

        is_any_modality = rearrange(is_any_modality, '... -> ... 1')

        x = self.layernorm(x)

        gamma, beta = self.to_film(cond).chunk(2, dim = -1)

        text_tokens = x * (self.layernorm_gamma + 1.)

        modality_tokens = x * (gamma + 1.) + beta

        x = torch.where(is_any_modality, modality_tokens, text_tokens)

        # attention or feedforwards

        out = self.fn(x, **kwargs)

        multiple_returns = isinstance(out, tuple)

        if multiple_returns:
            out, *rest = out

        # take care of conditioning output separately for text vs modality

        text_out = out * (self.layerscale + 1.)

        modalities_out = out * self.to_ada_ln_zero(cond).sigmoid()

        conditioned_out = torch.where(is_any_modality, modalities_out, text_out)

        # take care of function returning cache

        if not multiple_returns:
            return conditioned_out

        return (conditioned_out, *rest)

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
        Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        Linear(dim_inner, dim)
    )

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        softcap_value = 50.,
        use_flex_attn = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        assert not (use_flex_attn and not exists(flex_attention)), 'flex attention is only available on torch 2.5.0 (nightly) onwards'
        self.use_flex_attn = use_flex_attn

        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        )

        self.softcap_value = softcap_value

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            Linear(dim_inner, dim, bias = False)
        )

    def forward(
        self,
        x,
        attn_mask: Tensor | None = None, # for manual masking
        rotary_emb: Tensor | None = None,
        cache: Tensor | None = None,
        causal = False,
        block_mask = None, # only passed in for flex attention
        return_kv_cache = False
    ):
        device, input_is_cuda, is_decoding_with_cache = x.device, x.is_cuda, exists(cache)

        should_use_flex_attn = self.use_flex_attn and input_is_cuda

        # handle maybe mask
        # if receiving kv cache, assume decoding and turn off all masking

        if is_decoding_with_cache:
            block_mask = attn_mask = None

        assert not (exists(block_mask) and exists(attn_mask))
        assert not (not self.use_flex_attn and exists(block_mask)), 'you cannot pass in the `block_mask` if `use_flex_attn` was not set to be `True`'

        # pre rmsnorm and project to queries, keys, values

        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        # handle cache being passed in

        if exists(cache):
            cached_k, cached_v = cache
            k = torch.cat((cached_k, k), dim = -2)
            v = torch.cat((cached_v, v), dim = -2)

        # maybe kv cache

        if return_kv_cache:
            kv_cache = torch.stack((k, v))

        # rotary embeddings

        if exists(rotary_emb):
            q, k = tuple(apply_rotary_emb(rotary_emb[..., -t.shape[-2]:, :], t) for t in (q, k))

        # whether to use flex attention or not

        if should_use_flex_attn:
            assert not causal, 'causal mask should be constructed in transformer'

            flex_attn_kwargs = dict(block_mask = block_mask)

            if self.softcap_value > 0.:
                flex_attn_kwargs.update(score_mod = softcap_score_mod(self.softcap_value))

            out = flex_attention(q, k, v, **flex_attn_kwargs)

        else:
            q = q * self.scale
            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            sim = softclamp(sim, self.softcap_value)

            mask_value = -torch.finfo(sim.dtype).max

            if causal:
                i, j = sim.shape[-2:]
                causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
                sim = sim.masked_fill(causal_mask, mask_value)

            if exists(attn_mask):
                sim = einx.where('b i j, b h i j, -> b h i j', attn_mask, sim, mask_value)

            attn = sim.softmax(dim = -1)

            attn = self.dropout(attn)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # combine heads and out

        out = self.to_out(out)

        if not return_kv_cache:
            return out

        return out, kv_cache

class Transformer(Module):
    @beartype
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
        ff_kwargs: dict = dict(),
        unet_skips = True,
        use_flex_attn = False
    ):
        super().__init__()
        self.use_flex_attn = use_flex_attn

        self.dim = dim
        self.dim_head = dim_head

        self.to_time_cond = nn.Sequential(
            RandomFourierEmbed(dim),
            Linear(dim + 1, dim * 4),
            nn.SiLU()
        )

        layers = ModuleList([])

        for ind in range(depth):
            is_latter_half = ind >= (depth // 2)

            skip_proj = Linear(dim * 2, dim, bias = False) if is_latter_half and unet_skips else None

            attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout, use_flex_attn = use_flex_attn, **attn_kwargs)

            ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor, **ff_kwargs)

            attn = AdaptiveWrapper(attn, dim = dim, dim_cond = dim * 4)
            ff = AdaptiveWrapper(ff, dim = dim, dim_cond = dim * 4)

            layers.append(ModuleList([skip_proj, attn, ff]))

        self.layers = layers
        self.norm = RMSNorm(dim)

    def forward(
        self,
        x,
        times: Float[''] | Float['b'] | Float['b n'] | None = None,
        attn_mask: Bool['b i j'] | None = None,
        modality_positions: RawModalityPositions | Int['b n 2'] | None = None,
        is_any_modality: bool | Bool['b n'] | None = None,
        rotary_emb: Tensor | None = None,
        cache: Tensor | None = None,
        modality_only = False,
        causal_mask: bool = False,
        return_kv_cache = False
    ):
        batch, seq_len, device, input_is_cuda = x.shape[0], x.shape[-2], x.device, x.is_cuda

        is_decoding_with_cache = exists(cache)
        needs_masking = not is_decoding_with_cache

        should_use_flex_attn = input_is_cuda and needs_masking and self.use_flex_attn

        assert not (exists(attn_mask) and exists(modality_positions))

        # handle time

        cond = None

        if exists(times):
            if times.ndim == 0:
                times = repeat(times, ' -> b', b = batch)

            cond = self.to_time_cond(times)

        # create the specialized mask needed for autoregressive text + bidirectional flow attention

        attn_mask_kwargs = dict()

        if needs_masking:
            if causal_mask:
                if should_use_flex_attn:
                    block_mask = create_block_mask(causal, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len, device = device)
                    attn_mask_kwargs.update(block_mask = block_mask)
                else:
                    attn_mask_kwargs.update(causal = True)

            if exists(modality_positions):
                assert not causal_mask

                if should_use_flex_attn:
                    transfusion_mask_fn = transfusion_attn_mask(modality_positions)
                    block_mask = create_block_mask(transfusion_mask_fn, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len, device = device)
                    attn_mask_kwargs.update(block_mask = block_mask)
                else:
                    attn_mask = naive_attn_mask(seq_len, modality_positions, device = device)
                    attn_mask_kwargs.update(attn_mask = attn_mask)

        if not exists(is_any_modality) and exists(modality_positions):
            is_any_modality = modality_positions_to_is_modality_mask(seq_len, modality_positions).any(dim = 1)
            is_any_modality = reduce(is_any_modality, 'b t n -> b n', 'any')

        # handle kv caching

        if is_decoding_with_cache:
            cache_length = first(cache).shape[-2]

            x = x[..., cache_length:, :]
            cond = cond[..., cache_length:, :]

            if torch.is_tensor(is_any_modality):
                is_any_modality = is_any_modality[..., cache_length:]

        # adaptive layernorm kwargs, which handles text and modality tokens differently

        adaptive_kwargs = dict(
            cond = cond,
            modality_only = modality_only,
            is_any_modality = is_any_modality
        )

        # handle cache

        cache = default(cache, (None,))
        iter_cache = iter(cache)

        # transformer layers as usual, using mask from above

        skips = []
        new_cache = []

        depth = len(self.layers)

        for ind, (skip_proj, attn, ff) in enumerate(self.layers):
            layer = ind + 1

            # skip connection

            is_first_half = layer <= (depth // 2)
            is_later_half = not is_first_half

            if is_first_half:
                skips.append(x)

            if is_later_half and exists(skip_proj):
                skip = skips.pop()

                residual = x
                x = torch.cat((x, skip), dim = -1)
                x = skip_proj(x) + residual

            # attention and feedforward

            attn_out, kv_cache = attn(
                x,
                rotary_emb = rotary_emb,
                cache = next(iter_cache, None),
                return_kv_cache = True,
                **attn_mask_kwargs,
                **adaptive_kwargs
            )

            new_cache.append(kv_cache)

            x = attn_out + x
            x = ff(x, **adaptive_kwargs) + x

        out = self.norm(x)

        if not return_kv_cache:
            return out

        return out, torch.stack(new_cache)

# classes

class Transfusion(Module):
    @beartype
    def __init__(
        self,
        *,
        num_text_tokens,
        transformer: dict | Transformer,
        dim_latent: int | tuple[int, ...] | None = None,
        channel_first_latent = False,
        modality_encoder: Module | tuple[Module, ...] | None = None,
        modality_decoder: Module | tuple[Module, ...] | None = None,
        modality_token_transform: tuple[ModalityTokenTransform, ...] | ModalityTokenTransform | None = None,
        modality_default_shape: tuple[int, ...] | tuple[tuple[int, ...], ...] | None = None,
        modality_validate_num_dim: int | tuple[int, ...] | None = None,
        to_modality_shape_fn: Callable | tuple[Callable, ...] = default_to_modality_shape_fn,
        ignore_index = -1,
        flow_loss_weight = 1.,
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
    ):
        super().__init__()

        # transformer

        if isinstance(transformer, dict):
            transformer = Transformer(**transformer)

        self.transformer = transformer
        dim = transformer.dim

        self.dim = dim

        # latent and model dimension not the same
        # make it work for 1 modality for now

        dim_latent = default(dim_latent, dim)

        self.dim_latents = cast_tuple(dim_latent)

        # whether the latents are accepted to be channel first or channel last
        # if channel first, will be rearrange(c ... -> ... c -> (...) c)

        self.channel_first_latent = channel_first_latent

        # number of modalities

        self.num_modalities = len(self.dim_latents)

        # functions for converting the sampled language model meta string back to modality shape of tuple[int, ...]

        self.to_modality_shape_fn = cast_tuple(to_modality_shape_fn, self.num_modalities)

        # specifying the number of dimensions for the modality, which will be hard validated

        self.modality_validate_num_dim = cast_tuple(modality_validate_num_dim, self.num_modalities)
        assert len(self.modality_validate_num_dim) == self.num_modalities

        # modality encoders and decoders

        modality_encoder = cast_tuple(modality_encoder, 1 if exists(modality_encoder) else self.num_modalities)
        modality_decoder = cast_tuple(modality_decoder, 1 if exists(modality_decoder) else self.num_modalities)

        self.modality_encoder = ModuleList(modality_encoder)
        self.modality_decoder = ModuleList(modality_decoder)

        assert len(self.modality_encoder) == self.num_modalities
        assert len(self.modality_decoder) == self.num_modalities

        # default token lengths for respective modality
        # fallback if the language model does not come up with valid dimensions

        if not exists(modality_default_shape) or is_bearable(modality_default_shape, tuple[int, ...]):
            modality_default_shape = (modality_default_shape,) * self.num_modalities

        self.modality_default_shape = modality_default_shape

        assert len(self.modality_default_shape) == self.num_modalities

        # entire "sentence" start and end id

        num_text_special_ids = 2

        self.sos_id, self.eos_id = num_text_tokens, (num_text_tokens + 1)

        # modality meta, start and end tokens - termed [mom] [som] [eom] in this repo

        num_modality_special_ids = self.num_modalities * 2
        som_eom_tensor = torch.arange(num_modality_special_ids) + num_text_tokens + num_text_special_ids # shift to the very end
        som_eom_tensor = rearrange(som_eom_tensor, '(start_end m) -> start_end m', start_end = 2)

        # modality meta, start and end ids

        self.som_ids, self.eom_ids = som_eom_tensor.tolist()

        # char tokenizing for modality meta information

        meta_token_offset = num_text_tokens + num_text_special_ids + num_modality_special_ids

        self.meta_id = meta_token_offset

        self.char_tokenizer = partial(char_tokenize, offset = meta_token_offset + 1)
        self.decode_chars = partial(decode_chars, offset = meta_token_offset + 1)

        num_meta_tokens = 128 + 1

        # modality transforms

        modality_token_transform = cast_tuple(modality_token_transform, self.num_modalities)
        modality_token_transform = [default(transform, identity) for transform in modality_token_transform]
        self.modality_token_transform = [Rearrange(maybe_einops_eq) if isinstance(maybe_einops_eq, str) else maybe_einops_eq for maybe_einops_eq in modality_token_transform]

        assert len(self.modality_token_transform) == self.num_modalities

        self.latent_to_model_projs = ModuleList([Linear(dim_latent, dim) if dim_latent != dim else nn.Identity() for dim_latent in self.dim_latents])

        # relative positions

        self.rotary_emb = RotaryEmbedding(transformer.dim_head)

        # embeddings and un-embeddings

        effective_num_text_tokens = num_text_tokens + num_text_special_ids + num_modality_special_ids + num_meta_tokens

        self.text_embed = nn.Embedding(effective_num_text_tokens, dim)

        self.to_text_logits = Linear(dim, effective_num_text_tokens, bias = False)

        self.model_to_latent_preds = ModuleList([Linear(dim, dim_latent, bias = False) for dim_latent in self.dim_latents])

        # loss related

        self.ignore_index = ignore_index
        self.flow_loss_weight = flow_loss_weight

        # flow sampling related

        self.odeint_fn = partial(odeint, **odeint_kwargs)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        prompt: ModalitySample | None = None,
        max_length = 2048,
        text_temperature = 1.5,
        text_min_p = 0.1,
        cache_kv = False,
        fixed_modality_shape: tuple[int, ...] | None = None,
        init_modality_noise: Float['n d'] | None = None,
        modality_steps = 16,
        return_unprocessed_modalities = False
    ) -> ModalitySample:

        device = self.device

        init_text_seq = tensor([self.sos_id], device = device)
        modality_sample = [init_text_seq, *default(prompt, [])]

        # take care of moving to device

        modality_sample = tree_map_tensor(modality_sample, lambda t: t.to(device))
        modality_sample = tree_map_tensor(modality_sample, lambda t: rearrange(t, '-> 1') if t.ndim == 0 else t)

        *_, last_modality_sample = modality_sample
        assert last_modality_sample.dtype in (torch.int, torch.long), 'prompt must be text tokens'

        curr_length = 0
        curr_modality_id = None
        modality_shape = None
        num_past_modalities = 0  # starts off with no modalities in output

        text_is_greedy = text_temperature == 0.
        is_decoding_text = True  # starts off with text decoding, and alternates with modalities depending on [som] tokens detected

        def maybe_transition_to_modality_decoding(seq):
            nonlocal modality_shape
            nonlocal is_decoding_text
            nonlocal curr_modality_id

            sampled_token_id = seq[-1]

            if sampled_token_id not in self.som_ids:
                return

            curr_modality_id = self.som_ids.index(sampled_token_id)

            if exists(fixed_modality_shape):
                modality_shape = fixed_modality_shape
                return

            # get the tokens after the modality meta id

            maybe_meta_tensor = get_tokens_since_rightmost_id(seq, self.meta_id)

            default_shape = self.modality_default_shape[curr_modality_id]
            maybe_modality_validate_num_dim = self.modality_validate_num_dim[curr_modality_id]

            meta_str_to_modality_shape = self.to_modality_shape_fn[curr_modality_id]

            if maybe_meta_tensor.numel() > 0:
                meta_tensor = maybe_meta_tensor[:-1]
                meta_str = self.decode_chars(meta_tensor)

                if not meta_str.isdigit() or int(meta_str) <= 0:

                    assert exists(default_shape), f'invalid modality meta information detected, please set `modality_default_shape` in order to properly fallback'
                    modality_shape = default_shape
                else:
                    modality_shape = meta_str_to_modality_shape(meta_str)

            modality_shape = default(modality_shape, default_shape)

            assert exists(modality_shape), f'language model did not produce a proper modality shape for modality type {curr_modality_id} - please set a fallback shape with `modality_default_shape`'
            assert not exists(maybe_modality_validate_num_dim) or maybe_modality_validate_num_dim == len(modality_shape), f'expected modality type {curr_modality_id} to have {maybe_modality_validate_num_dim} dimensions but language model produced a shape of {modality_shape}'

            is_decoding_text = False

        # determine if to transition from start

        maybe_transition_to_modality_decoding(last_modality_sample)

        cache = None

        with tqdm(total = max_length) as pbar:

            while curr_length <= max_length:

                if is_decoding_text:
                    pbar.set_description('decoding text')

                    *_, seq = modality_sample

                    logits, new_kv_cache = self.forward(
                        [modality_sample],
                        return_loss = False,
                        cache = cache,
                        decoding_text_or_modality = 'text',
                        return_kv_cache = True
                    )

                    logits = logits[0][-1]

                    if text_is_greedy:
                        sampled = logits.argmax(dim = -1, keepdim = True)
                    else:
                        logits = min_p_filter(logits, min_p = text_min_p)

                        probs = (logits / text_temperature).softmax(dim = -1)
                        sampled = torch.multinomial(probs, 1)

                    seq = torch.cat((seq, sampled), dim = -1)
                    modality_sample[-1] = seq

                    pbar.update(1)
                    curr_length += 1

                    if cache_kv:
                        cache = new_kv_cache

                    sampled_token_id = sampled.item()

                    if sampled_token_id == self.eos_id:
                        break

                    maybe_transition_to_modality_decoding(seq)

                else:
                    assert exists(curr_modality_id)
                    pbar.set_description(f'decoding modality [{curr_modality_id}]')

                    latent_dim = self.dim_latents[curr_modality_id]

                    maybe_modality_decoder = self.modality_decoder[curr_modality_id]

                    modality_length = math.prod(modality_shape)

                    if exists(init_modality_noise):
                        noise = init_modality_noise[:modality_length, :latent_dim]
                    else:
                        assert exists(modality_length)
                        noise = torch.randn((modality_length, latent_dim), device = device)

                    assert noise.shape == (modality_length, latent_dim)

                    new_kv_cache = None

                    def ode_step_fn(step_times, denoised):
                        nonlocal new_kv_cache

                        step_times = rearrange(step_times, ' -> 1 1') # batch size of 1
                        step_times = F.pad(step_times, (num_past_modalities, 0), value = 1.) # past decoded modalities receive a time conditioning of 1.

                        embeds, new_kv_cache = self.forward(
                            [[*modality_sample, (curr_modality_id, denoised)]],
                            times = step_times,
                            return_embed = True,
                            cache = cache,
                            return_kv_cache = True,
                            decoding_text_or_modality = 'modality'
                        )

                        to_flow_pred = self.model_to_latent_preds[curr_modality_id]
                        flow = to_flow_pred(embeds)

                        return flow[0, -modality_length:]

                    times = torch.linspace(0, 1, modality_steps, device = device)
                    trajectory = self.odeint_fn(ode_step_fn, noise, times)

                    # add the sampled modality tokens

                    sampled_modality = trajectory[-1]

                    # reshape

                    sampled_modality = sampled_modality.reshape(*modality_shape, latent_dim)

                    modality_sample.append((curr_modality_id, sampled_modality))

                    # add the appropriate [eom]

                    eom_id = self.eom_ids[curr_modality_id]
                    modality_sample.append(tensor([eom_id], device = device))

                    # set kv cache if needed

                    if cache_kv:
                        cache = new_kv_cache

                    # back to decoding text

                    pbar.update(modality_length)
                    curr_length += modality_length

                    num_past_modalities += 1
                    curr_modality_id = None
                    modality_length = None

                    is_decoding_text = True

        if return_unprocessed_modalities:
            return modality_sample

        # post process modalities

        processed_modality_sample = []

        for sample in modality_sample:
            if not isinstance(sample, tuple):
                processed_modality_sample.append(sample)
                continue

            modality_id, modality = sample
            maybe_modality_decoder = self.modality_decoder[modality_id]

            if self.channel_first_latent:
                modality = rearrange(modality, '... d -> d ...')

            if exists(maybe_modality_decoder):
                modality = maybe_modality_decoder(modality)

            processed_modality_sample.append((modality_id, modality))

        return processed_modality_sample

    def forward_text(
        self,
        text: Int['b n'],
        return_loss = True,
        return_embed = False,
        cache: Tensor | None = None,
        return_kv_cache = False
    ) -> (
        Float[''],
        Float['b n d'],
        tuple[Float['b n d'], list[Float['...']]]
    ):

        device = self.device
        text = text.to(device)

        text, labels = text[:, :-1], text[:, 1:]

        # embed text

        text = text.masked_fill(text == -1, 0)
        tokens = self.text_embed(text)

        # rotary

        seq_len = tokens.shape[-2]
        pos = torch.arange(seq_len, device = device)

        rotary_emb = self.rotary_emb(pos)

        # attention

        embed, kv_cache = self.transformer(
            tokens,
            rotary_emb = rotary_emb,
            causal_mask = True,
            cache = cache,
            return_kv_cache = True
        )

        # text unembedding

        logits = self.to_text_logits(embed)

        if not return_loss:
            if not return_kv_cache:
                return logits

            return logits, kv_cache

        loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss

    def forward_modality(
        self,
        modalities: Float['b ...'],
        modality_type: int | None = None,
        return_loss = True
    ) -> Float['']:

        if self.num_modalities > 1:
            assert exists(modality_type), '`modality_type` must be explicitly passed in on forward when training on greater than 1 modality'

        modality_type = default(modality_type, 0)

        transform = self.modality_token_transform[modality_type]
        latent_to_model_fn = self.latent_to_model_projs[modality_type]
        model_to_flow_pred_fn = self.model_to_latent_preds[modality_type]

        tokens = transform(modalities)

        # maybe channel first

        if self.channel_first_latent:
            tokens = rearrange(tokens, 'b d ... -> b (...) d')

        # rotary

        batch, seq_len, device = tokens.shape[0], tokens.shape[-2], tokens.device

        pos = torch.arange(seq_len, device = device)

        rotary_emb = self.rotary_emb(pos)

        # times

        times = torch.rand((batch,), device = device)
        padded_times = rearrange(times, 'b -> b 1 1')

        noise = torch.randn_like(tokens)

        noised_tokens = padded_times * tokens + (1. - padded_times) * noise

        flow = tokens - noise

        # attention

        noised_tokens = latent_to_model_fn(noised_tokens)

        embed = self.transformer(
            noised_tokens,
            times = times,
            rotary_emb = rotary_emb,
            modality_only = True,
        )

        pred_flow = model_to_flow_pred_fn(embed)

        if not return_loss:
            return pred_flow

        # flow loss

        return F.mse_loss(pred_flow, flow)

    def forward(
        self,
        modalities: (
            list[ModalitySample] |
            Int['b n'] |
            Float['b ...']
        ),
        times: Float['b m'] | None = None,
        modality_length_to_times_fn: Callable[[Int['b m']], Float['b m']] | None = None, # allows a researcher to customize the times (noise level) based on the modality lengths in a given sample 
        modality_type: int | None = None,
        cache: Tensor | None = None,
        decoding_text_or_modality: Literal['text', 'modality'] | None = None,
        return_loss = True,
        return_breakdown = False,
        return_embed = False,
        return_kv_cache = False,
    ) -> (
        Float['b n l'] |
        Float['b n d'] |
        tuple[Float['b n _'], list[Float['...']]] |
        Float[''] |
        tuple[Float[''], LossBreakdown]
    ):
        is_decoding = exists(decoding_text_or_modality)

        is_text_only = torch.is_tensor(modalities) and modalities.dtype in (torch.int, torch.long)
        is_modality_only = torch.is_tensor(modalities) and modalities.dtype == torch.float

        return_loss &= (not return_embed or not is_decoding)

        if is_text_only:

            forward_text_kwargs = dict(
                return_loss = return_loss,
                return_embed = return_embed,
                cache = cache,
                return_kv_cache = return_kv_cache
            )

            return self.forward_text(modalities, **forward_text_kwargs)

        if is_modality_only:
            assert return_loss
            return self.forward_modality(modalities, modality_type = modality_type)

        device = self.device
        tensor_ = partial(tensor, device = device)

        # add "sentence" start and end tokens when training

        if return_loss:
            for modality in modalities:
                prepend(modality, tensor_([self.sos_id]))
                modality.append(tensor_([self.eos_id]))

        # process list of text and modalities interspersed with one another

        modality_positions = []
        modality_tokens = []
        text = []

        for batch_modalities in modalities:
            batch_modality_positions = []
            batch_modality_tokens = []
            batch_text = []
            offset = 0

            for modality in batch_modalities:
                # if non-text modality detected and not given as a tuple
                # cast to (int, Tensor) where int is defaulted to type 0 (convenience for one modality)

                if torch.is_tensor(modality) and modality.dtype == torch.float:
                    modality = (0, modality)

                is_text = not isinstance(modality, tuple)

                if is_text:
                    modality_tensor = modality
                else:
                    modality_type, modality_tensor = modality

                    if not is_decoding:
                        modality_transform = self.modality_token_transform[modality_type]
                        maybe_modality_encode = self.modality_encoder[modality_type]

                        modality_tensor = modality_transform(modality_tensor)

                        if exists(maybe_modality_encode):

                            with torch.no_grad():
                                maybe_modality_encode.eval()
                                modality_tensor = maybe_modality_encode(modality_tensor).detach()

                        if self.channel_first_latent:
                            modality_tensor = rearrange(modality_tensor, 'd ... -> ... d')

                    assert 0 <= modality_type < self.num_modalities, f'received a modality index that is out of range. only {self.num_modalities} modalities specified'
                    assert self.dim_latents[modality_type] == modality_tensor.shape[-1], f'mismatch for modality latent dimension - expected {self.dim_latents[modality_type]} but received {modality_tensor.shape[-1]} - modality shape is {tuple(modality_tensor.shape)}, perhaps you need to set `channel_first_latent = True`'

                # auto move modality tensor to device of model

                modality_tensor = modality_tensor.to(device)

                if modality_tensor.dtype in (torch.int, torch.long) and modality_tensor.ndim == 0:
                    modality_tensor = rearrange(modality_tensor, '-> 1')

                # handle text

                if is_text:
                    assert modality_tensor.ndim == 1
                    text_length = modality_tensor.shape[0]

                    batch_text.append(modality_tensor)
                    offset += text_length
                    continue

                # otherwise handle a modality

                modality_shape_tuple = tuple(modality_tensor.shape[:-1])
                modality_length = math.prod(modality_shape_tuple)

                text_tensor = torch.full((modality_length,), -1, device = device) # text is all -1 here, so text labels are not learned on

                # add the [som] and [eom] tokens for the modality type

                som_id, eom_id = self.som_ids[modality_type], self.eom_ids[modality_type]

                # start by just storing the token length of the modality

                modality_shape_str = join([*map(str, modality_shape_tuple)], ',')
                modality_meta_info = self.char_tokenizer(modality_shape_str, device = device)

                text_tensor = torch.cat((
                    tensor_([self.meta_id]),
                    modality_meta_info,
                    tensor_([som_id]),
                    text_tensor,
                    tensor_([eom_id])
                ))

                batch_text.append(text_tensor)
                modality_tensor = rearrange(modality_tensor, '... d -> (...) d')

                batch_modality_tokens.append(modality_tensor)

                batch_modality_positions.append((modality_type, offset + 1, modality_length)) # offset + 1 due to extra [som] token

                offset += modality_length + 2 + len(modality_meta_info) + 1 # +2 due to [som] and [eom] - then account for meta start id and modality shape information (or eventually any meta information about modality)

            text.append(torch.cat(batch_text))
            modality_tokens.append(batch_modality_tokens)
            modality_positions.append(batch_modality_positions)

        text = pad_sequence(text, padding_value = -1)

        # if returning loss, split text for next token prediction

        if return_loss:
            text, text_labels = text[:, :-1], text[:, 1:]

        # derive is_modality mask for flow on the right tokens + flow loss

        batch, seq_len, device = *text.shape, text.device

        assert len(modality_positions) == batch

        if isinstance(modality_positions, list):
            modality_positions = modality_positions_to_tensor(modality_positions, device = device)

        if modality_positions.shape[-1] == 2: # Int['b m 2'] -> Int['b m 3'] if type is not given (one modality)
            modality_positions = F.pad(modality_positions, (1, 0), value = 0)

        # for now use dummy padding modality position info if empty (all zeros)

        if modality_positions.numel() == 0:
            modality_positions = F.pad(modality_positions, (0, 0, 0, 1))

        # embed the list of modality tokens into a sequence of Float['b n d'] at right offsets and lengths as dictated by modalities info tensor

        if torch.is_tensor(modality_tokens):
            modality_tokens = [modality_tokens]

        # embed the modality tokens into one Tensor if not given as one

        if isinstance(modality_tokens, list) and isinstance(first(modality_tokens), list): # detect list[list[tensor]]
            modality_tokens = [embed_modality_tokens(seq_len, dim_latent, modality_tokens, modality_positions, modality_id) for modality_id, dim_latent in enumerate(self.dim_latents)]

        # sort the modalities tensor and sanitize, readying for noising of modalities

        modality_positions, sorted_indices = order_modality_positions_by_seq_offset(modality_positions)

        num_modalities = modality_positions.shape[-2]

        is_modalities = modality_positions_to_is_modality_mask(seq_len, modality_positions, num_modalities = self.num_modalities, device = device)

        is_any_modality = reduce(is_modalities, 'b t m n -> b n', 'any')

        # embed text

        text = text.masked_fill(text == -1, 0)

        text_tokens = self.text_embed(text)

        # noise the modality tokens

        if not exists(times):
            modality_length_to_times_fn = default(default_modality_length_to_time_fn, modality_length_to_times_fn)

            if exists(modality_length_to_times_fn):
                times = modality_length_to_times_fn(modality_positions[..., -1])

        times = einsum(is_modalities.float(), times, 'b t m n, b m -> b t n')

        if return_loss:
            noised_modality_tokens = []
            flows = []

            for modality_id, one_modality_tokens in enumerate(modality_tokens):
                noise = torch.randn_like(one_modality_tokens)

                one_times = times[:, modality_id]
                padded_times = rearrange(one_times, 'b n -> b n 1')

                one_noised_modality_tokens = one_modality_tokens * padded_times + noise * (1. - padded_times)

                # the flow is the (data - noise)

                one_flow = one_modality_tokens - noise

                # append

                flows.append(one_flow)
                noised_modality_tokens.append(one_noised_modality_tokens)

            modality_tokens = noised_modality_tokens

        # project the modality tokens to model

        modality_tokens = [fn(one_modality_tokens) for fn, one_modality_tokens in zip(self.latent_to_model_projs, modality_tokens)]

        modality_tokens = sum(modality_tokens)

        # intersperse the modalities with the text for the joint transformer + flow system

        tokens = einx.where('b n, b n d, b n d', is_any_modality, modality_tokens, text_tokens)

        # derive rotary positions

        rotary_positions = derive_rotary_positions_from_modality_positions(seq_len, modality_positions)

        rotary_emb = self.rotary_emb(rotary_positions)
        rotary_emb = rearrange(rotary_emb, 'b n d -> b 1 n d')

        # take care of cache

        is_any_modality_when_decoding = None

        if exists(cache):
            assert exists(decoding_text_or_modality)
            is_any_modality_when_decoding = decoding_text_or_modality == 'modality'
            modality_positions = None

        # times

        times = reduce(times, 'b t n -> b n', 'sum')

        # attention

        embed, kv_cache = self.transformer(
            tokens,
            times = times,
            rotary_emb = rotary_emb,
            modality_positions = modality_positions,
            is_any_modality = is_any_modality_when_decoding,
            cache = cache,
            return_kv_cache = True
        )

        # early return for embedding for decoding modality

        if return_embed:
            if not return_kv_cache:
                return embed

            return embed, kv_cache

        # text unembedding

        text_logits = self.to_text_logits(embed)

        if not return_loss:
            if not return_kv_cache:
                return text_logits

            return text_logits, kv_cache

        # calculate total tokens for weighing the loss

        total_tokens = (text_labels != self.ignore_index).sum()

        # text autoregressive loss

        text_labels = text_labels.masked_fill(is_any_modality, self.ignore_index)

        text_loss = F.cross_entropy(
            rearrange(text_logits, 'b n l -> b l n'),
            text_labels,
            ignore_index = self.ignore_index
        )

        text_loss_weight = (text_labels != self.ignore_index).sum() / total_tokens

        # flow loss

        pred_flows = [fn(embed) for fn in self.model_to_latent_preds]

        flow_losses = []
        modality_loss_weights = []

        for flow, pred_flow, is_one_modality in zip(flows, pred_flows, is_modalities.unbind(dim = 1)):

            flow_loss = F.mse_loss(
                pred_flow,
                flow,
                reduction = 'none'
            )

            is_one_modality = reduce(is_one_modality, 'b m n -> b n', 'any')

            flow_loss = flow_loss[is_one_modality].mean()

            modality_loss_weight = is_one_modality.sum() / total_tokens

            modality_loss_weights.append(modality_loss_weight)

            flow_losses.append(flow_loss)

        # only the token positions that are not modalities have autoregressive loss

        total_loss = (
            text_loss * text_loss_weight +
            (torch.stack(flow_losses) * torch.stack(modality_loss_weights)).sum() * self.flow_loss_weight
        )

        if not return_breakdown:
            return total_loss

        return total_loss, LossBreakdown(total_loss, text_loss, flow_losses)
