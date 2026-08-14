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
p - positions
"""

import os
import math
from collections import defaultdict

from itertools import chain
from functools import partial, wraps, cache
from dataclasses import dataclass
from typing import NamedTuple, Callable, Literal

import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor, is_tensor, cat, stack, atleast_1d
from torch.nn import Module, ModuleList, Linear

from torch.utils.data import Dataset, DataLoader
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

from torchdiffeq import odeint

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, einsum, pack, unpack

from ema_pytorch import EMA

from axial_positional_embedding import ContinuousAxialPositionalEmbedding

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from tqdm import tqdm
from loguru import logger

from torch_einops_utils import (
    pack_with_inverse,
    tree_map_tensor,
    tree_map_tensor_to_device,
    temp_eval,
    reverse_cumsum,
    pad_left_at_dim,
    pad_right_at_dim,
    pad_sequence,
    batched_index_select
)

from .modality_processing import (
    ModalitySample,
    GetPredFlows,
    get_model_output_to_flow_fn,
    evaluate_modality_pos_emb,
    DEFAULT_PROCESSING_STRATEGY,
    get_processing_strategy
)

# tensor typing

import jaxtyping
from jaxtyping import jaxtyped
from beartype import beartype
from beartype.door import is_bearable

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

# types

Scalar = Float['']

ModalityTokenTransform = str | Callable | None

RawModalityPositions = list[list[tuple[int, int, int]]]

class LossBreakdown(NamedTuple):
    total: Scalar
    text: Scalar
    flow: list[Scalar]
    velocity: list[Scalar] | None = None
    recon: list[Scalar] | None = None

class ModalityInfo(NamedTuple):
    encoder: Module | None
    decoder: Module | None
    latent_to_model: Module
    model_to_latent: Module
    add_pos_emb: bool
    pos_emb_mlp: Module | None
    num_dim: int | None
    dim_latent: int
    default_shape: tuple[int, ...]
    som_id: int
    eom_id: int
    to_shape_fn: Callable | None
    channel_first_latent: bool
    modality_type: int

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def first(it):
    return it[0]

def join(arr, delimiter = ''):
    return delimiter.join(arr)

def divisible_by(num, den):
    return (num % den) == 0

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def is_int_tensor(t):
    return is_tensor(t) and t.dtype in (torch.int, torch.long)

def set_dropout_(model: Module, prob: float):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = prob

def add_temp_batch_dim(fn: Callable):
    @wraps(fn)
    def inner(t: Tensor, *args, **kwargs) -> Tensor:
        t = rearrange(t, '... -> 1 ...')
        out = fn(t, *args, **kwargs)
        out = rearrange(out, '1 ... -> ...')
        return out
    return inner

# maybe typecheck

typecheck = jaxtyped(typechecker = beartype) if os.environ.get('TYPECHECK', '').lower() in ('1', 'true') else identity

# default function for constituting modality shape from string

def default_to_modality_shape_fn(maybe_shape_str) -> tuple[int, ...]:
    return tuple([*map(int, maybe_shape_str.split(','))])

# default function for translating modality length to times (noise level, where 0 is highest noise)

def random_modality_length_to_time_fn(num_modalities: Int['b']) -> Float['b m']:
    batch, device = num_modalities.shape[0], num_modalities.device
    total_modalities = num_modalities.amax().item()
    return torch.rand((batch, total_modalities), device = device)

def default_modality_length_to_time_fn(num_modalities: Int['b']) -> Float['b m']:
    batch, device = num_modalities.shape[0], num_modalities.device
    total_modalities = num_modalities.amax().item()

    if total_modalities == 0:
        return torch.empty((batch, 0), device = device, dtype = torch.float)

    rand_num_modalities = torch.floor(torch.rand_like(num_modalities.float()) * num_modalities)
    seq = torch.arange(total_modalities, device = device)

    prev_decoded_modality = einx.less('m, b -> b m', seq, rand_num_modalities)
    curr_modality_rand_time = torch.rand_like(num_modalities.float())

    # in paper, they fix previous decoded modalities to 500 / 1000 steps for discrete ddpm, here using flow matching with times 0 - 1 so corresponds to 0.5
    return einx.where('b m, , b -> b m', prev_decoded_modality, 0.5, curr_modality_rand_time)

# pretty print

def concat_contiguous_text(
    modality_sample: ModalitySample
) -> ModalitySample:
    """ within a modality sample, any two tensors of type int / long will be concatted together if next to each other, so all text is followed by a modality, and all modality followed by text """

    output = []

    for modality in modality_sample:
        if (
            len(output) > 0 and
            is_int_tensor(output[-1]) and is_int_tensor(modality) and
            output[-1].dtype == modality.dtype
        ):
            packed_text, _ = pack((output[-1], modality), '*')
            output[-1] = packed_text

        else:
            output.append(modality)

    return output

def print_modality_sample(
    modality_sample: ModalitySample
):
    output = []

    for sample in modality_sample:
        if isinstance(sample, tuple):
            modality_type, sample = sample
            output.append((f'modality:{modality_type}', sample.shape))
        elif is_int_tensor(sample):
            output.append(('text', sample.shape))
        else:
            output.append(('modality', sample.shape))

    logger.info(output)

# character based tokenizer

def char_tokenize(
    text: str,
    device = None,
    offset = 0
) -> Tensor:
    tokenized = tensor([*map(ord, text)], device = device) + offset
    return tokenized.long()

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

    after_right_mask = reverse_cumsum(mask.int(), dim = 0) == 0
    return t[after_right_mask]

# tensor helpers

def l2norm(t):
    return F.normalize(t, dim = -1)

def softclamp(t, value = 50.):
    return (t / value).tanh() * value

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def append_dims(t, ndims):
    return t.reshape(*t.shape, *((1,) * ndims))

def is_empty(t):
    return t.numel() == 0

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, dim = -1, keepdim = True):
    noise = gumbel_noise(t)
    return (t + noise).argmax(dim = dim, keepdim = keepdim)

# dataloader related

def collate_fn(data):
    return [*map(list, data)]

@typecheck
def create_dataloader(dataset: Dataset, **kwargs) -> DataLoader:
    return DataLoader(dataset, collate_fn = collate_fn, **kwargs)

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

@typecheck
def modality_positions_to_tensor(
    modalities: RawModalityPositions,
    pad_value = 0,
    device = None
) -> Int['b m 2'] | Int['b m 3']:

    processed = []

    for modality in modalities:
        if len(modality) == 0:
            modality = torch.empty((0, 3), device = device, dtype = torch.long)
        else:
            modality = tensor(modality, device = device)
            modality = rearrange(modality, '... d -> (...) d')

        processed.append(modality)

    modalities = pad_sequence(processed, dim = -2, value = pad_value)

    if modalities.ndim == 2:
        modalities = modalities.reshape(*modalities.shape, 3)

    return modalities.long()

# sanitizing modalities tensor, making sure it is ordered

@typecheck
def order_modality_positions_by_seq_offset(
    modalities: Int['b m 3']
) -> tuple[Int['b m 3'], Int['b m']]:

    modality_type, offsets, lengths = modalities.unbind(dim = -1)

    no_modality_mask = lengths <= 0 # there may be uneven number of modalities per batch sample
    offsets_to_sort = offsets.masked_fill(no_modality_mask, 1e10)
    _, sorted_indices = offsets_to_sort.sort(dim = -1)

    # sort by ascending offset

    modalities = batched_index_select(modalities, sorted_indices)
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

    _, offsets, lengths = modalities.unbind(dim = -1)
    seq = torch.arange(seq_len, device = device)

    is_extra_modality_token = (
        einx.greater('i, b m -> b m i', seq, offsets) &
        einx.less('j, b m -> b m j', seq, offsets + lengths)
    )

    is_any_extra = reduce(is_extra_modality_token, 'b m n -> b n', 'any')

    return seq - is_any_extra.cumsum(dim = -1)

# modality tokens are given as list of tensors, can be then be embedded into the modality tokens for attending alongside text tokens

@typecheck
def embed_modality_tokens(
    seq_len: int,
    dim: int,
    modality_tokens: list[list[Float['...']]],
    modalities: Int['b m 3'],
    modality_id: int,
    channel_first: bool
) -> Float['b n d']:

    batch, device = modalities.shape[0], modalities.device

    shape = (batch, seq_len, dim) if not channel_first else (batch, dim, seq_len)
    output = torch.zeros(shape, device = device)

    for batch_ind, (one_modality, one_modality_token) in enumerate(zip(modalities, modality_tokens)):
        for (modality_type, offset, length), batch_modality_token in zip(one_modality, one_modality_token):

            if modality_id != modality_type or length <= 0:
                continue

            modality_shape = batch_modality_token.shape

            if channel_first:
                mod_dim, *mod_axial_shape = modality_shape
                batch_modality_token = rearrange(batch_modality_token, 'd ... -> d (...)')
            else:
                *mod_axial_shape, mod_dim = modality_shape
                batch_modality_token = rearrange(batch_modality_token, '... d -> (...) d')

            mod_length = math.prod(mod_axial_shape)

            assert length == mod_length, f'received a modality of shape {modality_shape} but sequence length in modalities info is {length}'
            assert dim == mod_dim, f'received modality [{modality_id}] with shape {modality_shape} but expected dimension of {dim}'

            if channel_first:
                output[batch_ind, :, offset:(offset + length)] = batch_modality_token
            else:
                output[batch_ind, offset:(offset + length), :] = batch_modality_token

    return output

# functions for managing modality token mask

@typecheck
def modality_positions_to_is_modality_mask(
    seq_len: int,
    modalities: Int['b m 3'],
    offset: Int['2'] | None = None,
    device = None,
    num_modalities = 1
) -> Bool['b t m n']:

    device = modalities.device

    if exists(offset):
        offset = pad_left_at_dim(offset, 1, dim = 0)
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

@typecheck
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

# unet encoder related function

def stack_same_shape_tensors_with_inverse(tensors: list[Tensor]):

    shape_tensors_dict = defaultdict(list)
    shape_batch_dict = defaultdict(int) # also store a shape -> num tensors dictionary to validate inverse function input
    inverse_index_list = []

    for tensor in tensors:
        shape = tuple(tensor.shape)
        batch_el = shape_batch_dict[shape]

        shape_tensors_dict[shape].append(tensor)
        shape_batch_dict[shape] += 1

        inverse_index_list.append((shape, batch_el))

    # stack all the tensors with same shapes to have a batch dimension

    shape_tensors_dict = {shape: torch.stack(same_shape_tensors) for shape, same_shape_tensors in shape_tensors_dict.items()}

    # inverse function

    def inverse(
        indexed_tensors: dict[tuple[int, ...], Tensor]
    ) -> list[Tensor]:

        out_shape_batch_dict = {shape: len(tensors) for shape, tensors in indexed_tensors.items()}

        assert out_shape_batch_dict == shape_batch_dict

        inversed = []

        for shape, batch_el in inverse_index_list:
            tensor = indexed_tensors[shape][batch_el]
            inversed.append(tensor)

        return inversed

    return shape_tensors_dict, inverse

def filter_with_inverse(cond, inp):

    indices = set()
    filtered = []

    for ind, el in enumerate(inp):
        if cond(el):
            indices.add(ind)
            filtered.append(el)

    def inverse(inverse_inp):
        assert len(inverse_inp) == len(filtered)

        output = []
        inverse_inp_index = 0

        for ind, el in enumerate(inp):
            if ind not in indices:
                output.append(el)
                continue

            inverse_inp_el = inverse_inp[inverse_inp_index]
            output.append(inverse_inp_el)
            inverse_inp_index += 1

        return output

    return filtered, inverse

def apply_fn_modality_type(
    fn: Callable,
    modalities: ModalitySample | list[ModalitySample],
    modality_type = 0,
    return_untransformed = False
) -> ModalitySample | list[ModalitySample]:

    modalities, tree_spec = tree_flatten(modalities, is_leaf = lambda el: isinstance(el, tuple))

    # standardize tuples to (<modality_type>, <modality_tensor>)

    modalities = [(0, m) if (is_tensor(m) and m.is_floating_point()) else m for m in modalities]

    # filter for specific modality type to transform

    modalities, inverse_filter = filter_with_inverse(lambda el: isinstance(el, tuple) and el[0] == modality_type, modalities)

    # remove the type

    modalities = [m for _, m in modalities]

    # batch process

    stacked_modalities, inverse_stack = stack_same_shape_tensors_with_inverse(modalities)

    out = {shape: fn(batched_modalities) for shape, batched_modalities in stacked_modalities.items()}

    out = inverse_stack(out)

    # add back the type

    if return_untransformed:
        out = [(modality_type, transformed_m, prev_m) for transformed_m, prev_m in zip(out, modalities)]
    else:
        out = [(modality_type, transformed_m) for transformed_m in out]

    # replace transformed modalities and untree flatten

    out = inverse_filter(out)

    return tree_unflatten(out, tree_spec)

# modality processing strategies live in `modality_processing.py`, pickable via `Transfusion(..., modality_processing = ...)`

# sampling related functions

# min_p for text
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

# random fourier embedding

class RandomFourierEmbed(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        self.dim = dim
        self.register_buffer('weights', torch.randn(dim // 2))

    @typecheck
    def forward(
        self,
        times: Float['b n'] | Float['b']
    ) -> Float['b n {self.dim+1}']:

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

    @typecheck
    def forward_text(
        self,
        x: Float['b n {self.dim}'],
        **kwargs
    ):
        x = self.layernorm(x)

        x = x * (self.layernorm_gamma + 1.)

        out = self.fn(x, **kwargs)

        (out, *rest), tree_spec = tree_flatten(out)

        out = out * (self.layerscale + 1.)

        out = tree_unflatten((out, *rest), tree_spec)

        return out

    @typecheck
    def forward_modality(
        self,
        x: Float['b n {self.dim}'],
        cond: Float['... {self.dim_cond}'],
        **kwargs
    ):
        x = self.layernorm(x)

        gamma, beta = self.to_film(cond).chunk(2, dim = -1)

        modality_tokens = x * (gamma + 1.) + beta

        # attention or feedforwards

        out = self.fn(modality_tokens, **kwargs)

        (out, *rest), tree_spec = tree_flatten(out)

        # take care of conditioning output separately for text vs modality

        modalities_out = out * self.to_ada_ln_zero(cond).sigmoid()

        # take care of function returning cache or value residual

        modalities_out = tree_unflatten((modalities_out, *rest), tree_spec)

        return modalities_out

    @typecheck
    def forward(
        self,
        x: Float['b n {self.dim}'],
        cond: Float['... {self.dim_cond}'] | None = None,
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

        (out, *rest), tree_spec = tree_flatten(out)

        # take care of conditioning output separately for text vs modality

        text_out = out * (self.layerscale + 1.)

        modalities_out = out * self.to_ada_ln_zero(cond).sigmoid()

        conditioned_out = torch.where(is_any_modality, modalities_out, text_out)

        # take care of function returning cache or value residual

        conditioned_out = tree_unflatten((conditioned_out, *rest), tree_spec)

        return conditioned_out

# attention

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return l2norm(x) * self.scale * (self.gamma + 1.) # use unit offset from Ohad Rubin

# attention residual

class AttentionResidual(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

        self.scale = dim ** -0.5
        self.norm_keys = RMSNorm(dim)

        self.pseudo_queries = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.pseudo_queries, std = 0.02)

    def forward(
        self,
        hiddens
    ):
        if isinstance(hiddens, list):
            hiddens = stack(hiddens)

        # cross attention

        values = hiddens
        keys = self.norm_keys(values)

        # sim[b, n, l] = sum_d pseudo_queries[d] * keys[l, b, n, d]
        # `torch.mv` instead of einsum - the ellipsis einsum path has been observed to
        # read garbage for these shapes, producing nan

        keys, unpack_keys = pack_with_inverse(keys, '* d')

        sim = torch.mv(keys, self.pseudo_queries)
        sim = unpack_keys(rearrange(sim, 'n -> n 1'))
        sim = rearrange(sim, 'l b n 1 -> b n l') * self.scale

        # attention and aggregate

        attn = sim.softmax(dim = -1)

        return einsum(attn, values, '... l, l ... d -> ... d')

class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return F.gelu(gates) * x

class FeedForward(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 4.,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor * 2 / 3)
        self.net = nn.Sequential(
            Linear(dim, dim_inner * 2),
            GEGLU(),
            nn.Dropout(dropout),
            Linear(dim_inner, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        softcap_value = 50.,
        use_flex_attn = False,
        gate_values = True,
        laser = False,
        laser_softclamp_value = 15.,
        learned_value_residual_mix = False,
        qk_rmsnorm = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        assert not (use_flex_attn and not exists(flex_attention)), 'flex attention is only available on torch 2.5.0 (nightly) onwards'
        self.use_flex_attn = use_flex_attn

        self.to_qk = nn.Sequential(
            Linear(dim, dim_inner * 2, bias = False),
            Rearrange('b n (qk h d) -> qk b h n d', qk = 2, h = heads)
        )

        # qk rmsnorm keeps the attention logits bounded regardless of the magnitude of the
        # inputs (text tokens, noised modalities, previously decoded modalities all share
        # one sequence), preventing attention from blowing up into nan during sampling

        self.qk_rmsnorm = qk_rmsnorm
        self.q_norm = RMSNorm(dim_head)
        self.k_norm = RMSNorm(dim_head)

        self.to_v = nn.Sequential(
            Linear(dim, dim_inner, bias = False),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        self.to_learned_value_residual = nn.Sequential(
            nn.Linear(dim, heads),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1') # add head dimension
        ) if learned_value_residual_mix else always(0.5)

        self.to_gates = nn.Sequential(
            nn.Linear(dim, heads, bias = False),
            Rearrange('b n h -> b h n 1', h = heads)
        ) if gate_values else None

        self.softcap_value = softcap_value

        self.laser = laser
        self.laser_softclamp_value = laser_softclamp_value

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
        return_kv_cache = False,
        return_values = False,
        value_residual: Tensor | None = None
    ):
        device, input_is_cuda, is_decoding_with_cache = x.device, x.is_cuda, exists(cache)

        should_use_flex_attn = self.use_flex_attn and input_is_cuda

        # handle maybe mask
        # if receiving kv cache, assume decoding and turn off all masking
        # (an explicit `attn_mask` is still honored - `sample_many` excludes the padded kv cache entries of other samples this way)

        if is_decoding_with_cache and not exists(attn_mask):
            block_mask = attn_mask = None

        assert not (exists(block_mask) and exists(attn_mask))
        assert not (not self.use_flex_attn and exists(block_mask)), 'you cannot pass in the `block_mask` if `use_flex_attn` was not set to be `True`'

        # project to queries, keys, values

        q, k, v = (*self.to_qk(x), self.to_v(x))

        # qk rmsnorm - normalize queries and keys separately so attention logits stay bounded

        if self.qk_rmsnorm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # value residual

        orig_v = v

        if exists(value_residual):
            mix = self.to_learned_value_residual(x)
            v = v * mix + value_residual * (1. - mix)

        # rotary embeddings

        if exists(rotary_emb):
            q, k = tuple(apply_rotary_emb(rotary_emb, t, freqs_seq_dim = -2) for t in (q, k))

        # handle cache being passed in

        if exists(cache):
            cached_k, cached_v = cache
            k = cat((cached_k, k), dim = -2)
            v = cat((cached_v, v), dim = -2)

        # maybe kv cache

        if return_kv_cache:
            kv_cache = stack((k, v))

        # laser attention

        if self.laser:
            v = softclamp(v, self.laser_softclamp_value)
            v = v.exp()

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

            mask_value = max_neg_value(sim)

            if causal:
                i, j = sim.shape[-2:]
                causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
                sim = sim.masked_fill(causal_mask, mask_value)

            if exists(attn_mask):
                sim = einx.where('b i j, b h i j, -> b h i j', attn_mask, sim, mask_value)

            attn = sim.softmax(dim = -1)

            attn = self.dropout(attn)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # laser attention

        if self.laser:
            out = log(out)

        # maybe gate values

        if exists(self.to_gates):
            out = out * self.to_gates(x).sigmoid()

        # combine heads and out

        out = self.to_out(out)

        if return_values:
            out = (out, orig_v)

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
        attn_laser = False,
        unet_skips = True,
        use_flex_attn = False,
        qk_rmsnorm = True,
        use_value_residual = False
    ):
        super().__init__()

        self.use_flex_attn = use_flex_attn
        self.use_value_residual = use_value_residual

        self.dim = dim
        self.dim_head = dim_head

        self.to_time_cond = nn.Sequential(
            RandomFourierEmbed(dim),
            Linear(dim + 1, dim * 4),
            nn.SiLU()
        )

        # layers

        layers = ModuleList([])

        for ind in range(depth):
            is_first = ind == 0

            is_latter_half = ind >= (depth / 2)

            skip_proj = Linear(dim * 2, dim, bias = False) if is_latter_half and unet_skips else None

            attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout, use_flex_attn = use_flex_attn, learned_value_residual_mix = not is_first and use_value_residual, laser = attn_laser, qk_rmsnorm = qk_rmsnorm, **attn_kwargs)

            ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor, **ff_kwargs)

            attn = AdaptiveWrapper(attn, dim = dim, dim_cond = dim * 4)
            ff = AdaptiveWrapper(ff, dim = dim, dim_cond = dim * 4)

            attn_res = AttentionResidual(dim = dim)

            layers.append(ModuleList([skip_proj, attn, ff, attn_res]))

        self.layers = layers
        self.norm = RMSNorm(dim)

    @typecheck
    def forward(
        self,
        x,
        times: Scalar | Float['b'] | Float['b n'] | None = None,
        attn_mask: Bool['b i j'] | None = None,
        modality_positions: RawModalityPositions | Int['b m 3'] | None = None,
        is_any_modality: bool | Bool['b n'] | None = None,
        rotary_emb: Tensor | None = None,
        cache: Tensor | None = None,
        decode_length: int | None = None,
        modality_only = False,
        causal_mask = False,
        return_hiddens = False,
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
                    block_mask = create_block_mask(causal, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len, _compile = True, device = device)
                    attn_mask_kwargs.update(block_mask = block_mask)
                else:
                    attn_mask_kwargs.update(causal = True)

            if exists(modality_positions):
                assert not causal_mask

                if should_use_flex_attn:
                    transfusion_mask_fn = transfusion_attn_mask(modality_positions)
                    block_mask = create_block_mask(transfusion_mask_fn, B = batch, H = None, Q_LEN = seq_len, KV_LEN = seq_len, _compile = True, device = device)
                    attn_mask_kwargs.update(block_mask = block_mask)
                else:
                    attn_mask = naive_attn_mask(seq_len, modality_positions, device = device)
                    attn_mask_kwargs.update(attn_mask = attn_mask)

        # explicit mask is honored even when decoding with a cache - `sample_many` excludes the padded kv cache entries of other samples this way

        if exists(attn_mask):
            attn_mask_kwargs.update(attn_mask = attn_mask)

        if not exists(is_any_modality) and exists(modality_positions):
            is_any_modality = modality_positions_to_is_modality_mask(seq_len, modality_positions).any(dim = 1)
            is_any_modality = reduce(is_any_modality, 'b t n -> b n', 'any')

        # handle kv caching

        if is_decoding_with_cache:
            assert exists(decode_length)

            x = x[..., -decode_length:, :]

            if exists(cond):
                cond = cond[..., -decode_length:, :]

            if is_tensor(is_any_modality):
                is_any_modality = is_any_modality[..., -decode_length:]

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
        value_residual = None

        new_cache = []
        hiddens = [x]

        depth = len(self.layers)

        for ind, (skip_proj, attn, ff, attn_res) in enumerate(self.layers):
            layer = ind + 1

            # skip connection

            is_first_half = layer <= (depth // 2)
            is_later_half = not is_first_half

            if is_first_half:
                skips.append(x)

            if is_later_half and exists(skip_proj):
                skip = skips.pop()

                residual = x
                x = cat((x, skip), dim = -1)
                x = skip_proj(x) + residual

            # attention and feedforward

            (attn_out, attn_values), kv_cache = attn(
                x,
                rotary_emb = rotary_emb,
                cache = next(iter_cache, None),
                return_kv_cache = True,
                return_values = True,
                value_residual = value_residual,
                **attn_mask_kwargs,
                **adaptive_kwargs
            )

            value_residual = default(value_residual, attn_values) if self.use_value_residual else None

            new_cache.append(kv_cache)

            x = attn_out + x

            ff_out = ff(x, **adaptive_kwargs)

            x = ff_out + x

            hiddens.append(x)

            x = attn_res(hiddens)

        assert len(skips) == 0

        out = self.norm(x)

        if return_hiddens:
            hiddens.append(out)

        if not return_kv_cache and not return_hiddens:
            return out

        ret = (out,)

        if return_hiddens:
            ret = (*ret, hiddens)

        if return_kv_cache:
            ret = (*ret, stack(new_cache))

        return ret

# classes

@dataclass
class _SamplingState:
    # per-sample state for `sample_many` - each sample walks the same state machine as `sample_one`

    sample: ModalitySample # parts assembled so far - text tensors and (modality_type, modality) tuples
    curr_seq: Tensor # the text tensor currently being built (the last part of `sample`)
    last_token: Tensor | None # last sampled text token, embedded on the next text decoding step

    phase: str = 'text' # 'text', 'modality', or 'done'
    cache: Tensor | None = None # kv cache of everything decoded so far
    uncond_cache: Tensor | None = None # kv cache of the null-text history, for classifier free guidance
    tokens_seen: int = 0 # rotary position of the next token to decode - +1 per text token, +1 per modality instance
    num_past_modalities: int = 0 # number of previously decoded modalities, for the unconditional time conditioning
    curr_modality_id: int | None = None # modality type currently being decoded
    modality_shape: tuple[int, ...] | None = None # shape of the modality currently being decoded
    modality_length: int | None = None # number of latent tokens of the modality currently being decoded
    dim_latent: int | None = None # latent dimension of the modality currently being decoded
    num_tokens: int = 0 # total decoded tokens, for the max length budget


class Transfusion(Module):
    @beartype
    def __init__(
        self,
        *,
        num_text_tokens,
        transformer: dict | Transformer,
        model_output_clean = False, # https://arxiv.org/abs/2511.13720
        dim_latent: int | tuple[int, ...] | None = None,
        channel_first_latent: bool | tuple[bool, ...] = False,
        add_pos_emb: bool | tuple[bool, ...] = False,
        modality_encoder: Module | tuple[Module, ...] | None = None,
        modality_decoder: Module | tuple[Module, ...] | None = None,
        pre_post_transformer_enc_dec: tuple[Module, Module] | tuple[tuple[Module, Module], ...] | None = None,
        modality_default_shape: tuple[int, ...] | tuple[tuple[int, ...], ...] | None = None,
        fallback_to_default_shape_if_invalid = False,
        modality_num_dim: int | tuple[int, ...] | None = None,
        to_modality_shape_fn: Callable | tuple[Callable, ...] = default_to_modality_shape_fn,
        ignore_index = -1,
        flow_loss_weight = 1.,
        text_loss_weight = 1.,
        velocity_consistency_loss_weight = 0.1,
        reconstruction_loss_weight = 0.,
        modality_encoder_decoder_requires_batch_dim = True, # whether the modality encoder / decoder requires batch dimension, will auto assume it is needed
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        eps = 1e-2,
        prob_uncond = 0.1,
        modality_processing: str = DEFAULT_PROCESSING_STRATEGY # which strategy to use for processing a batch of interspersed text + modalities, see `PROCESSING_STRATEGIES` in `modality_processing.py`
    ):
        super().__init__()

        self.modality_processing = modality_processing
        get_processing_strategy(modality_processing) # validate early

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

        # number of modalities

        self.num_modalities = len(self.dim_latents)

        # whether the latents are accepted to be channel first or channel last
        # if channel first, will be rearrange(c ... -> ... c -> (...) c)

        self.channel_first_latent = cast_tuple(channel_first_latent, self.num_modalities)
        assert len(self.channel_first_latent) == self.num_modalities

        # functions for converting the sampled language model meta string back to modality shape of tuple[int, ...]

        self.to_modality_shape_fn = cast_tuple(to_modality_shape_fn, self.num_modalities)

        # default token lengths for respective modality
        # fallback if the language model does not come up with valid dimensions

        if not exists(modality_default_shape) or is_bearable(modality_default_shape, tuple[int, ...]):
            modality_default_shape = (modality_default_shape,) * self.num_modalities

        self.modality_default_shape = modality_default_shape

        assert len(self.modality_default_shape) == self.num_modalities

        self.fallback_to_default_shape_if_invalid = fallback_to_default_shape_if_invalid

        # default `modality_num_dim` to `len(modality_default_shape)` if latter is specified but former not

        modality_num_dim = default(modality_num_dim, tuple(len(shape) if exists(shape) else None for shape in self.modality_default_shape))

        # specifying the number of dimensions for the modality, which will be hard validated

        self.modality_num_dim = cast_tuple(modality_num_dim, self.num_modalities)

        assert len(self.modality_num_dim) == self.num_modalities

        assert all([not exists(ndim) or not exists(shape) or len(shape) == ndim for ndim, shape in zip(self.modality_num_dim, self.modality_default_shape)])

        # whether to add an extra axial positional embedding per modality

        self.add_pos_emb = cast_tuple(add_pos_emb, self.num_modalities)
        assert len(self.add_pos_emb) == self.num_modalities

        self.pos_emb_mlp = ModuleList([])

        for modality_add_pos_emb, modality_ndim in zip(self.add_pos_emb, self.modality_num_dim):

            if not modality_add_pos_emb:
                self.pos_emb_mlp.append(None)
                continue

            assert exists(modality_ndim), '`modality_num_dim` must be set if you wish to automatically inject axial positional embeddings'

            pos_generating_mlp = ContinuousAxialPositionalEmbedding(
                dim = dim,
                num_axial_dims = modality_ndim,
            )

            self.pos_emb_mlp.append(pos_generating_mlp)

        # modality encoders and decoders

        modality_encoder = cast_tuple(modality_encoder, 1 if exists(modality_encoder) else self.num_modalities)
        modality_decoder = cast_tuple(modality_decoder, 1 if exists(modality_decoder) else self.num_modalities)

        self.modality_encoder = ModuleList(modality_encoder)
        self.modality_decoder = ModuleList(modality_decoder)

        assert len(self.modality_encoder) == self.num_modalities
        assert len(self.modality_decoder) == self.num_modalities

        # auto handle batch dimension for modality encoder / decoder

        self.maybe_add_temp_batch_dim = add_temp_batch_dim if modality_encoder_decoder_requires_batch_dim else identity

        # store number of text tokens

        self.num_text_tokens = num_text_tokens

        # entire "sentence" start and end id, plus null text id for classifier free guidance

        num_text_special_ids = 3

        self.sos_id, self.eos_id, self.null_text_id = num_text_tokens, num_text_tokens + 1, num_text_tokens + 2

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

        # prepare pre-post transformer encoder / decoder, for the learnable unets as in paper

        if is_bearable(pre_post_transformer_enc_dec, tuple[Module, Module]):
            pre_post_transformer_enc_dec = (pre_post_transformer_enc_dec,)

        pre_post_transformer_enc_dec = cast_tuple(pre_post_transformer_enc_dec, self.num_modalities)
        assert len(pre_post_transformer_enc_dec) == self.num_modalities

        # latent to model and back
        # by default will be Linear, with or without rearranges depending on channel_first_latent setting
        # can also be overridden for the unet down/up as in the paper with `pre_post_transformer_enc_dec: tuple[Module, Module]`

        latent_to_model_projs = []
        model_to_latent_projs = []

        for (
            dim_latent,
            one_channel_first_latent,
            enc_dec,
         ) in zip(
            self.dim_latents,
            self.channel_first_latent,
            pre_post_transformer_enc_dec
        ):

            pre_attend_enc, post_attend_dec = default(enc_dec, (None, None))

            latent_to_model_proj = Linear(dim_latent, dim) if dim_latent != dim else nn.Identity()
            model_to_latent_proj = Linear(dim, dim_latent, bias = False)

            if one_channel_first_latent:
                latent_to_model_proj = nn.Sequential(Rearrange('b d ... -> b ... d'), latent_to_model_proj)
                model_to_latent_proj = nn.Sequential(model_to_latent_proj, Rearrange('b ... d -> b d ...'))

                if exists(pre_attend_enc):
                    pre_attend_enc = nn.Sequential(pre_attend_enc, Rearrange('b d ... -> b ... d'))

                if exists(post_attend_dec):
                    post_attend_dec = nn.Sequential(Rearrange('b ... d -> b d ...'), post_attend_dec)

            latent_to_model_projs.append(default(pre_attend_enc, latent_to_model_proj))
            model_to_latent_projs.append(default(post_attend_dec, model_to_latent_proj))

        self.latent_to_model_projs = ModuleList(latent_to_model_projs)
        self.model_to_latent_projs = ModuleList(model_to_latent_projs)

        # relative positions

        self.rotary_emb = RotaryEmbedding(transformer.dim_head)

        # embeddings and un-embeddings

        effective_num_text_tokens = num_text_tokens + num_text_special_ids + num_modality_special_ids + num_meta_tokens

        self.text_embed = nn.Embedding(effective_num_text_tokens, dim)

        self.to_text_logits = Linear(dim, effective_num_text_tokens, bias = False)

        text_only_mask = torch.arange(effective_num_text_tokens) < num_text_tokens
        self.register_buffer('text_only_logits_mask', text_only_mask, persistent = False)

        # loss related

        self.ignore_index = ignore_index
        self.flow_loss_weight = flow_loss_weight
        self.text_loss_weight = text_loss_weight

        # velocity consistency weight - only added if EMA model is passed in during training

        self.velocity_consistency_loss_weight = velocity_consistency_loss_weight

        # additional reconstruction loss, through the decoder

        self.has_recon_loss = reconstruction_loss_weight > 0.
        self.reconstruction_loss_weight = reconstruction_loss_weight

        # whether model is outputting clean

        self.model_output_clean = model_output_clean
        self.eps = eps

        # flow sampling related

        self.odeint_fn = partial(odeint, **odeint_kwargs)

        self.prob_uncond = prob_uncond

        # dummy loss

        self.register_buffer('zero', tensor(0.), persistent = False)

    @property
    def device(self):
        return next(self.parameters()).device

    @cache
    def get_modality_info(
        self,
        modality_type: int | None = None
    ) -> ModalityInfo:

        modality_type = default(modality_type, 0)

        modality_encoder = self.modality_encoder[modality_type]
        modality_decoder = self.modality_decoder[modality_type]
        latent_to_model = self.latent_to_model_projs[modality_type]
        model_to_latent = self.model_to_latent_projs[modality_type]

        add_pos_emb = self.add_pos_emb[modality_type]
        pos_emb_mlp = self.pos_emb_mlp[modality_type]
        modality_num_dim = self.modality_num_dim[modality_type]

        dim_latent = self.dim_latents[modality_type]

        default_shape = self.modality_default_shape[modality_type]

        som_id = self.som_ids[modality_type]
        eom_id = self.eom_ids[modality_type]

        to_shape_fn = self.to_modality_shape_fn[modality_type]

        channel_first_latent = self.channel_first_latent[modality_type]

        return ModalityInfo(
            encoder = modality_encoder,
            decoder = modality_decoder,
            latent_to_model = latent_to_model,
            model_to_latent = model_to_latent,
            add_pos_emb = add_pos_emb,
            pos_emb_mlp = pos_emb_mlp,
            num_dim = modality_num_dim,
            dim_latent = dim_latent,
            default_shape = default_shape,
            som_id = som_id,
            eom_id = eom_id,
            to_shape_fn = to_shape_fn,
            channel_first_latent = channel_first_latent,
            modality_type = modality_type
        )

    def get_all_modality_info(self) -> list[ModalityInfo]:
        return [self.get_modality_info(i) for i in range(self.num_modalities)]

    def get_modality_shape(
        self,
        modality: Float['...'],
        modality_type: int | None  = None
    ) -> tuple[int, ...]:

        mod = self.get_modality_info(modality_type)

        if mod.channel_first_latent:
            modality = rearrange(modality, 'c ... -> ... c')

        return tuple(modality.shape[:-1])

    def get_modality_shape_from_seq(
        self,
        seq: Int['n'],
        modality_id: int,
        fixed_modality_shape: tuple[int, ...] | None = None
    ) -> tuple[int, ...]:
        """ determine the shape of the modality to be decoded, either from the meta string tokens after the [mom] token, or fall back to the default shape """

        modality_shape = fixed_modality_shape

        maybe_meta_tensor = get_tokens_since_rightmost_id(seq, self.meta_id)

        mod = self.get_modality_info(modality_id)

        default_shape = mod.default_shape
        maybe_modality_num_dim = mod.num_dim
        meta_str_to_modality_shape = mod.to_shape_fn

        if maybe_meta_tensor.numel() > 0:
            meta_tensor = maybe_meta_tensor[:-1]
            meta_str = self.decode_chars(meta_tensor)

            if not meta_str.isdigit() or int(meta_str) <= 0:

                assert exists(default_shape), 'invalid modality meta information detected, please set `modality_default_shape` in order to properly fallback'
                modality_shape = default_shape
            else:
                modality_shape = meta_str_to_modality_shape(meta_str)

        modality_shape = default(modality_shape, default_shape)

        if self.fallback_to_default_shape_if_invalid:

            if exists(maybe_modality_num_dim) and len(modality_shape) != maybe_modality_num_dim:
                logger.warning(f'invalid modality shape {modality_shape} for modality {modality_id}. falling back to default shape {default_shape}')
                modality_shape = default_shape

        assert exists(modality_shape), f'language model did not produce a proper modality shape for modality type {modality_id} - please set a fallback shape with `modality_default_shape`'
        assert not exists(maybe_modality_num_dim) or maybe_modality_num_dim == len(modality_shape), f'expected modality type {modality_id} to have {maybe_modality_num_dim} dimensions but language model produced a shape of {modality_shape}'

        return modality_shape

    def parameters_without_encoder_decoder(self):
        return (
            set(self.parameters()) -
            set(self.modality_encoder.parameters()) -
            set(self.modality_decoder.parameters())
        )

    def muon_parameters(self):
        params = []

        for m in self.modules():
            if isinstance(m, Attention):
                params.extend([
                    *m.to_v.parameters(),
                    *m.to_out.parameters(),
                ])
            elif isinstance(m, FeedForward):
                params.extend([
                    m.net[0].weight,
                    m.net[-1].weight
                ])

        return params

    def create_dataloader(
        self,
        *args,
        **kwargs
    ):
        return create_dataloader(*args, **kwargs)

    def create_ema(
        self,
        beta = 0.99,
        *ema_kwargs
    ) -> EMA:

        ema = EMA(
            self,
            beta = beta,
            forward_method_names = (
                'sample',
                'sample_one',
                'sample_many',
                'generate_text_only',
                'generate_modality_only'
            )
        )

        return ema

    @torch.no_grad()
    @temp_eval
    @typecheck
    def sample_one(
        self,
        prompt: ModalitySample | Tensor | tuple[int, Float['...']] | None = None,
        max_length = 2048,
        text_temperature = 1.0,
        text_min_p = 0.1,
        cache_kv = False,
        fixed_modality_shape: tuple[int, ...] | None = None,
        init_modality_noise: Float['n d'] | None = None,
        modality_steps = 16,
        return_unprocessed_modalities = False,
        cfg_scale = 3.
    ) -> ModalitySample:

        device = self.device

        # handle edge case where there are no text tokens

        if self.num_text_tokens == 0:
            logger.warning(f'you have `num_text_tokens` set to 0, so `sample` will be forwarded to `generate_modality_only(batch_size: int, modality_type: int)` method')

            return self.generate_modality_only(batch_size = 1)

        # take care of prompt being a raw tensor, either text or raw modality (image, video, actions, latents, etc)

        if is_tensor(prompt) and prompt.is_floating_point(): # is modality with type 0 implicit
            prompt = (0, prompt)

        if is_int_tensor(prompt): # is text only prompt
            prompt = [prompt]

        elif isinstance(prompt, tuple):
            modality_type, modality = prompt

            mod = self.get_modality_info(modality_type)

            if exists(mod.encoder):
                with torch.no_grad():
                    mod.encoder.eval()
                    modality = self.maybe_add_temp_batch_dim(mod.encoder)(modality).detach()

            modality_shape_tuple = self.get_modality_shape(modality, modality_type)
            modality_shape_str = join([*map(str, modality_shape_tuple)], ',')
            modality_meta_info = self.char_tokenizer(modality_shape_str, device = device)

            prompt = [
                tensor([self.meta_id]),
                modality_meta_info,
                tensor([mod.som_id]),
                (modality_type, modality),
                tensor([mod.eom_id]),
            ]

        # sos

        init_text_seq = tensor([self.sos_id], device = device)

        # just take care of prompt being zero dimensions

        modality_sample = [init_text_seq, *default(prompt, [])]

        # take care of moving to device

        modality_sample = tree_map_tensor_to_device(modality_sample, device)
        modality_sample = tree_map_tensor(atleast_1d, modality_sample)

        modality_sample = concat_contiguous_text(modality_sample)

        *_, last_modality_sample = modality_sample

        curr_length = 0
        curr_modality_id = None
        modality_shape = None

        num_past_modalities = sum(not is_tensor(part) for part in modality_sample) # any modalities in the prompt count as previously decoded, for the time conditioning

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

            modality_shape = self.get_modality_shape_from_seq(seq, curr_modality_id, fixed_modality_shape)

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
                        decode_length = 1,
                        decoding_text_or_modality = 'text',
                        return_kv_cache = True
                    )

                    logits = logits[0][-1]

                    if text_is_greedy:
                        sampled = logits.argmax(dim = -1, keepdim = True)
                    else:
                        logits = logits / text_temperature
                        logits = min_p_filter(logits, min_p = text_min_p)

                        probs = logits.softmax(dim = -1)
                        sampled = torch.multinomial(probs, 1)

                    seq = torch.cat((seq, sampled), dim = -1)
                    modality_sample[-1] = seq

                    pbar.update(1)
                    curr_length += 1

                    if cache_kv:
                        cache = new_kv_cache

                    sampled_token_id = sampled.item()

                    if sampled_token_id == self.eos_id:
                        logger.info(f'detecting an end of string token [{self.eos_id}], terminating sampling early')
                        break

                    maybe_transition_to_modality_decoding(seq)

                else:
                    assert exists(curr_modality_id)
                    pbar.set_description(f'decoding modality [{curr_modality_id}]')

                    mod = self.get_modality_info(curr_modality_id)

                    modality_length = math.prod(modality_shape)

                    if exists(init_modality_noise):
                        noise = init_modality_noise[:modality_length, :mod.dim_latent]
                    else:
                        assert exists(modality_length)
                        noise = torch.randn((modality_length, mod.dim_latent), device = device)

                    assert noise.shape == (modality_length, mod.dim_latent)

                    noise = noise.reshape(*modality_shape, mod.dim_latent)

                    if mod.channel_first_latent:
                        noise = rearrange(noise, '... d -> d ...')

                    new_kv_cache = None

                    use_cfg = cfg_scale != 1.

                    if use_cfg:
                        # prepare unconditional kv cache for CFG

                        uncond_history = [
                            torch.full_like(item, self.null_text_id) if is_int_tensor(item) else item
                            for item in modality_sample
                        ]

                        with torch.no_grad():
                            _, uncond_cache = self.forward(
                                [uncond_history],
                                return_loss = False,
                                return_kv_cache = True,
                                return_embed = True,
                                decoding_text_or_modality = 'modality'
                            )

                    def ode_step_fn(step_times, denoised):
                        nonlocal new_kv_cache

                        # Conditional Input (Text + Image)
                        cond_input = [[*modality_sample, (curr_modality_id, denoised)]]

                        step_times = rearrange(step_times, ' -> 1 1') # batch size of 1
                        step_times = pad_left_at_dim(step_times, num_past_modalities, dim = -1, value = 1.) # past decoded modalities receive a time conditioning of 1.

                        (embeds_cond, get_pred_flows_cond), new_kv_cache = self.forward(
                            cond_input,
                            times = step_times,
                            return_embed = True,
                            cache = cache,
                            decode_length = modality_length,
                            return_kv_cache = True,
                            decoding_text_or_modality = 'modality'
                        )

                        parse_cond = get_pred_flows_cond[curr_modality_id][-1]
                        parsed_cond = parse_cond(embeds_cond, need_splice = not exists(cache))
                        cond_flow = add_temp_batch_dim(mod.model_to_latent)(parsed_cond)

                        if not use_cfg:
                            return cond_flow

                        uncond_input = [[*uncond_history, (curr_modality_id, denoised)]]

                        # unconditional forward

                        (embeds_uncond, get_pred_flows_uncond), _ = self.forward(
                            uncond_input,
                            times = step_times, # same time
                            return_embed = True,
                            cache = uncond_cache,
                            decode_length = modality_length,
                            return_kv_cache = True,
                            decoding_text_or_modality = 'modality'
                        )

                        parse_uncond = get_pred_flows_uncond[curr_modality_id][-1]
                        parsed_uncond = parse_uncond(embeds_uncond, need_splice = not exists(uncond_cache))
                        uncond_flow = add_temp_batch_dim(mod.model_to_latent)(parsed_uncond)

                        final_flow = uncond_flow + cfg_scale * (cond_flow - uncond_flow)

                        return final_flow

                    times = torch.linspace(0, 1, modality_steps, device = device)

                    trajectory = self.odeint_fn(ode_step_fn, noise, times)

                    # add the sampled modality tokens

                    sampled_modality = trajectory[-1]

                    modality_sample.append((curr_modality_id, sampled_modality))

                    # add the appropriate [eom]

                    eom_id = mod.eom_id
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

        logger.info(f'sampling stopped at length: {curr_length} / {max_length}')

        if return_unprocessed_modalities:
            return modality_sample

        # post process modality sample, decoding modality types if they have a decoder

        for mod in self.get_all_modality_info():
            decoder_fn = default(mod.decoder, nn.Identity())

            with torch.no_grad():
                decoder_fn.eval()
                modality_sample = apply_fn_modality_type(decoder_fn, modality_sample, modality_type = mod.modality_type)

        return modality_sample

    # `sample_one` is kept around as `sample` for backwards compatibility

    sample = sample_one

    @torch.no_grad()
    @temp_eval
    @typecheck
    def sample_many(
        self,
        prompts: list[ModalitySample | Tensor | tuple[int, Float['...']] | None] | None = None,
        max_length = 2048,
        text_temperature = 1.0,
        text_min_p = 0.1,
        fixed_modality_shape: tuple[int, ...] | None = None,
        init_modality_noise: Float['n d'] | None = None,
        modality_steps = 16,
        return_unprocessed_modalities = False,
        cfg_scale = 3.
    ) -> list[ModalitySample]:

        """
        batched version of `sample_one` - decodes a batch of interleaved text + modality samples in parallel.

        each sample independently walks the same state machine as `sample_one`, but all samples
        currently in the same phase share a single forward pass:

        - text phase - one kv-cached forward for all text decoding samples (one new token each),
          with per-sample attention masks excluding the padded kv cache entries of the others
        - modality phase - one joint `odeint` trajectory for all modality decoding samples (one
          forward pass per ode evaluation), each sample with its own shape, length and modality type

        the kv cache is always used, so `sample_many` follows the `sample_one(..., cache_kv = True)`
        code path exactly.

        a batch is a list of prompts, where each prompt takes the same form as the `prompt` of
        `sample_one` - a list of text tensors / (modality_type, modality) tuples, a raw tensor or
        tuple, or `None` for an empty prompt. a list of raw prompts is a batch of independent
        prompts, one sample per element.
        """

        device = self.device

        # `None` and raw prompts (tensors / tuples) become a batch of one
        # a list is kept as-is: a list of raw prompts is a batch of prompts, a list of lists is a batch of samples

        if not exists(prompts):
            prompts = [None]
        elif not isinstance(prompts, list):
            prompts = [prompts]

        # build the initial samples - [sos] + prompt, replicating `sample_one`'s prompt handling

        states = []
        sample_seq_lens = []

        for prompt in prompts:
            # take care of raw tensors, either text or raw modality

            if is_tensor(prompt) and prompt.is_floating_point():
                prompt = (0, prompt)

            if is_int_tensor(prompt):
                prompt = [prompt]

            elif isinstance(prompt, tuple):
                modality_type, modality = prompt

                mod = self.get_modality_info(modality_type)

                if exists(mod.encoder):
                    with torch.no_grad():
                        mod.encoder.eval()
                        modality = self.maybe_add_temp_batch_dim(mod.encoder)(modality).detach()

                modality_shape_tuple = self.get_modality_shape(modality, modality_type)
                modality_shape_str = join([*map(str, modality_shape_tuple)], ',')
                modality_meta_info = self.char_tokenizer(modality_shape_str, device = device)

                prompt = [
                    tensor([self.meta_id], device = device),
                    modality_meta_info,
                    tensor([mod.som_id], device = device),
                    (modality_type, modality),
                    tensor([mod.eom_id], device = device),
                ]

            elif not exists(prompt):
                prompt = []

            if isinstance(prompt, list):
                prompt = [part for part in prompt if exists(part)]

            sample = [tensor([self.sos_id], device = device), *prompt]

            sample = tree_map_tensor_to_device(sample, device)
            sample = tree_map_tensor(atleast_1d, sample)
            sample = concat_contiguous_text(sample)

            # count the tokens in the sample - and the rotary position collapse of each modality
            # instance (all tokens of a modality share a single position)

            seq_len = 0
            position_collapse = 0
            num_past_modalities = 0

            for part in sample:
                if is_tensor(part):
                    seq_len += part.numel()
                    continue

                modality_type, modality = part
                num_past_modalities += 1

                mod = self.get_modality_info(modality_type)
                axial_shape = modality.shape[1:] if mod.channel_first_latent else modality.shape[:-1]
                modality_len = math.prod(axial_shape)

                seq_len += modality_len
                position_collapse += modality_len - 1

            last_part = sample[-1]
            last_token = atleast_1d(last_part[-1]) if is_tensor(last_part) else None

            state = _SamplingState(
                sample = sample,
                curr_seq = last_part if is_tensor(last_part) else tensor([self.sos_id], device = device),
                last_token = last_token
            )

            # the rotary position of the next token to decode, following `sample_one`'s convention
            # (the modalities in the prompt count as previously decoded, for the time conditioning)

            state.tokens_seen = seq_len - position_collapse
            state.num_past_modalities = num_past_modalities

            states.append(state)
            sample_seq_lens.append(seq_len)

        # one batched forward over all the prompts - builds the kv caches and gives the logits for
        # sampling the first token of each sample (the same forward `sample_one` uses for its first
        # text decoding step, so the kv caches are laid out identically)

        logits, (init_cache_tensor, _) = self.forward(
            [state.sample for state in states],
            return_loss = False,
            decoding_text_or_modality = 'text',
            return_kv_cache = True
        )

        text_is_greedy = text_temperature == 0.

        def sample_text_token(logits):
            if text_is_greedy:
                return logits.argmax(dim = -1, keepdim = True)

            logits = logits / text_temperature
            logits = min_p_filter(logits, min_p = text_min_p)

            probs = logits.softmax(dim = -1)
            return torch.multinomial(probs, 1)

        def maybe_transition_to_modality(state, sampled_token_id):
            if sampled_token_id not in self.som_ids:
                return False

            curr_modality_id = self.som_ids.index(sampled_token_id)
            modality_shape = self.get_modality_shape_from_seq(state.curr_seq, curr_modality_id, fixed_modality_shape)

            mod = self.get_modality_info(curr_modality_id)

            state.curr_modality_id = curr_modality_id
            state.modality_shape = modality_shape
            state.modality_length = math.prod(modality_shape)
            state.dim_latent = mod.dim_latent
            state.phase = 'modality'

            return True

        # determine if to transition from start (a prompt ending in [som])

        for state in states:
            if exists(state.last_token) and state.last_token.item() in self.som_ids:
                maybe_transition_to_modality(state, state.last_token.item())

        # slice out the kv cache per sample, sample the first token

        for ind, (state, seq_len) in enumerate(zip(states, sample_seq_lens)):
            state.cache = init_cache_tensor[:, :, ind:ind + 1, :, :seq_len]

            if state.phase == 'text':
                sampled = sample_text_token(logits[ind, seq_len - 1])

                state.curr_seq = cat((state.curr_seq, sampled))
                state.sample[-1] = state.curr_seq
                state.last_token = sampled
                # note: `tokens_seen` is not advanced here - the first sampled token only gets its
                # kv cache (and its rotary position) on the first text decoding step
                state.num_tokens += 1

                sampled_token_id = sampled.item()

                if sampled_token_id == self.eos_id:
                    logger.info(f'detecting an end of string token [{self.eos_id}], terminating sampling early')
                    state.phase = 'done'
                    continue

                maybe_transition_to_modality(state, sampled_token_id)

            if state.num_tokens > max_length:
                state.phase = 'done'

        # one batched text decoding step - all samples in the text phase share a single forward pass

        def pad_kv_caches(caches, max_seq_len):
            # right-pad each sample's kv cache to the group's longest, so they can be batched
            # (the padded entries are excluded from attention via the per-sample masks)
            # kv cache layout is (layers, key/value, batch, heads, seq, dim_head)

            _, _, batch, _, _, _ = caches[0].shape
            assert batch == 1, 'each sample cache holds a batch of one'

            batch_dim = 2 # index of the batch dim in the kv cache layout

            padded_caches = []

            for cache in caches:
                cache_seq_len = cache.shape[-2] # the seq dim of the kv cache layout

                if cache_seq_len == max_seq_len:
                    padded_caches.append(cache)
                else:
                    padded_caches.append(pad_right_at_dim(cache, max_seq_len - cache_seq_len, dim = -2))

            return cat(padded_caches, dim = batch_dim)

        def step_text(group):
            group_batch = len(group)

            # per-sample cache lengths, and the group's longest

            cache_seq_lens = [state.cache.shape[-2] for state in group]
            max_seq_len = max(cache_seq_lens)

            cache_tensor = pad_kv_caches([state.cache for state in group], max_seq_len)

            # the new token for each sample - the last sampled one

            x = cat([self.text_embed(state.last_token[None]) for state in group], dim = 0)

            # each sample sits at its own absolute rotary position

            positions = tensor([state.tokens_seen for state in group], device = device, dtype = torch.long)
            rotary_emb = rearrange(self.rotary_emb(positions), 'b d -> b 1 1 d')

            # each sample only attends to its own real kv cache entries - its own prefix and its own new token

            mask = torch.zeros((group_batch, 1, max_seq_len + 1), dtype = torch.bool, device = device)
            mask[:, 0, max_seq_len] = True

            for ind, (seq_len, state) in enumerate(zip(cache_seq_lens, group)):
                mask[ind, 0, :seq_len] = True

            embed, new_cache = self.transformer(
                x,
                cache = cache_tensor,
                decode_length = 1,
                rotary_emb = rotary_emb,
                attn_mask = mask,
                return_kv_cache = True
            )

            logits = self.to_text_logits(embed)[:, -1]

            sampled = sample_text_token(logits)

            # update each sample and transition out of the text phase as needed
            # (new cache layout is (layers, key/value, batch, heads, seq, dim_head) - each sample's
            # newly decoded token sits at the padded `max_seq_len` position of the batched cache)

            for ind, (seq_len, state) in enumerate(zip(cache_seq_lens, group)):
                token = sampled[ind]

                new_kv = new_cache[:, :, ind:ind + 1]
                state.cache = cat((new_kv[..., :seq_len, :], new_kv[..., max_seq_len:max_seq_len + 1, :]), dim = -2)

                state.curr_seq = cat((state.curr_seq, token))
                state.sample[-1] = state.curr_seq
                state.last_token = token
                state.tokens_seen += 1
                state.num_tokens += 1

                pbar.update(1)

                sampled_token_id = token.item()

                if sampled_token_id == self.eos_id:
                    logger.info(f'detecting an end of string token [{self.eos_id}], terminating sampling early')
                    state.phase = 'done'
                    continue

                if state.num_tokens > max_length:
                    state.phase = 'done'
                    continue

                if maybe_transition_to_modality(state, sampled_token_id):
                    pbar.set_description(f'decoding modality [{state.curr_modality_id}]')

        # one batched modality decoding step - all samples in the modality phase share a single
        # joint odeint trajectory, with one forward pass per ode evaluation

        def step_modality(group):
            group_batch = len(group)

            mods = [self.get_modality_info(state.curr_modality_id) for state in group]

            # per-sample modalities and caches, and the group's maxima - the joint odeint
            # trajectory spans the max modality length and latent dim across the group

            modality_lengths = [state.modality_length for state in group]
            dim_latents = [state.dim_latent for state in group]
            cache_seq_lens = [state.cache.shape[-2] for state in group]

            max_modality_length = max(modality_lengths)
            max_dim_latent = max(dim_latents)
            max_seq_len = max(cache_seq_lens)

            # initial noise for the joint odeint state - each sample at its own (length, dim) slice

            noise = torch.zeros((group_batch, max_modality_length, max_dim_latent), device = device)

            for ind, (modality_length, dim_latent, state) in enumerate(zip(modality_lengths, dim_latents, group)):
                if exists(init_modality_noise):
                    one_noise = init_modality_noise[:modality_length, :dim_latent]
                else:
                    one_noise = torch.randn((modality_length, dim_latent), device = device)

                assert one_noise.shape == (modality_length, dim_latent)

                noise[ind, :modality_length, :dim_latent] = one_noise

            # prepare the unconditional kv caches for classifier free guidance
            # (the past decoded modalities are conditioned at a time of 1.)

            use_cfg = cfg_scale != 1.

            if use_cfg:
                for state in group:
                    uncond_history = [
                        torch.full_like(item, self.null_text_id) if is_int_tensor(item) else item
                        for item in state.sample
                    ]

                    with torch.no_grad():
                        _, (uncond_cache, _) = self.forward(
                            [uncond_history],
                            times = torch.ones((1, state.num_past_modalities), device = device),
                            return_loss = False,
                            return_kv_cache = True,
                            return_embed = True,
                            decoding_text_or_modality = 'modality'
                        )

                    state.uncond_cache = uncond_cache

            # batch structure - constant across all ode evaluations

            positions = tensor([state.tokens_seen for state in group], device = device, dtype = torch.long)
            rotary_emb = repeat(self.rotary_emb(positions), 'b d -> b 1 l d', l = max_modality_length)

            # each sample attends to its own prefix and its own modality block only

            mask = torch.zeros((group_batch, max_modality_length, max_seq_len + max_modality_length), dtype = torch.bool, device = device)

            for ind, (seq_len, modality_length, state) in enumerate(zip(cache_seq_lens, modality_lengths, group)):
                mask[ind, :, :seq_len] = True
                mask[ind, :, max_seq_len:max_seq_len + modality_length] = True

            cache_tensor = pad_kv_caches([state.cache for state in group], max_seq_len)

            if use_cfg:
                uncond_seq_lens = [state.uncond_cache.shape[-2] for state in group]
                max_uncond_seq_len = max(uncond_seq_lens)

                uncond_mask = torch.zeros((group_batch, max_modality_length, max_uncond_seq_len + max_modality_length), dtype = torch.bool, device = device)

                for ind, (uncond_seq_len, modality_length, state) in enumerate(zip(uncond_seq_lens, modality_lengths, group)):
                    uncond_mask[ind, :, :uncond_seq_len] = True
                    uncond_mask[ind, :, max_uncond_seq_len:max_uncond_seq_len + modality_length] = True

                uncond_cache_tensor = pad_kv_caches([state.uncond_cache for state in group], max_uncond_seq_len)

            def project_to_model(state, mod, latent):
                # latent space (l, d) -> model space (l, dim), handling channel first

                if mod.channel_first_latent:
                    latent = rearrange(latent.reshape(*state.modality_shape, state.dim_latent), '... d -> d ...')
                else:
                    latent = latent.reshape(*state.modality_shape, state.dim_latent)

                return mod.latent_to_model(latent[None])[0].reshape(state.modality_length, self.dim)

            def parse_flow(ind, embed, state, mod, step_times, denoised):
                # the flow is computed in model space (prediction minus the projected noised
                # modality), then projected back into latent space - matching `sample_one` exactly
                # (projecting the prediction first would not be equivalent, as the latent and model
                # projections do not compose to the identity)

                out = embed[ind, :state.modality_length].reshape(*state.modality_shape, self.dim)
                noised = project_to_model(state, mod, denoised[ind, :state.modality_length, :state.dim_latent]).reshape(*state.modality_shape, self.dim)

                if self.model_output_clean:
                    out = (out - noised) / (1. - step_times).clamp_min(self.eps)

                out = mod.model_to_latent(out.reshape(*state.modality_shape, self.dim)[None])[0]

                if mod.channel_first_latent:
                    out = rearrange(out, 'd ... -> (...) d')
                else:
                    out = out.reshape(state.modality_length, state.dim_latent)

                return out

            new_kv_cache = None

            def ode_step_fn(step_times, denoised):
                nonlocal new_kv_cache

                # project each sample's denoised modality into the model space

                x = torch.zeros((group_batch, max_modality_length, self.dim), device = device)

                for ind, (modality_length, dim_latent, state, mod) in enumerate(zip(modality_lengths, dim_latents, group, mods)):
                    x[ind, :modality_length] = project_to_model(state, mod, denoised[ind, :modality_length, :dim_latent])

                # all tokens of the current modality instance share the step time conditioning

                times_cond = torch.full((group_batch, max_modality_length), step_times, device = device)

                embed, new_kv_cache = self.transformer(
                    x,
                    times = times_cond,
                    cache = cache_tensor,
                    decode_length = max_modality_length,
                    rotary_emb = rotary_emb,
                    attn_mask = mask,
                    is_any_modality = True,
                    return_kv_cache = True
                )

                # parse out the flow for each sample

                cond_flows = torch.zeros((group_batch, max_modality_length, max_dim_latent), device = device)

                for ind, (modality_length, dim_latent, state, mod) in enumerate(zip(modality_lengths, dim_latents, group, mods)):
                    cond_flows[ind, :modality_length, :dim_latent] = parse_flow(ind, embed, state, mod, step_times, denoised)

                if not use_cfg:
                    return cond_flows

                # unconditional forward, with the same batch structure

                embed_uncond, _ = self.transformer(
                    x,
                    times = times_cond,
                    cache = uncond_cache_tensor,
                    decode_length = max_modality_length,
                    rotary_emb = rotary_emb,
                    attn_mask = uncond_mask,
                    is_any_modality = True,
                    return_kv_cache = True
                )

                uncond_flows = torch.zeros((group_batch, max_modality_length, max_dim_latent), device = device)

                for ind, (modality_length, dim_latent, state, mod) in enumerate(zip(modality_lengths, dim_latents, group, mods)):
                    uncond_flows[ind, :modality_length, :dim_latent] = parse_flow(ind, embed_uncond, state, mod, step_times, denoised)

                return uncond_flows + cfg_scale * (cond_flows - uncond_flows)

            times = torch.linspace(0, 1, modality_steps, device = device)

            trajectory = self.odeint_fn(ode_step_fn, noise, times)

            final = trajectory[-1]

            # commit the final kv cache, append the sampled modality and the [eom] token

            for ind, (seq_len, modality_length, state, mod) in enumerate(zip(cache_seq_lens, modality_lengths, group, mods)):
                new_kv = new_kv_cache[:, :, ind:ind + 1]
                state.cache = cat((new_kv[..., :seq_len, :], new_kv[..., max_seq_len:max_seq_len + modality_length, :]), dim = -2)

                sampled_modality = final[ind, :modality_length, :state.dim_latent]

                if mod.channel_first_latent:
                    sampled_modality = rearrange(sampled_modality.reshape(*state.modality_shape, state.dim_latent), '... d -> d ...')
                else:
                    sampled_modality = sampled_modality.reshape(*state.modality_shape, state.dim_latent)

                state.sample.append((state.curr_modality_id, sampled_modality))

                eom_id = mod.eom_id
                state.curr_seq = tensor([eom_id], device = device)
                state.sample.append(state.curr_seq)
                state.last_token = tensor([eom_id], device = device)
                state.tokens_seen += 1
                state.num_tokens += modality_length
                state.num_past_modalities += 1
                state.phase = 'text'

                if state.num_tokens > max_length:
                    state.phase = 'done'

                pbar.update(modality_length)

        # phase-grouped scheduling - all samples in the same phase share a forward pass

        with tqdm(total = len(states) * max_length) as pbar:
            while not all(state.phase == 'done' for state in states):
                text_group = [state for state in states if state.phase == 'text']

                while text_group:
                    pbar.set_description('decoding text')
                    step_text(text_group)
                    text_group = [state for state in text_group if state.phase == 'text']

                modality_group = [state for state in states if state.phase == 'modality']

                while modality_group:
                    step_modality(modality_group)
                    modality_group = [state for state in modality_group if state.phase == 'modality']

        for state in states:
            logger.info(f'sampling stopped at length: {state.num_tokens} / {max_length}')

        samples = [state.sample for state in states]

        if return_unprocessed_modalities:
            return samples

        # post process modality samples, decoding modality types if they have a decoder

        for mod in self.get_all_modality_info():
            decoder_fn = default(mod.decoder, nn.Identity())

            with torch.no_grad():
                decoder_fn.eval()
                samples = apply_fn_modality_type(decoder_fn, samples, modality_type = mod.modality_type)

        return samples

    @typecheck
    def forward_text(
        self,
        text: Int['b n'],
        return_loss = True,
        return_embed = False,
        cache: Tensor | None = None,
        return_hiddens = False,
        return_kv_cache = False
    ) -> (
        Scalar |
        Float['b n d'] |
        tuple[Float['b n d'], list[Float['...']]]
    ):

        device = self.device
        text = text.to(device)

        if return_loss:
            text, labels = text[:, :-1], text[:, 1:]

        # embed text

        text = text.masked_fill(text == -1, 0)
        tokens = self.text_embed(text)

        # handle cache and tokens_seen

        raw_cache, tokens_seen = default(cache, (None, 0))

        # rotary

        seq_len = tokens.shape[-2]

        pos = torch.arange(tokens_seen, tokens_seen + seq_len, device = device)

        rotary_emb = self.rotary_emb(pos)

        # attention

        transformer_out = self.transformer(
            tokens,
            rotary_emb = rotary_emb,
            causal_mask = True,
            cache = raw_cache,
            decode_length = seq_len if exists(raw_cache) else None,
            return_kv_cache = return_kv_cache,
            return_hiddens = True
        )

        embed, hiddens, *maybe_kv_cache = transformer_out
        kv_cache = (maybe_kv_cache[0], tokens_seen + seq_len) if return_kv_cache else None

        # text unembedding

        logits = self.to_text_logits(embed)

        if not return_loss:
            ret = (logits,)

            if return_kv_cache:
                ret = (*ret, kv_cache)

            if return_hiddens:
                ret = (*ret, hiddens)

            return ret[0] if len(ret) == 1 else ret

        logits = logits.masked_fill(~self.text_only_logits_mask, max_neg_value(logits))

        loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels,
            ignore_index = self.ignore_index
        )

        if not return_hiddens:
            return loss

        return loss, hiddens

    @torch.no_grad()
    @temp_eval
    @typecheck
    def generate_text_only(
        self,
        prompt: Int['b n'],
        seq_len: int,
        temperature = 1.0,
        min_p = 0.1,
        cache_kv = True
    ) -> Int['b no']:

        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        curr_out = out
        cache = None

        for _ in tqdm(range(sample_num_times)):
            logits, next_cache = self.forward_text(curr_out, return_loss = False, return_kv_cache = True, cache = cache)

            if cache_kv:
                cache = next_cache

            logits = logits[:, -1]

            if temperature == 0.:
                sample = logits.argmax(dim = -1, keepdim = True)
            else:
                logits = logits / temperature
                logits = min_p_filter(logits, min_p = min_p)
                logits.masked_fill_(~self.text_only_logits_mask, max_neg_value(logits))
                sample = gumbel_sample(logits, dim = -1)

            out = cat((out, sample), dim = -1)

            if cache_kv:
                curr_out = sample
            else:
                curr_out = out

        return out[..., prompt_seq_len:]

    @typecheck
    def forward_modality(
        self,
        modalities: Float['b ...'],
        times: Float['b'] | None = None,
        modality_type: int | None = None,
        encode_modality: bool = True,
        velocity_consistency_ema_model: Transfusion | None = None,
        velocity_consistency_delta_time = 1e-5,
        return_loss = True,
        return_loss_breakdown = False
    ) -> Scalar | Float['b ...']:
        requires_velocity_consistency = exists(velocity_consistency_ema_model)

        modalities = modalities.to(self.device)
        orig_modalities = modalities

        if self.num_modalities > 1:
            assert exists(modality_type), '`modality_type` must be explicitly passed in on forward when training on greater than 1 modality'

        modality_type = default(modality_type, 0)

        mod = self.get_modality_info(modality_type)

        # maybe modality encode

        if encode_modality and exists(mod.encoder):
            with torch.no_grad():
                mod.encoder.eval()
                modalities = mod.encoder(modalities).detach()

        # shapes and device

        tokens = modalities

        batch, device = tokens.shape[0], tokens.device

        # times

        if not exists(times):
            times = torch.rand((batch,), device = device)

        if return_loss:

            if requires_velocity_consistency:
                orig_times = times.clone()
                times = times * (1. - velocity_consistency_delta_time) # make sure times are max of 1. - small delta, for velocity consistency

            padded_times = append_dims(times, tokens.ndim - 1)

            noise = torch.randn_like(tokens)

            noised_tokens = padded_times * tokens + (1. - padded_times) * noise

            flow = tokens - noise

        else:
            noised_tokens = tokens

        # save the noised and times

        model_output_to_flow = identity

        if self.model_output_clean:
            model_output_to_flow = get_model_output_to_flow_fn(noised_tokens, times, self.eps)

        # from latent to model tokens

        noised_tokens = mod.latent_to_model(noised_tokens)

        # axial positions

        if mod.add_pos_emb:
            assert exists(mod.num_dim), f'modality_num_dim must be set for modality {modality_type} if further injecting axial positional embedding'

            _, *axial_dims, _ = noised_tokens.shape

            assert len(axial_dims) == mod.num_dim, f'received modalities of ndim {len(axial_dims)} but expected {modality_num_dim}'

        # maybe transform

        noised_tokens, inverse_pack_axial_dims = pack_with_inverse(noised_tokens, 'b * d')

        # maybe add axial pos emb

        if mod.add_pos_emb:
            axial_pos_emb = mod.pos_emb_mlp(tensor(axial_dims), flatten = True)
            noised_tokens = noised_tokens + axial_pos_emb

        # attention

        embed = self.transformer(
            noised_tokens,
            times = times,
            modality_only = True,
        )

        embed = inverse_pack_axial_dims(embed)

        model_output = mod.model_to_latent(embed)

        pred_flow = model_output_to_flow(model_output)

        if not return_loss:
            return pred_flow

        # flow loss

        flow_loss = F.mse_loss(pred_flow, flow)

        # maybe velocity consistency loss

        velocity_loss = self.zero

        if requires_velocity_consistency:

            with torch.no_grad():
                flow_with_delta_time = velocity_consistency_ema_model.forward_modality(
                    modalities = modalities,
                    modality_type = modality_type,
                    times = orig_times + velocity_consistency_delta_time,
                    encode_modality = False, # modality already encoded
                    return_loss = False
                )

            velocity_loss = F.mse_loss(flow, flow_with_delta_time)

        # maybe recon loss

        recon_loss = self.zero

        if self.has_recon_loss:
            assert encode_modality

            recon = noise + pred_flow * (1. - padded_times)

            if exists(mod.decoder):
                with torch.no_grad():
                    mod.decoder.eval()
                    recon = mod.decoder(recon)

            recon_loss = F.mse_loss(
                recon,
                orig_modalities
            )

        # total loss

        total_loss = (
            flow_loss +
            velocity_loss * self.velocity_consistency_loss_weight +
            recon_loss * self.reconstruction_loss_weight
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (flow_loss, velocity_loss, recon_loss)

    @torch.no_grad()
    @temp_eval
    @typecheck
    def generate_modality_only(
        self,
        batch_size: int = 1,
        modality_type: int | None = None,
        fixed_modality_shape: tuple[int, ...] | None = None,
        modality_steps = 16,
        return_unprocessed_modalities = False
    ) -> Tensor:

        device = self.device

        if self.num_modalities > 1:
            assert exists(modality_type), '`modality_type` must be explicitly passed in on forward when training on greater than 1 modality'

        mod = self.get_modality_info(modality_type)

        modality_shape = default(fixed_modality_shape, mod.default_shape)

        assert exists(modality_shape)

        noise = torch.randn((batch_size, *modality_shape, mod.dim_latent), device = device)

        if mod.channel_first_latent:
            noise = rearrange(noise, 'b ... d -> b d ...')

        def ode_step_fn(step_times, denoised):

            step_times = repeat(step_times, ' -> b', b = batch_size)

            flow = self.forward_modality(
                denoised,
                times = step_times,
                modality_type = modality_type,
                encode_modality = False,
                return_loss = False
            )

            return flow

        times = torch.linspace(0., 1., modality_steps, device = device)
        trajectory = self.odeint_fn(ode_step_fn, noise, times)

        # add the sampled modality tokens

        sampled_modality = trajectory[-1]

        # decode

        if exists(mod.decoder):
            mod.decoder.eval()
            sampled_modality = mod.decoder(sampled_modality)

        return sampled_modality

    @typecheck
    def forward(
        self,
        modalities: (
            list[ModalitySample] |
            Int['b n'] |
            Float['b ...']
        ),
        times: Float['b m'] | None = None,
        num_modalities_to_times_fn: Callable[[Int['b']], Float['b m']] | None = None, # allows a researcher to customize the times (noise level) based on the modality lengths in a given sample
        modality_type: int | None = None,
        cache: Tensor | None = None,
        decode_length: int | None = None,
        decoding_text_or_modality: Literal['text', 'modality'] | None = None,
        velocity_consistency_ema_model: Transfusion | EMA | None = None,
        velocity_consistency_delta_time = 1e-3,
        return_only_pred_flows = False,
        return_loss = True,
        return_breakdown = False,
        return_embed = False,
        return_hiddens = False,
        return_kv_cache = False,
        return_times = False,
        prob_uncond: float | None = None
    ) -> (
        Float['b _ l'] |
        tuple[Float['b _ d'], GetPredFlows] |
        tuple[tuple[Float['b _ _'], GetPredFlows], Tensor] |
        Scalar |
        tuple[Scalar, LossBreakdown] |
        list[Float['b _ _']] |
        tuple[Float['b _ l'], Tensor] |
        list[list[Tensor]]
    ):

        is_decoding = exists(decoding_text_or_modality)

        is_text_only = is_int_tensor(modalities)
        is_modality_only = is_tensor(modalities) and modalities.is_floating_point()

        # handle ema model being passed in for velocity consistency loss

        if isinstance(velocity_consistency_ema_model, EMA):
            assert isinstance(velocity_consistency_ema_model.ema_model, Transfusion)
            velocity_consistency_ema_model = velocity_consistency_ema_model.ema_model

        need_velocity_matching = not is_decoding and exists(velocity_consistency_ema_model)

        # return loss

        return_loss &= not (return_embed or is_decoding)

        if is_text_only:

            forward_text_kwargs = dict(
                return_loss = return_loss,
                return_embed = return_embed,
                cache = cache,
                return_hiddens = return_hiddens,
                return_kv_cache = return_kv_cache
            )

            return self.forward_text(modalities, **forward_text_kwargs)

        if is_modality_only:
            assert return_loss

            forward_modality_kwargs = dict(
                modality_type = modality_type,
                velocity_consistency_ema_model = velocity_consistency_ema_model
            )

            return self.forward_modality(modalities, **forward_modality_kwargs)

        batch = len(modalities)
        device = self.device
        tensor_ = partial(tensor, device = device)

        # save a copy for ema model for velocity matching
        velocity_modalities = modalities

        if need_velocity_matching:
            if isinstance(velocity_modalities, list):
                velocity_modalities = [modality.copy() for modality in velocity_modalities]

        # defensively shallow copy out inner lists to prevent in-place mutation of user input
        if isinstance(modalities, list):
            modalities = [list(batch) if isinstance(batch, list) else batch for batch in modalities]

        # add "sentence" start and end tokens when training

        if return_loss or need_velocity_matching:
            if isinstance(modalities, list):
                for i, modality in enumerate(modalities):
                    modalities[i] = [
                        tensor_([self.sos_id]),
                        *modality,
                        tensor_([self.eos_id])
                    ]

        # classifier free guidance

        prob_uncond = default(prob_uncond, self.prob_uncond)

        if self.training and prob_uncond > 0 and isinstance(modalities, list):
            rand_mask = torch.rand(len(modalities), device = self.device) < prob_uncond

            def to_uncond_sample(sample):
                uncond = []

                for item in sample:
                    if is_int_tensor(item):
                        item = torch.full_like(item, self.null_text_id)

                    uncond.append(item)

                return uncond

            modalities = [to_uncond_sample(sample) if is_uncond else sample for sample, is_uncond in zip(modalities, rand_mask)]

        # need axial pos emb

        need_axial_pos_emb = any(self.add_pos_emb)

        # standardize modalities to be tuple - type 0 modality is implicit if not given
        # also store modality lengths for determining noising times

        num_modalities = []

        for batch_modalities in modalities:
            batch_num_modalities = 0

            for ind, modality in enumerate(batch_modalities):

                if is_tensor(modality) and modality.is_floating_point():
                    modality = (0, modality)

                if not isinstance(modality, tuple):
                    continue

                modality_type, modality_tensor = modality
                batch_modalities[ind] = modality
                batch_num_modalities += 1

            num_modalities.append(batch_num_modalities)

        num_modalities = tensor_(num_modalities)

        # determine the times

        if not exists(times):
            if is_empty(num_modalities) or num_modalities.amax().item() == 0:
                times = torch.empty((batch, 0), device = device, dtype = torch.float)
            else:
                num_modalities_to_times_fn = default(num_modalities_to_times_fn, default_modality_length_to_time_fn)

                if exists(num_modalities_to_times_fn):
                    times = num_modalities_to_times_fn(num_modalities)

        # if needs velocity matching, make sure times are in the range of 0 - (1. - <velocity consistency delta time>)

        if need_velocity_matching:
            orig_times = times.clone()
            times = times * (1. - velocity_consistency_delta_time)

        # process list of text and modalities interspersed with one another

        modality_positions = []
        modality_tokens = []
        modality_pos_emb = []

        text = []

        modalities = tree_map_tensor_to_device(modalities, device)

        # for all modalities, batch process same shaped modalities of the same type

        if not is_decoding:
            for mod in self.get_all_modality_info():
                encode_fn = default(mod.encoder, nn.Identity())

                with torch.no_grad():
                    encode_fn.eval()
                    modalities = apply_fn_modality_type(encode_fn, modalities, modality_type = mod.modality_type)

        # process the whole batch of interspersed text + modalities in batched ops
        # (strategy chosen via `modality_processing`, see `PROCESSING_STRATEGIES`)

        process_modality_batch_fn = get_processing_strategy(self.modality_processing)

        (
            text,
            modality_tokens,
            modality_positions,
            modality_pos_emb,
            flows,
            get_pred_flows,
            get_recon_losses,
            pos_emb_max_axial_dims,
            total_tokens
        ) = process_modality_batch_fn(
            modalities,
            times,
            self,
            need_axial_pos_emb = need_axial_pos_emb,
            return_loss = return_loss,
            return_embed = return_embed
        )

        # handle modality positional embedding - lazy evaluation from factorized positional embedding of maximum axial dims

        if need_axial_pos_emb:
            modality_pos_emb = evaluate_modality_pos_emb(modality_pos_emb, pos_emb_max_axial_dims, self, self.dim, device)

        # handle training mode and removal of last token

        if return_loss:
            modality_tokens = modality_tokens[:, :-1]

            if need_axial_pos_emb:
                modality_pos_emb = modality_pos_emb[:, :-1]

        # if returning loss, split text for next token prediction

        if return_loss:
            text, text_labels = text[:, :-1], text[:, 1:]

        # derive is_modality mask for flow on the right tokens + flow loss

        batch, seq_len, device = *text.shape, text.device

        assert len(modality_positions) == batch

        if isinstance(modality_positions, list):
            modality_positions = modality_positions_to_tensor(modality_positions, device = device)

        if modality_positions.shape[-1] == 2: # Int['b m 2'] -> Int['b m 3'] if type is not given (one modality)
            modality_positions = pad_left_at_dim(modality_positions, 1, dim = -1)

        # for now use dummy padding modality position info if empty (all zeros)

        if modality_positions.numel() == 0:
            modality_positions = pad_right_at_dim(modality_positions, 1, dim = -2)

        # sort the modalities tensor and sanitize, readying for noising of modalities

        modality_positions, sorted_indices = order_modality_positions_by_seq_offset(modality_positions)

        is_modalities = modality_positions_to_is_modality_mask(seq_len, modality_positions, num_modalities = self.num_modalities, device = device)

        is_any_modality = reduce(is_modalities, 'b t m n -> b n', 'any')

        # embed text

        text = text.masked_fill(text == -1, 0)

        text_tokens = self.text_embed(text)

        # maybe add the axial positional embedding

        if need_axial_pos_emb:
            modality_tokens = modality_tokens + modality_pos_emb

        # intersperse the modalities with the text for the joint transformer + flow system

        tokens = einx.where('b n, b n d, b n d', is_any_modality, modality_tokens, text_tokens)

        # take care of cache

        raw_cache, tokens_seen = default(cache, (None, 0))
        is_any_modality_when_decoding = None

        if exists(raw_cache):
            assert exists(decode_length), '`decode_length` must be passed in on forward for modality sampling. think of a cleaner way on some future date'
            assert exists(decoding_text_or_modality)

            if decoding_text_or_modality == 'text':
                decode_length = 1

            is_any_modality_when_decoding = decoding_text_or_modality == 'modality'
            modality_positions = None

        # derive rotary positions

        if exists(raw_cache):
            if is_any_modality_when_decoding:
                # all tokens in a modality instance share the exact same rotary position

                rotary_positions = torch.full((decode_length,), tokens_seen, device = device, dtype = torch.long)
                next_tokens_seen = tokens_seen + 1
            else:
                rotary_positions = torch.arange(tokens_seen, tokens_seen + decode_length, device = device)
                next_tokens_seen = tokens_seen + decode_length
        else:
            rotary_positions = derive_rotary_positions_from_modality_positions(seq_len, modality_positions) + tokens_seen
            next_tokens_seen = (rotary_positions[..., -1].max() + 1).item()

        rotary_emb = self.rotary_emb(rotary_positions)

        if rotary_emb.ndim == 3:
            rotary_emb = rearrange(rotary_emb, 'b n d -> b 1 n d')

        # times

        times_per_token = einsum(is_modalities.float(), times, 'b t m n, b m -> b t n')

        times_cond = reduce(times_per_token, 'b t n -> b n', 'sum')

        # attention

        embed, hiddens, *maybe_kv_cache = self.transformer(
            tokens,
            times = times_cond,
            rotary_emb = rotary_emb,
            modality_positions = modality_positions,
            is_any_modality = is_any_modality_when_decoding,
            cache = raw_cache,
            decode_length = decode_length,
            return_hiddens = True,
            return_kv_cache = return_kv_cache
        )

        kv_cache = (maybe_kv_cache[0], next_tokens_seen) if return_kv_cache else None

        # helper for appending auxiliary returns

        def maybe_pack_aux(out):
            ret = (out,)

            if return_kv_cache:
                ret = (*ret, kv_cache)

            if return_hiddens:
                ret = (*ret, hiddens)

            if return_times:
                ret = (*ret, times)

            if len(ret) == 1:
                return ret[0]

            return ret

        # early return for embedding for decoding modality

        if return_embed:
            return maybe_pack_aux((embed, get_pred_flows))

        # text unembedding

        text_logits = self.to_text_logits(embed)

        if not return_loss:
            return maybe_pack_aux(text_logits)

        # flow loss

        pred_flows = []
        recon_losses = []

        for modality_id in range(self.num_modalities):
            mod = self.get_modality_info(modality_id)

            modality_get_pred_flows = get_pred_flows[modality_id]
            modality_get_recon_losses = get_recon_losses[modality_id]

            modality_pred_flows = []
            modality_recon_losses = []

            for get_pred_flow, get_recon_loss in zip(modality_get_pred_flows, modality_get_recon_losses):

                pred_flow = get_pred_flow(embed)
                pred_flow = add_temp_batch_dim(mod.model_to_latent)(pred_flow)
                modality_pred_flows.append(pred_flow)

                if not self.has_recon_loss:
                    continue

                modality_recon_losses.append(get_recon_loss(pred_flow))

            pred_flows.append(modality_pred_flows)
            recon_losses.append(modality_recon_losses)

        # early return for velocity consistency ema model

        if return_only_pred_flows:
            return pred_flows

        # text autoregressive loss

        text_labels = text_labels.masked_fill(is_any_modality, self.ignore_index)

        # ignore "Image -> Null" mappings.
        text_labels = text_labels.masked_fill(text_labels == self.null_text_id, self.ignore_index)

        text_loss = F.cross_entropy(
            rearrange(text_logits, 'b n l -> b l n'),
            text_labels,
            ignore_index = self.ignore_index
        )

        text_loss_weight = (text_labels != self.ignore_index).sum() / total_tokens

        # calculate flow losses

        flow_losses = []

        modality_loss_weights = []

        for modality_id, (pred_flow, is_one_modality) in enumerate(zip(pred_flows, is_modalities.unbind(dim = 1))):
            mod = self.get_modality_info(modality_id)

            is_one_modality = reduce(is_one_modality, 'b m n -> b n', 'any')
            modality_loss_weight = is_one_modality.sum() / total_tokens

            modality_loss_weights.append(modality_loss_weight)

            # modality type not present in this batch - nothing to compute a flow loss on

            if not pred_flow:
                continue

            modality_flows = flows[modality_id]

            pack_pattern = 'd *' if mod.channel_first_latent else '* d'

            modality_pred_flow, _ = pack(pred_flow, pack_pattern)
            modality_flows, _ = pack(modality_flows, pack_pattern)

            flow_loss = F.mse_loss(
                modality_pred_flow,
                modality_flows
            )

            flow_losses.append(flow_loss)

        modality_loss_weights = stack(modality_loss_weights)

        # only the token positions that are not modalities have autoregressive loss

        total_loss = (
            text_loss * text_loss_weight * self.text_loss_weight +
            (stack(flow_losses) * modality_loss_weights).sum() * self.flow_loss_weight
        )

        # whether to handle velocity consistency
        # for straightening the flow, from consistency flow matching paper https://arxiv.org/abs/2407.02398

        velocity_match_losses = None

        if need_velocity_matching:

            with torch.no_grad():
                velocity_consistency_ema_model.eval()

                ema_pred_flows = velocity_consistency_ema_model(
                    velocity_modalities,
                    times = orig_times + velocity_consistency_delta_time,
                    return_only_pred_flows = True
                )

            velocity_match_losses = []

            for modality_id, (ema_pred_flow, pred_flow) in enumerate(zip(ema_pred_flows, pred_flows)):

                if not pred_flow:
                    continue

                mod = self.get_modality_info(modality_id)

                pack_pattern = 'd *' if mod.channel_first_latent else '* d'
                pred_flow, _ = pack(pred_flow, pack_pattern)
                ema_pred_flow, _ = pack(ema_pred_flow, pack_pattern)

                velocity_match_loss = F.mse_loss(
                    pred_flow,
                    ema_pred_flow
                )

                velocity_match_losses.append(velocity_match_loss)

            total_loss = (
                total_loss +
                (stack(velocity_match_losses) * modality_loss_weights).sum() * self.velocity_consistency_loss_weight
            )

        # maybe reconstruction loss

        if self.has_recon_loss:
            averaged_recon_losses = [
                sum(modality_recon_loss) / len(modality_recon_loss) if len(modality_recon_loss) > 0 else self.zero
                for modality_recon_loss in recon_losses
            ]

            total_loss = (
                total_loss +
                (stack(averaged_recon_losses) * modality_loss_weights).sum() * self.reconstruction_loss_weight
            )

        # return total loss and maybe breakdown

        if not return_breakdown and not return_hiddens and not return_times:
            return total_loss

        ret = (total_loss,)

        if return_breakdown:
            breakdown = LossBreakdown(total_loss, text_loss, flow_losses, velocity_match_losses, recon_losses)
            ret = (*ret, breakdown)

        if return_hiddens:
            ret = (*ret, hiddens)

        if return_times:
            ret = (*ret, times)

        return ret

# Self-Masked Representation Training
# following the similar formula as in 'Self-Flow' from Chefer et al. at Black Forest Labs
# https://bfl.ai/research/self-flow

def default_rep_loss_fn(pred, target):
    cos_sim = F.cosine_similarity(pred, target, dim = -1)
    return 1. - cos_sim.mean()

class SelfMaskedRepTraining(Module):
    def __init__(
        self,
        net: Transfusion,
        ema_beta = 0.999,
        rep_loss_weight = 0.1,
        student_layer = -3,
        teacher_layer = -1,
        loss_fn = default_rep_loss_fn,
        use_asymmetric_dropout = True,
        student_dropout_rate = 0.1,
        teacher_dropout_rate = 0.
    ):
        super().__init__()
        assert not use_asymmetric_dropout or student_dropout_rate > teacher_dropout_rate, 'student must have greater dropout rate than teacher to ensure teacher has a better view'

        self.student = net
        self.teacher = net.create_ema(beta = ema_beta)

        self.rep_loss_weight = rep_loss_weight
        self.has_ssl_loss = rep_loss_weight > 0

        self.use_asymmetric_dropout = use_asymmetric_dropout
        self.student_dropout_rate = student_dropout_rate
        self.teacher_dropout_rate = teacher_dropout_rate

        self.student_layer = student_layer
        self.teacher_layer = teacher_layer
        self.loss_fn = loss_fn

        # prediction head logic

        dim = net.dim

        self.student_predict_head = nn.Sequential(
            RMSNorm(dim),
            FeedForward(dim)
        )

        self.register_buffer('zero', tensor(0.))

    def parameters(self):
        return chain(
            self.student.parameters(),
            self.student_predict_head.parameters()
        )

    def update_teacher(self):
        self.teacher.update()

    def forward(
        self,
        *args,
        **kwargs
    ):
        # student pass with masked inputs (via asymmetric dropout)

        if self.use_asymmetric_dropout:
            set_dropout_(self.student, self.student_dropout_rate)

        student_loss, student_hiddens, student_times = self.student(
            *args,
            return_loss = True,
            return_hiddens = True,
            return_times = True,
            **kwargs
        )

        if not self.has_ssl_loss:
            return student_loss, (student_loss, self.zero)

        # extract student representation at layer l

        student_rep = student_hiddens[self.student_layer]

        # teacher pass with unmasked (cleaner) inputs

        if self.use_asymmetric_dropout:
            set_dropout_(self.teacher.ema_model, self.teacher_dropout_rate)

        with torch.no_grad():
            _, teacher_hiddens = self.teacher.ema_model(
                *args,
                times = student_times,
                return_loss = True,
                return_hiddens = True,
                **kwargs
            )

            teacher_rep = teacher_hiddens[self.teacher_layer]

        # squeeze out the extra stream dimension if it exists

        if student_rep.ndim == 4:
            student_rep = rearrange(student_rep, 'b n 1 d -> b n d')

        if teacher_rep.ndim == 4:
            teacher_rep = rearrange(teacher_rep, 'b n 1 d -> b n d')

        # predict teacher representation

        student_pred = self.student_predict_head(student_rep)

        self_flow_loss = self.loss_fn(student_pred, teacher_rep)

        # losses

        total_loss = student_loss + self_flow_loss * self.rep_loss_weight

        return total_loss, (student_loss, self_flow_loss)
