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

from functools import partial
from typing import NamedTuple

import torch
from torch import nn, Tensor, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear
from torch.nn.utils.rnn import pad_sequence

import einx
from einops import rearrange, repeat, reduce, einsum, pack
from einops.layers.torch import Rearrange

from transfusion_pytorch.tensor_typing import Float, Int, Bool

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

pad_sequence = partial(pad_sequence, batch_first = True)

# maybe flex attention

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
except ImportError:
    flex_attention = None

# constants

RawModalityPositions = list[list[tuple[int, int]]]

class LossBreakdown(NamedTuple):
    total: Float['']
    text: Float['']
    diffusion: list[Float['']]

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def first(it):
    return it[0]

def divisible_by(num, den):
    return (num % den) == 0

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

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
    return modalities

# sanitizing modalities tensor, making sure it is ordered

def order_modality_positions_by_seq_offset(
    modalities: Int['b m 3']
) -> tuple[Int['b m 3'], Int['b m']]:

    type, offsets, lengths = modalities.unbind(dim = -1)

    no_modality_mask = lengths <= 0 # there may be uneven number of modalities per batch sample
    offsets_to_sort = offsets.masked_fill(no_modality_mask, 1e10)
    _, sorted_indices = offsets_to_sort.sort(dim = -1)

    # sort by ascending offset and do a final mask of both offset and length to 0

    modalities = einx.get_at('b [mi] ..., b mo -> b mo ...', modalities, sorted_indices)
    modalities = einx.where('b m, b m ..., -> b m ...', ~no_modality_mask, modalities, 0.)

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
        self.layerscale = nn.Parameter(torch.zeros(dim))

        # modalities will get the adaptive layernorm + ada-ln zero

        self.to_film = Linear(dim_cond, dim * 2)
        self.to_ada_ln_zero = Linear(dim_cond, dim)

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

        text_out = out * (self.layerscale + 1.)
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
        attn_mask: Tensor | None = None,
        rotary_emb: Tensor | None = None,
        block_mask = None,
    ):
        assert not (exists(block_mask) and exists(attn_mask))

        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        # rotary embeddings

        if exists(rotary_emb):
            q, k = tuple(apply_rotary_emb(rotary_emb, t) for t in (q, k))

        # whether to use flex attention or not

        if self.use_flex_attn:

            flex_attn_kwargs = dict(block_mask = block_mask)

            if self.softcap_value > 0.:
                flex_attn_kwargs.update(score_mod = softcap_score_mod(self.softcap_value))

            out = flex_attention(q, k, v, **flex_attn_kwargs)

        else:
            q = q * self.scale
            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            sim = softclamp(sim, self.softcap_value)

            if exists(attn_mask):
                mask_value = -torch.finfo(sim.dtype).max
                sim = einx.where('b i j, b h i j, -> b h i j', attn_mask, sim, mask_value)

            attn = sim.softmax(dim = -1)

            attn = self.dropout(attn)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # combine heads and out

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
        ff_kwargs: dict = dict(),
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

        for _ in range(depth):
            attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout, use_flex_attn = use_flex_attn, **attn_kwargs)

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
        modality_positions: RawModalityPositions | Int['b n 2'] | None = None,
        is_any_modality: Bool['b n'] | None = None,
        rotary_emb: Tensor | None = None
    ):
        seq_len, device = x.shape[-2], x.device
        assert exists(attn_mask) ^ exists(modality_positions)

        # handle time

        if times.ndim == 0:
            times = repeat(times, ' -> b', b = batch)

        cond = self.to_time_cond(times)

        # create the specialized mask needed for autoregressive text + bidirectional diffusion attention

        attn_mask_kwargs = dict()

        if exists(modality_positions):
            if self.use_flex_attn:
                transfusion_mask_fn = transfusion_attn_mask(modality_positions)
                block_mask = create_block_mask(transfusion_mask_fn, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len, device = device)
                attn_mask_kwargs.update(block_mask = block_mask)
            else:
                attn_mask = naive_attn_mask(seq_len, modality_positions, device = device)
                attn_mask_kwargs.update(attn_mask = attn_mask)

        if not exists(is_any_modality):
            assert exists(modality_positions)
            is_any_modality = modality_positions_to_is_modality_mask(seq_len, modality_positions).any(dim = 1)
            is_any_modality = reduce(is_any_modality, 'b t n -> b n', 'any')

        adaptive_kwargs = dict(cond = cond, is_any_modality = is_any_modality)

        # transformer layers as usual, using mask from above

        for attn, ff in self.layers:
            x = attn(x, rotary_emb = rotary_emb, **attn_mask_kwargs, **adaptive_kwargs) + x
            x = ff(x, **adaptive_kwargs) + x

        return self.norm(x)

# classes

class Transfusion(Module):
    def __init__(
        self,
        *,
        num_text_tokens,
        transformer: dict | Transformer,
        dim_latent: int | tuple[int, ...] | None = None,
        modality_token_transform: tuple[str | callable, ...] | None = None,
        ignore_index = -1,
        diffusion_loss_weight = 1.,
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

        # modality transforms

        modality_token_transform = [default(transform, identity) for transform in modality_token_transform]
        self.modality_token_transform = [Rearrange(maybe_einops_eq) if isinstance(maybe_einops_eq, str) else maybe_einops_eq for maybe_einops_eq in modality_token_transform]

        # number of modalities

        self.num_modalities = len(self.dim_latents)

        self.latent_to_model_projs = ModuleList([Linear(dim_latent, dim) if dim_latent != dim else nn.Identity() for dim_latent in self.dim_latents])

        # relative positions

        self.rotary_emb = RotaryEmbedding(transformer.dim_head)

        # embeddings and un-embeddings

        self.text_embed = nn.Embedding(num_text_tokens, dim)

        self.to_text_logits = Linear(dim, num_text_tokens, bias = False)

        self.model_to_latent_preds = ModuleList([Linear(dim, dim_latent, bias = False) for dim_latent in self.dim_latents])

        # loss related

        self.ignore_index = ignore_index
        self.diffusion_loss_weight = diffusion_loss_weight

    def forward(
        self,
        text: Int['b n'],
        modality_tokens: list[list[Float['_ _']]] | list[Float['b n _']] | Float['b n _'],
        modality_positions: RawModalityPositions | Int['b m 2'] | Int['b m 3'],
        times: Float['b m'] | None = None,
        return_loss = True
    ) -> (
        Float['b n l'] |
        tuple[Float[''], LossBreakdown]
    ):

        # if returning loss, split text for next token prediction

        if return_loss:
            text, text_labels = text[:, :-1], text[:, 1:]

        # derive is_modality mask for diffusion on the right tokens + diffusion loss

        batch, seq_len, device = *text.shape, text.device

        assert len(modality_positions) == batch

        if isinstance(modality_positions, list):
            modality_positions = modality_positions_to_tensor(modality_positions, device = device)

        if modality_positions.shape[-1] == 2: # Int['b m 2'] -> Int['b m 3'] if type is not given (one modality)
            modality_positions = F.pad(modality_positions, (1, 0), value = 0)

        # embed the list of modality tokens into a sequence of Float['b n d'] at right offsets and lengths as dictated by modalities info tensor

        if torch.is_tensor(modality_tokens):
            modality_tokens = [modality_tokens]

        # transform the modality tokens from the vae encoder output into (batch, seq, feature) shape, if needed

        transformed_modality_tokens = []

        for batch_modality_tokens, batch_modality_position in zip(modality_tokens, modality_positions):
            batch_transformed = []

            for one_tokens, one_position in zip(batch_modality_tokens, batch_modality_position):
                modality_type, _, _ = one_position
                post_encode_transform = self.modality_token_transform[modality_type]
                transformed = post_encode_transform(one_tokens)
                batch_transformed.append(transformed)

            transformed_modality_tokens.append(batch_transformed)

        modality_tokens = transformed_modality_tokens

        # embed the modality tokens into one Tensor if not given as one

        if isinstance(modality_tokens, list) and isinstance(first(modality_tokens), list): # detect list[list[tensor]]
            modality_tokens = [embed_modality_tokens(seq_len, dim_latent, modality_tokens, modality_positions, modality_id) for modality_id, dim_latent in enumerate(self.dim_latents)]

        # sort the modalities tensor and sanitize, readying for noising of modalities

        modality_positions, sorted_indices = order_modality_positions_by_seq_offset(modality_positions)

        num_modalities = modality_positions.shape[-2]

        is_modalities = modality_positions_to_is_modality_mask(seq_len, modality_positions, num_modalities = self.num_modalities, device = device)

        is_any_modality = reduce(is_modalities, 'b t m n -> b n', 'any')

        # embed text

        text_tokens = self.text_embed(text)

        # noise the modality tokens

        if not exists(times):
            times = torch.rand((batch, num_modalities), device = device)

        if return_loss:
            noised_modality_tokens = []
            flows = []

            times = einsum(is_modalities.float(), times, 'b t m n, b m -> b t n')

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

        # intersperse the modalities with the text for the joint transformer + diffusion system

        tokens = einx.where('b n, b n d, b n d', is_any_modality, modality_tokens, text_tokens)

        # derive rotary positions

        rotary_positions = derive_rotary_positions_from_modality_positions(seq_len, modality_positions)

        rotary_emb = self.rotary_emb(rotary_positions)
        rotary_emb = rearrange(rotary_emb, 'b n d -> b 1 n d')

        # attention

        embed = self.transformer(
            tokens,
            times = reduce(times, 'b t n -> b n', 'sum'),
            rotary_emb = rotary_emb,
            modality_positions = modality_positions
        )

        # text unembedding

        text_logits = self.to_text_logits(embed)

        if not return_loss:
            return text_logits

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

        # diffusion loss

        pred_flows = [fn(embed) for fn in self.model_to_latent_preds]

        diffusion_losses = []
        modality_loss_weights = []

        for flow, pred_flow, is_one_modality in zip(flows, pred_flows, is_modalities.unbind(dim = 1)):

            diffusion_loss = F.mse_loss(
                pred_flow,
                flow,
                reduction = 'none'
            )

            is_one_modality = reduce(is_one_modality, 'b m n -> b n', 'any')

            diffusion_loss = diffusion_loss[is_one_modality].mean()

            modality_loss_weight = is_one_modality.sum() / total_tokens

            modality_loss_weights.append(modality_loss_weight)

            diffusion_losses.append(diffusion_loss)

        # only the token positions that are not modalities have autoregressive loss

        total_loss = (
            text_loss * text_loss_weight +
            (torch.stack(diffusion_losses) * torch.stack(modality_loss_weights)).sum() * self.diffusion_loss_weight
        )

        return total_loss, LossBreakdown(total_loss, text_loss, diffusion_losses)
