from __future__ import annotations

"""
modality processing strategies for `Transfusion`

processing turns a batch of interspersed text + modality samples into one joint sequence,
with the bookkeeping the training / sampling loop needs (positions, flow and loss closures,
positional embeddings).

strategies live in the `PROCESSING_STRATEGIES` registry below, pickable via
`Transfusion(..., modality_processing = ...)`. `'auto'` (the default) measures the candidate
strategies on the actual batch and dispatches to the fastest.

  naive    per-instance loop, the reference baseline
  grouped  one batched noise + noising + projection per same (type, shape) group
  flat     one noise + noising + projection for the whole modality type, any shapes
  hybrid   same-shaped groups batched, singletons of a type via the flat path
  auto     router that times the candidates on the batch and picks the fastest

to add a strategy: write `process_modality_batch_<name>(...)` with the same signature, register
it in `PROCESSING_STRATEGIES`, and it will be picked up by `benchmark_processing.py` and the
equivalence checks in the test suite automatically.
"""

import math
import time
import statistics

from collections import defaultdict
from functools import partial
from typing import Callable, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, tensor, is_tensor, cat, stack

from einops import rearrange

from loguru import logger

from torch_einops_utils import (
    pack_with_inverse,
    pad_at_dim,
    pad_sequence
)

# tensor typing (mirrors transfusion.py, kept local to avoid a circular import)

import jaxtyping

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]

Float = TorchTyping(jaxtyping.Float)
Int   = TorchTyping(jaxtyping.Int)

# types

ModalitySample = list[Int[''] | Int['_'] | Float['...'] | tuple[int, Float['...']]]

GetPredFlows = dict[int, list[Callable[[Tensor], Tensor]]]

# small helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def join(arr, delimiter = ''):
    return delimiter.join(arr)

def is_int_tensor(t):
    return is_tensor(t) and t.dtype in (torch.int, torch.long)

def append_dims(t, ndims):
    return t.reshape(*t.shape, *((1,) * ndims))

def add_temp_batch_dim(fn: Callable):
    def inner(t: Tensor, *args, **kwargs) -> Tensor:
        t = rearrange(t, '... -> 1 ...')
        out = fn(t, *args, **kwargs)
        out = rearrange(out, '1 ... -> ...')
        return out
    return inner

# decorator for model output to flow

def get_model_output_to_flow_fn(
    noised: Tensor,
    times: Tensor,
    eps = 5e-2,
    return_decorator = False
):
    if times.ndim == 0:
        times = rearrange(times, '-> 1')

    def to_flow(out):
        nonlocal noised
        noised = noised.reshape_as(out)
        padded_times = append_dims(times, out.ndim - 1)

        flow = (out - noised) / (1. - padded_times).clamp_min(eps)
        return flow

    if not return_decorator:
        return to_flow

    def decorator(fn):
        def inner(embed, *args, **kwargs):
            out = fn(embed, *args, **kwargs)
            return to_flow(out)
        return inner

    return decorator

class ModalityRecord(NamedTuple):
    batch_index: int
    modality_type: int
    tensor: Tensor
    time: Tensor
    scatter_offset: int
    length: int
    axial_shape: tuple[int, ...]

class ProcessedModalityBatch(NamedTuple):
    text: Int['b n']
    modality_tokens: Float['b n d']
    modality_positions: list[list[tuple[int, int, int]]]
    modality_pos_emb: list | None
    flows: dict[int, list[Tensor]]
    get_pred_flows: GetPredFlows
    get_recon_losses: dict[int, list[Callable[[Tensor], Tensor]]]
    pos_emb_max_axial_dims: dict[int, list[Tensor]]
    total_tokens: int | None

def validate_modality(modality_tensor: Tensor, modality_type: int, model) -> None:
    # check the modality sample against the modality info for that type

    assert 0 <= modality_type < model.num_modalities, f'received a modality index that is out of range. only {model.num_modalities} modalities specified'

    mod = model.get_modality_info(modality_type)
    channel_dim = 0 if mod.channel_first_latent else -1

    assert mod.dim_latent == modality_tensor.shape[channel_dim], f'mismatch for modality latent dimension - expected {mod.dim_latent} but received {modality_tensor.shape[-1]} - modality shape is {tuple(modality_tensor.shape)}, perhaps you need to set `channel_first_latent` to the correct value'
    assert mod.num_dim == (len(modality_tensor.shape) - 1), f'mismatch for modality number of dimensions - expected {mod.num_dim} but received {len(modality_tensor.shape) - 1} {modality_tensor.shape}'

def model_to_pred_flow(batch_index, start_index, modality_length, unpack_fn):
    # for parsing out the predicted flow from the flattened sequence of tokens coming out of the transformer

    def inner(embed: Float['b n d'], need_splice = True) -> Float['...']:
        embed = embed[batch_index]

        if need_splice:
            if embed.shape[0] < (start_index + modality_length):
                embed = embed[-modality_length:]
            else:
                embed = embed[start_index:(start_index + modality_length)]

        embed = unpack_fn(embed)
        return embed

    return inner

def get_recon_loss(noise, times, modality):
    # for going from predicted flow -> reconstruction

    def inner(pred_flow):
        recon_modality = noise + pred_flow * (1. - times)
        return F.mse_loss(modality, recon_modality)

    return inner

def get_recon_loss_lazy(noise, noised, times, shape, start, end, slice_):
    # like `get_recon_loss`, but slices the noise / noised out of the flat per-type tensors
    # only when the loss is actually evaluated (reconstruction loss is off by default)

    def inner(pred_flow):
        noise_instance = slice_(noise, start, end).reshape(shape)
        noised_instance = slice_(noised, start, end).reshape(shape)
        recon_modality = noise_instance + pred_flow * (1. - times)
        return F.mse_loss(noised_instance, recon_modality)

    return inner

def group_records_by_shape(records) -> dict[tuple[int, ...], list[ModalityRecord]]:
    shape_groups = defaultdict(list)

    for record in records:
        shape_groups[record.axial_shape].append(record)

    return shape_groups

def scan_batch_for_structure(
    modalities: list[ModalitySample],
    times,
    model,
    *,
    need_axial_pos_emb: bool,
    return_embed: bool
):
    # shared pass 1 - walk each sample for structure only, no gpu allocations in the hot path
    # constant meta tensors ([meta], [som], [eom]) cached per modality type, shape meta strings tokenized once per unique string

    device = model.device

    tensor_ = partial(tensor, device = device)

    meta_tensors = {
        modality_type: (
            tensor_([model.meta_id]),
            tensor_([model.get_modality_info(modality_type).som_id]),
            tensor_([model.get_modality_info(modality_type).eom_id])
        )
        for modality_type in range(model.num_modalities)
    }

    meta_token_cache = {}

    text_chunks = [] # per sample, list of (offset, int tensor) to be scattered into the text buffer
    modality_positions = []
    modality_pos_emb = []
    pos_emb_max_axial_dims: dict[int, list[Tensor]] = defaultdict(list)

    modality_records = []
    total_lens = []

    for batch_index, batch_modalities in enumerate(modalities):

        offset = 0
        modality_index = 0
        sample_text_chunks = []
        sample_modality_positions = []
        sample_modality_pos_emb = []

        for modality in batch_modalities:
            is_text = not isinstance(modality, tuple)

            if is_text:
                modality_tensor = modality
            else:
                modality_type, modality_tensor, *_ = modality
                validate_modality(modality_tensor, modality_type, model)

            # auto ward against scalars (lone start end tokens)

            if is_int_tensor(modality_tensor) and modality_tensor.ndim == 0:
                modality_tensor = rearrange(modality_tensor, '-> 1')

            # handle text

            if is_text:
                assert modality_tensor.ndim == 1 and is_int_tensor(modality_tensor)
                text_length = modality_tensor.shape[0]

                sample_text_chunks.append((offset, modality_tensor))
                offset += text_length

                if need_axial_pos_emb:
                    sample_modality_pos_emb.append(('zeros', text_length))

                continue

            # otherwise handle a modality
            # each modality instance gets its own noise level, indexed by its position in the sample

            mod = model.get_modality_info(modality_type)

            modality_time = times[batch_index, modality_index]
            modality_index += 1

            axial_shape = modality_tensor.shape[1:] if mod.channel_first_latent else modality_tensor.shape[:-1]
            modality_length = math.prod(axial_shape)

            precede_modality_tokens = succeed_modality_tokens = 0

            if not return_embed:
                # add the [meta] [shape] [som] ... [eom] tokens

                modality_shape_str = join([*map(str, axial_shape)], ',')
                modality_meta_info = meta_token_cache.get(modality_shape_str)

                if not exists(modality_meta_info):
                    modality_meta_info = model.char_tokenizer(modality_shape_str, device = device)
                    meta_token_cache[modality_shape_str] = modality_meta_info

                precede_modality_tokens = len(modality_meta_info) + 2
                succeed_modality_tokens = 1

                meta_tensor, som_tensor, eom_tensor = meta_tensors[modality_type]

                sample_text_chunks.extend((
                    (offset, meta_tensor),
                    (offset + 1, modality_meta_info),
                    (offset + precede_modality_tokens - 1, som_tensor),
                    (offset + precede_modality_tokens + modality_length, eom_tensor)
                ))

            scatter_offset = offset + precede_modality_tokens

            sample_modality_positions.append((modality_type, scatter_offset, modality_length))

            # handle axial positional embedding

            if need_axial_pos_emb:

                if exists(mod.pos_emb_mlp):
                    pos_emb_max_axial_dims[modality_type].append(tensor(axial_shape))
                    sample_modality_pos_emb.append((modality_type, axial_shape, (precede_modality_tokens, succeed_modality_tokens)))

                else:
                    sample_modality_pos_emb.append(('zeros', precede_modality_tokens + modality_length + succeed_modality_tokens))

            offset += modality_length + precede_modality_tokens + succeed_modality_tokens

            modality_records.append(ModalityRecord(batch_index, modality_type, modality_tensor, modality_time, scatter_offset, modality_length, axial_shape))

        total_lens.append(offset)
        text_chunks.append(sample_text_chunks)
        modality_positions.append(sample_modality_positions)

        if need_axial_pos_emb:
            modality_pos_emb.append(sample_modality_pos_emb)

    return modality_records, text_chunks, modality_positions, modality_pos_emb, pos_emb_max_axial_dims, total_lens

def process_modality_batch_naive(
    modalities: list[ModalitySample],
    times: Float['b m'],
    model,
    *,
    need_axial_pos_emb: bool,
    return_loss: bool,
    return_embed: bool
) -> ProcessedModalityBatch:

    # reference implementation, mirroring the original per-instance loop in `Transfusion.forward`
    # note: the original (incorrectly) read only the first column of `times` for all modalities in a sample

    device = model.device
    dim = model.dim

    tensor_ = partial(tensor, device = device)

    modality_positions = []
    modality_tokens = []
    modality_pos_emb = []

    text = []

    flows = defaultdict(list)

    get_pred_flows: GetPredFlows = defaultdict(list)

    get_recon_losses = defaultdict(list)

    pos_emb_max_axial_dims: dict[int, list[Tensor]] = defaultdict(list)

    for batch_index, batch_modalities in enumerate(modalities):

        modality_index = 0
        batch_modality_positions = []
        batch_modality_tokens = []
        batch_modality_pos_emb = []

        batch_text = []

        offset = 0

        for modality in batch_modalities:

            # if non-text modality detected and not given as a tuple
            # cast to (int, Tensor) where int is defaulted to type 0 (convenience for one modality)

            is_text = not isinstance(modality, tuple)

            if is_text:
                modality_tensor = modality
            else:
                modality_type, modality_tensor, *_ = modality
                mod = model.get_modality_info(modality_type)
                validate_modality(modality_tensor, modality_type, model)

            # auto ward against scalars (lone start end tokens)

            if is_int_tensor(modality_tensor) and modality_tensor.ndim == 0:
                modality_tensor = rearrange(modality_tensor, '-> 1')

            # handle text

            if is_text:
                assert modality_tensor.ndim == 1 and is_int_tensor(modality_tensor)
                text_length = modality_tensor.shape[0]

                batch_text.append(modality_tensor)
                zeros = torch.zeros(text_length, dim, device = device)

                batch_modality_tokens.append(zeros)

                offset += text_length

                if need_axial_pos_emb:
                    batch_modality_pos_emb.append(zeros)

                continue

            # otherwise handle a modality

            modality_time = times[batch_index, modality_index]

            # noise

            if return_loss:
                noise = torch.randn_like(modality_tensor)

                noised_modality = modality_tensor * modality_time + noise * (1. - modality_time)

                # the flow is the (data - noise)

                modality_flow = modality_tensor - noise

                # append to flow for loss

                flows[modality_type].append(modality_flow)

                modality_tensor = noised_modality

                # store function for deriving reconstruction loss from decoder

                get_recon_losses[modality_type].append(get_recon_loss(noise, modality_time, modality_tensor))

            # go through maybe encoder

            modality_tensor = add_temp_batch_dim(mod.latent_to_model)(modality_tensor)

            # gather the modality length

            modality_shape_tuple = modality_tensor.shape[:-1]
            modality_length = math.prod(modality_shape_tuple)

            text_tensor = torch.full((modality_length,), -1, device = device) # text is all -1 here, so text labels are not learned on

            # only add modality meta information when not returning embedding, which only occurs when sampling modality

            succeed_modality_tokens = precede_modality_tokens = 0

            if not return_embed:
                # add the [som] and [eom] tokens for the modality type

                som_id, eom_id = mod.som_id, mod.eom_id

                # start by just storing the token length of the modality

                modality_shape_str = join([*map(str, modality_shape_tuple)], ',')
                modality_meta_info = model.char_tokenizer(modality_shape_str, device = device)

                precede_modality_tokens = len(modality_meta_info) + 2
                succeed_modality_tokens = 1

                text_tensor = cat((
                    tensor_([model.meta_id]),
                    modality_meta_info,
                    tensor_([som_id]),
                    text_tensor,
                    tensor_([eom_id])
                ))

            batch_modality_positions.append((modality_type, offset + precede_modality_tokens, modality_length)) # offset + preceding meta tag length (which includes the modality start token)

            # store parsing out back to shape

            modality_tensor, unpack_modality_shape = pack_with_inverse(modality_tensor, '* d')

            inverse_fn = model_to_pred_flow(batch_index, offset + precede_modality_tokens, modality_length, unpack_modality_shape)

            # maybe decorate the function if model output is predicting clean

            if model.model_output_clean:
                decorator = get_model_output_to_flow_fn(modality_tensor, modality_time, model.eps, return_decorator = True)
                inverse_fn = decorator(inverse_fn)

            # store function for extracting flow later

            get_pred_flows[modality_type].append(inverse_fn)

            # increment offset

            offset += modality_length + precede_modality_tokens + succeed_modality_tokens # +2 due to [som] and [eom] - then account for meta start id and modality shape information (or eventually any meta information about modality)

            modality_tensor = pad_at_dim(modality_tensor, (precede_modality_tokens, succeed_modality_tokens), dim = -2)

            batch_modality_tokens.append(modality_tensor)

            batch_text.append(text_tensor)

            # handle axial positional embedding

            if need_axial_pos_emb:

                if exists(mod.pos_emb_mlp):
                    pos_emb_max_axial_dims[modality_type].append(tensor(modality_shape_tuple))
                    pos_emb = (modality_type, modality_shape_tuple, (precede_modality_tokens, succeed_modality_tokens))

                else:
                    pos_emb = torch.zeros(text_tensor.shape[0], dim, device = device)

                batch_modality_pos_emb.append(pos_emb)

        text.append(cat(batch_text))

        if need_axial_pos_emb:
            modality_pos_emb.append(batch_modality_pos_emb)

        modality_tokens.append(cat(batch_modality_tokens))
        modality_positions.append(batch_modality_positions)

        modality_index += 1 # original code incremented this per sample, not per modality instance - so all modalities in a sample read times[batch, sample_ordinal] (a bug kept for reference)

    total_tokens = sum([t.numel() for t in text]) if return_loss else None

    text = pad_sequence(text, value = -1)

    modality_tokens = pad_sequence(modality_tokens, dim = -2, value = 0.)

    if not need_axial_pos_emb:
        modality_pos_emb = None

    return ProcessedModalityBatch(
        text = text,
        modality_tokens = modality_tokens,
        modality_positions = modality_positions,
        modality_pos_emb = modality_pos_emb,
        flows = flows,
        get_pred_flows = get_pred_flows,
        get_recon_losses = get_recon_losses,
        pos_emb_max_axial_dims = pos_emb_max_axial_dims,
        total_tokens = total_tokens
    )

class ProcessedRecord(NamedTuple):
    packed: Tensor
    noise: Tensor | None
    noised: Tensor | None
    flow: Tensor | None
    shape: tuple[int, ...] | None # original tensor shape, for reshaping flat noise slices back
    start: int | None
    end: int | None
    slice_: Callable | None # token-axis slicing function of the flat per-type tensors
    time: Tensor

def process_type_flat(records: list[ModalityRecord], model, dim, return_loss) -> dict[ModalityRecord, ProcessedRecord]:
    # process all instances of one modality type (of any shapes) as a single flat tensor:
    # one random noise, one noising, one latent projection for the whole type. the projected
    # tensor is always (S, d) so it slices along dim 0, but for channel first the noise /
    # noised / flow live in (c, S) so their token axis is dim 1. the flow targets keep their
    # flat slice shape - the flow loss packs them by value - and the recon loss closures
    # slice lazily, so there is no per-instance slicing work in the hot path (reconstruction
    # loss is off by default anyway)

    mod = model.get_modality_info(records[0].modality_type)
    channel_first = mod.channel_first_latent

    # flatten each instance to its token sequence first - (length, d) for channel last,
    # (c, length) for channel first - then concatenate along the token axis

    def flatten(record):
        if channel_first:
            return record.tensor.reshape(record.tensor.shape[0], record.length)
        return record.tensor.reshape(record.length, record.tensor.shape[-1])

    combined = cat([flatten(record) for record in records], dim = 1 if channel_first else 0)

    if return_loss:
        times_ = stack([record.time for record in records])
        flat_times = times_.repeat_interleave(tensor([record.length for record in records], device = combined.device))

        if channel_first:
            flat_times = flat_times.view(1, -1)
        else:
            flat_times = append_dims(flat_times, 1)

        noise = torch.randn_like(combined)
        noised = combined * flat_times + noise * (1. - flat_times)
        flow = combined - noise
    else:
        noised = combined
        noise = flow = None

    # single latent projection for the whole type - `latent_to_model` is a linear over the
    # last dim, but expects a batch dim when channel first, so add one

    if channel_first:
        projected = mod.latent_to_model(noised[None, ...])[0]
    else:
        projected = mod.latent_to_model(noised)

    if channel_first:
        slice_ = lambda t, start, end: t[:, start:end]
    else:
        slice_ = lambda t, start, end: t[start:end]

    processed_by_record = {}

    offset = 0

    for record in records:
        start, end = offset, offset + record.length
        offset = end

        packed = projected[start:end].reshape(record.length, dim)

        if return_loss:
            processed_by_record[record] = ProcessedRecord(packed, noise, noised, slice_(flow, start, end), record.tensor.shape, start, end, slice_, record.time)
        else:
            processed_by_record[record] = ProcessedRecord(packed, None, None, None, None, None, None, None, record.time)

    return processed_by_record

def process_group_stacked(shape_records: list[ModalityRecord], model, dim, return_loss) -> dict[ModalityRecord, ProcessedRecord]:
    # process a group of 2+ same-shaped instances with one batched noise, noising and projection

    stacked = stack([record.tensor for record in shape_records])

    if return_loss:
        times_ = stack([record.time for record in shape_records])
        padded_times = append_dims(times_, stacked.ndim - 1)

        noise = torch.randn_like(stacked)
        noised = stacked * padded_times + noise * (1. - padded_times)
        flow = stacked - noise
    else:
        noised = stacked
        noise = flow = None

    mod = model.get_modality_info(shape_records[0].modality_type)

    projected = mod.latent_to_model(noised) # single projection for the whole group

    processed_by_record = {}

    for ind, record in enumerate(shape_records):
        packed = projected[ind].reshape(record.length, dim)

        if return_loss:
            processed_by_record[record] = ProcessedRecord(packed, noise[ind], noised[ind], flow[ind], None, None, None, None, record.time)
        else:
            processed_by_record[record] = ProcessedRecord(packed, None, None, None, None, None, None, None, record.time)

    return processed_by_record

def process_instance(record: ModalityRecord, model, dim, return_loss) -> dict[ModalityRecord, ProcessedRecord]:
    # process a single instance directly - a stack or cat would be a wasted copy

    mod = model.get_modality_info(record.modality_type)

    if return_loss:
        noise = torch.randn_like(record.tensor)
        noised = record.tensor * record.time + noise * (1. - record.time)
        flow = record.tensor - noise
    else:
        noised = record.tensor
        noise = flow = None

    # note: `latent_to_model` needs a batch dim when channel first

    if mod.channel_first_latent:
        projected = mod.latent_to_model(noised[None, ...])[0]
    else:
        projected = mod.latent_to_model(noised)

    packed = projected.reshape(record.length, dim)

    if return_loss:
        return {record: ProcessedRecord(packed, noise, noised, flow, None, None, None, None, record.time)}

    return {record: ProcessedRecord(packed, None, None, None, None, None, None, None, record.time)}

def build_record_closures(
    records: list[ModalityRecord],
    processed_by_record: dict[ModalityRecord, ProcessedRecord],
    model,
    dim,
    modality_type,
    return_loss,
    flows,
    get_pred_flows,
    get_recon_losses,
    packed_by_record
):
    # build the flow extraction functions and loss closures in scan order, so per type lists stay aligned

    for record in records:

        processed = processed_by_record[record]

        # packing and unpacking a modality is a plain reshape - no per instance pack needed

        def unpack_fn(embed, shape = record.axial_shape, dim_ = dim):
            return embed.reshape(*shape, dim_)

        inverse_fn = model_to_pred_flow(record.batch_index, record.scatter_offset, record.length, unpack_fn)

        # maybe decorate the function if model output is predicting clean

        if model.model_output_clean:
            decorator = get_model_output_to_flow_fn(processed.packed, record.time, model.eps, return_decorator = True)
            inverse_fn = decorator(inverse_fn)

        get_pred_flows[modality_type].append(inverse_fn)

        if return_loss:
            flows[modality_type].append(processed.flow)

            if processed.slice_ is not None:
                # flat-style processing: the noise / noised live in the flat per-type tensor, slice lazily

                get_recon_losses[modality_type].append(get_recon_loss_lazy(processed.noise, processed.noised, processed.time, processed.shape, processed.start, processed.end, processed.slice_))

            else:
                get_recon_losses[modality_type].append(get_recon_loss(processed.noise, processed.time, processed.noised))

        packed_by_record[record] = processed.packed

def process_modality_batch(
    modalities: list[ModalitySample],
    times: Float['b m'],
    model,
    *,
    need_axial_pos_emb: bool,
    return_loss: bool,
    return_embed: bool
) -> ProcessedModalityBatch:

    device = model.device
    dim = model.dim

    batch = len(modalities)

    modality_records, text_chunks, modality_positions, modality_pos_emb, pos_emb_max_axial_dims, total_lens = scan_batch_for_structure(
        modalities,
        times,
        model,
        need_axial_pos_emb = need_axial_pos_emb,
        return_embed = return_embed
    )

    # pass 2 - group all modality instances by (modality type, shape) and process in parallel:
    # one random noise, one noising operation, one latent to model projection per group

    records_by_type = defaultdict(list)

    for record in modality_records:
        records_by_type[record.modality_type].append(record)

    grouped_instances = sum(len(shape_records) - 1 for records in records_by_type.values() for shape_records in group_records_by_shape(records).values() if len(shape_records) > 1)
    singleton_groups = sum(1 for records in records_by_type.values() for shape_records in group_records_by_shape(records).values() if len(shape_records) == 1)

    logger.debug(f'process_modality_batch: {len(records_by_type)} modality types, {len(modality_records)} modality instances detected, '
                 f'{grouped_instances} instances processed in batched groups, {singleton_groups} processed per-instance')

    flows = defaultdict(list)
    get_pred_flows: GetPredFlows = defaultdict(list)
    get_recon_losses = defaultdict(list)

    processed_by_record = {}
    packed_by_record = {}

    for modality_type, records in records_by_type.items():

        shape_groups = group_records_by_shape(records)

        for shape_records in shape_groups.values():

            if len(shape_records) > 1:
                # group of 2+ same-shaped instances - process all at once:
                # one noise, one noising, one projection for the whole group,
                # then scatter each result back into its position in the sequence

                processed_by_record.update(process_group_stacked(shape_records, model, dim, return_loss))

            else:
                # group of one - a stack would be a wasted copy, process the instance directly

                processed_by_record.update(process_instance(shape_records[0], model, dim, return_loss))

        # build the flow extraction functions and loss closures in scan order, so per type lists stay aligned

        build_record_closures(records, processed_by_record, model, dim, modality_type, return_loss, flows, get_pred_flows, get_recon_losses, packed_by_record)

    # pass 3 - assemble each sample into a single pre-allocated buffer with per-chunk scatter,
    # replacing per-chunk allocations, padding and cats

    max_len = max(total_lens)

    text_bufs = torch.full((batch, max_len), -1, device = device)
    modality_bufs = torch.zeros((batch, max_len, dim), device = device)

    for batch_index, sample_chunks in enumerate(text_chunks):
        for offset, chunk in sample_chunks:
            text_bufs[batch_index, offset:(offset + chunk.shape[0])] = chunk

    for record in modality_records:
        packed = packed_by_record[record]
        modality_bufs[record.batch_index, record.scatter_offset:(record.scatter_offset + record.length)] = packed

    total_tokens = sum(total_lens) if return_loss else None

    if not need_axial_pos_emb:
        modality_pos_emb = None

    return ProcessedModalityBatch(
        text = text_bufs,
        modality_tokens = modality_bufs,
        modality_positions = modality_positions,
        modality_pos_emb = modality_pos_emb,
        flows = flows,
        get_pred_flows = get_pred_flows,
        get_recon_losses = get_recon_losses,
        pos_emb_max_axial_dims = pos_emb_max_axial_dims,
        total_tokens = total_tokens
    )

def process_modality_batch_flat(
    modalities: list[ModalitySample],
    times: Float['b m'],
    model,
    *,
    need_axial_pos_emb: bool,
    return_loss: bool,
    return_embed: bool
) -> ProcessedModalityBatch:

    # per modality type, concatenate all instances (of any shape) into one tensor along the
    # token axis, then process the whole type with a single random noise, single noising and
    # single latent projection. the noise, noising and projection are all elementwise / linear
    # over the last dim, so nothing about the grouping by shape helps - this drops the kernel
    # count per type from one-per-group to a small constant.

    device = model.device
    dim = model.dim

    batch = len(modalities)

    modality_records, text_chunks, modality_positions, modality_pos_emb, pos_emb_max_axial_dims, total_lens = scan_batch_for_structure(
        modalities,
        times,
        model,
        need_axial_pos_emb = need_axial_pos_emb,
        return_embed = return_embed
    )

    records_by_type = defaultdict(list)

    for record in modality_records:
        records_by_type[record.modality_type].append(record)

    flows = defaultdict(list)
    get_pred_flows: GetPredFlows = defaultdict(list)
    get_recon_losses = defaultdict(list)

    processed_by_record = {}
    packed_by_record = {}

    for modality_type, records in records_by_type.items():

        processed_by_record.update(process_type_flat(records, model, dim, return_loss))

        build_record_closures(records, processed_by_record, model, dim, modality_type, return_loss, flows, get_pred_flows, get_recon_losses, packed_by_record)

    # pass 3 - assemble each sample into a single pre-allocated buffer with per-chunk scatter

    max_len = max(total_lens)

    text_bufs = torch.full((batch, max_len), -1, device = device)
    modality_bufs = torch.zeros((batch, max_len, dim), device = device)

    for batch_index, sample_chunks in enumerate(text_chunks):
        for offset, chunk in sample_chunks:
            text_bufs[batch_index, offset:(offset + chunk.shape[0])] = chunk

    for record in modality_records:
        packed = packed_by_record[record]
        modality_bufs[record.batch_index, record.scatter_offset:(record.scatter_offset + record.length)] = packed

    total_tokens = sum(total_lens) if return_loss else None

    if not need_axial_pos_emb:
        modality_pos_emb = None

    return ProcessedModalityBatch(
        text = text_bufs,
        modality_tokens = modality_bufs,
        modality_positions = modality_positions,
        modality_pos_emb = modality_pos_emb,
        flows = flows,
        get_pred_flows = get_pred_flows,
        get_recon_losses = get_recon_losses,
        pos_emb_max_axial_dims = pos_emb_max_axial_dims,
        total_tokens = total_tokens
    )

def process_modality_batch_hybrid(
    modalities: list[ModalitySample],
    times: Float['b m'],
    model,
    *,
    need_axial_pos_emb: bool,
    return_loss: bool,
    return_embed: bool
) -> ProcessedModalityBatch:

    # best of both: same (type, shape) groups of 2+ are processed with one batched noise /
    # noising / projection (no concatenation copies), while all singleton groups of a type are
    # collected and processed together with the flat path (one noise / noising / projection
    # for the whole set) instead of per-instance.

    device = model.device
    dim = model.dim

    batch = len(modalities)

    modality_records, text_chunks, modality_positions, modality_pos_emb, pos_emb_max_axial_dims, total_lens = scan_batch_for_structure(
        modalities,
        times,
        model,
        need_axial_pos_emb = need_axial_pos_emb,
        return_embed = return_embed
    )

    records_by_type = defaultdict(list)

    for record in modality_records:
        records_by_type[record.modality_type].append(record)

    flows = defaultdict(list)
    get_pred_flows: GetPredFlows = defaultdict(list)
    get_recon_losses = defaultdict(list)

    processed_by_record = {}
    packed_by_record = {}

    for modality_type, records in records_by_type.items():

        shape_groups = group_records_by_shape(records)

        for shape_records in shape_groups.values():
            if len(shape_records) > 1:
                processed_by_record.update(process_group_stacked(shape_records, model, dim, return_loss))

        singleton_records = [shape_records[0] for shape_records in shape_groups.values() if len(shape_records) == 1]

        if singleton_records:
            processed_by_record.update(process_type_flat(singleton_records, model, dim, return_loss))

        build_record_closures(records, processed_by_record, model, dim, modality_type, return_loss, flows, get_pred_flows, get_recon_losses, packed_by_record)

    # pass 3 - assemble each sample into a single pre-allocated buffer with per-chunk scatter

    max_len = max(total_lens)

    text_bufs = torch.full((batch, max_len), -1, device = device)
    modality_bufs = torch.zeros((batch, max_len, dim), device = device)

    for batch_index, sample_chunks in enumerate(text_chunks):
        for offset, chunk in sample_chunks:
            text_bufs[batch_index, offset:(offset + chunk.shape[0])] = chunk

    for record in modality_records:
        packed = packed_by_record[record]
        modality_bufs[record.batch_index, record.scatter_offset:(record.scatter_offset + record.length)] = packed

    total_tokens = sum(total_lens) if return_loss else None

    if not need_axial_pos_emb:
        modality_pos_emb = None

    return ProcessedModalityBatch(
        text = text_bufs,
        modality_tokens = modality_bufs,
        modality_positions = modality_positions,
        modality_pos_emb = modality_pos_emb,
        flows = flows,
        get_pred_flows = get_pred_flows,
        get_recon_losses = get_recon_losses,
        pos_emb_max_axial_dims = pos_emb_max_axial_dims,
        total_tokens = total_tokens
    )

def evaluate_modality_pos_emb(
    modality_pos_emb,
    pos_emb_max_axial_dims,
    model,
    dim,
    device
):
    # lazily evaluate the modality positional embedding from the factorized positional embedding of maximum axial dims

    if not exists(modality_pos_emb):
        return None

    pos_emb_max_axial_dims = {mod_type: stack(sizes, dim = -1).amax(dim = -1) for mod_type, sizes in pos_emb_max_axial_dims.items()}
    factorized_pos_emb = {mod_type: model.get_modality_info(mod_type).pos_emb_mlp(max_size, return_factorized = True) for mod_type, max_size in pos_emb_max_axial_dims.items()}

    evaluated_pos_emb = []

    for batch_modality_pos_emb in modality_pos_emb:
        evaluated_batch_pos_emb = []

        for maybe_pos_emb_config in batch_modality_pos_emb:

            if is_tensor(maybe_pos_emb_config):
                evaluated_batch_pos_emb.append(maybe_pos_emb_config)
                continue

            if maybe_pos_emb_config[0] == 'zeros':
                _, length = maybe_pos_emb_config
                evaluated_batch_pos_emb.append(torch.zeros(length, dim, device = device))
                continue

            mod_type, mod_size, padding = maybe_pos_emb_config

            mod_info = model.get_modality_info(mod_type)
            mod_factorized_pos_emb = factorized_pos_emb[mod_type]

            mod_pos_emb = mod_info.pos_emb_mlp.combine_factorized(mod_factorized_pos_emb, mod_size, flatten = True)
            mod_pos_emb = pad_at_dim(mod_pos_emb, padding, dim = -2) # handle padding for preceding and succeeding meta tokens

            evaluated_batch_pos_emb.append(mod_pos_emb)

        evaluated_pos_emb.append(cat(evaluated_batch_pos_emb, dim = -2))

    return pad_sequence(evaluated_pos_emb, dim = -2, value = 0.)

# registry

PROCESSING_STRATEGIES = {
    'naive': process_modality_batch_naive,
    'grouped': process_modality_batch,
    'flat': process_modality_batch_flat,
    'hybrid': process_modality_batch_hybrid,
    'auto': None # set below once the auto router is defined
}

DEFAULT_PROCESSING_STRATEGY = 'auto'

# router - autodetect the fastest strategy for the batch structure at hand
# `'naive'` is excluded from routing: it is the reference baseline and never wins

ROUTING_CANDIDATES = ('grouped', 'flat', 'hybrid')

ROUTING_WARMUP = 1
ROUTING_ITERS = 3
ROUTING_MAX_CACHE = 64

def _sync_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

def structure_signature(
    modalities: list[ModalitySample],
    model,
    *,
    need_axial_pos_emb: bool,
    return_loss: bool,
    return_embed: bool
):
    # cheap pure-python pass over the batch, mirroring `scan_batch_for_structure`'s element
    # interpretation - yields the cache key for the routing decision

    type_shape_counts = defaultdict(lambda: defaultdict(int))

    total_tokens = 0
    batch_size = 0

    for batch_modalities in modalities:
        batch_size += 1

        for modality in batch_modalities:
            is_text = not isinstance(modality, tuple)

            if is_text:
                modality_tensor = modality

                if not is_int_tensor(modality_tensor):
                    modality_type = 0 # bare float tensor, treated as a type 0 modality
                    is_text = False
            else:
                modality_type, modality_tensor, *_ = modality

            # auto ward against scalars (lone start end tokens)

            if is_int_tensor(modality_tensor) and modality_tensor.ndim == 0:
                modality_tensor = rearrange(modality_tensor, '-> 1')

            if is_text:
                total_tokens += modality_tensor.shape[0]
                continue

            mod = model.get_modality_info(modality_type)
            axial_shape = modality_tensor.shape[1:] if mod.channel_first_latent else modality_tensor.shape[:-1]

            total_tokens += math.prod(axial_shape)
            type_shape_counts[modality_type][axial_shape] += 1

    structure = tuple(
        (modality_type, shape, count)
        for modality_type in sorted(type_shape_counts)
        for shape, count in sorted(type_shape_counts[modality_type].items())
    )

    return (
        str(model.device),
        model.dim,
        batch_size,
        total_tokens,
        need_axial_pos_emb,
        return_loss,
        return_embed,
        structure
    )

class StrategyRouter:
    def __init__(
        self,
        candidates = ROUTING_CANDIDATES,
        warmup = ROUTING_WARMUP,
        iters = ROUTING_ITERS,
        max_cache = ROUTING_MAX_CACHE
    ):
        self.candidates = tuple(candidates)
        self.warmup = warmup
        self.iters = iters
        self.max_cache = max_cache
        self.cache = {}

    def clear(self):
        self.cache.clear()

    def measure(self, modalities, times, model, *, need_axial_pos_emb, return_loss, return_embed):
        # time every candidate strategy on the actual batch and return the fastest

        device = model.device
        kwargs = dict(need_axial_pos_emb = need_axial_pos_emb, return_loss = return_loss, return_embed = return_embed)

        best = self.candidates[0]
        best_time = float('inf')

        for name in self.candidates:
            fn = get_processing_strategy(name)

            for _ in range(self.warmup):
                fn(modalities, times, model, **kwargs)

            samples = []

            for _ in range(self.iters):
                _sync_device(device)
                start = time.perf_counter()
                fn(modalities, times, model, **kwargs)
                _sync_device(device)
                samples.append(time.perf_counter() - start)

            median_time = statistics.median(samples)

            if median_time < best_time:
                best, best_time = name, median_time

        return best

    def route(self, modalities, times, model, *, need_axial_pos_emb, return_loss, return_embed):
        # pick the strategy for this batch - measured once per distinct batch structure, cached after

        key = structure_signature(
            modalities,
            model,
            need_axial_pos_emb = need_axial_pos_emb,
            return_loss = return_loss,
            return_embed = return_embed
        )

        if key in self.cache:
            return self.cache[key]

        if not key[-1]:
            # no modalities in the batch - every strategy is identical work, skip measuring
            strategy = 'hybrid'
        else:
            strategy = self.measure(
                modalities,
                times,
                model,
                need_axial_pos_emb = need_axial_pos_emb,
                return_loss = return_loss,
                return_embed = return_embed
            )

        self.cache[key] = strategy

        if len(self.cache) > self.max_cache:
            self.cache.pop(next(iter(self.cache))) # evict the oldest entry

        return strategy

ROUTER = StrategyRouter()

def process_modality_batch_auto(
    modalities: list[ModalitySample],
    times: Float['b m'],
    model,
    *,
    need_axial_pos_emb: bool,
    return_loss: bool,
    return_embed: bool
) -> ProcessedModalityBatch:

    # autodetect the fastest strategy for this batch structure (measured once, then cached)

    strategy = ROUTER.route(
        modalities,
        times,
        model,
        need_axial_pos_emb = need_axial_pos_emb,
        return_loss = return_loss,
        return_embed = return_embed
    )

    return get_processing_strategy(strategy)(
        modalities,
        times,
        model,
        need_axial_pos_emb = need_axial_pos_emb,
        return_loss = return_loss,
        return_embed = return_embed
    )

PROCESSING_STRATEGIES['auto'] = process_modality_batch_auto

def get_processing_strategy(name: str):
    assert name in PROCESSING_STRATEGIES, f'unknown modality processing strategy `{name}`, available: {list(PROCESSING_STRATEGIES)}'
    return PROCESSING_STRATEGIES[name]

def assert_strategies_equivalent(
    model,
    modalities,
    times,
    need_axial_pos_emb,
    return_loss,
    return_embed,
    strategy_names: list[str] | None = None
):
    # verify every strategy produces identical outputs (deterministic noise via mocked `torch.randn_like`)
    # used by the test suite and the benchmark before timing

    from unittest import mock

    strategy_names = default(strategy_names, list(PROCESSING_STRATEGIES))

    kwargs = dict(need_axial_pos_emb = need_axial_pos_emb, return_loss = return_loss, return_embed = return_embed)

    outputs = {}

    with mock.patch('torch.randn_like', side_effect = lambda t: torch.zeros_like(t)):
        for name in strategy_names:
            outputs[name] = get_processing_strategy(name)(modalities, times, model, **kwargs)

    reference = outputs[strategy_names[0]]

    for name in strategy_names[1:]:
        candidate = outputs[name]

        assert torch.equal(candidate.text, reference.text), f'{name}: text mismatch'
        assert torch.equal(candidate.modality_tokens, reference.modality_tokens), f'{name}: modality tokens mismatch'
        assert candidate.modality_positions == reference.modality_positions, f'{name}: positions mismatch'
        assert candidate.total_tokens == reference.total_tokens, f'{name}: total tokens mismatch'

        embed = torch.randn(len(modalities), reference.text.shape[-1], model.dim, device = model.device)

        for modality_type in reference.get_pred_flows:
            assert modality_type in candidate.get_pred_flows, f'{name}: missing modality type {modality_type} in pred flows'

            for pred_reference, pred_candidate in zip(reference.get_pred_flows[modality_type], candidate.get_pred_flows[modality_type]):
                assert torch.allclose(pred_reference(embed), pred_candidate(embed)), f'{name}: pred flow closures mismatch'

        if return_loss:
            for modality_type in reference.flows:
                for flow_reference, flow_candidate in zip(reference.flows[modality_type], candidate.flows[modality_type]):
                    assert torch.equal(flow_reference.reshape(-1), flow_candidate.reshape(-1)), f'{name}: flow targets mismatch'

    return outputs
