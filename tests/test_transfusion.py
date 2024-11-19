import pytest
from functools import partial
from copy import deepcopy

import torch
from torch import nn, randint, randn, tensor, cuda

from einops import rearrange

import torch._dynamo
torch._dynamo.config.suppress_errors = True

cuda_available = cuda.is_available()

from transfusion_pytorch.transfusion import (
    Transfusion,
    flex_attention,
    exists
)

@pytest.mark.parametrize('cache_kv', (False, True))
@pytest.mark.parametrize('use_flex_attn', (False, True))
def test_transfusion(
    cache_kv: bool,
    use_flex_attn: bool,
):

    if use_flex_attn and (not exists(flex_attention) or not cuda_available):
        return pytest.skip()

    text_tokens = 8
    randint_ = partial(randint, 0, text_tokens)

    model = Transfusion(
        num_text_tokens = text_tokens,
        dim_latent = (384, 192), # specify multiple latent dimensions
        modality_default_shape = ((32,), (64,)),
        transformer = dict(
            dim = 512,
            depth = 2,
            use_flex_attn = use_flex_attn
        )
    )

    if use_flex_attn:
        model = model.cuda()

    # then for the Tensors of type float, you can pass a tuple[int, Tensor] and specify the modality index in the first position

    text_images_and_audio = [
        [randint_((16,)), (0, randn(4, 384)), randint_((8,)), (1, randn(6, 192))],
        [randint_((16,)), randn(7, 384), randint_((5,)), (1, randn(2, 192)), randint_((9,))]
    ]

    loss = model(text_images_and_audio)

    loss.backward()

    # after much training

    prime = [tensor(model.som_ids[0])]

    one_multimodal_sample = model.sample(prime, max_length = 128, cache_kv = cache_kv)


@pytest.mark.parametrize('use_flex_attn', (False, True))
def test_auto_modality_transform(
    use_flex_attn: bool
):

    if use_flex_attn and (not exists(flex_attention) or not cuda_available):
        return pytest.skip()

    text_tokens = 8
    randint_ = partial(randint, 0, text_tokens)

    model = Transfusion(
        num_text_tokens = text_tokens,
        dim_latent = 384,
        channel_first_latent = True,
        modality_default_shape = (32,),
        transformer = dict(
            dim = 512,
            depth = 2,
            use_flex_attn = use_flex_attn
        )
    )

    text_and_images = [
        [randint_((16,)), randn(384, 2, 2), randint_((8,)), randn(384, 2, 2)],
        [randint_((16,)), randn(384, 2, 2), randint_((5,)), randn(384, 2, 2), randint_((9,))]
    ]

    loss = model(text_and_images)

    loss.backward()

    # after much training

    prime = [tensor(model.som_ids[0])]

    one_multimodal_sample = model.sample(prime, max_length = 128)

@pytest.mark.parametrize('use_flex_attn', (False, True))
@pytest.mark.parametrize('return_loss', (False, True))
def test_text(
    use_flex_attn: bool,
    return_loss: bool
):

    if use_flex_attn and (not exists(flex_attention) or not cuda_available):
        return pytest.skip()

    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = 384,
        channel_first_latent = True,
        modality_default_shape = (32,),
        transformer = dict(
            dim = 512,
            depth = 2,
            use_flex_attn = use_flex_attn
        )
    )

    if use_flex_attn:
        model = model.cuda()

    text = randint(0, 256, (2, 1024))

    model(text, return_loss = return_loss)

@pytest.mark.parametrize('channel_first', (False, True))
def test_modality_only(
    channel_first: bool
):

    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = (384, 192),
        channel_first_latent = channel_first,
        modality_default_shape = (32,),
        transformer = dict(
            dim = 512,
            depth = 2,
            use_flex_attn = False
        )
    )

    images = randn(2, 8, 8, 192)

    if channel_first:
        images = rearrange(images, 'b ... d -> b d ...')

    loss = model(images, return_loss = True, modality_type = 1)

    loss.backward()


@pytest.mark.parametrize('custom_time_fn', (False, True))
def test_text_image_end_to_end(
    custom_time_fn: bool
):
    mock_vae_encoder = nn.Conv2d(3, 384, 3, padding = 1)
    mock_vae_decoder = nn.Conv2d(384, 3, 3, padding = 1)

    model = Transfusion(
        num_text_tokens = 4,
        dim_latent = 384,
        channel_first_latent = True,
        modality_default_shape = ((4, 4),),
        modality_encoder = mock_vae_encoder,
        modality_decoder = mock_vae_decoder,
        transformer = dict(
            dim = 512,
            depth = 8
        )
    )

    text_and_images = [
        [
            randint(0, 4, (16,)),
            randn(3, 8, 8),
            randint(0, 4, (8,)),
            randn(3, 7, 7)
        ],
        [
            randint(0, 4, (16,)),
            randn(3, 8, 5),
            randint(0, 4, (5,)),
            randn(3, 2, 16),
            randint(0, 4, (9,))
        ]
    ]

    # allow researchers to experiment with different time distributions across multiple modalities in a sample

    def modality_length_to_times(modality_length):
        has_modality = modality_length > 0
        return torch.where(has_modality, torch.ones_like(modality_length), 0.)

    time_fn = modality_length_to_times if custom_time_fn else None

    # forward

    loss = model(
        text_and_images,
        modality_length_to_times_fn = time_fn
    )

    loss.backward()

    # after much training

    one_multimodal_sample = model.sample()

def test_velocity_consistency():
    mock_encoder = nn.Conv2d(3, 384, 3, padding = 1)
    mock_decoder = nn.Conv2d(384, 3, 3, padding = 1)

    model = Transfusion(
        num_text_tokens = 12,
        dim_latent = 384,
        channel_first_latent = True,
        modality_default_shape = ((4, 4)),
        modality_num_dim = 2,
        modality_encoder = mock_encoder,
        modality_decoder = mock_decoder,
        transformer = dict(
            dim = 512,
            depth = 1
        )
    )

    ema_model = deepcopy(model)

    text_and_images = [
        [
            randint(0, 12, (16,)),
            randn(3, 8, 8),
            randint(0, 12, (8,)),
            randn(3, 7, 7)
        ],
        [
            randint(0, 12, (16,)),
            randn(3, 8, 5),
            randint(0, 12, (5,)),
            randn(3, 2, 16),
            randint(0, 12, (9,))
        ]
    ]

    def modality_length_to_times(modality_length):
        has_modality = modality_length > 0
        return torch.where(has_modality, torch.ones_like(modality_length), 0.)

    loss, breakdown = model(
        text_and_images,
        velocity_consistency_ema_model = ema_model,
        modality_length_to_times_fn = modality_length_to_times,
        return_breakdown = True
    )

    loss.backward()

    assert exists(breakdown.velocity)

def test_axial_pos_emb():
    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = (384, 192),                    # specify multiple latent dimensions
        modality_default_shape = ((2, 2), (2,)),    # default shapes for first and second modality
        fallback_to_default_shape_if_invalid = True,
        add_pos_emb = True,
        modality_num_dim = (2, 1),
        transformer = dict(
            dim = 512,
            depth = 8
        )
    )

    # then for the Tensors of type float, you can pass a tuple[int, Tensor] and specify the modality index in the first position

    # any torch.long is text, torch.float is modalities

    text_images_and_audio = [
        [randint(0, 256, (16,)), (0, randn(2, 3, 384)), randint(0, 256, (8,)), (1, randn(6, 192))],
        [randint(0, 256, (16,)), randn(1, 4, 384), randint(0, 256, (5,)), (1, randn(2, 192)), randint(0, 256, (9,))]
    ]

    loss = model(text_images_and_audio)

    loss.backward()

    # after much training

    one_multimodal_sample = model.sample()
