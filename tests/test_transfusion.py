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
    exists,
    stack_same_shape_tensors_with_inverse,
    filter_with_inverse,
    apply_fn_modality_type,
    SelfMaskedRepTraining
)

@pytest.mark.parametrize('cache_kv', (False, True))
@pytest.mark.parametrize('use_flex_attn', (False, True))
@pytest.mark.parametrize('num_residual_streams', (1, 4))
@pytest.mark.parametrize('reconstruction_loss_weight', (0., 0.1))
@pytest.mark.parametrize('model_output_clean', (False, True))
def test_transfusion(
    cache_kv: bool,
    use_flex_attn: bool,
    num_residual_streams: int,
    reconstruction_loss_weight: float,
    model_output_clean: bool
):

    if use_flex_attn and (not exists(flex_attention) or not cuda_available):
        return pytest.skip()

    text_tokens = 8
    randint_ = partial(randint, 0, text_tokens)

    model = Transfusion(
        num_text_tokens = text_tokens,
        dim_latent = (16, 16),
        modality_default_shape = ((8,), (4,)),
        reconstruction_loss_weight = reconstruction_loss_weight,
        model_output_clean = model_output_clean,
        transformer = dict(
            dim = 16,
            depth = 1,
            use_flex_attn = use_flex_attn,
            num_residual_streams = num_residual_streams
        ),
    )

    if use_flex_attn:
        model = model.cuda()

    # then for the Tensors of type float, you can pass a tuple[int, Tensor] and specify the modality index in the first position

    text_images_and_audio = [
        [randint_((16,)), (0, randn(4, 16)), randint_((8,)), (1, randn(6, 16))],
        [randint_((16,)), randn(7, 16), randint_((5,)), (1, randn(2, 16)), randint_((9,))]
    ]

    loss = model(text_images_and_audio)

    loss.backward()

    # after much training

    prime = [tensor(model.som_ids[0])]

    one_multimodal_sample = model.sample(prime, max_length = 4, cache_kv = cache_kv)


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
        dim_latent = 16,
        channel_first_latent = True,
        modality_default_shape = (2, 2),
        transformer = dict(
            dim = 16,
            depth = 1,
            use_flex_attn = use_flex_attn
        )
    )

    text_and_images = [
        [randint_((16,)), randn(16, 2, 2), randint_((8,)), randn(16, 2, 2)],
        [randint_((16,)), randn(16, 2, 2), randint_((5,)), randn(16, 2, 2), randint_((9,))]
    ]

    loss = model(text_and_images)

    loss.backward()

    # after much training

    prime = [tensor(model.som_ids[0])]

    one_multimodal_sample = model.sample(prime, max_length = 4)

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
        dim_latent = 16,
        channel_first_latent = True,
        modality_default_shape = (8,),
        transformer = dict(
            dim = 16,
            depth = 1,
            use_flex_attn = use_flex_attn
        )
    )

    if use_flex_attn:
        model = model.cuda()

    text = randint(0, 256, (2, 64))

    model(text, return_loss = return_loss)

@pytest.mark.parametrize('channel_first', (False, True))
def test_modality_only(
    channel_first: bool
):

    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = (16, 16),
        channel_first_latent = channel_first,
        modality_default_shape = (8,),
        transformer = dict(
            dim = 16,
            depth = 1,
            use_flex_attn = False
        )
    )

    images = randn(2, 4, 4, 16)

    if channel_first:
        images = rearrange(images, 'b ... d -> b d ...')

    loss = model(images, return_loss = True, modality_type = 1)

    loss.backward()

    model.generate_modality_only(modality_type = 1)

@pytest.mark.parametrize('custom_time_fn', (False, True))
def test_text_image_end_to_end(
    custom_time_fn: bool
):
    mock_vae_encoder = nn.Conv2d(3, 16, 3, padding = 1)
    mock_vae_decoder = nn.Conv2d(16, 3, 3, padding = 1)

    model = Transfusion(
        num_text_tokens = 4,
        dim_latent = 16,
        channel_first_latent = True,
        modality_default_shape = ((4, 4),),
        modality_encoder = mock_vae_encoder,
        modality_decoder = mock_vae_decoder,
        transformer = dict(
            dim = 16,
            depth = 1
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

    def num_modalities_to_times(num_modalities):
        batch = num_modalities.shape[0]
        device = num_modalities.device
        total_modalities = num_modalities.amax().item()
        return torch.ones((batch, total_modalities), device = device)

    time_fn = num_modalities_to_times if custom_time_fn else None

    # forward

    loss = model(
        text_and_images,
        num_modalities_to_times_fn = time_fn
    )

    loss.backward()

    # after much training

    one_multimodal_sample = model.sample(max_length = 4)

def test_velocity_consistency():
    mock_encoder = nn.Conv2d(3, 16, 3, padding = 1)
    mock_decoder = nn.Conv2d(16, 3, 3, padding = 1)

    model = Transfusion(
        num_text_tokens = 12,
        dim_latent = 16,
        channel_first_latent = True,
        modality_default_shape = (4, 4),
        modality_encoder = mock_encoder,
        modality_decoder = mock_decoder,
        transformer = dict(
            dim = 16,
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

    loss, breakdown = model(
        text_and_images,
        velocity_consistency_ema_model = ema_model,
        return_breakdown = True
    )

    loss.backward()

    assert exists(breakdown.velocity)

def test_axial_pos_emb():
    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = (16, 16),                      # specify multiple latent dimensions
        modality_default_shape = ((2, 2), (2,)),    # default shapes for first and second modality
        fallback_to_default_shape_if_invalid = True,
        add_pos_emb = True,
        modality_num_dim = (2, 1),
        transformer = dict(
            dim = 16,
            depth = 1
        )
    )

    # then for the Tensors of type float, you can pass a tuple[int, Tensor] and specify the modality index in the first position

    # any torch.long is text, torch.float is modalities

    text_images_and_audio = [
        [randint(0, 256, (16,)), (0, randn(2, 3, 16)), randint(0, 256, (8,)), (1, randn(6, 16))],
        [randint(0, 256, (16,)), randn(1, 4, 16), randint(0, 256, (5,)), (1, randn(2, 16)), randint(0, 256, (9,))]
    ]

    loss = model(text_images_and_audio)

    loss.backward()

    # after much training

    one_multimodal_sample = model.sample(max_length = 4)

# unet related

def test_modality_only_with_unet():

    model = Transfusion(
        num_text_tokens = 10,
        dim_latent = 4,
        modality_default_shape = (14, 14),
        pre_post_transformer_enc_dec = (
            nn.Conv2d(4, 16, 3, 2, 1),
            nn.ConvTranspose2d(16, 4, 3, 2, 1, output_padding = 1),
        ),
        channel_first_latent = True,
        add_pos_emb = True,
        modality_num_dim = 2,
        velocity_consistency_loss_weight = 0.1,
        transformer = dict(
            dim = 16,
            depth = 1,
            dim_head = 8,
            heads = 2
        )
    )

    x = torch.randn(1, 4, 14, 14)

    loss = model(x)
    loss.backward()

    sampled = model.generate_modality_only()

def test_stack_similar_shape_fn():
    from torch import zeros

    data = [
        zeros(3, 5),
        zeros(2, 3),
        zeros(3, 5),
        zeros(2, 3),
        zeros(4, 5),
        zeros(4, 5)
    ]

    plus_one = lambda x: x + 1

    data = [d + i for i, d in enumerate(data)]
    data_plus_one = [plus_one(d) for d in data]

    stacked_tensors, inverse = stack_same_shape_tensors_with_inverse(data)

    stacked_tensors = {k: plus_one(v) for k, v in stacked_tensors.items()}

    batch_processed_data_plus_one = inverse(stacked_tensors)

    assert all([torch.allclose(tensor1, tensor2) for tensor1, tensor2 in zip(data_plus_one, batch_processed_data_plus_one)])

def test_filter_with_inverse():
    x = [0, 1, 2, 3, 4]
    is_even = lambda el: (el % 2) == 0

    x_even, inverse = filter_with_inverse(is_even, x)
    x_even_times_ten = [el * 10 for el in x_even]

    y = inverse(x_even_times_ten)
    assert y == [0, 1, 20, 3, 40]

def test_apply_fn_modality_type():
    from torch import zeros

    modalities = [
        [zeros(3, 5)],
        [zeros(1, 5)],
        [(1, zeros(3, 5))],
        [(1, zeros(2, 5))],
        [(0, zeros(1, 5)), (1, zeros(3, 5))],
    ]

    modalities = apply_fn_modality_type(lambda x: x + 1, modalities)

    modalities = apply_fn_modality_type(lambda x: x + 2, modalities, modality_type = 1)

    assert (modalities[0][0][-1] == 1).all()
    assert (modalities[2][0][-1] == 2).all()


def test_zero_dimensional():

    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = 16,
        modality_default_shape = (),
        transformer = dict(
            dim = 16,
            depth = 1,
            num_residual_streams = 1
        )
    )

    # any torch.long is text, torch.float is modalities

    text_and_embeds = [
        [randint(0, 256, (16,)), randn(16), randint(0, 256, (8,)), randn(16)],
        [randint(0, 256, (16,)), randn(16), randint(0, 256, (5,)), randn(16), randint(0, 256, (9,))]
    ]

    loss = model(text_and_embeds)

    loss.backward()

    # after much training

    one_multimodal_sample = model.sample(prompt = randn(16), max_length = 4)

def test_self_flow():
    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = 16,
        modality_default_shape = (),
        transformer = dict(
            dim = 16,
            depth = 1,
            num_residual_streams = 1
        )
    )

    self_flow_wrapper = SelfMaskedRepTraining(
        model,
        use_asymmetric_dropout = True,
        student_dropout_rate = 0.1,
        teacher_dropout_rate = 0.,
        rep_loss_weight = 0.1,
        student_layer = -1,
        teacher_layer = -1,
    )

    text_and_embeds = [
        [randint(0, 256, (16,)), randn(16), randint(0, 256, (8,)), randn(16)],
        [randint(0, 256, (16,)), randn(16), randint(0, 256, (5,)), randn(16), randint(0, 256, (9,))]
    ]

    total_loss, (student_loss, self_flow_loss) = self_flow_wrapper(text_and_embeds)
    total_loss.backward()

    self_flow_wrapper.update_teacher()

    assert exists(self_flow_loss) and exists(student_loss)
    assert total_loss.shape == ()

@pytest.mark.parametrize('cache_kv', (False, True))
@pytest.mark.parametrize('prob_uncond', (0.0, 0.5, 1.0))
def test_classifier_free_guidance(
    cache_kv: bool,
    prob_uncond: float
):
    text_tokens = 16

    model = Transfusion(
        num_text_tokens = text_tokens,
        dim_latent = 8,
        prob_uncond = prob_uncond,
        modality_default_shape = (4,),
        transformer = dict(
            dim = 16,
            depth = 1,
            use_flex_attn = False
        )
    )

    # dataset batch with text and image modalities

    text_and_images = [
        [randint(0, text_tokens, (8,)), randn(4, 8), randint(0, text_tokens, (4,))],
        [randint(0, text_tokens, (6,)), randn(4, 8), randint(0, text_tokens, (5,))]
    ]

    # train forward pass with CFG drop

    model.train()
    loss = model(text_and_images)
    loss.backward()

    # sample with CFG scale > 1

    prompt = [randint(0, text_tokens, (4,))]
    sample_cfg1 = model.sample(prompt, max_length = 8, cfg_scale = 1.0, cache_kv = cache_kv)
    sample_cfg3 = model.sample(prompt, max_length = 8, cfg_scale = 3.0, cache_kv = cache_kv)

    assert len(sample_cfg1) > 0
    assert len(sample_cfg3) > 0

def test_e2e_multimodal_cfg_sampling():
    from transfusion_pytorch.transfusion import random_modality_length_to_time_fn

    text_tokens = 16

    model = Transfusion(
        num_text_tokens = text_tokens,
        dim_latent = 8,
        prob_uncond = 0.2,
        modality_default_shape = (4,),
        transformer = dict(
            dim = 16,
            depth = 1,
            use_flex_attn = False
        )
    )

    text_and_images = [
        [randint(0, text_tokens, (8,)), randn(4, 8), randint(0, text_tokens, (4,))],
        [randint(0, text_tokens, (6,)), randn(4, 8), randint(0, text_tokens, (5,))]
    ]

    loss = model(text_and_images, num_modalities_to_times_fn = random_modality_length_to_time_fn)
    loss.backward()

    # sample starting with a [som] token to force modality generation followed by text

    prime = [tensor([model.som_ids[0]])]
    sample = model.sample(prime, max_length = 16, cfg_scale = 2.5, cache_kv = True)

    assert len(sample) >= 3


def test_e2e_self_flow_with_cfg():
    model = Transfusion(
        num_text_tokens = 32,
        dim_latent = 16,
        prob_uncond = 0.1,
        modality_default_shape = (4,),
        transformer = dict(
            dim = 16,
            depth = 1,
            num_residual_streams = 1
        )
    )

    wrapper = SelfMaskedRepTraining(
        model,
        use_asymmetric_dropout = True,
        student_dropout_rate = 0.1,
        teacher_dropout_rate = 0.,
        rep_loss_weight = 0.1,
        student_layer = -1,
        teacher_layer = -1
    )

    data = [
        [randint(0, 32, (12,)), randn(4, 16), randint(0, 32, (6,))]
    ]

    total_loss, (student_loss, self_flow_loss) = wrapper(data)
    total_loss.backward()
    wrapper.update_teacher()

    assert total_loss.ndim == 0
