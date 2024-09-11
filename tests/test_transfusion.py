import pytest
from functools import partial
from torch import randint, randn, tensor

from transfusion_pytorch.transfusion import (
    Transfusion,
    flex_attention,
    exists
)

@pytest.mark.parametrize('cache_kv', (False, True))
@pytest.mark.parametrize('use_flex_attn', (False, True))
def test_transfusion(
    cache_kv: bool,
    use_flex_attn: bool
):

    if use_flex_attn and not exists(flex_attention):
        return pytest.skip()

    text_tokens = 8
    randint_ = partial(randint, 0, text_tokens)

    model = Transfusion(
        num_text_tokens = text_tokens,
        dim_latent = (384, 192), # specify multiple latent dimensions
        modality_default_length = (32, 64),
        transformer = dict(
            dim = 512,
            depth = 2,
            use_flex_attn = use_flex_attn
        )
    )

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

    if use_flex_attn and not exists(flex_attention):
        return pytest.skip()

    text_tokens = 8
    randint_ = partial(randint, 0, text_tokens)

    model = Transfusion(
        num_text_tokens = text_tokens,
        dim_latent = 384,
        modality_token_transform = 'c h w -> (h w) c',
        modality_default_length = 32,
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
def test_auto_modality_transform(
    use_flex_attn: bool
):

    if use_flex_attn and not exists(flex_attention):
        return pytest.skip()

    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = 384,
        modality_token_transform = 'c h w -> (h w) c',
        modality_default_length = 32,
        transformer = dict(
            dim = 512,
            depth = 2,
            use_flex_attn = use_flex_attn
        )
    )

    text = randint(0, 256, (2, 1024))

    loss = model(text)

    loss.backward()
