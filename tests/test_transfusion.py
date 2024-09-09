import pytest
from functools import partial
from torch import randint, randn
from transfusion_pytorch import Transfusion

@pytest.mark.parametrize('cache_kv', (False, True))
def test_transfusion(
    cache_kv: bool
):

    text_tokens = 8
    randint_ = partial(randint, 0, text_tokens)

    model = Transfusion(
        num_text_tokens = text_tokens,
        dim_latent = (384, 192), # specify multiple latent dimensions
        transformer = dict(
            dim = 512,
            depth = 8
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

    one_multimodal_sample = model.sample(max_length = 128, cache_kv = cache_kv)
