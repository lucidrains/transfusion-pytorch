from torch import randint, randn
from transfusion_pytorch import Transfusion

def test_transfusion():
    model = Transfusion(
        num_text_tokens = 256,
        dim_latent = (384, 192), # specify multiple latent dimensions
        transformer = dict(
            dim = 512,
            depth = 8
        )
    )

    # then for the Tensors of type float, you can pass a tuple[int, Tensor] and specify the modality index in the first position

    text_images_and_audio = [
        [randint(0, 256, (16,)), (0, randn(4, 384)), randint(0, 256, (8,)), (1, randn(6, 192))],
        [randint(0, 256, (16,)), randn(7, 384), randint(0, 256, (5,)), (1, randn(2, 192)), randint(0, 256, (9,))]
    ]

    loss = model(text_images_and_audio)

    loss.backward()

    # after much training

    one_multimodal_sample = model.sample(max_length = 128)
