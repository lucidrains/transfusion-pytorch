<img src="./transfusion.png" width="400px"></img>

## Transfusion - Pytorch (wip)

Pytorch implementation of [Transfusion](https://www.arxiv.org/abs/2408.11039), "Predict the Next Token and Diffuse Images with One Multi-Modal Model", from MetaAI.

Once completed, will also extend this to flow matching, as well as audio, video, perhaps even policies.

## Install

```bash
$ pip install transfusion-pytorch
```

## Usage

One modality, say images

```python
import torch
from transfusion_pytorch import Transfusion

model = Transfusion(
    num_text_tokens = 256,
    dim_latent = 192,
    transformer = dict(
        dim = 512,
        depth = 8
    )
)

text_ids = torch.randint(0, 256, (2, 1024))

modality_tokens = [[
    torch.randn(6, 192),
    torch.randn(4, 192)
], [
    torch.randn(5, 192),
    torch.randn(3, 192)
]]

modality_positions = [[
    (2, 6),
    (10, 4)
], [
    (2, 5),
    (10, 3)
]] # (offset, length)

loss, breakdown = model(
    text_ids,
    modality_tokens = modality_tokens,
    modality_positions = modality_positions
)

loss.backward()
```

Multiple modalities

```python
import torch
from transfusion_pytorch import Transfusion

model = Transfusion(
    num_text_tokens = 256,
    dim_latent = (384, 192),
    transformer = dict(
        dim = 512,
        depth = 8
    )
)

text_ids = torch.randint(0, 256, (2, 1024))

modality_tokens = [[
    torch.randn(6, 384),
    torch.randn(4, 192)
], [
    torch.randn(5, 192),
    torch.randn(3, 384)
]]

modality_positions = [[
    (0, 2, 6),
    (1, 10, 4)
], [
    (1, 2, 5),
    (0, 10, 3)
]] # (type, offset, length)

loss, breakdown = model(
    text_ids,
    modality_tokens = modality_tokens,
    modality_positions = modality_positions
)

loss.backward()
```

## Citations

```bibtex
@inproceedings{Zhou2024TransfusionPT,
    title  = {Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model},
    author = {Chunting Zhou and Lili Yu and Arun Babu and Kushal Tirumala and Michihiro Yasunaga and Leonid Shamis and Jacob Kahn and Xuezhe Ma and Luke Zettlemoyer and Omer Levy},
    year   = {2024},
    url    = {https://api.semanticscholar.org/CorpusID:271909855}
}
```

```bibtex
@misc{Rubin2024,
    author  = {Ohad Rubin},
    url     = {https://medium.com/@ohadrubin/exploring-weight-decay-in-layer-normalization-challenges-and-a-reparameterization-solution-ad4d12c24950}
}
```
