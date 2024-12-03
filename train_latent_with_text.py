from shutil import rmtree
from pathlib import Path

import torch
from torch import nn, tensor, Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from einops import rearrange

import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from transfusion_pytorch import Transfusion, print_modality_sample

# hf related

from datasets import load_dataset
from diffusers.models import AutoencoderKL

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder = "vae")

class Encoder(Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        with torch.no_grad():
            latent = self.vae.encode(image * 2 - 1)

        return 0.18215 * latent.latent_dist.sample()

class Decoder(Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        latents = (1 / 0.18215) * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        return (image / 2 + 0.5).clamp(0, 1)

# results folder

rmtree('./results', ignore_errors = True)
results_folder = Path('./results')
results_folder.mkdir(exist_ok = True, parents = True)

# constants

SAMPLE_EVERY = 100

with open("./data/flowers/labels.txt", "r") as file:
    content = file.read()

LABELS_TEXT = content.split('\n')

# functions

def divisible_by(num, den):
    return (num % den) == 0

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens: Tensor) -> str:
    return "".join(list(map(decode_token, tokens.tolist())))

def encode_tokens(str: str) -> Tensor:
    return tensor([*bytes(str, 'UTF-8')])

# encoder / decoder

model = Transfusion(
    num_text_tokens = 256,
    dim_latent = 4,
    channel_first_latent = True,
    modality_default_shape = (8, 8),
    modality_encoder = Encoder(vae),
    modality_decoder = Decoder(vae),
    pre_post_transformer_enc_dec = (
        nn.Conv2d(4, 128, 3, 2, 1),
        nn.ConvTranspose2d(128, 4, 3, 2, 1, output_padding = 1),
    ),
    add_pos_emb = False,
    modality_num_dim = 2,
    reconstruction_loss_weight = 0.1,
    transformer = dict(
        dim = 128,
        depth = 8,
        dim_head = 64,
        heads = 8,
    )
).cuda()

ema_model = model.create_ema(0.9)

class FlowersDataset(Dataset):
    def __init__(self, image_size):
        self.ds = load_dataset("nelorth/oxford-flowers")['train']

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.PILToTensor(),
            T.Lambda(lambda t: t / 255.)
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        pil = sample['image']

        labels_int = sample['label']
        labels_text = LABELS_TEXT[labels_int]

        tensor = self.transform(pil)
        return encode_tokens(labels_text), tensor

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

dataset = FlowersDataset(128)

dataloader = model.create_dataloader(dataset, batch_size = 4, shuffle = True)

iter_dl = cycle(dataloader)

optimizer = Adam(model.parameters(), lr = 8e-4)

# train loop

for step in range(1, 100_000 + 1):

    for _ in range(4):
        loss = model.forward(next(iter_dl))
        (loss / 4).backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()
    optimizer.zero_grad()

    ema_model.update()

    print(f'{step}: {loss.item():.3f}')

    if divisible_by(step, SAMPLE_EVERY):
        sample = ema_model.sample()

        print_modality_sample(sample)

        if len(sample) < 3:
            continue

        text_tensor, maybe_image, *_ = sample

        if not isinstance(maybe_image, tuple):
            continue

        _, image = maybe_image
        text_tensor = text_tensor[text_tensor < 256] # todo: offer a utility function for removing meta tags and special tokens

        text = decode_tokens(text_tensor)
        filename = str(results_folder / f'{step}.{text}.png')

        save_image(
            image.detach().cpu(),
            filename
        )
