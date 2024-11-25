from shutil import rmtree
from pathlib import Path

import torch
from torch import tensor
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

# results folder

rmtree('./results', ignore_errors = True)
results_folder = Path('./results')
results_folder.mkdir(exist_ok = True, parents = True)

# constants

SAMPLE_EVERY = 100

# functions

def divisible_by(num, den):
    return (num % den) == 0

# encoder / decoder

class Encoder(Module):
    def forward(self, x):
        x = rearrange(x, '... c (h p1) (w p2) -> ... h w (p1 p2 c)', p1 = 4, p2 = 4)
        return x * 2 - 1

class Decoder(Module):
    def forward(self, x):
        x = rearrange(x, '... h w (p1 p2 c) -> ... c (h p1) (w p2)', p1 = 4, p2 = 4, c = 3)
        return ((x + 1) * 0.5).clamp(min = 0., max = 1.)

model = Transfusion(
    num_text_tokens = 10,
    dim_latent = 4 * 4 * 3,
    channel_first_latent = False,
    modality_default_shape = (16, 16),
    modality_encoder = Encoder(),
    modality_decoder = Decoder(),
    add_pos_emb = True,
    modality_num_dim = 2,
    velocity_consistency_loss_weight = 0.1,
    reconstruction_loss_weight = 0.1,
    transformer = dict(
        dim = 64,
        depth = 4,
        dim_head = 32,
        heads = 8
    )
).cuda()

ema_model = model.create_ema()

class FlowersDataset(Dataset):
    def __init__(self):
        self.ds = load_dataset("nelorth/oxford-flowers")['train']

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pil = self.ds[idx]['image']
        image_tensor = T.PILToTensor()(pil)
        return T.Resize((64, 64))(image_tensor / 255.)

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

dataset = FlowersDataset()

dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
iter_dl = cycle(dataloader)

optimizer = Adam(model.parameters(), lr = 8e-4)

# train loop

for step in range(1, 100_000 + 1):

    loss = model.forward_modality(next(iter_dl))
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()
    optimizer.zero_grad()

    ema_model.update()

    print(f'{step}: {loss.item():.3f}')

    if divisible_by(step, SAMPLE_EVERY):
        image = ema_model.generate_modality_only(batch_size = 64)

        save_image(
            rearrange(image, '(gh gw) c h w -> c (gh h) (gw w)', gh = 8).detach().cpu(),
            str(results_folder / f'{step}.png')
        )
