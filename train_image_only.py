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

rmtree('./results', ignore_errors = True)
results_folder = Path('./results')
results_folder.mkdir(exist_ok = True, parents = True)

# functions

def divisible_by(num, den):
    return (num % den) == 0

# encoder / decoder

class Encoder(Module):
    def forward(self, x):
        x = rearrange(x, '... 1 (h p1) (w p2) -> ... h w (p1 p2)', p1 = 2, p2 = 2)
        return x * 2 - 1

class Decoder(Module):
    def forward(self, x):
        x = rearrange(x, '... h w (p1 p2) -> ... 1 (h p1) (w p2)', p1 = 2, p2 = 2, h = 14)
        return ((x + 1) * 0.5).clamp(min = 0., max = 1.)

model = Transfusion(
    num_text_tokens = 10,
    dim_latent = 4,
    modality_default_shape = (14, 14),
    modality_encoder = Encoder(),
    modality_decoder = Decoder(),
    add_pos_emb = True,
    modality_num_dim = 2,
    velocity_consistency_loss_weight = 0.1,
    transformer = dict(
        dim = 64,
        depth = 4,
        dim_head = 32,
        heads = 8
    )
).cuda()

ema_model = model.create_ema()

class MnistDataset(Dataset):
    def __init__(self):
        self.mnist = torchvision.datasets.MNIST(
            './data',
            download = True
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        pil, labels = self.mnist[idx]
        digit_tensor = T.PILToTensor()(pil)
        return (digit_tensor / 255).float()

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

dataset = MnistDataset()

dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
iter_dl = cycle(dataloader)

optimizer = Adam(model.parameters(), lr = 8e-4)

# train loop

for step in range(1, 100_000 + 1):

    loss = model(next(iter_dl), velocity_consistency_ema_model = ema_model)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()
    optimizer.zero_grad()

    ema_model.update()

    print(f'{step}: {loss.item():.3f}')

    if divisible_by(step, 500):
        image = ema_model.generate_modality_only(batch_size = 64)

        save_image(
            rearrange(image, '(gh gw) 1 h w -> 1 (gh h) (gw w)', gh = 8).detach().cpu(),
            str(results_folder / f'{step}.png')
        )
