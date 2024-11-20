from shutil import rmtree
from pathlib import Path

import torch
from torch import nn, tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from einops import rearrange
from einops.layers.torch import Rearrange

import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from tqdm import tqdm

from transfusion_pytorch import Transfusion, print_modality_sample

rmtree('./results', ignore_errors = True)
results_folder = Path('./results')
results_folder.mkdir(exist_ok = True, parents = True)

# functions

def divisible_by(num, den):
    return (num % den) == 0

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

# dataset

class MnistDataset(Dataset):
    def __init__(self):
        self.mnist = torchvision.datasets.MNIST(
            './data/mnist',
            download = True
        )   

        self.transform = T.Compose([
            T.PILToTensor(),
            T.RandomResizedCrop((28, 28), scale = (0.8, 1.))
        ])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        pil, labels = self.mnist[idx]
        digit_tensor = self.transform(pil)
        return tensor(labels), (digit_tensor / 255).float()

dataset = MnistDataset()

# contrived encoder / decoder with layernorm at bottleneck

autoencoder_train_steps = 15000
dim_latent = 16

class Normalize(Module):
    def forward(self, x):
        return F.normalize(x, dim = -1)

encoder = nn.Sequential(
    nn.Conv2d(1, 4, 3, padding = 1),
    nn.Conv2d(4, 8, 4, 2, 1),
    nn.ReLU(),
    nn.Dropout(0.05),
    nn.Conv2d(8, dim_latent, 1),
    Rearrange('b d ... -> b ... d'),
    Normalize()
).cuda()

decoder = nn.Sequential(
    Rearrange('b ... d -> b d ...'),
    nn.Conv2d(dim_latent, 8, 1),
    nn.ReLU(),
    nn.ConvTranspose2d(8, 4, 4, 2, 1),
    nn.Conv2d(4, 1, 3, padding = 1),
).cuda()

# train autoencoder

autoencoder_optimizer = Adam([*encoder.parameters(), *decoder.parameters()], lr = 3e-4)
autoencoder_dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

autoencoder_iter_dl = cycle(autoencoder_dataloader)

print('training autoencoder')

with tqdm(total = autoencoder_train_steps) as pbar:
    for _ in range(autoencoder_train_steps):
        _, images = next(autoencoder_iter_dl)
        images = images.cuda()

        latents = encoder(images)
        latents = latents.lerp(torch.randn_like(latents), torch.rand_like(latents) * 0.2) # add a bit of noise to latents
        reconstructed = decoder(latents)

        loss = F.mse_loss(images, reconstructed)

        loss.backward()

        pbar.set_description(f'loss: {loss.item():.5f}')

        autoencoder_optimizer.step()
        autoencoder_optimizer.zero_grad()

        pbar.update()

# transfusion

model = Transfusion(
    num_text_tokens = 10,
    dim_latent = dim_latent,
    modality_default_shape = (14, 14),
    modality_encoder = encoder,
    modality_decoder = decoder,
    add_pos_emb = True,
    modality_num_dim = 2,
    transformer = dict(
        dim = 64,
        depth = 4,
        dim_head = 32,
        heads = 8
    )
).cuda()

# training transfusion

dataloader = model.create_dataloader(dataset, batch_size = 16, collate_fn = collate_fn, shuffle = True)
iter_dl = cycle(dataloader)

optimizer = Adam(model.parameters_without_encoder_decoder(), lr = 3e-4)

# train loop

transfusion_train_steps = 25_000

print('training transfusion with autoencoder')

with tqdm(total = transfusion_train_steps) as pbar:
    for index in range(transfusion_train_steps):
        step = index + 1

        model.train()

        loss = model(next(iter_dl))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description(f'loss: {loss.item():.3f}')

        pbar.update()

        # eval

        if divisible_by(step, 500):
            one_multimodal_sample = model.sample(max_length = 10)

            print_modality_sample(one_multimodal_sample)

            if len(one_multimodal_sample) < 2:
                continue

            maybe_label, maybe_image, *_ = one_multimodal_sample

            filename = f'{step}.{maybe_label[1].item()}.png'

            save_image(
                maybe_image[1].cpu().clamp(min = 0., max = 1.),
                str(results_folder / filename),
            )
