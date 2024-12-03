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

from transfusion_pytorch.transfusion import Transfusion, print_modality_sample

rmtree('./results', ignore_errors = True)
results_folder = Path('./results')
results_folder.mkdir(exist_ok = True, parents = True)

# constants

IMAGE_AFTER_TEXT = True   # False for captioning, True for text-to-image
USE_PROMPT = False        # whether to use prompting, or synthesize from start token 
NUM_TRAIN_STEPS = 20_000
SAMPLE_EVERY = 250
CHANNEL_FIRST = True

# functions

def divisible_by(num, den):
    return (num % den) == 0

# encoder / decoder

class Encoder(Module):
    def forward(self, x):
        x = rearrange(x, '... 1 (h p1) (w p2) -> ... h w (p1 p2)', p1 = 2, p2 = 2)

        if CHANNEL_FIRST:
            x = rearrange(x, 'b ... d -> b d ...')

        return x * 2 - 1

class Decoder(Module):
    def forward(self, x):

        if CHANNEL_FIRST:
            x = rearrange(x, 'b d ... -> b ... d')

        x = rearrange(x, '... h w (p1 p2) -> ... 1 (h p1) (w p2)', p1 = 2, p2 = 2)
        return ((x + 1) * 0.5).clamp(min = 0., max = 1.)

model = Transfusion(
    num_text_tokens = 10,
    dim_latent = 4,
    modality_default_shape = (14, 14),
    modality_encoder = Encoder(),
    modality_decoder = Decoder(),
    add_pos_emb = True,
    modality_num_dim = 2,
    channel_first_latent = CHANNEL_FIRST,
    transformer = dict(
        dim = 64,
        depth = 4,
        dim_head = 32,
        heads = 8,
    )
).cuda()

ema_model = model.create_ema()

class MnistDataset(Dataset):
    def __init__(self):
        self.mnist = torchvision.datasets.MNIST(
            './data/mnist',
            download = True
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        pil, labels = self.mnist[idx]
        digit_tensor = T.PILToTensor()(pil)
        output =  tensor(labels), (digit_tensor / 255).float()

        if IMAGE_AFTER_TEXT:
            return output

        first, second = output
        return second, first

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

def collate_fn(data):
    data = [*map(list, data)]
    return data

dataset = MnistDataset()
dataloader = model.create_dataloader(dataset, batch_size = 16, shuffle = True)

iter_dl = cycle(dataloader)

optimizer = Adam(model.parameters(), lr = 3e-4)

# train loop

for step in range(1, NUM_TRAIN_STEPS + 1):
    model.train()

    loss = model(next(iter_dl))
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()
    optimizer.zero_grad()

    ema_model.update()

    print(f'{step}: {loss.item():.3f}')

    # eval

    if divisible_by(step, SAMPLE_EVERY):

        if not USE_PROMPT:
            # sampling from start to finish

            one_multimodal_sample = ema_model.sample(max_length = 384)
        else:
            # sampling using prompt
            # which differs depending on which comes first, text or images

            if IMAGE_AFTER_TEXT:

                text_label = torch.randint(0, 10, ()).cuda()
                one_multimodal_sample = ema_model.sample(prompt = text_label, max_length = 384)

            else:

                rand_batch = next(iter_dl)
                rand_image = rand_batch[0][0]

                one_multimodal_sample = ema_model.sample(prompt = rand_image, max_length = 384)

        # make sure modality sample overall order of modalities look correct

        print_modality_sample(one_multimodal_sample)

        if len(one_multimodal_sample) < 2:
            continue

        if IMAGE_AFTER_TEXT:
            maybe_label, maybe_image, *_ = one_multimodal_sample
        else:
            _, maybe_image, maybe_label = one_multimodal_sample

        filename = f'{step}.{maybe_label[1].item()}.png'

        save_image(
            maybe_image[1].cpu(),
            str(results_folder / filename),
        )
