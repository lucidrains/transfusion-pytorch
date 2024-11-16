import torch
from torch import randint, randn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from transfusion_pytorch import Transfusion, print_modality_sample

def divisible_by(num, den):
    return (num % den) == 0

model = Transfusion(
    num_text_tokens = 8,
    dim_latent = 16,
    modality_default_shape = (2,),
    transformer = dict(
        dim = 64,
        depth = 1,
        dim_head = 8,
        heads = 2
    )
).cuda()

class MockDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return torch.ones((1,)).long(), randn(2, 16)

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

def collate_fn(data):
    data = [*map(list, data)]
    return data

mock_dataset = MockDataset()

dataloader = DataLoader(mock_dataset, batch_size = 4, collate_fn = collate_fn)
iter_dl = cycle(dataloader)

optimizer = Adam(model.parameters(), lr = 3e-4)

# train loop

for step in range(1, 10_000 + 1):

    loss = model(next(iter_dl))
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()
    optimizer.zero_grad()

    print(f'{step}: {loss.item():.3f}')

    # eval

    if divisible_by(step, 100):
        one_multimodal_sample = model.sample()
        print_modality_sample(one_multimodal_sample)
