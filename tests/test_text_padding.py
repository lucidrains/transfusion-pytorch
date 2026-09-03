import torch
from torch import nn, tensor

from transfusion_pytorch.transfusion import Transformer, Transfusion, naive_attn_mask, text_attn_mask


class NoRotary(nn.Module):
    def forward(self, positions):
        return None


def make_text_model():
    torch.manual_seed(0)

    model = Transfusion(
        num_text_tokens = 16,
        dim_latent = 8,
        modality_default_shape = (2,),
        transformer = dict(
            dim = 16,
            depth = 2,
            dim_head = 8,
            heads = 2
        )
    ).eval()

    # Removing position features makes the per-token comparison independent
    # of where a sample was padded in the batch.
    model.rotary_emb = NoRotary()
    return model


def test_text_attn_mask_excludes_padding_keys_and_keeps_rows_valid():
    text = tensor([
        [-1, -1, 2, 3, -1],
        [4, 5, -1, -1, -1]
    ])

    mask = text_attn_mask(text)

    assert mask.shape == (2, 5, 5)
    assert mask[0, 2].tolist() == [False, False, True, False, False]
    assert mask[0, 3].tolist() == [False, False, True, True, False]
    assert mask[1, 4].tolist() == [True, True, False, False, True]
    assert mask.any(dim = -1).all()


def test_forward_text_padding_matches_individual_sequences_and_loss():
    model = make_text_model()
    padded = tensor([
        [-1, -1, 1, 2, 3],
        [-1, 4, 5, 6, 7]
    ])

    with torch.no_grad():
        padded_logits = model.forward_text(padded, return_loss = False)
        padded_loss = model.forward_text(padded, return_loss = True)

        individual_logits = []
        individual_losses = []

        for row in padded:
            valid = row[row != -1].unsqueeze(0)
            individual_logits.append(model.forward_text(valid, return_loss = False))
            individual_losses.append(model.forward_text(valid, return_loss = True))

    for index, (row, logits) in enumerate(zip(padded, individual_logits)):
        valid_positions = (row != -1).nonzero(as_tuple = True)[0]
        assert torch.allclose(
            padded_logits[index, valid_positions],
            logits[0],
            atol = 1e-5,
            rtol = 1e-5
        )

    # Cross entropy averages over non-padding labels, so combine the
    # individual losses with their respective numbers of next-token targets.
    target_counts = torch.tensor([2., 3.])
    expected_loss = (torch.stack(individual_losses) * target_counts).sum() / target_counts.sum()
    assert torch.allclose(padded_loss, expected_loss, atol = 1e-5, rtol = 1e-5)


def test_cached_forward_text_preserves_padding_mask_per_sample():
    model = make_text_model()
    padded = tensor([
        [-1, -1, 1, 2, 3],
        [-1, 4, 5, 6, 7]
    ])
    next_tokens = tensor([[8], [9]])

    with torch.no_grad():
        _, cache = model.forward_text(
            padded,
            return_loss = False,
            return_kv_cache = True
        )
        assert len(cache) == 3
        assert torch.equal(cache[2], padded != -1)

        cached_logits, next_cache = model.forward_text(
            next_tokens,
            return_loss = False,
            cache = cache,
            return_kv_cache = True
        )

        assert len(next_cache) == 3
        expected_mask = torch.cat(
            (padded != -1, torch.ones_like(next_tokens, dtype = torch.bool)),
            dim = -1
        )
        assert torch.equal(next_cache[2], expected_mask)

        for index, row in enumerate(padded):
            valid = row[row != -1]
            baseline = model.forward_text(
                torch.cat((valid, next_tokens[index]), dim = 0).unsqueeze(0),
                return_loss = False
            )
            assert torch.allclose(cached_logits[index, 0], baseline[0, -1], atol = 1e-5, rtol = 1e-5)


def test_naive_multimodal_mask_excludes_padding_keys():
    modalities = tensor([
        [[0, 2, 2]],
        [[0, 0, 2]]
    ])
    padding_mask = tensor([
        [False, False, True, True, True, True],
        [True, True, True, False, False, False]
    ])

    mask = naive_attn_mask(
        6,
        modalities,
        padding_mask = padding_mask
    )

    # Modality queries are bidirectional, but must not see another sample's
    # padded columns.  A padded query retains only its diagonal entry.
    assert not mask[0, 2:4, :2].any()
    assert not mask[0, 2:4, 4:].any()
    assert mask[0, 0].tolist() == [True, False, False, False, False, False]
    assert mask.any(dim = -1).all()


def test_transformer_multimodal_padding_matches_unpadded_samples():
    torch.manual_seed(1)
    transformer = Transformer(
        dim = 8,
        depth = 1,
        dim_head = 4,
        heads = 2
    ).eval()

    tokens = torch.randn(2, 6, 8)
    padding_mask = tensor([
        [False, False, True, True, True, True],
        [True, True, True, False, False, False]
    ])
    modalities = tensor([
        [[0, 2, 2]],
        [[0, 0, 2]]
    ])

    with torch.no_grad():
        padded = transformer(
            tokens,
            times = torch.zeros(2),
            modality_positions = modalities,
            padding_mask = padding_mask
        )

        for index in range(2):
            valid_indices = padding_mask[index].nonzero(as_tuple = True)[0]
            unpadded_tokens = tokens[index, valid_indices].unsqueeze(0)
            unpadded_modalities = modalities[index:index + 1].clone()
            unpadded_modalities[..., 1] -= valid_indices[0]

            unpadded = transformer(
                unpadded_tokens,
                times = torch.zeros(1),
                modality_positions = unpadded_modalities,
                padding_mask = torch.ones((1, valid_indices.numel()), dtype = torch.bool)
            )

            assert torch.allclose(
                padded[index, valid_indices],
                unpadded[0],
                atol = 1e-5,
                rtol = 1e-5
            )
