import time

import pytest

import torch
from torch import nn, randint, randn, tensor

from transfusion_pytorch.transfusion import Transfusion
from transfusion_pytorch.modality_processing import (
    PROCESSING_STRATEGIES,
    ROUTING_CANDIDATES,
    DEFAULT_PROCESSING_STRATEGY,
    get_processing_strategy,
    assert_strategies_equivalent,
    structure_signature,
    StrategyRouter,
    ROUTER,
    process_modality_batch_auto
)

def build_model(**overrides):
    kwargs = dict(
        num_text_tokens = 8,
        dim_latent = 16,
        modality_default_shape = (4, 4),
        model_output_clean = False,
        transformer = dict(
            dim = 16,
            depth = 1,
            use_flex_attn = False
        )
    )
    kwargs.update(overrides)
    return Transfusion(**kwargs).eval()

def make_batch(batch = 2, text_len = 7, image_shape = (4, 4), num_images = 1, modality_type = 0):
    sample = [randint(0, 8, (text_len,))]

    for _ in range(num_images):
        sample.append((modality_type, randn(*image_shape, 16)))
        sample.append(randint(0, 8, (3,)))

    return [sample for _ in range(batch)]

# registry and defaults

def test_auto_is_default_and_registered():
    assert DEFAULT_PROCESSING_STRATEGY == 'auto'
    assert 'auto' in PROCESSING_STRATEGIES
    assert callable(PROCESSING_STRATEGIES['auto'])

    model = build_model()
    assert model.modality_processing == 'auto'

def test_routing_candidates_exclude_naive():
    assert 'naive' not in ROUTING_CANDIDATES
    assert all(candidate in PROCESSING_STRATEGIES for candidate in ROUTING_CANDIDATES)

def test_explicit_strategy_override():
    for strategy in ('naive', 'grouped', 'flat', 'hybrid', 'auto'):
        model = build_model(modality_processing = strategy)
        assert model.modality_processing == strategy

def test_unknown_strategy_raises():
    with pytest.raises(AssertionError):
        build_model(modality_processing = 'not-a-strategy')

def test_get_processing_strategy():
    assert callable(get_processing_strategy('auto'))
    assert get_processing_strategy('auto') is process_modality_batch_auto

    with pytest.raises(AssertionError):
        get_processing_strategy('not-a-strategy')

# structure signature

def test_structure_signature_deterministic_and_content_agnostic():
    model = build_model()

    batch_a = make_batch(text_len = 7, image_shape = (4, 4), num_images = 2)
    batch_b = make_batch(text_len = 7, image_shape = (4, 4), num_images = 2) # same structure, different token values

    times = tensor([[0.5, 0.9], [0.5, 0.9]])

    key_a = structure_signature(batch_a, model, need_axial_pos_emb = False, return_loss = True, return_embed = False)
    key_b = structure_signature(batch_b, model, need_axial_pos_emb = False, return_loss = True, return_embed = False)

    assert key_a == key_b
    assert key_a == structure_signature(batch_a, model, need_axial_pos_emb = False, return_loss = True, return_embed = False) # repeated calls stable

def test_structure_signature_distinguishes_batch_structure():
    model = build_model()

    base = structure_signature(make_batch(num_images = 1), model, need_axial_pos_emb = False, return_loss = True, return_embed = False)

    different = {
        'different shape': structure_signature(make_batch(image_shape = (5, 5)), model, need_axial_pos_emb = False, return_loss = True, return_embed = False),
        'different count': structure_signature(make_batch(num_images = 2), model, need_axial_pos_emb = False, return_loss = True, return_embed = False),
        'different batch size': structure_signature(make_batch(batch = 3), model, need_axial_pos_emb = False, return_loss = True, return_embed = False),
        'different modality type': structure_signature(make_batch(modality_type = 1), build_model(dim_latent = (16, 16), modality_default_shape = ((4, 4), (4, 4))), need_axial_pos_emb = False, return_loss = True, return_embed = False),
    }

    for label, key in different.items():
        assert key != base, f'signature failed to distinguish: {label}'

def test_structure_signature_distinguishes_flags():
    model = build_model()
    batch = make_batch()

    base = structure_signature(batch, model, need_axial_pos_emb = False, return_loss = True, return_embed = False)

    assert structure_signature(batch, model, need_axial_pos_emb = True, return_loss = True, return_embed = False) != base
    assert structure_signature(batch, model, need_axial_pos_emb = False, return_loss = False, return_embed = True) != base
    assert structure_signature(batch, model, need_axial_pos_emb = False, return_loss = True, return_embed = True) != base

def test_structure_signature_text_only():
    model = build_model()
    batch = [[randint(0, 8, (7,))] for _ in range(2)]

    key = structure_signature(batch, model, need_axial_pos_emb = False, return_loss = True, return_embed = False)

    assert key[-1] == () # no modalities in the structure
    assert key[3] > 0    # but text tokens are counted

def test_structure_signature_bare_modality_and_scalar_tokens():
    # bare float tensors are treated as type 0 modalities, scalar ints as text (matching the scan)

    model = build_model()

    bare = [[randint(0, 8, (7,)), randn(4, 4, 16), randint(0, 8, (3,))] for _ in range(2)]
    explicit = [[randint(0, 8, (7,)), (0, randn(4, 4, 16)), randint(0, 8, (3,))] for _ in range(2)]

    key_bare = structure_signature(bare, model, need_axial_pos_emb = False, return_loss = True, return_embed = False)
    key_explicit = structure_signature(explicit, model, need_axial_pos_emb = False, return_loss = True, return_embed = False)

    assert key_bare == key_explicit

    scalar = [[randint(0, 8, (7,)), randint(0, 8, ()), (0, randn(4, 4, 16)), randint(0, 8, (3,))] for _ in range(2)]
    key_scalar = structure_signature(scalar, model, need_axial_pos_emb = False, return_loss = True, return_embed = False)

    assert key_scalar[3] == key_explicit[3] + 2 # the two scalar tokens count as one text token each

# router mechanics

def test_router_picks_fastest(monkeypatch):
    # replace every candidate with a function that sleeps a controlled duration -
    # the router must return the fastest one

    sleeps = {'grouped': 0.05, 'flat': 0.01, 'hybrid': 0.1}

    def make_slow(sleep):
        def slow_strategy(*args, **kwargs):
            time.sleep(sleep)
        return slow_strategy

    monkeypatch.setitem(PROCESSING_STRATEGIES, 'grouped', make_slow(sleeps['grouped']))
    monkeypatch.setitem(PROCESSING_STRATEGIES, 'flat', make_slow(sleeps['flat']))
    monkeypatch.setitem(PROCESSING_STRATEGIES, 'hybrid', make_slow(sleeps['hybrid']))

    router = StrategyRouter(warmup = 0, iters = 1)

    model = build_model()
    batch = make_batch(num_images = 2)

    chosen = router.route(batch, tensor([[0.5, 0.5], [0.5, 0.5]]), model, need_axial_pos_emb = False, return_loss = True, return_embed = False)

    assert chosen == 'flat'

def test_router_measure_called_once_per_structure(monkeypatch):
    router = StrategyRouter(warmup = 0, iters = 1)

    calls = []
    original_measure = router.measure

    def counting_measure(*args, **kwargs):
        calls.append(kwargs)
        return original_measure(*args, **kwargs)

    monkeypatch.setattr(router, 'measure', counting_measure)

    model = build_model()
    batch = make_batch(num_images = 2)
    times = tensor([[0.5, 0.5], [0.5, 0.5]])

    kwargs = dict(need_axial_pos_emb = False, return_loss = True, return_embed = False)

    first = router.route(batch, times, model, **kwargs)
    second = router.route(batch, times, model, **kwargs)
    third = router.route(make_batch(num_images = 3), tensor([[0.5] * 3, [0.5] * 3]), model, **kwargs)

    assert len(calls) == 2 # measured for the first structure, reused for the second call, measured again for the new structure
    assert first == second

def test_router_skips_measurement_for_text_only(monkeypatch):
    # text only batches are routed without timing anything

    router = StrategyRouter()

    for name in ROUTING_CANDIDATES:
        monkeypatch.setitem(PROCESSING_STRATEGIES, name, lambda *args, **kwargs: pytest.fail(f'{name} should not be measured for a text-only batch'))

    model = build_model()
    batch = [[randint(0, 8, (7,))] for _ in range(2)]

    chosen = router.route(batch, torch.ones(2, 1), model, need_axial_pos_emb = False, return_loss = True, return_embed = False)

    assert chosen == 'hybrid'
    assert router.cache  # but the decision is still cached

def test_router_cache_eviction():
    router = StrategyRouter(warmup = 0, iters = 1, max_cache = 2)

    model = build_model()
    times = tensor([[0.5], [0.5]])

    kwargs = dict(need_axial_pos_emb = False, return_loss = True, return_embed = False)

    shapes = [(4, 4), (5, 5), (6, 6)]

    for shape in shapes:
        router.route(make_batch(image_shape = shape), times, model, **kwargs)

    assert len(router.cache) <= 2

    oldest_shape = shapes[0]
    assert router.route(make_batch(image_shape = oldest_shape), times, model, **kwargs) is not None # re-routing an evicted structure just re-measures

def test_router_clear():
    router = StrategyRouter(warmup = 0, iters = 1)

    model = build_model()
    batch = make_batch(num_images = 2)
    kwargs = dict(need_axial_pos_emb = False, return_loss = True, return_embed = False)

    router.route(batch, tensor([[0.5, 0.5], [0.5, 0.5]]), model, **kwargs)
    assert router.cache

    router.clear()
    assert not router.cache

def test_router_restricted_candidates(monkeypatch):
    # only the given candidates are timed

    calls = []

    def tracked(name):
        def fn(*args, **kwargs):
            calls.append(name)
        return fn

    for name in ROUTING_CANDIDATES:
        monkeypatch.setitem(PROCESSING_STRATEGIES, name, tracked(name))

    router = StrategyRouter(candidates = ('grouped',), warmup = 0, iters = 1)

    model = build_model()
    batch = make_batch(num_images = 2)

    chosen = router.route(batch, tensor([[0.5, 0.5], [0.5, 0.5]]), model, need_axial_pos_emb = False, return_loss = True, return_embed = False)

    assert chosen == 'grouped'
    assert set(calls) == {'grouped'}

def test_global_router_and_auto_strategy_are_wired():
    assert isinstance(ROUTER, StrategyRouter)
    assert PROCESSING_STRATEGIES['auto'] is process_modality_batch_auto

# auto strategy correctness - equivalent to every explicit strategy on every profile

def test_auto_equivalence_all_profiles():
    # the auto strategy must produce outputs identical to every explicit strategy,
    # regardless of which one the router happens to pick

    profiles = []

    model = build_model()
    profiles.append((model, make_batch(num_images = 2), torch.ones(2, 2)))

    model = build_model(dim_latent = 16, modality_default_shape = (4, 4), channel_first_latent = True)
    profiles.append((model, [
        [randint(0, 8, (7,)), (0, randn(16, 4, 4)), randint(0, 8, (3,)), (0, randn(16, 5, 6))],
        [randint(0, 8, (2,)), (0, randn(16, 6, 6)), (0, randn(16, 4, 4)), randint(0, 8, (9,))]
    ], torch.ones(2, 2)))

    model = build_model(modality_default_shape = ())
    profiles.append((model, [
        [randint(0, 8, (7,)), (0, randn(16)), randint(0, 8, (3,)), (0, randn(16))],
        [randint(0, 8, (2,)), (0, randn(16)), randint(0, 8, (9,))]
    ], torch.ones(2, 2)))

    model = build_model(add_pos_emb = True, modality_num_dim = 2)
    profiles.append((model, make_batch(num_images = 2), torch.ones(2, 2)))

    model = build_model(dim_latent = (16, 8), modality_default_shape = ((4, 4), (8,)), model_output_clean = True)
    profiles.append((model, [
        [randint(0, 8, (7,)), (0, randn(4, 4, 16)), randint(0, 8, (3,)), (1, randn(8, 8))],
        [randint(0, 8, (2,)), (0, randn(4, 4, 16)), (1, randn(8, 8)), randint(0, 8, (9,))]
    ], torch.ones(2, 2)))

    for model, batch, times in profiles:
        assert_strategies_equivalent(model, batch, times, need_axial_pos_emb = False, return_loss = True, return_embed = False)
        assert_strategies_equivalent(model, batch, times, need_axial_pos_emb = False, return_loss = False, return_embed = True)

        if model.modality_default_shape != ():
            assert_strategies_equivalent(model, batch, times, need_axial_pos_emb = True, return_loss = True, return_embed = False)

def test_auto_direct_equivalence(monkeypatch):
    # direct comparison of the auto strategy against the router's chosen strategy

    router = StrategyRouter(warmup = 0, iters = 1)

    monkeypatch.setattr('transfusion_pytorch.modality_processing.ROUTER', router)

    model = build_model()
    batch = make_batch(num_images = 2)
    times = tensor([[0.5, 0.5], [0.5, 0.5]])

    kwargs = dict(need_axial_pos_emb = False, return_loss = True, return_embed = False)

    out_auto = process_modality_batch_auto(batch, times, model, **kwargs)

    chosen = router.route(batch, times, model, **kwargs)
    out_chosen = get_processing_strategy(chosen)(batch, times, model, **kwargs)

    # same outputs (equivalence is verified with deterministic noise)

    from unittest import mock

    with mock.patch('torch.randn_like', side_effect = lambda t: torch.zeros_like(t)):
        out_auto = process_modality_batch_auto(batch, times, model, **kwargs)
        out_chosen = get_processing_strategy(chosen)(batch, times, model, **kwargs)

    assert torch.equal(out_auto.text, out_chosen.text)
    assert torch.equal(out_auto.modality_tokens, out_chosen.modality_tokens)
    assert out_auto.modality_positions == out_chosen.modality_positions

# auto strategy end to end through the model

def test_auto_end_to_end_training():
    model = Transfusion(
        num_text_tokens = 12,
        dim_latent = 384,
        channel_first_latent = True,
        modality_default_shape = (4, 4),
        add_pos_emb = True,
        modality_encoder = nn.Conv2d(3, 384, 3, padding = 1),
        transformer = dict(
            dim = 512,
            depth = 2
        ),
        modality_processing = 'auto'
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    for _ in range(3):
        batch = [
            [randint(0, 12, (16,)), randn(3, 8, 8), randint(0, 12, (8,)), randn(3, 7, 7)],
            [randint(0, 12, (16,)), randn(3, 8, 5), randint(0, 12, (5,)), randn(3, 2, 16), randint(0, 12, (9,))]
        ]

        loss = model(batch)

        assert torch.isfinite(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_auto_text_only_training():
    # text only training goes through the `is_text_only` forward path (a single int tensor),
    # which bypasses the modality processing entirely

    model = build_model()

    batch = randint(0, 8, (2, 7))

    loss = model(batch)

    assert torch.isfinite(loss)

def test_auto_text_only_router_shortcut(monkeypatch):
    # a batch of text only samples routed directly through the auto strategy - the router
    # must pick a strategy without measuring anything

    router = StrategyRouter()

    def fail_measure(*args, **kwargs):
        pytest.fail('should not measure strategies for a text-only batch')

    monkeypatch.setattr(router, 'measure', fail_measure)
    monkeypatch.setattr('transfusion_pytorch.modality_processing.ROUTER', router)

    model = build_model()
    batch = [[randint(0, 8, (7,))] for _ in range(2)]

    out = process_modality_batch_auto(batch, torch.ones(2, 1), model, need_axial_pos_emb = False, return_loss = True, return_embed = False)

    assert torch.equal(out.text, torch.stack([sample[0] for sample in batch])) # the text is assembled as-is
    assert torch.equal(out.modality_tokens, torch.zeros(2, 7, 16)) # no modality tokens
    assert out.modality_positions == [[], []]
    assert out.total_tokens == 14

def test_auto_decoding_path():
    # the return_embed path used when sampling modality tokens

    model = build_model()

    batch = make_batch(num_images = 2)

    embed, get_pred_flows = model(batch, return_embed = True, return_loss = False)

    assert torch.isfinite(embed).all()

    for modality_type, pred_flows in get_pred_flows.items():
        for pred_flow in pred_flows:
            assert torch.isfinite(pred_flow(embed)).all()

def test_auto_across_strategy_explicit_models_train_identically():
    # with the same seed, every strategy (explicit and auto) must produce identical training

    losses = {}

    for strategy in ('naive', 'grouped', 'flat', 'hybrid', 'auto'):
        torch.manual_seed(1)
        model = build_model(modality_processing = strategy)

        batch = make_batch(num_images = 2)
        times = torch.ones(2, 2)

        from unittest import mock

        with mock.patch('torch.randn_like', side_effect = lambda t: torch.zeros_like(t)):
            loss = model(batch)

        losses[strategy] = loss.item()

    assert len(set(losses.values())) == 1, losses

# unet style (downsampling) encoders - token lengths must be derived from the *projected*
# tokens, not the raw input shapes, matching the reference `naive` strategy

def build_unet_model(**overrides):
    kwargs = dict(
        num_text_tokens = 10,
        dim_latent = 4,
        modality_default_shape = (14, 14),
        pre_post_transformer_enc_dec = (
            nn.Conv2d(4, 16, 3, 2, 1),
            nn.ConvTranspose2d(16, 4, 3, 2, 1, output_padding = 1),
        ),
        channel_first_latent = True,
        transformer = dict(dim = 16, depth = 1, use_flex_attn = False)
    )
    kwargs.update(overrides)
    return Transfusion(**kwargs).eval()

def make_unet_batch():
    return [
        [randint(0, 10, (8,)), (0, randn(4, 14, 14)), randint(0, 10, (4,)), (0, randn(4, 12, 12)), randint(0, 10, (4,))],
        [randint(0, 10, (6,)), (0, randn(4, 14, 14)), randint(0, 10, (5,)), (0, randn(4, 16, 16))]
    ]

def test_unet_encoder_equivalence_all_strategies():
    # the stride-2 conv encoder downsamples 14x14 -> 7x7 - every strategy must derive the
    # token lengths from the projected tokens, matching `naive`

    model = build_unet_model()
    batch = make_unet_batch()
    times = torch.ones(2, 2)

    assert_strategies_equivalent(model, batch, times, need_axial_pos_emb = False, return_loss = True, return_embed = False)
    assert_strategies_equivalent(model, batch, times, need_axial_pos_emb = False, return_loss = False, return_embed = True)

def test_unet_encoder_equivalence_with_pos_emb():
    model = build_unet_model(add_pos_emb = True, modality_num_dim = 2)
    batch = make_unet_batch()
    times = torch.ones(2, 2)

    assert_strategies_equivalent(model, batch, times, need_axial_pos_emb = True, return_loss = True, return_embed = False)

def test_unet_encoder_positions_use_projected_lengths():
    # 14x14 with stride 2 conv -> 7x7 projected tokens, so the recorded length is 49 not 196

    model = build_unet_model()
    batch = make_unet_batch()
    times = torch.ones(2, 2)

    out = get_processing_strategy('grouped')(batch, times, model, need_axial_pos_emb = False, return_loss = True, return_embed = False)

    assert out.modality_positions[0] == [(0, 13, 49), (0, 72, 36)]
    assert out.modality_positions[1] == [(0, 11, 49), (0, 71, 64)]
    assert out.total_tokens == 249

def test_unet_encoder_flat_falls_back_for_nonlinear_projection():
    from transfusion_pytorch.modality_processing import latent_projection_is_linear

    model = build_unet_model()

    assert not latent_projection_is_linear(model.get_modality_info(0).latent_to_model)

    batch = make_unet_batch()
    times = torch.ones(2, 2)

    from unittest import mock

    with mock.patch('torch.randn_like', side_effect = lambda t: torch.zeros_like(t)):
        out_flat = get_processing_strategy('flat')(batch, times, model, need_axial_pos_emb = False, return_loss = True, return_embed = False)
        out_naive = get_processing_strategy('naive')(batch, times, model, need_axial_pos_emb = False, return_loss = True, return_embed = False)

    assert torch.equal(out_flat.modality_tokens, out_naive.modality_tokens)
    assert out_flat.modality_positions == out_naive.modality_positions

def test_unet_encoder_end_to_end_auto_training():
    torch.manual_seed(0)

    model = build_unet_model(modality_processing = 'auto').train()

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    batch = make_unet_batch()

    for _ in range(3):
        loss = model(batch)

        assert torch.isfinite(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
