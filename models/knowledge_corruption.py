import torch


# abc2 knowledge layout: [bs, 3, 4]
#   3 rows = parameters (a, b, c)
#   4 cols = [3 indicator cols (one-hot) | 1 value col]
# Active rows have non-zero indicator; inactive rows are all zeros.
# Fixed absolute sigmas and clamp ranges per parameter:
_ABC2_SIGMA_SCALE = torch.tensor([2.0, 6.0, 2.0])   # sigma_abs = sigma_rel * scale
_ABC2_CLAMP_MIN = torch.tensor([-1.0, 0.0, -1.0])
_ABC2_CLAMP_MAX = torch.tensor([1.0, 6.0, 1.0])


def _is_abc2(knowledge):
    """Check if tensor looks like abc2 knowledge: [bs, 3, 4]."""
    return knowledge.ndim == 3 and knowledge.shape[1] == 3 and knowledge.shape[2] == 4


def _corrupt_abc2_noisy(knowledge, sigma_rel, seed, clip=True):
    """Add Gaussian noise to the value column of active rows only."""
    bs = knowledge.shape[0]
    out = knowledge.clone()

    generator = torch.Generator(device=knowledge.device)
    generator.manual_seed(seed)

    # Active mask: a row is active if any indicator column is non-zero
    # indicator cols are [:, :, :3], value col is [:, :, 3]
    active = (knowledge[:, :, :3].abs().sum(dim=-1) > 0)  # [bs, 3] bool

    # sigma_abs per parameter row: [3]
    sigma_abs = sigma_rel * _ABC2_SIGMA_SCALE.to(knowledge.device)

    # Generate noise for all [bs, 3] value positions
    noise = torch.randn(bs, 3, generator=generator, device=knowledge.device)
    noise = noise * sigma_abs.unsqueeze(0)  # [bs, 3]

    # Apply noise only to value column of active rows
    value_col = out[:, :, 3]  # [bs, 3]
    value_col = value_col + noise * active.float()

    if clip:
        clamp_min = _ABC2_CLAMP_MIN.to(knowledge.device).unsqueeze(0)  # [1, 3]
        clamp_max = _ABC2_CLAMP_MAX.to(knowledge.device).unsqueeze(0)  # [1, 3]
        # Only clamp active rows; inactive stay zero
        value_col = torch.where(
            active,
            value_col.clamp(min=clamp_min.expand_as(value_col),
                            max=clamp_max.expand_as(value_col)),
            value_col,
        )

    out[:, :, 3] = value_col
    return out


def corrupt_knowledge(knowledge, regime="clean", seed=42, clip=True):
    """
    Corrupt knowledge tensors for evaluation robustness testing.

    Args:
        knowledge: torch.Tensor — for abc2: [bs, 3, 4]; other shapes also supported.
        regime: "clean" | "noisy_<sigma>" (e.g. "noisy_0.1") | "permuted"
        seed: random seed for reproducibility
        clip: whether to clamp noisy values to valid ranges (abc2 only)

    Returns:
        Corrupted knowledge tensor (same shape as input, never modifies input in-place)
    """
    if regime == "clean":
        return knowledge.clone() if isinstance(knowledge, torch.Tensor) else knowledge

    if not isinstance(knowledge, torch.Tensor) or not knowledge.is_floating_point():
        raise ValueError(
            f"Corruption regime '{regime}' requires a float tensor, "
            f"got {type(knowledge)}"
        )

    if regime == "permuted":
        generator = torch.Generator(device=knowledge.device)
        generator.manual_seed(seed)
        perm = torch.randperm(knowledge.shape[0], generator=generator,
                              device=knowledge.device)
        return knowledge[perm]

    if regime.startswith("noisy"):
        sigma_rel = float(regime.split("_", 1)[1])

        if _is_abc2(knowledge):
            return _corrupt_abc2_noisy(knowledge, sigma_rel, seed, clip=clip)

        # Fallback for non-abc2 tensors: generic per-feature-range noise
        generator = torch.Generator(device=knowledge.device)
        generator.manual_seed(seed)
        flat = knowledge.reshape(knowledge.shape[0], -1)
        feat_range = flat.max(dim=0).values - flat.min(dim=0).values
        feat_range = feat_range.clamp(min=1e-6)
        noise = torch.randn(flat.shape, generator=generator, device=knowledge.device)
        flat_corrupted = flat + sigma_rel * noise * feat_range
        return flat_corrupted.reshape(knowledge.shape)

    raise ValueError(f"Unknown corruption regime: '{regime}'")


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- abc2 tests ----
    # Build a fake abc2 tensor [bs=4, 3, 4]
    # Rows: indicator (one-hot 3x3) | value
    bs = 4
    k_abc2 = torch.zeros(bs, 3, 4)
    indicator = torch.eye(3)  # [3, 3]
    values = torch.tensor([0.5, 3.0, -0.5])  # a, b, c
    for i in range(bs):
        k_abc2[i, :, :3] = indicator
        k_abc2[i, :, 3] = values
    # Mask out row 2 (c) for samples 0,1 to simulate abc2 partial reveal
    k_abc2[0, 2, :] = 0.0
    k_abc2[1, 2, :] = 0.0

    # clean: returns clone, shape preserved
    c = corrupt_knowledge(k_abc2, "clean")
    assert c.shape == k_abc2.shape, "clean shape"
    assert torch.equal(c, k_abc2), "clean should equal input"
    assert c.data_ptr() != k_abc2.data_ptr(), "clean should be a clone"

    # noisy_0.1: shape preserved, only value col of active rows changed
    n1 = corrupt_knowledge(k_abc2, "noisy_0.1", seed=7)
    n2 = corrupt_knowledge(k_abc2, "noisy_0.1", seed=7)
    assert torch.equal(n1, n2), "noisy not deterministic"
    assert n1.shape == k_abc2.shape, "noisy shape"
    # Indicator columns (first 3) must be unchanged
    assert torch.equal(n1[:, :, :3], k_abc2[:, :, :3]), "noisy changed indicator cols"
    # Inactive rows must be unchanged (all zeros)
    assert torch.equal(n1[0, 2, :], torch.zeros(4)), "noisy changed inactive row"
    assert torch.equal(n1[1, 2, :], torch.zeros(4)), "noisy changed inactive row"
    # Active rows' value col should differ
    assert not torch.equal(n1[:, 0, 3], k_abc2[:, 0, 3]), "noisy should change active values"
    # Clamp check
    assert (n1[:, 0, 3] >= -1.0).all() and (n1[:, 0, 3] <= 1.0).all(), "a out of [-1,1]"
    assert (n1[:, 1, 3] >= 0.0).all() and (n1[:, 1, 3] <= 6.0).all(), "b out of [0,6]"
    # c only active for samples 2,3
    assert (n1[2:, 2, 3] >= -1.0).all() and (n1[2:, 2, 3] <= 1.0).all(), "c out of [-1,1]"

    # noisy_0.3: same structure checks
    n3 = corrupt_knowledge(k_abc2, "noisy_0.3", seed=7)
    assert n3.shape == k_abc2.shape, "noisy_0.3 shape"
    assert torch.equal(n3[:, :, :3], k_abc2[:, :, :3]), "noisy_0.3 changed indicator cols"

    # permuted: deterministic, same shape, same elements
    p1 = corrupt_knowledge(k_abc2, "permuted", seed=7)
    p2 = corrupt_knowledge(k_abc2, "permuted", seed=7)
    assert torch.equal(p1, p2), "permuted not deterministic"
    assert p1.shape == k_abc2.shape, "permuted shape"

    # ---- generic tensor fallback tests ----
    k_flat = torch.randn(8, 2)
    nf = corrupt_knowledge(k_flat, "noisy_0.1", seed=7)
    assert nf.shape == k_flat.shape, "generic noisy shape"
    assert not torch.equal(nf, k_flat), "generic noisy should differ"
    pf = corrupt_knowledge(k_flat, "permuted", seed=7)
    assert pf.shape == k_flat.shape, "generic permuted shape"

    # non-tensor raises
    try:
        corrupt_knowledge(["text"], "noisy_0.1")
        assert False, "should have raised"
    except ValueError:
        pass

    print("ALL PASS")
