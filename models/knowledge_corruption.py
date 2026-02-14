import torch


def corrupt_knowledge(knowledge, regime="clean", seed=42):
    """
    Corrupt knowledge tensors for evaluation robustness testing.

    Args:
        knowledge: torch.Tensor of shape [bs, D] or [bs, 1, D]
        regime: "clean" | "noisy_<sigma>" (e.g. "noisy_0.1") | "permuted"
        seed: random seed for reproducibility

    Returns:
        Corrupted knowledge tensor (same shape as input)
    """
    if regime == "clean":
        return knowledge

    if not isinstance(knowledge, torch.Tensor) or not knowledge.is_floating_point():
        raise ValueError(
            f"Corruption regime '{regime}' requires a float tensor, "
            f"got {type(knowledge)}"
        )

    generator = torch.Generator(device=knowledge.device)
    generator.manual_seed(seed)

    if regime.startswith("noisy"):
        sigma = float(regime.split("_", 1)[1])
        flat = knowledge.reshape(knowledge.shape[0], -1)  # [bs, D]
        feat_range = flat.max(dim=0).values - flat.min(dim=0).values  # [D]
        feat_range = feat_range.clamp(min=1e-6)
        noise = torch.randn(flat.shape, generator=generator, device=knowledge.device)
        flat_corrupted = flat + sigma * noise * feat_range
        return flat_corrupted.reshape(knowledge.shape)

    if regime == "permuted":
        perm = torch.randperm(knowledge.shape[0], generator=generator,
                              device=knowledge.device)
        return knowledge[perm]

    raise ValueError(f"Unknown corruption regime: '{regime}'")


if __name__ == "__main__":
    torch.manual_seed(0)
    k = torch.randn(8, 2)  # [bs=8, D=2]

    # clean passthrough
    assert torch.equal(corrupt_knowledge(k, "clean"), k)

    # noisy: deterministic + same shape
    n1 = corrupt_knowledge(k, "noisy_0.1", seed=7)
    n2 = corrupt_knowledge(k, "noisy_0.1", seed=7)
    assert torch.equal(n1, n2), "noisy not deterministic"
    assert n1.shape == k.shape
    assert not torch.equal(n1, k), "noisy should differ from clean"

    # permuted: deterministic + same shape + same elements
    p1 = corrupt_knowledge(k, "permuted", seed=7)
    p2 = corrupt_knowledge(k, "permuted", seed=7)
    assert torch.equal(p1, p2), "permuted not deterministic"
    assert p1.shape == k.shape
    assert set(range(8)) == set(
        [i for i in range(8) if any(torch.equal(p1[j], k[i]) for j in range(8))]
    )

    # 3D input
    k3 = torch.randn(4, 1, 3)
    assert corrupt_knowledge(k3, "noisy_0.3", seed=0).shape == k3.shape
    assert corrupt_knowledge(k3, "permuted", seed=0).shape == k3.shape

    # non-tensor raises
    try:
        corrupt_knowledge(["text"], "noisy_0.1")
        assert False, "should have raised"
    except ValueError:
        pass

    print("ALL PASS")