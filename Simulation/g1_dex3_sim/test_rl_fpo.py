# TL;DR: CPU-Test: FPO-Proxy mit gespeicherten Seeds ist reproduzierbar (Ratio == 1, Review S2).
"""python -m pytest Simulation/g1_dex3_sim/test_rl_fpo.py

Ein Attrappen-Modell zieht Rauschen und t wie Gr00tN1d6ActionHead.forward (randn auf dem
Geraet, Beta-Stichprobe mit CPU-Parametern). Ohne Seeds unterscheiden sich zwei Aufrufe,
mit denselben Seeds sind sie identisch, und der globale Zufallsstrom bleibt unberührt.
"""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from rl_finetune import compute_gae, fpo_logprob_proxy  # noqa: E402


class _FakeFlowModel:
    def __init__(self):
        self.beta = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
        self.w = torch.nn.Parameter(torch.tensor(0.7))

    def forward(self, inputs):
        actions = inputs["action"]
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = ((1 - self.beta.sample([actions.shape[0]])) * 0.999)[:, None, None]
        noisy = (1 - t) * noise + t * actions
        pred = self.w * noisy
        loss = (pred - (actions - noise)) ** 2 * inputs["action_mask"]
        return {"action_loss": loss, "action_mask": inputs["action_mask"]}


def _inputs(b=4):
    g = torch.Generator().manual_seed(0)
    a = torch.randn(b, 16, 28, generator=g)
    return {"action": a, "action_mask": torch.ones_like(a)}


def test_without_seeds_old_and_new_differ():
    m, inp = _FakeFlowModel(), _inputs()
    a = fpo_logprob_proxy(m, inp, 4, torch)
    b = fpo_logprob_proxy(m, inp, 4, torch)
    assert not torch.allclose(a, b)


def test_same_seeds_give_ratio_one_and_leave_global_rng_alone():
    m, inp = _FakeFlowModel(), _inputs()
    seeds = [11, 22, 33, 44]
    torch.manual_seed(123)
    before = torch.get_rng_state()
    old = fpo_logprob_proxy(m, inp, 4, torch, seeds=seeds)
    assert torch.equal(torch.get_rng_state(), before)
    new = fpo_logprob_proxy(m, inp, 4, torch, seeds=seeds)
    ratio = torch.exp(new - old)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=0, rtol=0)


def test_seeded_proxy_still_reacts_to_parameter_changes():
    m, inp = _FakeFlowModel(), _inputs()
    seeds = [1, 2]
    old = fpo_logprob_proxy(m, inp, 2, torch, seeds=seeds)
    with torch.no_grad():
        m.w += 0.1
    new = fpo_logprob_proxy(m, inp, 2, torch, seeds=seeds)
    assert not torch.allclose(old, new)


def test_gae_does_not_bootstrap_across_done():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.zeros(3, 1)
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    adv, ret = compute_gae(rewards, values, dones, torch.tensor([10.0]), 0.5, 1.0, torch)
    # Schritt 1 ist terminal: sein Return ist nur sein eigener Reward.
    assert float(ret[1, 0]) == pytest.approx(1.0)
    assert float(ret[0, 0]) == pytest.approx(1.0 + 0.5 * 1.0)
    assert float(ret[2, 0]) == pytest.approx(1.0 + 0.5 * 10.0)
