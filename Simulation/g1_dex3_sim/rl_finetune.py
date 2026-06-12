#!/usr/bin/env python3
"""
RL-Fine-tuning des GR00T-N1.6-Action-Heads auf Block-Stacking (FPO).

Status / Validierungsgrad
-------------------------
ERSTE ECHTE IMPLEMENTIERUNG. Der **Algorithmus-Kern** (FPO-Surrogat aus dem
Flow-Matching-Loss, GAE, PPO-Clipping, KL-Regularisierung gegen den BC-Checkpoint,
Optimierung NUR des Action-Heads) ist gegen die real gelesene GR00T-API verifiziert:
  - gr00t/model/gr00t_n1d6/gr00t_n1d6.py: action_head.forward(...) liefert
    "action_loss" (pro Element) — der likelihood-freie FPO-Proxy für log pi(a|s).
  - gr00t/policy/gr00t_policy.py: Processor -> collate_fn -> model.get_action /
    model.forward -> processor.decode_action (Obs->Inputs->Aktion-Pipeline).

NOCH am LIVE-Lauf zu bestaetigen (braucht RT-Core-GPU + Isaac Sim — die
Gruppe-0-Hardwareentscheidung aus docs/weiterfuehrend/reinforcement-learning-plan.md):
  1. Exakter Key, unter dem die gesampelte (normalisierte) Aktion in die collated
     model-inputs injiziert wird, damit action_head.forward sie als action_input.action
     sieht  (markiert mit  # >>> LIVE-CHECK).
  2. num_envs-Durchsatz mit 4-Kamera-Rendering (Render-FPS-Benchmark, Plan Gruppe 0).
  3. Checkpoint-/Embodiment-Lade-Pfade auf der Zielplattform.

Dieses Skript ist daher als Geruest gedacht, das auf der RT-Core-GPU iterativ
scharf gestellt wird — NICHT als bereits end-to-end gepruefter Trainer.

Algorithmus: FPO (Flow Policy Optimization, arXiv 2510.09976) — ersetzt den
PPO-Likelihood-Ratio durch exp(proxy_new - proxy_old) mit proxy = -L_flow_matching,
gemittelt ueber K (noise, t)-Ziehungen. Passt zur Flow-Matching-Policy von GR00T,
ohne explizite Likelihood. VLM bleibt eingefroren (wie im BC-Training).

Aufruf (im kombinierten Isaac-Sim + GR00T-Container, vgl. Dockerfile.vastai):
    python rl_finetune.py \
        --checkpoint /data/checkpoints/groot-g1dex3-checkpoint \
        --num-envs 16 --iterations 500 --rollout-steps 32
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FPO-RL-Fine-tuning des GR00T-Action-Heads")
    p.add_argument("--checkpoint", required=True, help="BC-Checkpoint (RL-Startpunkt)")
    p.add_argument("--asset-path", default=None,
                   help="Pfad zum g1_dex3.usd-Roboter-Asset (sonst cfg-Default in g1_dex3_cfg.py)")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--output-dir", default="/data/g1_dex3_rl")
    p.add_argument("--num-envs", type=int, default=16, help="Parallele Sim-Envs")
    p.add_argument("--iterations", type=int, default=500, help="RL-Iterationen (Rollout+Update)")
    p.add_argument("--rollout-steps", type=int, default=32, help="Env-Steps pro Rollout pro Env")
    p.add_argument("--epochs-per-iter", type=int, default=2, help="PPO-Epochen je Rollout")
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2, help="PPO/FPO-Clip-Epsilon")
    p.add_argument("--kl-coef", type=float, default=0.1, help="KL-Regularisierung gegen BC")
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--fpo-mc-samples", type=int, default=4, help="K (noise,t)-Ziehungen fuer den Proxy")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="gr00t-g1-dex3-rl")
    p.add_argument("--check", action="store_true",
                   help="Nur Imports/Aufbau pruefen, kein Training (CI/Smoke-Test).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Lazy, fehlertolerante Imports (Isaac Sim / GR00T sind nur im Sim-Container da)
# ---------------------------------------------------------------------------
def _import_stack():
    try:
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover
        sys.exit(f"[rl] PyTorch fehlt: {e}")
    try:
        # App-Start von Isaac Sim MUSS vor isaaclab-Importen passieren.
        from isaaclab.app import AppLauncher
    except ImportError as e:  # pragma: no cover
        sys.exit(
            "[rl] isaaclab nicht importierbar — dieses Skript braucht den kombinierten "
            f"Isaac-Sim + GR00T-Container (Simulation/Dockerfile.vastai). Detail: {e}"
        )
    return AppLauncher


# ---------------------------------------------------------------------------
# Value-Head (Critic) — GR00T hat keinen; wir lernen einen schlanken State-Critic.
# ---------------------------------------------------------------------------
def build_value_head(state_dim: int, device, torch):
    import torch.nn as nn

    net = nn.Sequential(
        nn.Linear(state_dim, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 1),
    ).to(device)
    return net


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, dones, last_values, gamma, lam, torch):
    """rewards/values/dones: (T, N). last_values: (N,). -> advantages, returns (T, N)."""
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


# ---------------------------------------------------------------------------
# FPO-Proxy: -L_flow_matching, gemittelt ueber K Ziehungen.
# ---------------------------------------------------------------------------
def fpo_logprob_proxy(model, inputs_with_action, k_samples, torch):
    """Likelihood-freier Stand-in fuer log pi(a|s) (FPO, arXiv 2510.09976).

    action_head.forward(...) zieht intern Rauschen + Zeit t und liefert den
    Flow-Matching-MSE als "action_loss". Wir mitteln -loss ueber K Ziehungen:
    hoeherer Proxy == niedrigerer FM-Loss == hoehere (implizite) Likelihood.
    """
    proxies = []
    for _ in range(k_samples):
        out = model.forward(inputs_with_action)  # >>> LIVE-CHECK: action-Key in inputs
        al = out["action_loss"]          # (B, horizon, dim), bereits * action_mask
        mask = out["action_mask"]
        per_sample = al.flatten(1).sum(dim=1) / (mask.flatten(1).sum(dim=1) + 1e-6)
        proxies.append(-per_sample)      # (B,)
    return torch.stack(proxies, dim=0).mean(dim=0)


def main() -> None:
    args = parse_args()
    AppLauncher = _import_stack()
    import torch

    # ── Isaac Sim hochfahren (headless) ───────────────────────────────────────
    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    # Erst NACH AppLauncher importierbar:
    from g1_dex3_blockstack_env import G1Dex3BlockstackEnv, G1Dex3BlockstackEnvCfg
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[rl] device={device}  num_envs={args.num_envs}", flush=True)

    # ── Env (vektorisiert, shaped reward) ─────────────────────────────────────
    cfg = G1Dex3BlockstackEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.reward_mode = "shaped"
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
        print(f"[rl] USD-Asset: {args.asset_path}", flush=True)
    env = G1Dex3BlockstackEnv(cfg)

    # ── Policy: aktuelle (trainierbar) + eingefrorene Referenz (fuer KL) ───────
    emb = EmbodimentTag(args.embodiment_tag)
    policy = Gr00tPolicy(model_path=args.checkpoint, embodiment_tag=emb)
    ref_policy = Gr00tPolicy(model_path=args.checkpoint, embodiment_tag=emb)
    model = policy.model
    ref_model = ref_policy.model
    for prm in ref_model.parameters():
        prm.requires_grad_(False)
    ref_model.eval()

    # VLM einfrieren, nur Action-Head/Projector trainieren (wie BC-Scope).
    trainable = []
    for name, prm in model.named_parameters():
        train_it = ("action_head" in name) or ("action_decoder" in name) or ("projector" in name)
        prm.requires_grad_(train_it)
        if train_it:
            trainable.append(prm)
    print(f"[rl] trainierbare Tensoren: {len(trainable)}", flush=True)

    value_head = build_value_head(cfg.observation_space, device, torch)
    optim = torch.optim.AdamW(trainable + list(value_head.parameters()), lr=args.lr)

    if args.check:
        print("[rl] --check: Aufbau OK (Env + Policy + Critic instanziiert).", flush=True)
        simulation_app.close()
        return

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # ── Trainings-Schleife ────────────────────────────────────────────────────
    obs_dict, _ = env.reset()
    for it in range(args.iterations):
        # Speicher fuer den Rollout
        buf_logp, buf_val, buf_rew, buf_done, buf_inputs, buf_state = [], [], [], [], [], []

        for _step in range(args.rollout_steps):
            obs = env.get_obs_batched()                  # batched GPU-Tensoren
            state = obs["state.joint_pos"].float().detach()
            raw = _obs_batched_to_policy_dict(obs, cfg.task_description, args.num_envs)

            # Sampling (no grad) + collated inputs fuers spaetere FPO-Loss-Recompute.
            with torch.no_grad():
                norm_action, collated, phys_action = _sample_action(policy, raw, torch)

            value = value_head(state).squeeze(-1)

            # >>> LIVE-CHECK: gesampelte (normalisierte) Aktion als action_input.action
            inputs_with_action = dict(collated)
            inputs_with_action["action"] = norm_action
            logp = fpo_logprob_proxy(model, inputs_with_action, args.fpo_mc_samples, torch)

            # Env-Step mit physischer Aktion (28-dim absolut)
            act_t = torch.as_tensor(phys_action, device=device, dtype=torch.float32)
            obs_dict, reward, terminated, time_out, _ = env.step(act_t)
            done = (terminated | time_out).float()

            buf_logp.append(logp.detach())
            buf_val.append(value.detach())
            buf_rew.append(reward.detach())
            buf_done.append(done)
            buf_inputs.append((inputs_with_action, norm_action.detach()))
            buf_state.append(state)

            if done.any():
                env.reset()

        # Bootstrap-Value + GAE
        last_obs = env.get_obs_batched()
        with torch.no_grad():
            last_val = value_head(last_obs["state.joint_pos"].float()).squeeze(-1)
        rewards = torch.stack(buf_rew)
        values = torch.stack(buf_val)
        dones = torch.stack(buf_done)
        old_logp = torch.stack(buf_logp)
        adv, returns = compute_gae(
            rewards, values, dones, last_val, args.gamma, args.gae_lambda, torch
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ── FPO/PPO-Update ────────────────────────────────────────────────────
        T, N = rewards.shape
        flat_idx = [(t, n) for t in range(T) for n in range(N)]
        for _epoch in range(args.epochs_per_iter):
            for start in range(0, len(flat_idx), args.minibatch_size):
                mb = flat_idx[start:start + args.minibatch_size]
                pol_loss = torch.zeros((), device=device)
                val_loss = torch.zeros((), device=device)
                kl_loss = torch.zeros((), device=device)
                for (t, n) in mb:
                    inp, _na = buf_inputs[t]
                    inp_n = _select_env(inp, n, torch)
                    new_logp = fpo_logprob_proxy(model, inp_n, args.fpo_mc_samples, torch)[0]
                    with torch.no_grad():
                        ref_logp = fpo_logprob_proxy(ref_model, inp_n, args.fpo_mc_samples, torch)[0]
                    ratio = torch.exp(new_logp - old_logp[t, n])
                    a = adv[t, n]
                    unclipped = ratio * a
                    clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a
                    pol_loss = pol_loss - torch.min(unclipped, clipped)
                    kl_loss = kl_loss + (ref_logp - new_logp)
                    v = value_head(buf_state[t][n:n + 1]).squeeze()
                    val_loss = val_loss + (v - returns[t, n]) ** 2
                m = max(len(mb), 1)
                loss = (pol_loss + args.kl_coef * kl_loss) / m + args.value_coef * val_loss / m
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable + list(value_head.parameters()), 1.0)
                optim.step()

        succ = float(env._episode_success.float().mean().item())
        mean_rew = float(rewards.mean().item())
        print(f"[rl] iter {it:04d}  reward={mean_rew:+.3f}  success={succ:.3f}", flush=True)
        if args.wandb:
            import wandb
            wandb.log({"iter": it, "reward_mean": mean_rew, "success_rate": succ})

        if (it + 1) % args.save_every == 0:
            import os
            ckpt = os.path.join(args.output_dir, f"rl-checkpoint-{it + 1}")
            os.makedirs(ckpt, exist_ok=True)
            model.save_pretrained(ckpt)
            print(f"[rl] Checkpoint gespeichert: {ckpt}", flush=True)

    simulation_app.close()


# ---------------------------------------------------------------------------
# Adapter Obs <-> Policy-Format (gegruendet auf client.py build_obs)
# ---------------------------------------------------------------------------
def _obs_batched_to_policy_dict(obs, task_description, num_envs):
    """Batched Env-Obs (GPU-Tensoren) -> GR00T-Roh-Obs-Dict im Eval-Format (numpy).

    Spiegelt client.py build_obs (Z. 180-201), aber batched N statt 1 und mit
    Zeitdimension T=1, die _unbatch_observation/_to_vla_step_data erwarten:
      video.<cam>:  (N, 1, H, W, 3) uint8
      state.<grp>:  (N, 1, D)       float32
      annotation:   list[str] der Laenge N
    Joint-Split exakt wie im Eval-Pfad: left_arm 0:7 | right_arm 7:14 |
    left_dex3 14:21 | right_dex3 21:28.

    >>> LIVE-CHECK: T-Dimension/Dtype-Konvention gegen den echten Processor bestaetigen.
    """
    import numpy as np

    jp = obs["state.joint_pos"].detach().cpu().numpy().astype(np.float32)  # (N, 28)

    def sta(a):  # (N, D) -> (N, 1, D)
        return a[:, None, :]

    raw = {
        "state.left_arm":   sta(jp[:, 0:7]),
        "state.right_arm":  sta(jp[:, 7:14]),
        "state.left_dex3":  sta(jp[:, 14:21]),
        "state.right_dex3": sta(jp[:, 21:28]),
        "annotation.human.task_description": [task_description] * num_envs,
    }
    for cam in ("cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"):
        vid = obs[f"video.{cam}"].detach().cpu().numpy().astype(np.uint8)  # (N, H, W, 3)
        raw[f"video.{cam}"] = vid[:, None, ...]  # (N, 1, H, W, 3)
    return raw


def _sample_action(policy, raw, torch):
    """Sampelt eine Aktion; gibt (normalisiert, collated_inputs, physisch) zurueck.

    Faithful-Replikat von Gr00tPolicy._get_action (gr00t_policy.py:326-352), aber
    wir BEHALTEN collated_inputs + die normalisierte Aktion, um spaeter den
    FPO-Flow-Matching-Loss mit Gradient nachzurechnen.
    """
    import numpy as np
    from gr00t.data.types import MessageType

    unbatched = policy._unbatch_observation(raw)
    processed, states = [], []
    for obs in unbatched:
        vla = policy._to_vla_step_data(obs)
        states.append(vla.states)
        processed.append(
            policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": vla}])
        )
    collated = policy.collate_fn(processed)

    with torch.no_grad():
        pred = policy.model.get_action(**collated)
    norm_action = pred["action_pred"].float()  # (N, horizon, action_dim), normalisiert

    batched_states = {
        k: np.stack([s[k] for s in states], axis=0)
        for k in policy.modality_configs["state"].modality_keys
    }
    phys_dict = policy.processor.decode_action(
        norm_action.detach().cpu().numpy(), policy.embodiment_tag, batched_states
    )
    phys = _assemble_phys_action(phys_dict)  # (N, 28) — erster Chunk-Step
    return norm_action, collated, phys


def _assemble_phys_action(action_dict):
    """Setzt die physischen Aktionsgruppen zu (N, 28) zusammen (erster Chunk-Step)."""
    import numpy as np

    order = ["action.left_arm", "action.right_arm", "action.left_dex3", "action.right_dex3"]
    parts = [action_dict[k] for k in order if k in action_dict]
    full = np.concatenate(parts, axis=-1) if parts else next(iter(action_dict.values()))
    if full.ndim == 3:        # (N, horizon, 28) -> ersten Step ausfuehren
        full = full[:, 0, :]
    return full.astype(np.float32)


def _select_env(inputs, n, torch):
    """Waehlt Env-Index n aus einem batched collated-inputs-Dict (Minibatch-Recompute).

    >>> LIVE-CHECK: bei verschachtelten eagle_*-Strukturen ggf. rekursiv slicen.
    """
    out = {}
    for k, v in inputs.items():
        out[k] = v[n:n + 1] if hasattr(v, "__getitem__") and not isinstance(v, str) else v
    return out


if __name__ == "__main__":
    main()
