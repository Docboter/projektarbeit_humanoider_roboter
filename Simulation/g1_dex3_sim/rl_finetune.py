#!/usr/bin/env python3
# TL;DR: FPO-RL-Trainer fuer den GR00T-Action-Head auf Block-Stacking; `server_rl_run.sh rl`.
"""
RL-Fine-tuning des GR00T-N1.6-Action-Heads auf Block-Stacking (FPO).

Status / Validierungsgrad
-------------------------
Der **Algorithmus-Kern** (FPO-Surrogat aus dem Flow-Matching-Loss, GAE,
PPO-Clipping, KL-Regularisierung gegen den BC-Checkpoint, Optimierung NUR des
Action-Heads) ist gegen die real gelesene GR00T-API verifiziert:
  - gr00t/model/gr00t_n1d6/gr00t_n1d6.py: action_head.forward(...) liefert
    "action_loss" (pro Element) — der likelihood-freie FPO-Proxy für log pi(a|s).
  - gr00t/policy/gr00t_policy.py: Processor -> collate_fn -> model.get_action /
    model.forward -> processor.decode_action (Obs->Inputs->Aktion-Pipeline).

END-TO-END auf Hardware gelaufen (RTX PRO 6000 Blackwell, Isaac Sim 6.0,
2026-08-08): eine vollstaendige Iteration mit num_envs=2 / rollout_steps=2 —
Aufbau, Obs-Konvertierung, Aktions-Sampling, Env-Step, GAE, FPO/PPO-Update,
backward + optim.step, Checkpoint-Schreiben. Alle drei urspruenglichen
LIVE-CHECK-Punkte sind damit erledigt.

NOCH offen (deshalb kein "fertiger" Trainer):
  - Lernverhalten: dass die Erfolgsrate ueber viele Iterationen steigt.
    Hyperparameter (lr, clip, kl_coef, fpo_mc_samples) sind ungetunt.
  - num_envs-Durchsatz mit 4-Kamera-Rendering (Render-FPS-Benchmark, Plan Gruppe 0).

Algorithmus: FPO (Flow Policy Optimization, arXiv 2510.09976) — ersetzt den
PPO-Likelihood-Ratio durch exp(proxy_new - proxy_old) mit proxy = -L_flow_matching,
gemittelt ueber K (noise, t)-Ziehungen. Passt zur Flow-Matching-Policy von GR00T,
ohne explizite Likelihood. VLM bleibt eingefroren (wie im BC-Training).

GPU-Aufteilung: Das eingefrorene Referenzmodell (fuer die KL gegen den BC-Checkpoint)
laeuft nur unter no_grad und wandert mit --ref-device auto auf eine zweite sichtbare
Karte. Das nimmt ~6-7 GB Gewichte plus einen transienten Forward von der Karte, die
sich Rendering, Policy, Optimizer-States und Aktivierungen ohnehin schon teilt. Mit nur
einer sichtbaren GPU faellt es automatisch auf das alte Verhalten zurueck.

Beobachtbarkeit: --live-view (bzw. LIVE_VIEW=1) blendet den laufenden Rollout als
MJPEG-Stream im Browser ein (live_view.py, Port 8900), --wandb-video-every N schneidet
zusaetzlich alle N Iterationen einen Rollout ins W&B-Dashboard. Beides ist opt-in und
im Aus-Zustand ein reiner Early-Return. Hintergrund: docs/weiterfuehrend/livestream-plan.md.

Aufruf (im kombinierten Isaac-Sim + GR00T-Container, vgl. Dockerfile.vastai):
    python rl_finetune.py \
        --checkpoint /data/checkpoints/groot-g1dex3-checkpoint \
        --num-envs 16 --iterations 500 --rollout-steps 32 \
        --live-view --wandb --wandb-video-every 10
"""

from __future__ import annotations

import argparse
import os
import sys

from live_view import LiveView


# ---------------------------------------------------------------------------
# Env-Defaults fuer die CLI
# ---------------------------------------------------------------------------
# Warum die Live-View-Flags ihre Defaults aus Env-Vars ziehen statt nur aus der
# Kommandozeile: server_rl_run.sh mountet Simulation/g1_dex3_sim LIVE in den Container,
# /scripts (entrypoint_rl.sh) dagegen NICHT — das steckt fest im Image. Liest dieses
# Skript LIVE_VIEW* selbst, genuegt ein `-e LIVE_VIEW=1` am docker exec und es braucht
# keinen Image-Rebuild (~30 min) nur um einen Schalter durchzureichen.
def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in ("", "0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


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
    p.add_argument("--ref-device", default=os.environ.get("RL_REF_DEVICE", "auto"),
                   help="Geraet fuer das eingefrorene Referenzmodell (KL): 'auto' = zweite "
                        "GPU, falls sichtbar; 'same' = wie die Policy; sonst z. B. 'cuda:1'. "
                        "Env: RL_REF_DEVICE.")
    p.add_argument("--num-envs", type=int, default=16, help="Parallele Sim-Envs")
    p.add_argument("--iterations", type=int, default=500, help="RL-Iterationen (Rollout+Update)")
    p.add_argument("--rollout-steps", type=int, default=32, help="Env-Steps pro Rollout pro Env")
    # Die folgenden drei sind die Speicher-Stellschrauben bei OOM (Env-Defaults, damit
    # sie ohne Image-Rebuild wirken — Begruendung oben bei _env_flag).
    p.add_argument("--epochs-per-iter", type=int, default=_env_int("RL_EPOCHS_PER_ITER", 2),
                   help="PPO-Epochen je Rollout. Env: RL_EPOCHS_PER_ITER.")
    p.add_argument("--minibatch-size", type=int, default=_env_int("RL_MINIBATCH_SIZE", 64),
                   help="Paare (t,env) je Update-Schritt. Env: RL_MINIBATCH_SIZE.")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2, help="PPO/FPO-Clip-Epsilon")
    p.add_argument("--kl-coef", type=float, default=0.1, help="KL-Regularisierung gegen BC")
    p.add_argument("--value-coef", type=float, default=0.5)
    # K skaliert Speicher UND Rechenzeit linear: die K Ziehungen muessen bis zur Bildung
    # von exp(mean_K(new) - old) alle im Graphen bleiben, lassen sich also nicht wie die
    # Zeitschritt-Gruppen einzeln zurueckrechnen. Erste Stellschraube bei OOM.
    p.add_argument("--fpo-mc-samples", type=int, default=_env_int("RL_FPO_MC_SAMPLES", 4),
                   help="K (noise,t)-Ziehungen fuer den Proxy. Env: RL_FPO_MC_SAMPLES.")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="gr00t-g1-dex3-rl")
    p.add_argument("--wandb-video-every", type=int, default=_env_int("RL_WANDB_VIDEO_EVERY", 0),
                   help="Alle N Iterationen einen Rollout als W&B-Video loggen "
                        "(0 = aus). Env: RL_WANDB_VIDEO_EVERY.")
    # ── Live-Ansicht im Browser (MJPEG, Spur B des Livestream-Plans) ─────────
    p.add_argument("--live-view", action="store_true", default=_env_flag("LIVE_VIEW"),
                   help="Live-Ansicht des Rollouts im Browser. Env: LIVE_VIEW.")
    p.add_argument("--live-view-port", type=int, default=_env_int("LIVE_VIEW_PORT", 8900),
                   help="HTTP-Port der Live-Ansicht. Env: LIVE_VIEW_PORT.")
    p.add_argument("--live-view-every-n", type=int, default=_env_int("LIVE_VIEW_EVERY_N", 1),
                   help="Nur jedes n-te Frame publizieren. Env: LIVE_VIEW_EVERY_N.")
    # Default sind die KALIBRIERTEN Policy-Kameras, nicht cam_scene: nur diese vier sind
    # per Overlay gegen die Dataset-Referenzframes justiert worden, und sie zeigen genau
    # das, was das Modell als Eingabe bekommt — fuer die Frage "naehert sich die Hand dem
    # Wuerfel?" also aussagekraeftiger als eine Uebersicht. cam_scene ist unvalidiert und
    # lieferte am 2026-08-08 fast nur Hintergrund (Diagnose: dump_camera_poses.py).
    p.add_argument("--live-view-cams",
                   default=os.environ.get("LIVE_VIEW_CAMS", "cam_left_high,cam_left_wrist"),
                   help="Kommagetrennte Kameras. Env: LIVE_VIEW_CAMS. Moeglich: "
                        "cam_left_high, cam_right_high, cam_left_wrist, cam_right_wrist, "
                        "cam_scene (unvalidiert).")
    # ── LIVE-Variante: WebRTC-Viewport (Spur A des Livestream-Plans) ─────────
    # Anders als die Live-Ansicht oben (MJPEG-Bilder im Browser) streamt das hier den
    # kompletten Isaac-Sim-Viewport: freie Kamera, Szene drehen, Isaac-Sim-UI. Geoeffnet
    # wird er vom nativen "Isaac Sim WebRTC Streaming Client" auf dem Arbeitsrechner.
    # Fuer einen tagelangen RL-Lauf ist Spur B der robustere Weg (zustandslos, beliebig
    # viele Zuschauer) — deshalb bleibt das hier opt-in und Default 0.
    p.add_argument("--livestream", type=int, default=_env_int("LIVESTREAM", 0),
                   help="0 = headless (Default), 1 = WebRTC oeffentlich, 2 = WebRTC "
                        "privat/lokal. Env: LIVESTREAM. Anleitung: "
                        "docs/simulation/live-ansicht.md")
    p.add_argument("--livestream-update-every-n", type=int,
                   default=_env_int("LIVESTREAM_UPDATE_EVERY_N", 1),
                   help="Nur mit --livestream: alle n Rollout-Steps simulation_app.update() "
                        "aufrufen, damit der Viewport nachzieht. 0 = nie (falls Isaac Lab "
                        "den Render-Loop selbst treibt). Env: LIVESTREAM_UPDATE_EVERY_N.")
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
def resolve_ref_device(spec: str, device: str, torch) -> str:
    """Geraet fuer das Referenzmodell bestimmen.

    'auto' legt es auf die zweite sichtbare GPU, sonst auf dieselbe wie die Policy.
    Sinn: Das Referenzmodell ist eingefroren und laeuft nur unter no_grad — es braucht
    also nur seine ~6-7 GB Gewichte plus einen transienten Forward, aber keinen
    Backward-Graphen. Genau dieser Ballast gehoert nicht auf die Karte, die sich
    Rendering, Policy, Optimizer-States und Aktivierungen ohnehin schon teilen.
    """
    if spec == "same" or not device.startswith("cuda"):
        return device
    if spec != "auto":
        return spec
    return "cuda:1" if torch.cuda.device_count() > 1 else device


def _to_device(obj, dev):
    """Collated Batch rekursiv auf ein anderes Geraet kopieren; Nicht-Tensoren bleiben."""
    if hasattr(obj, "to") and hasattr(obj, "device"):
        return obj.to(dev, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, dev) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, dev) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(v, dev) for v in obj)
    return obj


def fpo_logprob_proxy(model, inputs_with_action, k_samples, torch):
    """Likelihood-freier Stand-in fuer log pi(a|s) (FPO, arXiv 2510.09976).

    action_head.forward(...) zieht intern Rauschen + Zeit t und liefert den
    Flow-Matching-MSE als "action_loss". Wir mitteln -loss ueber K Ziehungen:
    hoeherer Proxy == niedrigerer FM-Loss == hoehere (implizite) Likelihood.
    """
    proxies = []
    for _ in range(k_samples):
        # inputs_with_action ist der innere collate-Batch inkl. 'action'/'action_mask';
        # model.forward(inputs: dict) erwartet genau diese Ebene (gr00t_n1d6.py:496).
        out = model.forward(inputs_with_action)
        al = out["action_loss"]          # (B, horizon, dim), bereits * action_mask
        mask = out["action_mask"]
        per_sample = al.flatten(1).sum(dim=1) / (mask.flatten(1).sum(dim=1) + 1e-6)
        proxies.append(-per_sample)      # (B,)
    return torch.stack(proxies, dim=0).mean(dim=0)


def main() -> None:
    args = parse_args()
    AppLauncher = _import_stack()
    import torch

    # ── Isaac Sim hochfahren (headless ODER LIVE-Variante) ────────────────────
    # headless UND livestream gleichzeitig zu setzen ist ein Fehler: Isaac Lab waehlt dann
    # das falsche Experience-File (IsaacLab#381) und der Stream bliebe schwarz. Jeder
    # Livestream-Modus impliziert Headless ohnehin — deshalb entweder/oder.
    if args.livestream:
        launcher_kwargs = {"livestream": args.livestream, "enable_cameras": True}
        # Kit-Settings (Ports/oeffentliche IP) baut die Shell-Seite in
        # Simulation/scripts/lib_livestream.sh und reicht sie als Env-Var durch — im
        # Server-Fall mit Default-Ports ist sie leer und wird gar nicht gebraucht.
        kit_args = os.environ.get("LIVESTREAM_KIT_ARGS", "").strip()
        if kit_args:
            launcher_kwargs["kit_args"] = kit_args
        print(
            f"[rl] LIVE-Variante aktiv (LIVESTREAM={args.livestream}): Isaac-Sim-Viewport "
            f"per WebRTC. Auf dem Arbeitsrechner mit dem 'Isaac Sim WebRTC Streaming "
            f"Client' auf Port {os.environ.get('LIVESTREAM_PORT', '49100')} verbinden.",
            flush=True,
        )
    else:
        launcher_kwargs = {"headless": True, "enable_cameras": True}
    app_launcher = AppLauncher(**launcher_kwargs)
    simulation_app = app_launcher.app

    # Erst NACH AppLauncher importierbar:
    from g1_dex3_blockstack_env import G1Dex3BlockstackEnv, G1Dex3BlockstackEnvCfg
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_device = resolve_ref_device(args.ref_device, device, torch)
    print(f"[rl] device={device}  num_envs={args.num_envs}", flush=True)
    if ref_device != device:
        print(f"[rl] Referenzmodell (KL) auf {ref_device} — entlastet {device}.", flush=True)
    elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
        print(
            "[rl] Nur EINE GPU sichtbar — Referenzmodell teilt sie sich mit der Policy. "
            "Fuer die Aufteilung beide Karten durchreichen (RL_GPUS).",
            flush=True,
        )

    # ── Env (vektorisiert, shaped reward) ─────────────────────────────────────
    cfg = G1Dex3BlockstackEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.reward_mode = "shaped"
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
        print(f"[rl] USD-Asset: {args.asset_path}", flush=True)
    env = G1Dex3BlockstackEnv(cfg)

    # ── Policy: aktuelle (trainierbar) + eingefrorene Referenz (fuer KL) ───────
    # EmbodimentTag aus einem String: N1.7 hat dafuer die Klassenmethode `resolve`
    # (Name ODER Wert, case-insensitive), N1.6 nur EmbodimentTag(wert)/EmbodimentTag[NAME].
    _resolve = getattr(EmbodimentTag, "resolve", None)
    if _resolve is not None:
        emb = _resolve(args.embodiment_tag)
    else:
        try:
            emb = EmbodimentTag(args.embodiment_tag)
        except ValueError:
            emb = EmbodimentTag[args.embodiment_tag.upper()]
    policy = Gr00tPolicy(model_path=args.checkpoint, embodiment_tag=emb, device=device)

    # ── N1.6-Sperre ───────────────────────────────────────────────────────────
    # Dieser Trainer repliziert Interna von N1.6: Gr00tN1d6ActionHead.forward liefert den
    # elementweisen "action_loss", der als FPO-Proxy fuer log pi(a|s) dient, und die
    # Aktions-/Zustandsmasken werden wie in processing_gr00t_n1d6.py von Hand gebaut.
    # Fuer N1.7 (Gr00tN1d7) ist davon nichts geprueft — und im Isaac-Sim-Python ist
    # ohnehin nur das N1.6-gr00t installiert. Also hier abbrechen, nicht erst mitten im
    # Update mit einem KeyError. Portierung: docs/weiterfuehrend/groot-n17-migration.md
    # (Phase 6).
    _model_cls = type(policy.model).__name__
    if _model_cls != "Gr00tN1d6":
        raise NotImplementedError(
            f"RL-Fine-tuning ist bislang nur fuer GR00T N1.6 implementiert, geladen wurde "
            f"'{_model_cls}' ({args.checkpoint}). Der FPO-Trainer haengt an "
            "Gr00tN1d6ActionHead.forward und den Prozessor-Masken aus "
            "processing_gr00t_n1d6.py. Portierung: Phase 6 in "
            "docs/weiterfuehrend/groot-n17-migration.md — bis dahin einen "
            "N1.6-Checkpoint verwenden."
        )

    ref_policy = Gr00tPolicy(model_path=args.checkpoint, embodiment_tag=emb, device=ref_device)
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

    # ── Live-Ansicht (opt-in) ─────────────────────────────────────────────────
    # Kostet keinen zusaetzlichen Render-Pass: cam_scene steht ohnehin in env.cameras
    # und wird jeden Step mitgerendert. Bewusst VOR dem --check-Return, damit der
    # Smoke-Test schon zeigt, ob Port und Pillow im Kit-Python in Ordnung sind.
    view_cams = [c.strip() for c in args.live_view_cams.split(",") if c.strip()] or ["cam_scene"]
    live = LiveView(
        enabled=args.live_view,
        port=args.live_view_port,
        every_n=args.live_view_every_n,
        cams=view_cams,
        title=f"G1 DEX3 — RL-Fine-tuning (FPO), {args.num_envs} Envs",
    )

    if args.check:
        print("[rl] --check: Aufbau OK (Env + Policy + Critic instanziiert).", flush=True)
        live.close()
        simulation_app.close()
        return

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # ── Trainings-Schleife ────────────────────────────────────────────────────
    # Bei aktiver LIVE-Variante den Viewport nachziehen: der Rollout ruft nur env.step(),
    # und ob Isaac Lab dabei von sich aus den Render-Loop treibt, haengt an der Isaac-Sim-
    # Version. Ein simulation_app.update() je n Steps macht das unabhaengig davon sichtbar.
    # 0 (bzw. livestream=0) schaltet den Aufruf komplett ab — dann kostet er auch nichts.
    ls_update_every = args.livestream_update_every_n if args.livestream else 0
    obs_dict, _ = env.reset()
    for it in range(args.iterations):
        # Speicher fuer den Rollout
        buf_logp, buf_val, buf_rew, buf_done, buf_inputs, buf_state = [], [], [], [], [], []
        # Option C (Livestream-Plan §5): nur in jeder N-ten Iteration mitschneiden —
        # rollout_steps Frames à ~1 MB, das lohnt nicht bei jeder Iteration.
        rec = [] if (
            args.wandb and args.wandb_video_every > 0 and it % args.wandb_video_every == 0
        ) else None

        for _step in range(args.rollout_steps):
            obs = env.get_obs_batched()                  # batched GPU-Tensoren
            live.publish_obs(obs, iteration=it, step=_step)
            if ls_update_every and _step % ls_update_every == 0:
                simulation_app.update()
            if rec is not None:
                fr = obs.get(f"video.{view_cams[0]}")
                if fr is not None:
                    rec.append(fr[0].detach().cpu().numpy())
            state = obs["state.joint_pos"].float().detach()
            raw = _obs_batched_to_policy_dict(obs, cfg.task_description, args.num_envs)

            # Sampling (no grad) + collated inputs fuers spaetere FPO-Loss-Recompute.
            with torch.no_grad():
                norm_action, collated, act_mask, phys_action = _sample_action(policy, raw, torch)

            value = value_head(state).squeeze(-1)

            # Gesampelte (normalisierte) Aktion + Maske in den INNEREN Batch legen:
            # action_head.prepare_input reicht den Batch unveraendert durch, und
            # Gr00tN1d6ActionHead.forward liest action_input.action UND .action_mask.
            # (LIVE-CHECK 2026-08-07 aufgeloest — Ebene und fehlende Maske korrigiert.)
            inputs_with_action = dict(collated)
            inputs_with_action["action"] = norm_action
            inputs_with_action["action_mask"] = act_mask
            # old_logp wird ohnehin detached gepuffert -> kein Graph noetig (spart im
            # Rollout den kompletten VLM-Aktivierungsgraph ueber alle Envs).
            with torch.no_grad():
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
        last_losses = (0.0, 0.0, 0.0)   # (policy, kl, value) des letzten Minibatch
        for _epoch in range(args.epochs_per_iter):
            for start in range(0, len(flat_idx), args.minibatch_size):
                mb = flat_idx[start:start + args.minibatch_size]
                # Nach Zeitschritt gruppieren: EIN Forward je t ueber alle benoetigten
                # Envs, danach die gebrauchten Env-Indizes herausgreifen.
                # Grund: aus den collated Eagle-Inputs laesst sich ein einzelner
                # Batch-Eintrag nicht schneiden — pixel_values ist ueber Envs UND
                # Kameras auf Dim 0 gepackt ((B*n_cams, C, H, W)), input_ids dagegen
                # (B, L). Naives [n:n+1] lieferte 1 statt n_cams Bildern, und Eagle
                # brach mit "size of tensor a (324) must match tensor b (81)" ab
                # (324 = 4 Kameras * 81 Bild-Token, 81 = 1 Kamera).
                # Nebeneffekt: identische Batch-Zusammensetzung wie beim Rollout,
                # old_logp und new_logp sind damit exakt vergleichbar.
                by_t = {}
                for (t, n) in mb:
                    by_t.setdefault(t, []).append(n)
                # Gradienten-Akkumulation je Zeitschritt-Gruppe statt EINES grossen
                # Graphen ueber das ganze Minibatch. Mathematisch identisch (die
                # Gradienten der Summe sind die Summe der Gradienten), aber der Graph
                # jeder Gruppe wird sofort nach ihrem backward() frei.
                # Vorher lebten alle Gruppen gleichzeitig: bei T=32/N=4/minibatch=64
                # umfasst ein Minibatch 16 Zeitschritte, mal fpo_mc_samples=4 sind das
                # 64 gehaltene Forward-Graphen — das war der OOM am 2026-08-08
                # (Traceback im Qwen3-lm_head, 54 GB im Prozess). Perfide dabei: ein
                # KLEINERES num_envs machte es schlimmer, weil dann mehr Zeitschritte
                # in dasselbe Minibatch passen.
                m = max(len(mb), 1)
                optim.zero_grad(set_to_none=True)
                pol_sum = kl_sum = val_sum = 0.0
                for t, ns in by_t.items():
                    inp, _na = buf_inputs[t]
                    idx = torch.as_tensor(ns, device=device, dtype=torch.long)
                    new_logp = fpo_logprob_proxy(model, inp, args.fpo_mc_samples, torch)[idx]
                    with torch.no_grad():
                        # Liegt das Referenzmodell auf der zweiten Karte, werden die Inputs
                        # EINMAL je Gruppe hinuebergeschoben — nicht je K-Ziehung, denn
                        # fpo_logprob_proxy laeuft K-mal ueber genau dieselben Inputs.
                        # Zurueck kommt nur ein (B,)-Vektor, der Rueckweg ist also gratis.
                        ref_inp = inp if ref_device == device else _to_device(inp, ref_device)
                        ref_logp = fpo_logprob_proxy(
                            ref_model, ref_inp, args.fpo_mc_samples, torch
                        ).to(device)[idx]
                    ratio = torch.exp(new_logp - old_logp[t, idx])
                    a = adv[t, idx]
                    unclipped = ratio * a
                    clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a
                    pol = -torch.min(unclipped, clipped).sum()
                    kl = (ref_logp - new_logp).sum()
                    v = value_head(buf_state[t][idx]).squeeze(-1)
                    vl = ((v - returns[t, idx]) ** 2).sum()
                    group_loss = (pol + args.kl_coef * kl) / m + args.value_coef * vl / m
                    group_loss.backward()
                    pol_sum += float(pol.detach()) / m
                    kl_sum += float(kl.detach()) / m
                    val_sum += float(vl.detach()) / m
                torch.nn.utils.clip_grad_norm_(trainable + list(value_head.parameters()), 1.0)
                optim.step()
                last_losses = (pol_sum, kl_sum, val_sum)

        succ = float(env._episode_success.float().mean().item())
        mean_rew = float(rewards.mean().item())
        pol_l, kl_l, val_l = last_losses
        print(
            f"[rl] iter {it:04d}  reward={mean_rew:+.3f}  success={succ:.3f}  "
            f"pol={pol_l:+.4f}  kl={kl_l:+.4f}  val={val_l:.4f}",
            flush=True,
        )
        live.update_meta(
            iteration=it, reward_mean=round(mean_rew, 4), success_rate=round(succ, 4),
            policy_loss=round(pol_l, 5), kl=round(kl_l, 5), value_loss=round(val_l, 5),
        )
        if args.wandb:
            import wandb
            payload = {"iter": it, "reward_mean": mean_rew, "success_rate": succ,
                       "policy_loss": pol_l, "kl": kl_l, "value_loss": val_l}
            if rec:
                # In DENSELBEN log()-Aufruf, nicht in einen zweiten: wandb zaehlt sonst
                # einen eigenen Step hoch und Video und Metriken landen versetzt.
                payload.update(_rollout_video_payload(rec))
            wandb.log(payload)

        if (it + 1) % args.save_every == 0:
            # os wird modulweit importiert. Ein lokales `import os` hier drin waere eine
            # Falle: es macht `os` fuer die GANZE Funktion lokal, und die erste kuenftige
            # Nutzung weiter oben stuerbte mit UnboundLocalError ab — mitten im Lauf.
            ckpt = os.path.join(args.output_dir, f"rl-checkpoint-{it + 1}")
            os.makedirs(ckpt, exist_ok=True)
            model.save_pretrained(ckpt)
            print(f"[rl] Checkpoint gespeichert: {ckpt}", flush=True)

    live.close()
    simulation_app.close()


# ---------------------------------------------------------------------------
# Option C aus dem Livestream-Plan (§5): Rollout als W&B-Artefakt archivieren
# ---------------------------------------------------------------------------
def _rollout_video_payload(frames) -> dict:
    """Mitgeschnittene Frames -> W&B-Log-Eintrag. Wirft nie.

    Ergaenzt die (fluechtige) Live-Ansicht um etwas Bleibendes: im W&B-Dashboard
    abrufbar, von ueberall erreichbar und in der Projektarbeit zitierbar.

    wandb.Video braucht fuer numpy-Eingaben moviepy — das ist im Isaac-Sim-Python NICHT
    installiert. Statt dafuer eine Dependency ins Image zu ziehen, faellt die Funktion
    auf einen Filmstreifen aus acht Einzelbildern zurueck (wandb.Image braucht nur PIL).
    """
    import numpy as np
    import wandb

    arr = np.stack(frames)  # (T, H, W, 3) uint8
    try:
        return {"rollout_video": wandb.Video(arr.transpose(0, 3, 1, 2), fps=10)}
    except Exception as e:
        print(f"[rl] wandb.Video nicht moeglich ({e}) — logge Filmstreifen.", flush=True)
        try:
            sel = arr[:: max(1, len(arr) // 8)][:8]
            return {"rollout_frames": [wandb.Image(f) for f in sel]}
        except Exception as e2:
            print(f"[rl] auch Filmstreifen fehlgeschlagen ({e2}) — ueberspringe.", flush=True)
            return {}


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

    Erzeugt das FLACHE Sim-Format. Die innere Gr00tPolicy braucht das verschachtelte
    Format — _flat_to_nested() konvertiert (LIVE-CHECK 2026-08-07 aufgeloest).
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


def _flat_to_nested(policy, raw):
    """Flaches Sim-Obs-Format -> verschachteltes Format der inneren Gr00tPolicy.

    GR00T kennt ZWEI Obs-Formate: das flache Sim-Format ('video.<cam>', 'state.<grp>',
    Sprache als list[str] der Laenge B) und das verschachtelte Policy-Format
    ({'video': {...}, 'state': {...}, 'language': {...}}). Die oeffentliche Sim-API
    konvertiert dazwischen; wir rufen policy._unbatch_observation direkt auf (um an
    collated_inputs zu kommen) und muessen die Konvertierung daher selbst machen.

    Faithful-Replikat von Gr00tSimPolicy._get_action (gr00t_policy.py:588-615). Die
    Schluessel kommen aus policy.modality_configs statt fest verdrahtet, damit ein
    anderes Embodiment nicht still das falsche Feld liest. Merke: video/state werden
    als f"{modality}.{key}" nachgeschlagen, Sprache dagegen unter dem BLANKEN key.
    """
    nested = {}
    for modality in ("video", "state", "language"):
        nested[modality] = {}
        for key in policy.modality_configs[modality].modality_keys:
            flat_key = key if modality == "language" else f"{modality}.{key}"
            if flat_key not in raw:
                raise KeyError(
                    f"Obs-Schluessel '{flat_key}' fehlt (Modalitaet '{modality}'). "
                    f"Vorhanden: {sorted(raw)}. _obs_batched_to_policy_dict anpassen."
                )
            arr = raw[flat_key]
            # Sprache: (B,) list[str] -> (B, 1) list[list[str]] (T-Dimension)
            nested[modality][key] = [[str(item)] for item in arr] if modality == "language" else arr
    return nested


def _action_spec(policy):
    """(modality_keys, action_horizon, action_dim) des Embodiments.

    Einzige Wahrheitsquelle fuer Aktions-Layout — dieselbe, aus der decode_action seine
    Grenzen zieht (processing_gr00t_n1d6.py:240-248). Wird von der Maske UND vom
    Zusammensetzen der physischen Aktion genutzt, damit beide nie auseinanderlaufen.
    """
    emb = policy.embodiment_tag.value
    acfg = policy.processor.modality_configs[emb]["action"]
    norm_params = policy.processor.state_action_processor.norm_params[emb]["action"]
    action_dim = sum(int(norm_params[k]["dim"].item()) for k in acfg.modality_keys)
    return list(acfg.modality_keys), len(acfg.delta_indices), action_dim


def _build_action_mask(policy, norm_action, torch):
    """Baut action_mask fuer eine gesampelte Aktion — Semantik wie im Prozessor.

    Der Prozessor erzeugt die Maske nur im TRAINING (processing_gr00t_n1d6.py:339-341):
        mask = ones_like(padded_action); mask[action_horizon:] = 0; mask[:, action_dim:] = 0
    Bei Inferenz ist actions={} -> weder 'action' noch 'action_mask' liegen im Batch. Fuer
    den FPO-Loss brauchen wir beide, muessen die Maske also selbst bauen.

    Eine All-Ones-Maske waere FALSCH: norm_action ist auf (max_action_horizon, max_action_dim)
    gepaddet, und der Loss wuerde die Padding-Spalten mitmitteln, in denen das Modell nichts
    Sinnvolles vorhersagt. Die echten Grenzen kommen aus derselben Quelle, aus der auch
    decode_action sie zieht (processing_gr00t_n1d6.py:240-248).
    """
    _keys, action_horizon, action_dim = _action_spec(policy)
    mask = torch.zeros_like(norm_action)
    mask[:, :action_horizon, :action_dim] = 1.0
    return mask


def _sample_action(policy, raw, torch):
    """Sampelt eine Aktion; gibt (normalisiert, fpo_inputs, action_mask, physisch) zurueck.

    Faithful-Replikat von Gr00tPolicy._get_action (gr00t_policy.py:326-352), aber
    wir BEHALTEN collated_inputs + die normalisierte Aktion, um spaeter den
    FPO-Flow-Matching-Loss mit Gradient nachzurechnen.

    fpo_inputs ist der INNERE Batch (collated["inputs"]): der Collator verpackt alles als
    BatchFeature({"inputs": batch}) (processing_gr00t_n1d6.py:102), weshalb die Referenz
    get_action(**collated) aufruft — das entfaltet zu get_action(inputs=batch). model.forward()
    nimmt denselben inneren Batch direkt entgegen, und dort hinein gehoert auch die Aktion.
    """
    import numpy as np
    from gr00t.data.types import MessageType

    unbatched = policy._unbatch_observation(_flat_to_nested(policy, raw))
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
    phys = _assemble_phys_action(policy, phys_dict)  # (N, action_dim) — erster Chunk-Step
    action_mask = _build_action_mask(policy, norm_action, torch)
    return norm_action, collated["inputs"], action_mask, phys


def _assemble_phys_action(policy, action_dict):
    """Setzt die physischen Aktionsgruppen zu (N, action_dim) zusammen (erster Chunk-Step).

    ACHTUNG Schluessel-Namen: decode_action() gibt die BLANKEN Gruppennamen zurueck
    ('left_arm', …). Das Praefix 'action.' haengt erst der ZMQ-Sim-Wrapper an
    (gr00t_policy.py:617) — client.py sieht es deshalb, wir hier NICHT.

    Fehlt eine Gruppe, wird hart abgebrochen. Die frueheren stillen Fallbacks
    ('if k in action_dict' + next(iter(...))) haben am 2026-08-07 aus einem simplen
    Namensfehler einen (N, 7)- statt (N, 28)-Tensor gemacht — sichtbar erst 30 s spaeter
    als CUDA device-side assert tief in _pre_physics_step.
    """
    import numpy as np

    keys, _horizon, action_dim = _action_spec(policy)
    missing = [k for k in keys if k not in action_dict]
    if missing:
        raise KeyError(
            f"Aktionsgruppen {missing} fehlen in decode_action-Ausgabe. "
            f"Vorhanden: {sorted(action_dict)}"
        )

    full = np.concatenate([np.asarray(action_dict[k], dtype=np.float32) for k in keys], axis=-1)
    if full.ndim == 3:        # (N, horizon, D) -> ersten Chunk-Step ausfuehren
        full = full[:, 0, :]
    if full.shape[-1] != action_dim:
        raise ValueError(
            f"Physische Aktion hat {full.shape[-1]} Dims, erwartet {action_dim}. "
            f"Die Env indiziert feste Gelenk-Indizes — zu schmal endet im CUDA-Assert."
        )
    return full.astype(np.float32)


if __name__ == "__main__":
    main()
