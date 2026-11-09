# TL;DR: Isaac-freie Eval-Metriken: kipp-robustes `lifted_clear` + Clopper-Pearson-KI.
"""Reine Rechenhilfen für die Closed-Loop-Eval (run_g1_dex3_sim_eval.py).

Isaac-frei (nur numpy + Standardbibliothek), damit die Tests auf der CPU laufen.

1. ``cube_lowest_z`` / ``lifted_clear_step``
   Die alte Metrik ``lifted`` vergleicht die SCHWERPUNKT-Höhe gegen das eingeschwungene
   Layout (> 2 cm). Ein 5-cm-Würfel, der auf eine Kante kippt, hebt seinen Schwerpunkt
   dabei um 2,5·(√2 − 1) ≈ 1,04 cm, auf eine Ecke um 2,5·(√3 − 1) ≈ 1,83 cm — knapp
   unter der Schwelle, aber ohne dass er je vom Tisch abhob (Review 2026-10-13, M3).
   ``lifted_clear`` misst stattdessen den TIEFSTEN Punkt des Würfels (Pose + halbe
   Kante): der liegt beim Kippen weiter auf dem Tisch, beim echten Anheben darüber.
   Zusätzlich muss im SELBEN Schritt eine Hand am Würfel sein (s. Kontakt-Proxy dort),
   sonst zählt ein geworfener oder abprallender Würfel mit.

2. ``clopper_pearson``
   Exaktes 95-%-Konfidenzintervall einer Erfolgsquote k/n (Review M4: 0/10 heißt
   0–30,8 %, 0/20 heißt 0–16,8 %). Ohne scipy: Bisektion auf der Binomial-CDF.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# Höhe der Würfel-UNTERSEITE über der Tischplatte, ab der ein Würfel als angehoben
# gilt. Derselbe Zahlenwert wie LIFT_THRESHOLD_M der alten Metrik, damit beide
# Varianten dieselbe Schwelle haben und sich nur im Bezugspunkt unterscheiden.
LIFT_CLEAR_THRESHOLD_M = 0.02

# Kontakt-Proxy: die Env hat KEINE Kontaktsensoren. Ersatz ist der Abstand der
# Fingerkuppen (env.get_contact_points_w, Fallback Handfläche/Handwurzel) zur
# Würfelmitte. 4 cm = halbe Kante 2,5 cm + 1,5 cm Spiel; derselbe Wert wie
# REACH_THRESHOLD_M ("reached") und der Replay-Greiftest "Würfel noch in der Hand".
HAND_CONTACT_PROXY_M = 0.04


# ---------------------------------------------------------------------------
# Kipp-robuste Anhebung
# ---------------------------------------------------------------------------

def cube_lowest_z(pos: Sequence[float] | np.ndarray,
                  quat_wxyz: Sequence[float] | np.ndarray,
                  half_edge: float) -> np.ndarray:
    """z des tiefsten Würfelpunkts (einer der 8 Ecken) für eine oder mehrere Posen.

    Args:
        pos:       (3,) oder (N, 3) Würfelmitte in Welt-/Env-Koordinaten.
        quat_wxyz: (4,) oder (N, 4) Orientierung, Isaac-Lab-Konvention (w, x, y, z).
                   Wird normiert, damit numerisches Driften nichts verfälscht.
        half_edge: halbe Kantenlänge [m] (CUBE_EDGE / 2 aus scene_geometry).

    Returns:
        (N,) bzw. Skalar-Array: Mitte z minus half_edge · Σ|R[2, i]|. Σ|R[2,i]| ist die
        halbe Ausdehnung des Würfels entlang Welt-z, in Einheiten der halben Kante:
        1 aufrecht, √2 auf der Kante, √3 auf der Ecke.
    """
    p = np.atleast_2d(np.asarray(pos, dtype=float))
    q = np.atleast_2d(np.asarray(quat_wxyz, dtype=float))
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Dritte Zeile der Rotationsmatrix (Welt-z der drei Körperachsen).
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    extent = np.abs(r20) + np.abs(r21) + np.abs(r22)
    low = p[:, 2] - half_edge * extent
    return low if np.ndim(pos) > 1 else low.reshape(())


def lifted_clear_step(pos: np.ndarray,
                      quat_wxyz: np.ndarray,
                      hand_dist: np.ndarray,
                      table_top_z: float,
                      half_edge: float,
                      lift_threshold: float = LIFT_CLEAR_THRESHOLD_M,
                      contact_threshold: float = HAND_CONTACT_PROXY_M) -> tuple[bool, float]:
    """Ein Schritt der Metrik ``lifted_clear``.

    Args:
        pos:        (N, 3) Würfelmitten.
        quat_wxyz:  (N, 4) Würfel-Orientierungen.
        hand_dist:  (N,) kleinster Abstand Hand-Kontaktpunkt → Würfelmitte je Würfel [m].
        table_top_z: z der Tischoberkante im selben Koordinatensystem wie ``pos``.

    Returns:
        (erfüllt, Freiraum): erfüllt = irgendein Würfel hat seine Unterseite mehr als
        ``lift_threshold`` über dem Tisch UND eine Hand an sich; Freiraum = größte
        Unterseiten-Höhe über dem Tisch unter den Würfeln MIT Handkontakt [m] (0, wenn
        keiner Kontakt hat) — die Zahl, die eine Episode ohne ``lifted_clear`` einordnet.
    """
    low = np.atleast_1d(cube_lowest_z(pos, quat_wxyz, half_edge))
    clearance = low - float(table_top_z)
    held = np.asarray(hand_dist, dtype=float) <= contact_threshold
    held_clearance = float(clearance[held].max()) if held.any() else 0.0
    ok = bool(np.any(held & (clearance > lift_threshold)))
    return ok, max(held_clearance, 0.0)


# ---------------------------------------------------------------------------
# Clopper-Pearson
# ---------------------------------------------------------------------------

def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X ≤ k) für X ~ Bin(n, p), über Logarithmen (stabil auch für n ~ 1000)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k + 1):
        log_term = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                    + i * lp + (n - i) * lq)
        total += math.exp(log_term)
    return min(total, 1.0)


def _bisect(f, lo: float, hi: float, iters: int = 100) -> float:
    """Nullstelle einer auf [lo, hi] monoton fallenden Funktion f (f(lo) ≥ 0 ≥ f(hi))."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exaktes zweiseitiges (1 − alpha)-Konfidenzintervall für eine Quote k/n.

    Untere Grenze p_u: P(X ≥ k | p_u) = alpha/2, also 1 − CDF(k − 1; p_u) = alpha/2.
    Obere Grenze p_o:  P(X ≤ k | p_o) = alpha/2.
    Beide Bedingungen sind in p monoton → Bisektion. Randfälle k = 0 (untere Grenze 0)
    und k = n (obere Grenze 1) wie in der Literatur.
    """
    if n <= 0:
        return 0.0, 1.0
    if not 0 <= k <= n:
        raise ValueError(f"k={k} liegt nicht in 0..{n}")
    half = alpha / 2.0
    lower = 0.0 if k == 0 else _bisect(
        lambda p: half - (1.0 - _binom_cdf(k - 1, n, p)), 0.0, 1.0)
    upper = 1.0 if k == n else _bisect(
        lambda p: _binom_cdf(k, n, p) - half, 0.0, 1.0)
    return lower, upper


def format_rate_ci(k: int, n: int, alpha: float = 0.05) -> str:
    """'k/n = 10 % [KI95 2–31 %]' — kompakte Zeile für die Zusammenfassung."""
    if n <= 0:
        return f"{k}/{n} (keine gültigen Episoden)"
    lo, hi = clopper_pearson(k, n, alpha)
    level = round(100 * (1 - alpha))
    return f"{k}/{n} = {k / n:.0%} [KI{level} {lo:.1%}–{hi:.1%}]"
