#!/usr/bin/env python3
# TL;DR: Erzeugt die Diagramme der Ausarbeitung (Lernkurven, Gate, Closed-Loop-Stufen, Fingerspanne, Orakel-Versatz) als PDF aus den CSVs unter latex/figures/daten/.
"""Abbildungen der Ausarbeitung aus den exportierten Messdaten.

Eingabe: ``latex/figures/daten/*.csv`` (Quelle und Lesart:
``latex/figures/daten/README.md``). Ausgabe: ``latex/figures/plots/*.pdf``. Die Zahlen werden nur
dargestellt, nicht verrechnet; einzige Ausnahme sind Anteile k/n.

Aufruf (aus dem Repo-Wurzelverzeichnis)::

    uv run --with matplotlib python latex/figures/make_figures.py
"""

import csv
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
DATA = HERE / "daten"
OUT = HERE / "plots"

# Kategoriale Farben in fester Reihenfolge (validierte Standardpalette, helle Fläche).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


def read(name):
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def de(x, nd=4):
    return f"{x:.{nd}f}".replace(".", ",")


def gate():
    rows = {r["checkpoint"]: r for r in read("gate_mse_haende.csv")}
    # Reihenfolge und Beschriftung; Farbe nach Generation (N1.6 blau, N1.7 orange, alt grau).
    sel = [
        ("N1.6 vision-v2 @44000 (Maßstab)", "N1.6 ohne Co-Training", BLUE, "//"),
        ("Lauf 1 @110000 (alt)", "N1.6 Lauf 1 (alt)", GRAY, ""),
        ("vision-v1 @35000 (alt)", "N1.6 vision-v1 (alt)", GRAY, ""),
        ("B N1.6 cube5 @44000", "N1.6 Co-Training 5 cm (B)", BLUE, ""),
        ("C N1.6 cube4 @44000", "N1.6 Co-Training 4 cm (C)", BLUE, ""),
        ("N1.7 vision-v1 @44000 (Maßstab)", "N1.7 ohne Co-Training", ORANGE, "//"),
        ("A N1.7 cube5 @44000", "N1.7 Co-Training 5 cm (A)", ORANGE, ""),
        ("D N1.7 cube4 @44000", "N1.7 Co-Training 4 cm (D)", ORANGE, ""),
    ]
    panels = [("echt", "Echtbild"), ("sim5", "Sim-Prüfset 5 cm"), ("sim4", "Sim-Prüfset 4 cm")]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.0), sharey=True)
    y = list(range(len(sel)))[::-1]
    for ax, (key, title) in zip(axes, panels):
        for yi, (ck, _, color, hatch) in zip(y, sel):
            r = rows[ck]
            m, lo, hi = (float(r[f"{key}_{s}"]) for s in ("mittel", "min", "max"))
            ax.barh(
                yi,
                m,
                height=0.62,
                color="white" if hatch else color,
                edgecolor=color,
                hatch=hatch,
                linewidth=1.0,
            )
            ax.errorbar(
                m, yi, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2
            )
        ax.set_title(title, fontsize=9, color=INK)
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: de(v, 3)))
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [s[1] for s in sel], fontsize=8)
    fig.supxlabel(
        "MSE Hände (Mittel der Seed-Mediane, Balken: Min–Max über 3 Seeds)", fontsize=8, color=INK2
    )
    fig.tight_layout()
    fig.savefig(OUT / "gate_mse_haende.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def closed_loop():
    rows = read("closed_loop.csv")
    pick = {
        "82": ("N1.6 ohne Co-Training (Lauf 82)", BLUE),
        "88+97": ("N1.6 Co-Training C, 60 Lagen (88+97)", ORANGE),
        "89": ("N1.7 Co-Training A (Lauf 89)", AQUA),
        "93": ("N1.7 Co-Training D (Lauf 93)", YELLOW),
    }
    stages = [
        ("reached", "reached"),
        ("touched", "touched"),
        ("lifted_ohne_artefakte", "lifted"),
        ("lifted_clear", "lifted_clear"),
        ("stacked", "stacked"),
    ]
    by_run = {r["chronik_lauf"]: r for r in rows if r["chronik_lauf"] in pick}
    sel = [(pick[k], by_run[k]) for k in pick]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    w = 0.19
    for i, ((label, color), r) in enumerate(sel):
        n = int(r["episoden"])
        for j, (col, _) in enumerate(stages):
            k = int(r[col])
            x = j + (i - 1.5) * w
            ax.bar(x, k / n, width=w - 0.02, color=color, label=label if j == 0 else None)
            if j >= 2:  # nur die kleinen Stufen direkt beschriften
                ax.text(
                    x,
                    k / n + 0.02,
                    f"{k}/{n}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=INK2,
                    rotation=90,
                )
    ax.set_xticks(range(len(stages)), [s[1] for s in stages])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Anteil der Episoden")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}".replace("%", " %")))
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(OUT / "closed_loop_stufen.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def span():
    rows = [r for r in read("closed_loop.csv") if r["chronik_lauf"] == "96"]
    names = {
        "C N1.6 cube4 @44000, 10x40 s": ("N1.6 Co-Training C", BLUE),
        "vision-v2 @44000, 10x40 s": ("N1.6 vision-v2 (ohne)", GRAY),
        "vision-v1 @35000, 10x40 s": ("N1.6 vision-v1 (alt)", GRAY),
        "Lauf 1 @110000, 10x40 s": ("N1.6 Lauf 1 (alt)", GRAY),
    }
    fig, ax = plt.subplots(figsize=(6.3, 2.0))
    for yi, r in enumerate(rows[::-1]):
        label, color = names[r["modell"]]
        m, lo, hi = (float(r[f"spanne_{s}_rad"]) for s in ("median", "min", "max"))
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(m, yi, "o", color=color, markersize=7, markeredgecolor="white")
        ax.text(hi + 0.04, yi, de(m, 2) + " rad", va="center", fontsize=7, color=INK2)
    ax.axvline(2.094, color=INK2, linestyle="--", linewidth=0.8)
    ax.text(2.094, len(rows) - 0.45, "Demo 2,094 rad", ha="right", fontsize=7, color=INK2)
    ax.set_yticks(range(len(rows)), [names[r["modell"]][0] for r in rows[::-1]], fontsize=8)
    ax.set_xlim(0, 2.3)
    ax.set_xlabel("Fingerspanne des Kommandos je Episode (Punkt: Median, Linie: Min–Max)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: de(v, 1)))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "fingerspanne_lauf96.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def sweep():
    rows = read("orakel_versatz_lauf62.csv")
    dxs = sorted({float(r["dx_cm"]) for r in rows})
    dys = sorted({float(r["dy_cm"]) for r in rows})
    cell = {
        (float(r["dx_cm"]), float(r["dy_cm"])): (int(r["lifted_clear_k"]), int(r["episoden"]))
        for r in rows
    }
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#eef4fc", "#184f95"])
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    for i, dx in enumerate(dxs):
        for j, dy in enumerate(dys):
            if (dx, dy) not in cell:
                continue
            k, n = cell[(dx, dy)]
            ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, color=ramp(k / n)))
            ax.text(
                j,
                i,
                f"{k}/{n}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if k / n > 0.45 else INK,
            )
    ax.set_xlim(-0.5, len(dys) - 0.5)
    ax.set_ylim(len(dxs) - 0.5, -0.5)
    ax.set_xticks(range(len(dys)), [de(v, 1) for v in dys])
    ax.set_yticks(range(len(dxs)), [de(v, 1) for v in dxs])
    ax.set_xlabel("dy: Würfel seitlich nach außen (cm)")
    ax.set_ylabel("dx: vom Roboter weg (cm)")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / "orakel_versatz_lauf62.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def lernkurven():
    rows = read("lernkurven_lauf3_lauf4.csv")
    val = read("val_mse_lauf3_lauf4.csv")

    def median15(xs):
        out = []
        for i in range(len(xs)):
            w = sorted(xs[max(0, i - 7) : i + 8])
            out.append(w[len(w) // 2])
        return out

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.6))
    for col, label, color in [
        ("loss_lauf3_n16", "Lauf 3 (N1.6)", BLUE),
        ("loss_lauf4_n17", "Lauf 4 (N1.7)", ORANGE),
    ]:
        pts = [(int(r["schritt"]), float(r[col])) for r in rows if r[col]]
        a1.plot(
            [p[0] for p in pts], median15([p[1] for p in pts]), color=color, lw=1.2, label=label
        )
        vp = [
            (int(r["schritt"]), float(r[col.replace("loss", "mse")]))
            for r in val
            if r[col.replace("loss", "mse")]
        ]
        a2.plot(
            [p[0] for p in vp], [p[1] for p in vp], "o-", color=color, lw=1.2, ms=4, label=label
        )
    a1.set_yscale("log")
    a1.set_title("Trainings-Loss (gleitender Median, 15 Werte)", fontsize=8, color=INK)
    a2.set_title("Validierungs-MSE, 20 Testepisoden × 750", fontsize=8, color=INK)
    for ax in (a1, a2):
        ax.set_xlabel("Trainingsschritt")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: de(v, 3)))
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "lernkurven_lauf3_lauf4.pdf", metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    gate()
    closed_loop()
    span()
    sweep()
    lernkurven()
    print("geschrieben:", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))
