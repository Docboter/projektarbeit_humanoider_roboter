# Fixes aus dem ersten Trainingsdurchlauf

**Erstellt:** 2026-06-03 · **Abgeleitet aus:**
[`lauf1-auswertung.md`](../ergebnisse/lauf1-auswertung.md) ·
**Betrifft:** Closed-Loop-Sim-Eval (Isaac Lab), nicht das Training selbst.

> Dieses Dokument sammelt die **konkreten Maßnahmen**, die aus der Auswertung des ersten
> kompletten Laufs (`g1_dex3_blockstacking_v1`, 175k Steps) folgen. Die Auswertung bleibt
> die Befund-Quelle; hier steht, **was deswegen geändert wurde** und **was noch offen ist**.

---

## 1. Ausgangslage (kurz)

Im Closed-Loop friert die Policy nach kurzem Anfahren ein (kein Greifen/Stapeln). Die
Diagnose über drei Tests (Closed-Loop / Replay / Open-Loop) ergab:

- **Modell ist fähig** (Open-Loop: gute Arm-Prädiktion auf echten Bildern),
- **Sim/Config ist korrekt** (Replay: korrekte Greifausführung),
- → wahrscheinliche Hauptursache: **visueller Domain-Gap** zwischen echten Trainingsbildern
  und synthetischen Sim-Renderings, bei **eingefrorenem Vision-Encoder**.

Der direkte Frame-Vergleich (Dataset-Referenz vs. Sim-Policy-Kameras) deckte konkrete,
**billig behebbare Asset-Unterschiede** auf — der Gap ist kleiner als zunächst angenommen
(weißer Hintergrund passt; es geht um Würfelfarbe, ein fehlendes Band und die Handfarbe).

---

## 2. Umgesetzte Fixes (Sim-Asset/Env, kein Retraining)

| # | Problem (Dataset vs. Sim) | Maßnahme | Datei | Status |
|---|---|---|---|---|
| 1 | Würfel **blau** statt **gelb** (`block_2`) | `diffuse_color` → gelb `(0.85,0.70,0.10)` | [g1_dex3_blockstack_env.py](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py) | ✅ |
| 2 | **schwarzes Stapel-Band fehlt** | statisches, dünnes schwarzes Cuboid `stack_band` ergänzt | [g1_dex3_blockstack_env.py](../../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py) | ✅ (Position genähert) |
| 3 | **Hände weiß** statt **schwarz** | schwarzes Material an die 16 Hand-Link-`/visuals` gebunden (Instancing-fest) | [recolor_hands_black.py](../../Simulation/g1_dex3_sim/recolor_hands_black.py) | ✅ (Skript, lokal verifiziert) |
| 4 | Asset musste manuell gesetzt werden | Default = `g1_dex3_blackhands.usd` + **`BLACK_HANDS`-Auto-Recolor** (selbstheilend, Fallback aufs Original) | s. §4 | ✅ |

**Einschätzung der Hebelwirkung** (auf den eingefrorenen Vision-Encoder): Handfarbe
**hoch** (Wrist-Kameras werden von der Hand dominiert) > Band **mittel-hoch** (Platzier-/
Stapel-Ziel) > Würfelfarbe **mittel**. Der einfachste Fix (Würfel) ist der schwächste
Hebel — ein belastbarer Test braucht **alle drei zusammen**.

---

## 3. Neue Werkzeuge

| Werkzeug | Zweck |
|---|---|
| [recolor_hands_black.py](../../Simulation/g1_dex3_sim/recolor_hands_black.py) | Offline-USD-Recolor (pxr): Hände schwarz. Instancing-fest, gezielt auf `left_hand_`/`right_hand_`. Läuft lokal (`pip install usd-core`) oder im Container. |
| [dump_policy_cams.py](../../Simulation/g1_dex3_sim/dump_policy_cams.py) | Speichert je ein Standbild der vier Policy-Kameras aus der Sim (für den Domain-Gap-Vergleich). |
| [compare_domain_gap.sh](../../Simulation/compare_domain_gap.sh) | Montiert Dataset- vs. Sim-Frames nebeneinander (beschriftet, je Kamera) zu einem Vergleichsbild. |
| [camera_reference/](../../Simulation/camera_reference/) | Echte Dataset-Referenzframes der vier Policy-Kameras (Dataset-Seite des Vergleichs). |

---

## 4. `BLACK_HANDS`-Mechanik (selbstheilend)

Die Launcher nutzen standardmäßig das schwarzhändige Asset und erzeugen es bei Bedarf
automatisch — ohne dass der Default je ins Leere zeigt:

- **Default:** `ASSET_PATH=…/g1_dex3_blackhands.usd`, `BLACK_HANDS=1`.
- **Bei Bedarf:** fehlt das schwarzhändige USD, wird es vor dem Sim-Start aus `g1_dex3.usd`
  per Recolor erzeugt (Isaac-Python via `isaaclab.sh -p`).
- **Fallback:** schlägt der Recolor fehl oder fehlt das Original, wird auf `g1_dex3.usd`
  zurückgefallen (kein Abbruch).
- **Abschalten:** `BLACK_HANDS=0` → Original-USD ohne Recolor.

Verdrahtet in: [entrypoint_sim.sh](../../Simulation/scripts/entrypoint_sim.sh),
[kisski_sim_submit.sh](../../Simulation/kisski_sim_submit.sh),
[kisski_replay_submit.sh](../../Simulation/kisski_replay_submit.sh); Defaults zusätzlich in
[dump_policy_cams.py](../../Simulation/g1_dex3_sim/dump_policy_cams.py) und
[g1_dex3_cfg.py](../../Simulation/g1_dex3_sim/g1_dex3_cfg.py).

---

## 5. Anwenden & verifizieren (Sim-Container, RT-Core-GPU)

1. **Frames mit den Fixes erzeugen** und gegen das Dataset legen:
   ```bash
   unset VIRTUAL_ENV
   ${ISAACLAB_PATH}/isaaclab.sh -p /workspace/g1_dex3_sim/dump_policy_cams.py \
       --headless --enable_cameras --out-dir /data/sim_cam_frames
   # sim_cam_*.png herunterladen, dann lokal:
   Simulation/compare_domain_gap.sh ./sim_cam_frames
   ```
   Prüfen: gelber Würfel ✔, schwarze Hände ✔, Band sichtbar ✔.
2. **Closed-Loop neu evaluieren** (Default nutzt jetzt das schwarzhändige Asset) und sehen,
   ob das Einfrieren verschwindet.

---

## 6. Offene Punkte / noch zu verifizieren

- **Band-Position/-Größe** sind eine Näherung → an die echte Dataset-Lage justieren.
- **Hand-Match** (`left_hand_`/`right_hand_`) und der **In-Container-Recolor**
  (`isaaclab.sh -p`) sind lokal nur über die pxr-Logik validiert, nicht im Container gefahren.
- **Würfelfarbe** ist der schwächste Hebel — Wirkung erst mit allen drei Fixes + neuer
  Closed-Loop-Eval beurteilbar.

## 7. Aus der Auswertung noch NICHT adressiert

- **Finger-Action-Qualität** (DEX3-Dims verrauscht, auch in-distribution) — Normalisierung
  der `*_dex3`-`ABSOLUTE`-Dims / mehr Greif-Demos. Siehe Auswertung §5–6.
- **Fehlende Eval-Metrik für den nächsten Trainingslauf** (`enable_open_loop_eval=true`,
  `eval_strategy="steps"`) + größere Batch-Size. Siehe Auswertung §6.4.
