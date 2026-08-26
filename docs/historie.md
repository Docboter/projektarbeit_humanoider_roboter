# Historie — ausgelagerte veraltete Doku-Inhalte

> **TL;DR:** Rohmaterial für ein späteres Changelog: hierher wandern überholte, aus der
> Haupt-Doku ausgelagerte Inhalte. Nichts auf dieser Seite ist aktueller Stand — für aktuelle
> Informationen die verlinkte Zieldoku lesen.

> **Aufbau:** Je Eintrag Datum, Herkunftsdatei und Grund der Ablösung; die Einträge sind
> chronologisch nach dem Datum des Originalinhalts sortiert.
>
> **Abgrenzung:** Die vier historischen Planungs-Docs unter
> [simulation/archiv/](simulation/archiv/) behalten ihr eigenes Warnbanner-Muster und bleiben
> dort. Die Lauf-Auswertungen unter [ergebnisse/](ergebnisse/README.md) sind bewusst
> historische Berichte und gehören nicht hierher; dasselbe gilt für die
> [Diagnose-Chronik](ergebnisse/diagnose-chronik.md) (fortlaufend ab Lauf 08).
>
> **Hinweis zu Links:** In den zitierten Originaltexten sind die auffälligsten relativen Links
> an den neuen Dateiort angepasst; vereinzelte Links können noch auf den ursprünglichen
> Ort der Quelldatei bezogen sein.

---

## 2026-08-18 — Restrukturierung der Dokumentation (diese Datei entsteht)

Umbau der `docs/`-Struktur; zugleich der erste Eintrag des Changelog-Rohmaterials.

**Aufgeteilt:**
- `weiterfuehrend/rl-anleitung.md` (2 197 Zeilen): operative RL-Anleitung (556 Zeilen) bleibt;
  das komplette Lauf-Protokoll (damals Läufe 08–34, seither fortgeführt: Kamera-Diagnosen, Kalibrier-Iterationen,
  Domain-Gap-Messläufe, Greif-Physik, `span`-Gate, `TUNE_VISUAL`-Closed-Loop) ist jetzt
  [ergebnisse/diagnose-chronik.md](ergebnisse/diagnose-chronik.md) (1 667 Zeilen). Der alte,
  in sich widersprüchliche Status-Kopf („nächster Schritt TUNE_VISUAL" vs. „nächster Schritt
  Co-Training") wurde durch einen kurzen Status ersetzt; Statusquelle ist
  [weiterfuehrend/reinforcement-learning-plan.md](weiterfuehrend/reinforcement-learning-plan.md).

**Aufgelöst (Inhalt unten in dieser Datei):**
- `docs/umgebungsanalyse.md` (Audit vom 2026-06-04, fast vollständig per Nachträgen überholt)
- `docs/training/fixes-aus-erstem-lauf.md` (abgeschlossene Fix-Runde vom 2026-06-03)
- `docs/ergebnisse/wandb-run-charts.html.md` (Auto-Konvertierungs-Stub ohne Eigeninhalt, gelöscht)

**Umbenannt / verschoben:**
- `ergebnisse/wandb-run-auswertung.md` → [ergebnisse/lauf1-zwischenstand.md](ergebnisse/lauf1-zwischenstand.md)
  (fügt sich ins `lauf-N`-Namensschema; war der Mid-Run-Health-Check von Lauf 1)
- `ergebnisse/baseline-unitree-g1.md` → [simulation/baseline-eval.md](simulation/baseline-eval.md)
  (Plan-/Setup-Dokument ohne Ergebnisse; löst zugleich die Namensverwechslung mit
  `ergebnisse/basismodell-referenz-eval.md`)
- Latenz-Messung + Denoising-Sweep aus `simulation/live-ansicht.md` →
  [simulation/inferenz-optimierung.md](simulation/inferenz-optimierung.md)

**Gekürzt / entschlackt:**
- `ergebnisse/domain-gap-analyse.md`: Neumessung 2026-08-08 ist jetzt der Hauptinhalt;
  Juni-Erstmessung, alter Aktionsplan und altes Roh-JSON → Eintrag unten. Zielkriterium
  einheitlich < 0,35 (der alte Zweitwert ≤ 0,25 stammte aus dem verworfenen Juni-Plan).
- `simulation/vastai-anleitung.md`: WebRTC-Abschnitt auf den vast.ai-spezifischen Kern
  (Port-Problem, Stop/Restart) reduziert; toter Browser-Client-Text (Port 8211) → Eintrag unten.
- `simulation/umsetzungsnotizen.md`: §12-Momentaufnahme und alte §14-Messwerte → Einträge
  unten; neuer Lesehilfe-Block im Kopf. §-Nummern unverändert.

**In-Place-Korrekturen (sachlich falsch gewordene Aussagen):**
- `training/kisski-hpc.md`: SIF-Suchreihenfolge war seit der Portabilitäts-Migration
  (Commit c032231) verkehrt beschrieben; `apptainer`-Beispielpfad personengebunden;
  „save_total_limit=5" war doppelt falsch (richtig: Env-Var `SAVE_TOTAL_LIMIT`,
  Default 10, KISSKI-Default 40).
- `training/wandb-offline-sync.md`: alle sechs `~/.project/dir.project/…`-Pfade auf das
  portable Schema (`KISSKI_SIF_DIR`/Repo-Pfad) umgestellt.
- `training/anleitung.md`: `git checkout training-luca-KISSKI` (2×) → aktueller Branch
  `training-luca-IKR-IS6.0`; Resume-FAQ auf Kernaussage + Link gekürzt.
- Resume-/Namespace-Warnung (Job 15271760) stand dreifach in anleitung.md, kisski-hpc.md und
  env-vars.md — kanonisch jetzt nur noch in [training/env-vars.md](training/env-vars.md).
- `fehlerbehebung.md`: Closed-Loop-Eintrag von Stand „Läufe 29–31" auf Stand „Läufe 29–34"
  gebracht (Lauf 32 bestätigt Domain-Gap, Lauf 34 misst 27,6 % statt 20,5 %; nächster
  Schritt Co-Training).
- `training/co-training.md` §2: Greifpunkt-aus-Fingerkinematik als Hauptweg der
  Würfelplatzierung überholt (→ Eintrag unten); Hauptweg ist die Würfellage aus den
  Realbildern ([simulation/wuerfellage-rekonstruktion.md](simulation/wuerfellage-rekonstruktion.md)).
- `simulation/basismodell-referenzaufgabe.md`: Checkbox-Stand konsistent gemacht; Phase 2
  (Isaac-Lab-Referenztask) explizit als nicht weiterverfolgt markiert.
- `portabilitaet.md` §2 (Migration) als zeitgebundener Abschnitt markiert.

---

## 2026-06-03 — Fixes aus dem ersten Lauf

Quelle: `docs/training/fixes-aus-erstem-lauf.md`, aufgelöst am 2026-08-18. Die drei
Sim-Asset-Fixes — Würfelfarbe (gelb statt blau), das schwarze Stapel-Band und die
`BLACK_HANDS`-Auto-Recolor-Mechanik — sind längst im Sim-Asset bzw. den Sim-Launchern aktiv
und operativ in [`docs/simulation/vastai-anleitung.md`](simulation/vastai-anleitung.md) und
den `env-vars`-Referenzen dokumentiert; dieses Dokument ist reine Historie eines
abgeschlossenen Fixes und wird hier nur noch als Protokoll aufbewahrt.

# Fixes aus dem ersten Trainingsdurchlauf

**Erstellt:** 2026-06-03 · **Abgeleitet aus:**
[`lauf1-auswertung.md`](ergebnisse/lauf1-auswertung.md) ·
**Betrifft:** Closed-Loop-Sim-Eval (Isaac Lab), nicht das Training selbst.

> Dieses Dokument sammelt die **konkreten Maßnahmen**, die aus der Auswertung des ersten
> kompletten Laufs (`g1_dex3_blockstacking_v1`, 175k Steps) folgen. Die Auswertung bleibt
> die Befund-Quelle; hier steht, **was deswegen geändert wurde** und **was noch offen ist**.

---

### 1. Ausgangslage (kurz)

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

### 2. Umgesetzte Fixes (Sim-Asset/Env, kein Retraining)

| # | Problem (Dataset vs. Sim) | Maßnahme | Datei | Status |
|---|---|---|---|---|
| 1 | Würfel **blau** statt **gelb** (`block_2`) | `diffuse_color` → gelb `(0.85,0.70,0.10)` | [g1_dex3_blockstack_env.py](../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py) | ✅ |
| 2 | **schwarzes Stapel-Band fehlt** | statisches, dünnes schwarzes Cuboid `stack_band` ergänzt | [g1_dex3_blockstack_env.py](../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py) | ✅ (Position genähert) |
| 3 | **Hände weiß** statt **schwarz** | schwarzes Material an die 16 Hand-Link-`/visuals` gebunden (Instancing-fest) | [recolor_hands_black.py](../Simulation/g1_dex3_sim/recolor_hands_black.py) | ✅ (Skript, lokal verifiziert) |
| 4 | Asset musste manuell gesetzt werden | Default = `g1_dex3_blackhands.usd` + **`BLACK_HANDS`-Auto-Recolor** (selbstheilend, Fallback aufs Original) | s. §4 | ✅ |

**Einschätzung der Hebelwirkung** (auf den eingefrorenen Vision-Encoder): Handfarbe
**hoch** (Wrist-Kameras werden von der Hand dominiert) > Band **mittel-hoch** (Platzier-/
Stapel-Ziel) > Würfelfarbe **mittel**. Der einfachste Fix (Würfel) ist der schwächste
Hebel — ein belastbarer Test braucht **alle drei zusammen**.

---

### 3. Neue Werkzeuge

| Werkzeug | Zweck |
|---|---|
| [recolor_hands_black.py](../Simulation/g1_dex3_sim/recolor_hands_black.py) | Offline-USD-Recolor (pxr): Hände schwarz. Instancing-fest, gezielt auf `left_hand_`/`right_hand_`. Läuft lokal (`pip install usd-core`) oder im Container. |
| [dump_policy_cams.py](../Simulation/g1_dex3_sim/dump_policy_cams.py) | Speichert je ein Standbild der vier Policy-Kameras aus der Sim (für den Domain-Gap-Vergleich). |
| [compare_domain_gap.sh](../Simulation/compare_domain_gap.sh) | Montiert Dataset- vs. Sim-Frames nebeneinander (beschriftet, je Kamera) zu einem Vergleichsbild. |
| [camera_reference/](../Simulation/camera_reference/) | Echte Dataset-Referenzframes der vier Policy-Kameras (Dataset-Seite des Vergleichs). |

---

### 4. `BLACK_HANDS`-Mechanik (selbstheilend)

Die Launcher nutzen standardmäßig das schwarzhändige Asset und erzeugen es bei Bedarf
automatisch — ohne dass der Default je ins Leere zeigt:

- **Default:** `ASSET_PATH=…/g1_dex3_blackhands.usd`, `BLACK_HANDS=1`.
- **Bei Bedarf:** fehlt das schwarzhändige USD, wird es vor dem Sim-Start aus `g1_dex3.usd`
  per Recolor erzeugt (Isaac-Python via `isaaclab.sh -p`).
- **Fallback:** schlägt der Recolor fehl oder fehlt das Original, wird auf `g1_dex3.usd`
  zurückgefallen (kein Abbruch).
- **Abschalten:** `BLACK_HANDS=0` → Original-USD ohne Recolor.

Verdrahtet in: [entrypoint_sim.sh](../Simulation/scripts/entrypoint_sim.sh),
[kisski_sim_submit.sh](../Simulation/kisski_sim_submit.sh),
[kisski_replay_submit.sh](../Simulation/kisski_replay_submit.sh); Defaults zusätzlich in
[dump_policy_cams.py](../Simulation/g1_dex3_sim/dump_policy_cams.py) und
[g1_dex3_cfg.py](../Simulation/g1_dex3_sim/g1_dex3_cfg.py).

---

### 5. Anwenden & verifizieren (Sim-Container, RT-Core-GPU)

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

### 6. Offene Punkte / noch zu verifizieren

- **Band-Position/-Größe** sind eine Näherung → an die echte Dataset-Lage justieren.
- **Hand-Match** (`left_hand_`/`right_hand_`) und der **In-Container-Recolor**
  (`isaaclab.sh -p`) sind lokal nur über die pxr-Logik validiert, nicht im Container gefahren.
- **Würfelfarbe** ist der schwächste Hebel — Wirkung erst mit allen drei Fixes + neuer
  Closed-Loop-Eval beurteilbar.

### 7. Aus der Auswertung noch NICHT adressiert

- **Finger-Action-Qualität** (DEX3-Dims verrauscht, auch in-distribution) — Normalisierung
  der `*_dex3`-`ABSOLUTE`-Dims / mehr Greif-Demos. Siehe Auswertung §5–6.
- **Fehlende Eval-Metrik für den nächsten Trainingslauf** + größere Batch-Size.
  Siehe Auswertung §6.4.
  > **Korrektur 2026-08-13:** Hier standen `enable_open_loop_eval=true` und
  > `eval_strategy="steps"`. Beide sind im Fork wirkungslos bzw. führen zum Absturz —
  > In-Training-Eval existiert dort nicht. Umgesetzt ist stattdessen die Auswertung **nach**
  > dem Lauf: [`checkpoint_sweep.py`](../Training/scripts/checkpoint_sweep.py). Begründung
  > und Code-Fundstellen: [train-test-split.md](training/train-test-split.md) (Hinweis 1).

---

## 2026-06-04 — Umgebungsanalyse / Audit (Zeitdokument, aufgelöst)

**Quelle:** `docs/umgebungsanalyse.md`, aufgelöst am 2026-08-18.
**Überholt weil:** Alle Empfehlungen des Audits sind umgesetzt (`TUNE_VISUAL`,
`USE_AUGMENTATION`, `TRAIN_TEST_SPLIT`, RL mit FPO), die Domain-Gap-Zahlen sind durch die
Neumessung vom 2026-08-08 ersetzt (siehe `ergebnisse/domain-gap-analyse.md`). Das Dokument
bestand zuletzt überwiegend aus Nachträgen zum eigenen Alt-Text.
**Bei Auslagerung noch offen (Status zuletzt 2026-06-04 bewertet, seither nicht nachgepflegt):**
DEX3-Fingerqualität (noisig auch real→real), Joint-Sign-Flip-Convention (Indices 17/19/24/26,
nie gegen den GR00T-Server verifiziert), Erreichbarkeit des Success-Kriteriums (3 Bedingungen
gleichzeitig bei Friction 3.0).

### Originaltext (Stand bei Auslagerung, inkl. aller Nachträge)

### Umgebungsanalyse: Training & Simulation

> Erstellt: 2026-06-04 — Automatische Analyse der Entwicklungsumgebung
>
> ⚠️ **Zeitdokument (Stand 2026-06-04).** Die Analyse als solche ist gültig geblieben, mehrere
> Einzelbefunde und alle Empfehlungen sind es nicht mehr. Was sich seither geändert hat, ist
> unten jeweils direkt am Befund vermerkt. Die drei wichtigsten Punkte vorweg:
> - Die **Domain-Gap-Zahlen dieses Dokuments (Mittel 0,260 / `cam_left_wrist` 0,427) sind
>   überholt.** Sie wurden vor der Kamera-Neukalibrierung gemessen; die Neumessung vom
>   2026-08-08 liegt bei **0,2229 / 0,3556** → [ergebnisse/domain-gap-analyse.md](ergebnisse/domain-gap-analyse.md).
> - Die **Empfehlungen sind abgearbeitet**: Vision-Encoder-Fine-tuning (`TUNE_VISUAL=1`),
>   Domain Randomization (`USE_AUGMENTATION=1`, Default) und der 80/20-Split
>   (`TRAIN_TEST_SPLIT=1`) sind implementiert.
> - **RL ist gebaut** — allerdings mit FPO statt der hier empfohlenen SAC/PPO
>   → [weiterfuehrend/reinforcement-learning-plan.md](weiterfuehrend/reinforcement-learning-plan.md).

---

### Was konzeptuell gut ist

- **Obs/Action-Alignment**: Training-Config und Sim sind perfekt synchron — 4 Kameras, 28-dim Joints, gleiche Modality-Keys (`video.cam_left_high` usw.), gleiche Chunk-Size (16). Das ist solide.
- **Containerisierung**: Die dreistufige Pipeline (Download → Convert → Train) ist sauber und reproduzierbar. Env-Var-Only-Config funktioniert auf vast.ai, KISSKI und lokal ohne Änderungen.
- **Diagnose**: Der Root-Cause (Domain Gap + frozen ViT) wurde durch Replay-Vergleich methodisch korrekt isoliert.

---

### Fundamentale Denkfehler

#### 1. Frozen Vision Encoder + Sim-Eval — strukturell inkompatibel

Das Modell wurde auf realen Teleop-Bildern trainiert, der Vision Encoder dabei eingefroren (`tune_visual = false`). Evaluiert wird dann mit synthetischen Isaac-Sim-Bildern. Diese Kombination war **von Anfang an zum Scheitern verurteilt** — nicht als unvorhergesehene Entdeckung, sondern als vorhersehbare Konsequenz des Setups.

Der eigentliche Denkfehler: Es wurde erheblicher Aufwand in die Sim-Pipeline investiert, bevor die fundamentale Frage beantwortet war: *Kann das Modell überhaupt auf Sim-Bilder generalisieren?* Ein kurzer Offline-Test (Sim-Frame durch den ViT, Embedding-Distanz zu Real-Frame messen) hätte das in unter einer Stunde gezeigt.

##### Domain-Gap-Messung (2026-06-04) — ⚠️ überholt

> **Diese Erstmessung ist nicht mehr gültig.** Sie entstand vor den Kamera-Fixes vom August 2026
> (falsch geschriebene Kamera-Prim-Orientierungen, unwirksame Head-Cam-FOV, falsch montiertes
> Stereo-Paar, DLSS-Leerframes). Die Neumessung vom 2026-08-08 auf korrekt kalibrierten Kameras
> ergibt **Mittelwert 0,2229** und **`cam_left_wrist` 0,3556** — der Gap ist kleiner als hier
> angenommen, die Rangfolge der Kameras bleibt aber bestehen. Aktuelle Zahlen und Interpretation:
> [ergebnisse/domain-gap-analyse.md](ergebnisse/domain-gap-analyse.md). Die folgenden Werte
> bleiben nur als Vergleichspunkt stehen.

Durchgeführt mit `google/siglip-so400m-patch14-224` (identische Gewichte zum frozen GR00T-ViT).
Frames: `Simulation/camera_reference/` (Real) vs. `Simulation/runs/202606002/02/` (Isaac Sim).
Script: [`Simulation/scripts/measure_domain_gap.py`](../Simulation/scripts/measure_domain_gap.py)

| Kamera | Cosine-Distanz Real→Sim | Bewertung |
|---|---|---|
| `cam_left_high` | 0.148 | moderat |
| `cam_right_high` | 0.214 | groß |
| `cam_right_wrist` | 0.249 | groß |
| `cam_left_wrist` | **0.427** | **kritisch** |
| **Mittelwert** | **0.260** | groß |

Baseline-Vergleich:
- Real-vs-Real (verschiedene Kameras): **0.273** — der mittlere Real→Sim-Gap liegt leicht darunter
- Sim-vs-Sim (verschiedene Kameras): **0.211**

**Interpretation:** Die Overhead-Kameras allein wären noch vertretbar (0.15–0.21). Der kritische Befund ist `cam_left_wrist` mit 0.427 — diese Kamera zeigt die Greifaktion aus nächster Nähe. Synthethische Hände, Tisch-Texturen und Beleuchtung weichen so stark von realen Bildern ab, dass der frozen ViT das Wrist-Signal nicht übertragen kann. Weil die Policy alle 4 Kameras gleichzeitig verarbeitet, blockiert dieser eine Eingang die Ausführung.

#### 2. Training ohne Validation = Blindflug

`eval_strategy = "no"` bedeutet: 175 000 Trainingsschritte, kein Checkpoint-Vergleich, keine Metriken auf einem Hold-Out-Set. Der verwendete Checkpoint (Step 175k) wurde arbiträr gewählt. Es ist unbekannt, ob 50k oder 100k besser gewesen wären. Für ein reproduzierbares Forschungsprojekt ist das eine echte Lücke.

> **Nachtrag:** Die Infrastruktur für den Hold-Out existiert inzwischen —
> `TRAIN_TEST_SPLIT=1` hält 20 % der Episoden zurück
> ([run_finetuning.sh:102](../Training/scripts/run_finetuning.sh#L102),
> [training/train-test-split.md](training/train-test-split.md)). Der Befund gilt damit für den
> 1. Lauf, nicht mehr für das Setup.

#### 3. DEX3-Fingerqualität als bekanntes, ungelöstes Problem

Die Open-Loop-Auswertung zeigt, dass die DEX3-Dimensionen noisig sind — selbst auf realen Bildern. Das ist dokumentiert, wurde aber nicht adressiert. Wenn die Fingersteuerung im einfachsten Fall (real → real) bereits schlecht ist, ist Sim-Eval der falsche nächste Schritt; der Fehler liegt früher.

---

### Konkrete technische Bugs

#### Episode-Length-Diskrepanz (behoben)

> **Status 2026-06-04: erledigt.** Die ursprüngliche Diskrepanz (Config 180 s, aber nur
> 1 200 Steps = 40 s ausgeführt) besteht nicht mehr. Der Eval-Loop berechnet
> `max_steps = episode_length_s * policy_hz`
> ([run_g1_dex3_sim_eval.py:207](../Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py#L207)),
> die Konfiguration wird also tatsächlich ausgeführt.
>
> **Nachtrag 2026-08:** Der Default steht inzwischen auf `episode_length_s = 300.0`
> ([g1_dex3_blockstack_env.py:411](../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py#L411))
> = 9 000 Steps = 5 Minuten; überschreibbar per `EPISODE_LENGTH_S`. Der Docstring-Kommentar am
> Dateikopf ([g1_dex3_blockstack_env.py:9](../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py#L9))
> nennt weiterhin „5400 Steps @ 30 Hz = 180 Sekunden" und ist damit erneut veraltet — rein
> kosmetisch, der Wert wird aus der Config gelesen.

#### WANDB_MODE=offline als Default

`entrypoint.sh` setzt `WANDB_MODE=offline` — auch auf vast.ai, wo Internet verfügbar ist.
W&B-Daten müssen dann manuell nachgesyncet werden.

> **Korrektur zum ursprünglichen Befund:** „hardcoded" trifft nicht (mehr) zu. Der Entrypoint
> setzt `export WANDB_MODE="${WANDB_MODE:-offline}"`
> ([entrypoint.sh:175](../Training/scripts/entrypoint.sh#L175)) — offline ist nur der Default und
> lässt sich per `WANDB_MODE=online` überschreiben. Der Sim-/RL-Entrypoint verhält sich genauso
> ([entrypoint_rl.sh:126](../Simulation/scripts/entrypoint_rl.sh#L126)).

Relevante Datei: [`Training/scripts/entrypoint.sh`](../Training/scripts/entrypoint.sh)

#### Joint Sign-Flip — Convention-Leak-Risiko

Die Indices 17, 19, 24, 26 (DEX3-Proximalgelenke) werden sowohl in Observations als auch Actions negiert, weil USD-Achsen und Dataset-Convention gegensätzlich sind. Es wird nie verifiziert, dass der GR00T-Server dieselbe Convention annimmt. Ein Mismatch würde sich als stilles Fehlverhalten äußern.

Relevante Datei: [`Simulation/g1_dex3_sim/client.py`](../Simulation/g1_dex3_sim/client.py)

#### Success-Kriterium möglicherweise unerreichbar

Alle drei Bedingungen müssen gleichzeitig erfüllt sein:
- xy-Distanz < 3 cm zwischen Blöcken
- Height > 8 cm (Stack bestätigt)
- Velocity < 5 cm/s (stabil)

Bei Friction = 3.0 und physikalischer Blockdrift im Sim kann "stabil" möglicherweise nie zuverlässig getriggert werden.

Relevante Datei: [`Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py`](../Simulation/g1_dex3_sim/run_g1_dex3_sim_eval.py)

---

### Empfehlungen

#### Sofort (ohne GPU)

1. ~~**Episode-Length-Bug fixen**~~ ✅ **Erledigt** — Config steht auf 180 s, Eval rechnet korrekt 5 400 Steps (→ oben); nur ein kosmetischer Docstring-Kommentar ist noch veraltet.
2. ~~**Domain-Gap quantifizieren**~~ ✅ **Erledigt** — Messergebnis: mittlere Cosine-Distanz 0.260, `cam_left_wrist` kritisch bei 0.427 (→ oben).

#### Nächster sinnvoller Schritt (mit GPU) — ⚠️ inzwischen umgesetzt

> Alle drei Empfehlungen dieses Abschnitts sind implementiert. Sie stehen als Begründung der
> damaligen Priorisierung, nicht mehr als offene Aufgaben.

**Option A — Vision Encoder fine-tunen:** ✅ **umgesetzt** als `TUNE_VISUAL=1`
(→ [run_finetuning_vision.sh](../Training/scripts/run_finetuning_vision.sh), eigener
`blockstacking_vision`-Namespace). Auswertung:
[ergebnisse/lauf2-vision-auswertung.md](ergebnisse/lauf2-vision-auswertung.md).
`tune_visual = true` in der Trainings-Config, auf einem Mix aus realen und Sim-ähnlichen Bildern trainieren. Direktester Fix, aber teurer (mehr VRAM, längere Laufzeit).

**Option B — Domain Randomization in der Sim:** ✅ **umgesetzt** — Bild-Augmentierung läuft per
`USE_AUGMENTATION=1` standardmäßig mit ([run_finetuning.sh:52](../Training/scripts/run_finetuning.sh#L52));
in der Sim-Env kommen Albedo- und Dome-Light-Regler dazu.
Texturen, Beleuchtung, Hintergründe randomisieren, sodass Sim-Bilder realen Bildern ähnlicher werden. Kein Retraining nötig, aber erfordert Arbeit an der Sim-Env.

**Eval-Set aufbauen:** ✅ **umgesetzt** als `TRAIN_TEST_SPLIT=1` (Ratio über `TRAIN_SPLIT_RATIO`,
Default 0.8) → [training/train-test-split.md](training/train-test-split.md).
~20 % der Episoden zurückhalten und nach dem Lauf [`checkpoint_sweep.py`](../Training/scripts/checkpoint_sweep.py) über alle Checkpoints fahren — damit Checkpoint-Selektion beim nächsten Lauf nicht mehr blind ist. *(Korrektur 2026-08-13: hier stand `eval_strategy = "steps"` mit `eval_steps = 5000`. In-Training-Eval existiert im Fork nicht — `factory.py:26` bricht mit `assert eval_strategy == "no"` ab. Begründung: [training/train-test-split.md](training/train-test-split.md) Hinweis 1.)*

#### Längerfristig — ⚠️ teilweise überholt

RL (SAC/PPO auf Sim) macht erst Sinn, nachdem entweder der Domain Gap gelöst ist oder das Reward-Signal rein propriozeptiv ist (Vision-unabhängig). Sonst lernt RL eine Policy, die nur im Sim funktioniert und genauso an Domain Gap scheitert.

> **Nachtrag 2026-08:** RL ist gebaut — mit **FPO** statt SAC/PPO und nur auf dem Action-Head
> ([rl_finetune.py](../Simulation/g1_dex3_sim/rl_finetune.py)). Die hier formulierte
> Reihenfolge-Warnung hat sich bestätigt: die Pipeline läuft end-to-end, der Lernerfolg ist
> unverifiziert, und die Diagnose der Läufe 29–31 verweist zurück auf Wahrnehmung/Politik statt
> auf RL → [weiterfuehrend/reinforcement-learning-plan.md](weiterfuehrend/reinforcement-learning-plan.md).

---

### Zusammenfassung

| Bereich | Status | Priorität |
|---------|--------|-----------|
| Obs/Action-Alignment Training↔Sim | ✅ korrekt | — |
| Containerisierung & Reproduzierbarkeit | ✅ solide | — |
| Episode-Length-Bug | ✅ behoben | — |
| Validation während Training | ⚠️ fehlt → ✅ `TRAIN_TEST_SPLIT=1` verfügbar | hoch |
| Domain Gap (frozen ViT + Sim-Eval) | ❌ strukturell → ⚠️ verkleinert (0,2229 nach Neumessung) | hoch |
| DEX3-Fingerqualität | ⚠️ noisig | mittel |
| WANDB_MODE offline hardcoded | ⚠️ unkomfortabel → überschreibbar (`WANDB_MODE=online`) | niedrig |
| Joint Sign-Flip Convention | ⚠️ Risiko | niedrig |

> Die Spalte „Status" trägt jeweils zuerst die Bewertung vom 2026-06-04, nach dem Pfeil den
> Stand August 2026.

**Die Infrastruktur ist gut** — kein prinzipieller Fehler im Container-Setup oder im Obs/Action-Alignment. Der Kern des Problems liegt im Training-Eval-Loop: Frozen ViT + Real-only Training + Sim-Eval ist kein valides Setup — der gemessene Domain Gap bestätigt das quantitativ und ist der eigentliche Hebel.

> **Nachtrag 2026-08:** Die Neumessung nach der Kamera-Kalibrierung senkt den Gap auf
> **0,2229 im Mittel** und **0,3556** für `cam_left_wrist` — unter die real↔real-Baseline
> (0,2726). Der Domain Gap bleibt ein Faktor, ist aber nicht mehr die alleinige Erklärung für
> die 0-%-Erfolgsrate. Die Diagnose der Läufe 29–31 zeigt zusätzlich, dass die Politik nur
> ~19 % der demonstrierten Greifbewegung kommandiert — eine Fehlerquelle im Modell selbst,
> unabhängig vom Rendering.

---

## 2026-06-04 — Domain-Gap-Erstmessung (überholt durch Neumessung 2026-08-08)

Überholt, weil zwischen Juni und August sowohl die Kamera-Neukalibrierung (Sichtfeld 47,2° → 75°,
Montagepunkt auf `d435_link`, parallele statt konvergierender Stereobasis) als auch der
Albedo-Fix „schwarze Hände" (`BLACK_HANDS=1`) das Rendering grundlegend verändert haben. Der
damals entworfene Aktionsplan (2-Kamera-Option zuerst, siehe Option 2 unten) wurde nie verfolgt —
stattdessen kam der `TUNE_VISUAL`/Co-Training-Pfad zum Einsatz.

Der folgende Inhalt ist die vollständige, unveränderte Erstmessung aus
`docs/ergebnisse/domain-gap-analyse.md` (Stand vor der Restrukturierung am 2026-08-18), archiviert
zur Nachvollziehbarkeit. Aktuell gültig ist ausschließlich die Neumessung vom 2026-08-08 im
Hauptdokument.

---

### Ergebnisse (Erstmessung, Juni 2026)

### Per-Kamera Real → Sim

| Kamera | Cosine-Distanz | Bewertung |
|---|---|---|
| `cam_left_high` | 0.148 | moderat |
| `cam_right_high` | 0.214 | groß |
| `cam_right_wrist` | 0.249 | groß |
| `cam_left_wrist` | **0.427** | **kritisch** |
| **Mittelwert** | **0.260** | groß |

### Baseline-Vergleiche

| Vergleich | Cosine-Distanz |
|---|---|
| Real → Sim (Mittelwert) | **0.260** |
| Real → Real (verschiedene Kameras) | 0.273 |
| Sim → Sim (verschiedene Kameras) | 0.211 |

---

### Interpretation (Erstmessung, Juni 2026)

### Gesamtbild

Der mittlere Real→Sim-Gap (0.260) liegt leicht **unterhalb** des Real→Real-Cross-View-Gaps (0.273). Das heißt: im Durchschnitt wirken Sim-Bilder auf den frozen ViT nicht wesentlich fremder als eine andere Kamera-Perspektive innerhalb des realen Datensatzes. Dieser Befund ist überraschend positiv — und würde für sich allein keine Katastrophe bedeuten.

### Der Blocker: cam_left_wrist

`cam_left_wrist` hat mit **0.427** einen kritischen Ausreißer, der rund **1.6× über dem Real→Real-Baseline** liegt. Diese Kamera zeigt die Greifaktion aus nächster Nähe: synthetische Roboterhand, Tischoberfläche und Nahbereichs-Beleuchtung weichen so stark vom realen Trainings-Material ab, dass der frozen ViT das Signal nicht überträgt.

Da GR00T alle 4 Kameras gleichzeitig als Input verarbeitet, genügt ein einziger kritisch-fremder Eingang, um die Politiksequenz zu destabilisieren — und das ist genau die handnahe Kamera, die für Greifentscheidungen entscheidend ist.

### Warum die Sim zu "sauber" ist

Die Sim-interne Varianz (0.211) ist **niedriger** als die Real-interne Varianz (0.273). Isaac Sim rendert konsistente Materialien, gleichförmige Beleuchtung und identische Hintergrundszenarien ohne die natürlichen Schwankungen realer Aufnahmen (Reflexionen, leichte Beleuchtungsänderungen, Kamerarauschen). Das macht die Sim-Verteilung im Embedding-Space eng und weit von der breiten Real-Verteilung entfernt.

---

### Handlungsoptionen (alter Aktionsplan, Juni 2026 — nicht in dieser Form verfolgt)

### Option 1: Domain Randomization (empfohlen als erster Schritt)

**Aufwand:** 1–2 Tage, kein GPU, kein Retraining
**Ziel:** Wrist-Gap von 0.427 auf ≤ 0.25 senken

In [`g1_dex3_blockstack_env.py`](../Simulation/g1_dex3_sim/g1_dex3_blockstack_env.py) für die Wrist-Kameras aktivieren:
- Textur-Randomization (Tischoberfläche, Roboterhand-Material)
- Beleuchtungs-Randomization (Intensität, Farbe, Position)
- Hintergrund-Randomization (optional)

Erfolg messbar mit `measure_domain_gap.py` nach jeder Änderung — kein GPU-Run nötig zum Testen.

**Risiko:** Wie viel Gap-Closure realistisch erreichbar ist, hängt vom Renderer ab. PhysX-basierte Isaac-Lab-Umgebungen haben begrenzte Randomization-Tiefe verglichen mit Ray-Tracing-Renderern.

### Option 2: 2-Kamera-Konfiguration (schnellster Weg zum messbaren Ergebnis)

**Aufwand:** ~1 Tag Konfiguration + 1 KISSKI-Run (≈24 h)

Die zwei Overhead-Kameras (`cam_left_high`: 0.148, `cam_right_high`: 0.214) liegen im vertretbaren Bereich. Ein neues Training nur mit diesen zwei Kameras eliminiert den kritischen Wrist-Blocker vollständig.

Änderungen für den nächsten Trainingslauf:
- Modality-Config auf `g1_dex3_2cam_config` umstellen (existiert bereits in [`g1_dex3_config.py`](../app/Groot-1.6/examples/G1_DEX3/g1_dex3_config.py))
- `TRAIN_TEST_SPLIT=1` setzen und nach dem Lauf [`checkpoint_sweep.py`](../Training/scripts/checkpoint_sweep.py) fahren (behebt den Blind-Checkpoint-Fehler). *Korrektur 2026-08-13: hier stand `eval_strategy = "steps"`, was im Fork in eine `assert`-Sperre läuft — siehe [train-test-split.md](training/train-test-split.md) Hinweis 1.*
- Sim-Eval analog auf 2 Kameras umstellen

**Risiko:** Ohne Wrist-Information sieht die Policy nicht, was die Hand gerade macht — feine Fingersteuerung wird schwieriger. Der Robot kann Blöcke lokalisieren, aber keine propriozeptive Greif-Korrektur aus Kamerabildern ableiten.

**Status: wurde nie verfolgt.** Stattdessen kam der `TUNE_VISUAL`/Co-Training-Pfad (Option 3 unten) zum Einsatz.

### Option 3: tune_visual = true (strukturell korrekte Lösung)

**Aufwand:** Sim-Frames + KISSKI-Run (≈48 h, mehr VRAM)

Setzt `tune_visual = true` im Training-Config und trainiert auf einem Mix aus echten und Sim-ähnlichen Bildern. Der ViT lernt die Sim-Embedding-Verteilung direkt.

Empfohlene Reihenfolge:
1. Erst Domain Randomization einbauen (Option 1)
2. Dann Sim-Frames aus der Randomized-Sim als Trainings-Augmentation hinzufügen
3. `tune_visual = true` mit erhöhtem VRAM-Budget auf KISSKI H100

Diese Option ist die richtige Langzeitlösung, lohnt sich aber erst nachdem Option 1 oder 2 einen Baseline-Erfolg gezeigt hat.

---

### Empfohlene Reihenfolge (alter Aktionsplan, Juni 2026)

```
Jetzt (ohne GPU):
  → Option 2 konfigurieren + KISSKI-Job submittieren (läuft autonom)

Parallel:
  → Option 1 entwickeln, mit measure_domain_gap.py iterativ testen

Wenn Option 2 Erfolg zeigt:
  → Option 3 als Follow-up mit tune_visual=true

Wenn Option 1 den Wrist-Gap auf ≤ 0.25 senkt:
  → Erneute Sim-Eval mit dem bestehenden Checkpoint (kein Retraining nötig!)
```

---

### Rohdaten (Erstmessung, Juni 2026)

JSON-Output aus dem Messungslauf:

```json
{
  "model": "google/siglip-so400m-patch14-224",
  "per_camera": {
    "cam_left_high": 0.1477,
    "cam_right_high": 0.2136,
    "cam_left_wrist": 0.4275,
    "cam_right_wrist": 0.2491
  },
  "mean_real_sim": 0.2595,
  "mean_real_real_crossview": 0.2726,
  "mean_sim_sim_crossview": 0.2111,
  "verdict": "Large domain gap",
  "action": "ViT fine-tuning strongly recommended"
}
```

---

## 2026-06-04 — Sim-Aufbau: Momentaufnahme "Aktueller Stand"

Snapshot-Tabelle aus `docs/simulation/umsetzungsnotizen.md` §12, festgehalten kurz nach der
Physik-Kalibrierungs-Session (§13). Durch die spätere Entwicklung (Isaac-Sim-6.0-Port,
Kamera-Neukalibrierung, Domain-Gap-Neumessung 2026-08-08) überholt — für den aktuellen Stand
siehe stattdessen §10–§11 und §13–§15 von umsetzungsnotizen.md sowie
`docs/ergebnisse/diagnose-chronik.md`.

### Ursprünglicher Text (§12, umsetzungsnotizen.md)

Pipeline vollständig validiert. Replay-Diagnose bestätigt: Sim-Config ist korrekt, Greif-Physik
funktioniert. Closed-Loop-Versagen ist nachweislich **Domain Gap** (eingefrorenem Vision-Encoder),
nicht die Sim — Details in §13 (`umsetzungsnotizen.md#13-physics-calibration-session-2026-06-04`) und
[`lauf1-auswertung.md`](ergebnisse/lauf1-auswertung.md).

| Komponente | Status |
|---|---|
| Pipeline (download → server → sim → eval) | ✅ end-to-end verifiziert |
| 4 Policy-Kameras (high + wrist) | ✅ alle zeigen Tisch/Hände/Würfel |
| Szenen-Übersichtskamera (Video) | ✅ |
| Roboter-Startpose (Dataset Frame 0) | ✅ |
| **Tischhöhe / Würfelposition** | ✅ Tisch 0,87 m; `block_z_surface = 0,915` → Würfel-Oberkante z=0,940 = tiefster Handpunkt |
| **Dex3-Finger (Sign-Convention-Fix)** | ✅ `middle_0`/`index_0` (Indices 17,19,24,26) negiert in Actions + Obs |
| **Greif-Physik** | ✅ `max_cube_lift = 2,8 cm` (Schwelle >2 cm = Greifen bestätigt) |
| Aktions-Tracking | ✅ 0,021 rad mittlerer Arm-Fehler |
| Finger-Aktuatoren | ✅ stiffness=60, effort=20 N·m, solver_iter=8 |
| Würfel-Reibung | ✅ static=3,0 / dynamic=2,5 |
| `Dockerfile.vastai` | ✅ aktuell; Rebuild via `./update_sim_image.sh --vastai` |
| Modell `checkpoint-175000` Closed-Loop | ❌ Domain Gap (eingefroren. Vision-Encoder vs. Sim-Bilder) |
| KISSKI Sim-Eval | ❌ Blockiert (RTX 5000 Turing < Ampere) |

**Iteration auf warmer Instanz** (ohne Rebuild): geänderte Sim-Dateien per `scp` nach
`/workspace/g1_dex3_sim/`, dann `NUM_EPISODES=2 PYTHONUNBUFFERED=1 bash /scripts/entrypoint_sim.sh`.
Für reproduzierbaren Stand: Image neu bauen + pushen.

---

## 2026-06-04 — Domain-Gap-Erstmessung (Werkzeug-Kontext aus umsetzungsnotizen §14)

Erste Domain-Gap-Messung (real vs. Isaac-Sim-Kamerabilder) mit den drei Diagnose-Werkzeugen
aus `docs/simulation/umsetzungsnotizen.md` §14 (`dump_policy_cams.py`, `compare_domain_gap.sh`,
`measure_domain_gap.py`). Stammt von vor dem Isaac-Sim-6.0-Port und vor der Kamera-
Neukalibrierung — durch die Neumessung vom 2026-08-08 ersetzt
(siehe `docs/ergebnisse/domain-gap-analyse.md`).

### Ursprünglicher Text (§14, umsetzungsnotizen.md)

Messergebnis des ersten Laufs (mittlere Cosine-Distanz 0.260, `cam_left_wrist` kritisch bei
0.427) ist in `umgebungsanalyse.md` festgehalten.

> ⚠️ **Diese Erstmessung ist überholt.** Sie stammt von vor dem Isaac-Sim-6.0-Port und vor der
> Kamerakalibrierung. Gültig ist die Neumessung vom 2026-08-08: Mittelwert **0,2229**,
> `cam_left_wrist` **0,3556** → `docs/ergebnisse/domain-gap-analyse.md`.

### Alt-Werte im Überblick

| Kamera / Aggregat | Erstmessung (Juni 2026, überholt) | Neumessung (2026-08-08, gültig) |
|---|---|---|
| Mittelwert (alle Kameras) | 0,260 | 0,2229 |
| `cam_left_wrist` (kritisch) | 0,427 | 0,3556 |

---

## 2026-08-07 — vast.ai-Anleitung: WebRTC-Browser-Client (Port 8211) entfallen

Der Isaac-Sim-6.0-Port entfernte den eingebauten HTTP-Browser-Client auf Port 8211 ersatzlos;
Nachfolger sind der native *Isaac Sim WebRTC Streaming Client* (Spur A) sowie seit 2026-08-17
der eigene `webview`-Browser-Client (Spur A2) — beide dokumentiert in
`docs/simulation/live-ansicht.md`.

Ausgeschnitten aus `docs/simulation/vastai-anleitung.md`, Abschnitt „Optional — Live-Stream des
3D-Viewports (WebRTC)" (vor der Kürzung ca. Zeilen 386–517):

### Die lange Veraltet-Warnbox (vor der Kürzung)

> ⚠️ **Dieser Abschnitt ist in Teilen veraltet und beschreibt einen auf Hardware ungetesteten
> Weg.** Zwei Punkte vorweg:
> 1. **Der Browser-Client auf Port 8211 existiert nicht mehr.** Er stammt aus Isaac Sim ≤ 5.x
>    und ist mit der Migration auf Isaac Sim 6.0 (2026-08-07) entfallen; das Image exponiert
>    den Port bewusst nicht mehr ([Dockerfile.vastai](../Simulation/Dockerfile.standalone),
>    `EXPOSE 49100 8900`). Alle 8211-Angaben unten sind gegenstandslos — es bleibt der native
>    „Isaac Sim WebRTC Streaming Client" auf `LIVESTREAM_PORT` (49100).
> 2. **Spur A (WebRTC) ist nicht auf Hardware verifiziert.** Der Abschnitt liest sich wie ein
>    erprobter Workflow, ist aber Konzept
>    → [../weiterfuehrend/livestream-plan.md](weiterfuehrend/livestream-plan.md).
>
> **Funktionierende Alternative:** „Spur B" — der MJPEG-Frame-Stream `LIVE_VIEW=1` auf Port
> 8900. Reines HTTP, beliebig viele Zuschauer, per `ssh -L` tunnelbar, kein NVENC nötig.
>
> **Update 2026-08-13:** Die Defekte D1–D4 sind behoben (gemeinsame
> [`lib_livestream.sh`](../Simulation/scripts/lib_livestream.sh), versionsabhängige
> Kit-Settings, `PUBLIC_IP` nur noch bei `LIVESTREAM=1`), und `LIVESTREAM` wirkt jetzt auf
> alle vier Läufe. Die **Bedienung steht in [live-ansicht.md](simulation/live-ansicht.md)** — dieser
> Abschnitt hier behandelt nur noch den vast.ai-Sonderfall (zufälliges Port-Mapping).
> Auf dem eigenen Server ist der Weg deutlich einfacher: `LIVESTREAM=2`, Ports frei wählbar.

### Allgemeiner Kontext (vor der Kürzung, jetzt in live-ansicht.md abgedeckt)

Standardmäßig läuft die Eval **headless** und produziert nur MP4s. Mit `LIVESTREAM=1` streamt
Isaac Sim stattdessen den **3D-Viewport live per WebRTC** — zum Zuschauen beim Greif-Verhalten
in Echtzeit.

> **Seit 2026-08-13: live *statt* Video.** Bei aktivem Livestream wird `--video-dir` leer
> übergeben, es entstehen also **keine MP4s** mehr (spart pro Step eine GPU→CPU-Kopie).
> Wer beides braucht: `LIVE_KEEP_VIDEO=1`.

> **GPU-Voraussetzung:** WebRTC braucht den **NVENC**-Hardware-Encoder. Alle für die Sim
> ohnehin geeigneten GPUs (L40, RTX 3090/4090, A6000) haben NVENC — A100/H100 sind bereits
> aus zwei Gründen ausgeschlossen (keine RT-Cores **und** kein NVENC).

### „Verbinden"-Abschnitt mit Browser-Client-Rest (vor der Kürzung)

```
### Verbinden

Der Entrypoint gibt beim Start die Client-URL aus. Generell:

~~**Browser (am einfachsten):**~~ ⚠️ **entfallen** — der eingebaute Browser-Client
(`http://<IP>:8211/streaming/webrtc-client`) existierte nur bis Isaac Sim 5.x. Unter Isaac
Sim 6.0 liefert diese URL garantiert nichts.

**Native Isaac Sim WebRTC Streaming Client** (von NVIDIA, läuft ohne lokale GPU) — der
verbleibende Weg für Spur A:
Server eintragen als `<PUBLIC_IP>:<extern-gemappter-49100>`.

> Nur **ein** Client gleichzeitig pro Instanz. Kein VPN/Tunnel-IP (ZeroTier etc.) nutzen —
> WebRTC braucht die echte öffentliche IP. UDP-Mapping (47998) ist auf vast.ai weniger
> zuverlässig als TCP; falls das Bild nicht durchkommt, auf die MP4-Aufzeichnung zurückfallen.
```

Der vast.ai-spezifische Kern dieses Abschnitts (Port-Problem, Stop/Neustart, sowie die
vast.ai-spezifische Verbinden-Notiz mit `<PUBLIC_IP>:<extern-gemappter-49100>`) bleibt in
`docs/simulation/vastai-anleitung.md` erhalten — nur der oben zitierte, inzwischen
gegenstandslose Teil wurde entfernt.

---

## 2026-08-13 — KISSKI: TRAIN_TEST_SPLIT/USE_AUGMENTATION wurden nicht durchgereicht

> ⚠️ **Bis 2026-08-13 wurden beide Schalter NICHT durchgereicht.** `kisski_submit.sh` baute die
> `--env`-Liste für Apptainer von Hand, und `TRAIN_TEST_SPLIT`/`USE_AUGMENTATION` fehlten darin.
> Apptainer erbt die Job-Umgebung nicht automatisch — ein `TRAIN_TEST_SPLIT=1 sbatch …` verpuffte
> also folgenlos, ohne Fehlermeldung. Wer ältere Läufe auswertet: sie sind **ohne** Split gelaufen,
> unabhängig davon, was beim Absenden gesetzt war. Der Job-Kopf zeigt die Werte jetzt an und warnt,
> wenn kein Split aktiv ist.

**Auflösung:** Seit 2026-08-13 baut `kisski_submit.sh` die `--env`-Liste vollständig, reicht
`TRAIN_TEST_SPLIT`/`USE_AUGMENTATION` (und weitere Schalter) an den Container durch und zeigt
alle aktiven Werte im Job-Kopf-Log an.

---

## 2026-08-14 → 2026-08-17 — Co-Training-Renderer: Greifpunkt aus der Fingerkinematik als Hauptweg (überholt)

**Quelle:** `docs/training/co-training.md` §2 („Warum der Renderer zwei Stufen hat"), ausgelagert am 2026-08-18.
**Überholt weil:** Der erste volle Renderlauf (60 Episoden, 2026-08-17) zeigte, dass der aus
`scan.json` rekonstruierte Greifpunkt bei ~der Hälfte der Griffe auf dem Transportweg liegt
(48/116 jenseits von 60 % der Episode) — der Arm griff beim echten Pick ins Leere. Seit
2026-08-17 ist die **Würfellage aus den Realbildern** (`server_rl_run.sh layout`,
`extract_block_layout.py`) der Hauptweg; der Scan-Greifpunkt ist nur noch Notnagel.
Aufarbeitung: `docs/simulation/wuerfellage-rekonstruktion.md` §2.1; Befund: co-training.md § 3.2a.

### Originaltext (Stand vor 2026-08-18)

> Der Datensatz speichert **keine Objektposen**. Spielt man die Aktionen ab, während die Env
> ihre Würfel zufällig auslegt, entstehen Paare, in denen der Arm dorthin greift, wo kein
> Würfel liegt — und der Encoder lernt, den Würfel zu **ignorieren**. Das wäre schlimmer als
> gar nichts.
>
> Deshalb:
>
> 1. **`scan`** — Episode abspielen, Kameras auf ein Zehntel der Auflösung, Übersichtskamera
>    aus. Gesucht wird je Hand der Moment der engsten Fingeröffnung und der Schwerpunkt der
>    drei Fingerkuppen dort. Das ist der Ort, an dem in der echten Aufnahme ein Würfel lag.
>    → `scan.json`
> 2. **`render`** — dieselben Episoden mit den Würfeln an genau diesen Punkten (x/y aus dem
>    Greifpunkt, z = Tischauflage) und in kalibrierter Auflösung 640 × 480. → der Datensatz
