# Augmentation — Bedienungsanleitung

> **TL;DR:** Baut und startet den `Augmentation/`-Container, der echte Trainingsvideos per
> NVIDIA Cosmos-Transfer2.5 stilvariiert und als Co-Training-fähigen Datensatz zusammenbaut.
> Läuft auf einem eigenen Docker-Server mit A100/H100-Klasse-GPU (~65 GB VRAM). Für die
> Trainings-seitige Einbindung (`USE_COTRAIN=1`) siehe [co-training.md](../training/co-training.md).

## 1. Warum Cosmos-Transfer2.5, und warum dieser Workflow?

**cosmos-transfer2.5** (nicht das ältere `cosmos-transfer1`) ist der aktiv gepflegte
NVIDIA-Nachfolger für Video-zu-Video-ControlNet-Transfer — kleineres 2B-Modell, Checkpoints
laden automatisch von HuggingFace beim ersten Lauf (kein separater 300-GB-Download-Schritt
wie bei transfer1), und das Repo bringt bereits Beispiel-Assets für Robotervideos mit
(`assets/robot_example/`).

Es gibt zwei Wege, damit Robotervideos zu augmentieren:

1. **Generisches Multimodal-ControlNet** (dieser Workflow) — Depth/Edge-Kontrollsignale
   werden aus dem Quellvideo selbst abgeleitet, kein zusätzlicher Datenprep nötig.
2. `robot_augmentation`-Submodul — erhält den Roboter pixelgenau, randomisiert nur den
   Hintergrund; braucht aber pro-Frame-Segmentierungsmasken des Roboters als Zusatzdaten.
   **Bewusst nicht gewählt** — der Maskierungs-Schritt ist zusätzlicher Scope, den dieses
   Projekt (noch) nicht braucht.

**Kein eigener Kontrollsignal-Schritt nötig** (frühere Version: erst cv2/Canny, dann
ffmpeg — beides an AV1-kodierten Quellvideos gescheitert, siehe §7 Historie): Cosmos-
Transfer2.5 erzeugt die Kantenerkennung selbst on-the-fly aus dem Quellvideo
(`cosmos_transfer2/config.py::EdgeConfig`: *"If None, edge is generated on-the-fly from
input video using CannyEdge Model"*), gesteuert über `AUGMENT_EDGE_THRESHOLD`
(`very_low`/`low`/`medium`/`high`/`very_high`, Default `medium`). Das Standardmodell ist
`AUGMENT_MODEL_VARIANT=edge` (immer verfügbar). Eine distillierte, schnellere Variante
(`edge/distilled`, `AUGMENT_NUM_STEPS=4` statt `35`) existiert, ist aber im Cosmos-Repo
hinter einem experimentellen Flag versteckt (`COSMOS_EXPERIMENTAL_CHECKPOINTS`) — wird
automatisch gesetzt, sobald `AUGMENT_MODEL_VARIANT` "distilled" enthält (siehe
`run_inference.sh`). Eine Depth-Variante (`AUGMENT_MODEL_VARIANT=depth`) ist als Basis-
Modell ebenfalls verfügbar, aber in diesem Container **nicht als Standardpfad getestet**
— offener Punkt, siehe §7.

## 2. Voraussetzung — Lizenz manuell akzeptieren (einmalig, außerhalb des Containers)

Cosmos-Transfer2.5-2B ist ein *gated* HuggingFace-Modell. Vor dem ersten Lauf:

1. https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B öffnen
2. Mit dem HF-Account einloggen, dessen Token als `HF_TOKEN` verwendet wird
3. „NVIDIA Open Model License Agreement" akzeptieren

Der Entrypoint prüft das beim Containerstart und bricht mit einer klaren Fehlermeldung ab,
statt nach Stunden Inferenz mit einem kryptischen 403 zu scheitern.

## 3. Build

```bash
docker build -t projekt-humanoider-roboter-augmentation Augmentation/
```

Baut in einer eigenen Stufe cosmos-transfer2.5 aus dem Quellcode (gepinnter Commit), fährt
dessen eigenes `just install`-Rezept, und legt daneben eine zweite, schlanke Python-venv fürs
Datensatz-Tooling an (Details: Kommentare im [Dockerfile](../../Augmentation/Dockerfile)).
**Noch nicht gegen einen echten Build verifiziert** — siehe „Offene Punkte" (§6).

## 4. Run

```bash
HF_TOKEN=hf_... AUGMENT_EPISODE_LIMIT=1 AUGMENT_CAMERAS=cam_left_high SKIP_INFERENCE=1 \
    ./Augmentation/setup_and_augment_DockerHub-pull.sh    # nur Specs generieren, Prompts prüfen

HF_TOKEN=hf_... AUGMENT_EPISODE_LIMIT=1 \
    ./Augmentation/setup_and_augment_DockerHub-pull.sh --resume   # billiger Testlauf, alle 4 Kameras

HF_TOKEN=hf_... AUGMENT_EPISODE_LIMIT=0 \
    ./Augmentation/setup_and_augment_DockerHub-pull.sh --resume   # voller, fortsetzbarer Lauf
```

`--resume` setzt denselben Container fort statt einen neuen zu erzeugen (Daten bleiben
erhalten — siehe `Augmentation/setup_and_augment_DockerHub-pull.sh --help`).

## 5. Env-Var-Referenz

| Variable | Default | Zweck |
|---|---|---|
| `HF_TOKEN` | — (Pflicht) | Zugriff auf den Datensatz UND das gated Modell (siehe §2) |
| `HF_HOME` | `/data/hf_cache` | Cosmos-Checkpoint-Cache, innerhalb `DATA_DIR` |
| `DATA_DIR` | `/data` | Wie im Training-Image |
| `NUM_GPU` | `1` | Cosmos' eigener Variablenname (`torchrun --nproc_per_node`) — bewusst **nicht** `NUM_GPUS` wie im Training-Image |
| `SKIP_DOWNLOAD` | `0` | Datensatz-Download überspringen, falls vorhanden |
| `SKIP_CONVERT` | `0` | v3.0→v2.1-Konvertierung überspringen, falls `modality.json` existiert |
| `SOURCE_DATASET_REPO` | `unitreerobotics/G1_Dex3_BlockStacking_Dataset` | HF-Datensatz-Repo |
| `AUGMENT_TRAIN_RATIO` | `0.8` | **Muss mit `TRAIN_SPLIT_RATIO`** im Training-Image übereinstimmen — dupliziertes Formel-Kontrakt, kein gemeinsames Artefakt (siehe `Training/scripts/lib_split.sh`) |
| `AUGMENT_EPISODE_LIMIT` | `2` | Kappung für billige erste Läufe; `0` = alle Train-Episoden |
| `AUGMENT_EPISODE_IDS` | `""` | Explizite, leerzeichengetrennte Indizes — überschreibt das Limit; bricht hart ab bei Index `>= n_train` |
| `AUGMENT_CAMERAS` | alle 4 Policy-Kameras | Teilmenge für billige/visuelle QS-Läufe; alle 4 nötig, damit eine Episode COTRAIN-nutzbar wird |
| `AUGMENT_VARIANTS` | `1` | Anzahl stilvariierter Kopien pro Quell-Episode/Kamera, je eine eigene Ausgabe-Episode |
| `AUGMENT_PROMPT_TEMPLATE` | eingebautes generisches Template | Erster konkreter Formulierungs-Durchgang — offener Punkt (§7) |
| `AUGMENT_EDGE_THRESHOLD` | `medium` | `very_low`/`low`/`medium`/`high`/`very_high` — steuert Cosmos' eigene on-the-fly-Kantenerkennung (niedriger = mehr erkannte Kanten inkl. Rauschen) |
| `AUGMENT_MODEL_VARIANT` | `edge` | Cosmos-Modellwahl laut `cosmos_transfer2/config.py::MODEL_CHECKPOINTS`: `depth`/`edge`/`seg`/`vis` immer verfügbar. `edge/distilled` (schneller, `AUGMENT_NUM_STEPS=4` statt `35`) existiert nur mit `COSMOS_EXPERIMENTAL_CHECKPOINTS=1` — wird automatisch gesetzt, wenn der Name "distilled" enthält. `depth/distilled` existiert **nicht** (nur edge hat eine distillierte Variante) |
| `AUGMENT_NUM_STEPS` | `35` | Diffusions-Schritte — Cosmos' eigener Default fürs Vollmodell. Bei `edge/distilled` reichen 4 |
| `AUGMENT_STRICT_FRAME_CHECK` | `1` | `1` = Episode bei Frame-/FPS-Abweichung verwerfen; `0` = trimmen (nur bei Frame-Überschuss möglich) |
| `AUGMENT_OUT_DIR` | `$DATA_DIR/augmentation/g1_dex3_cosmos_augmented` | Ziel-Datensatz (LeRobot v2.1) |
| `AUGMENT_HF_REPO` | `""` | Falls gesetzt: Upload nach Fertigstellung |
| `SKIP_INFERENCE` | `0` | Nach Spec-Generierung stoppen — Prompts vor der GPU-Zeit prüfen |
| `SKIP_ASSEMBLE` | `0` | Nach der Inferenz stoppen — Rohvideos vor dem Zusammenbau prüfen |
| `AUGMENT_OVERWRITE` | `0` | Bereits abgeschlossene Arbeitseinträge erneut ausführen |
| `SHELL_ON_ERROR` | `0` | Bei Fehler in eine Shell fallen statt abzubrechen |

## 6. Ins Training einbinden

Der fertige Datensatz unter `AUGMENT_OUT_DIR` (bzw. `AUGMENT_HF_REPO`, falls hochgeladen) ist
ein normaler LeRobot-v2.1-Datensatz mit **byte-identischer** `meta/modality.json` zum echten
Datensatz — genau der Vertrag, den `USE_COTRAIN=1` im Training-Image prüft
(`run_finetuning_cotrain.sh`, `cmp -s`). Keine Änderung im Training-Image nötig:

```bash
HF_TOKEN=hf_... WANDB_API_KEY=... \
USE_COTRAIN=1 COTRAIN_DATASET_PATH=/pfad/zum/augmentierten/datensatz COTRAIN_MIX_RATIO=0.25 \
    ./Training/setup_and_train_DockerHub-pull.sh
```

**`COTRAIN_MIX_RATIO` neu berechnen, nicht 0,25 blind übernehmen.** Die Formel aus
[co-training.md §4.2](../training/co-training.md#42-welches-mischungsverhältnis--025-gerendert)
gilt unverändert (`mix* = F_augmentiert / (F_augmentiert + F_echt)`), aber die konkrete Zahl
0,25 wurde für 60 gerenderte gegen 240 echte Episoden hergeleitet — bei anderer
`AUGMENT_EPISODE_LIMIT`/`AUGMENT_VARIANTS`-Kombination ergibt sich ein anderer Ausgleichspunkt.

## 7. Offene Punkte

1. **Checkpoint-Größe unbestätigt.** cosmos-transfer1 lud ~300 GB; das 2B-Modell sollte
   deutlich kleiner sein — vor dem ersten vollen Lauf `df -h` prüfen.
2. **„Parquet unverändert kopieren" setzt exakte Frame-/FPS-Erhaltung durch Cosmos voraus.**
   `AUGMENT_STRICT_FRAME_CHECK` + `ffprobe`-Vergleich in `assemble_dataset.py` ist das
   Sicherheitsnetz — am ersten echten Output verifizieren. Konkreter Verdächtiger laut
   `cosmos_transfer2/config.py`: Cosmos verarbeitet lange Videos in Chunks von
   `num_video_frames_per_chunk=93` Frames mit `num_conditional_frames` Überlappung —
   Rundungseffekte an Chunk-Grenzen könnten die Gesamt-Framezahl leicht verschieben.
3. **Inferenzkosten pro Episode/Kamera bei ~65 GB VRAM unbekannt** — `AUGMENT_EPISODE_LIMIT`
   (Default `2`) ist der eingebaute Knopf für billige erste Läufe.
4. **`AUGMENT_PROMPT_TEMPLATE`-Formulierung ist ein erster Entwurf**, nicht validiert. Die
   vier eingebauten Stil-Deskriptoren (Beleuchtung/Hintergrund/Tischfarbe) in
   `build_controlnet_specs.py` sind ein Ausgangspunkt für `AUGMENT_VARIANTS > 1`.
5. **`convert_v3_to_v2_standalone.py`s genaue Python-Abhängigkeiten sind ungeprüft** — das
   Dockerfile installiert eine plausible Teilmenge (pandas/pyarrow/jsonlines) in die
   Tools-venv; ggf. beim ersten Build nachschärfen.
6. **Cosmos' Ausgabe-Dateibenennung unter `-o <out_dir>` ist bis zum ersten echten Lauf
   unbekannt** — `assemble_dataset.py` sucht daher per Glob (`*.mp4`) statt einen festen
   Namen anzunehmen.
7. **Auflösung/FPS/Codec-Kompatibilität der Quellvideos mit Cosmos' Erwartung ist unbestätigt**
   — vor dem ersten vollen Lauf gegen `assets/robot_example/` im cosmos-transfer2.5-Repo
   gegenprüfen.
8. **`AUGMENT_MODEL_VARIANT=depth` ist als Cosmos-Basismodell verfügbar, aber in diesem
   Container nicht als Standardpfad getestet** — nur `edge` lief bisher real durch. Depth
   sollte laut Config genauso on-the-fly funktionieren (`VideoDepthAnything`), aber noch
   nicht verifiziert.
9. **Dockerfile-Übernahme aus cosmos-transfer2.5s eigenem Dockerfile ist eine Transkription,
   kein automatischer Abgleich** — bei zukünftigen `COSMOS_COMMIT`-Updates (`update_image.sh
   --update-commit`) ggf. `git show <sha>:Dockerfile` im cosmos-transfer2.5-Repo gegenprüfen,
   ob sich Basis-Image/System-Pakete geändert haben.
10. **`COSMOS_COMMIT` ist bewusst auf Tag `v1.5.0` gepinnt, nicht auf den neuesten Tag
    (`v1.5.4`) oder main-HEAD.** Beim ersten echten Build (2026-08-22) brach `just install
    cu128` mit `flash-attn ... doesn't have a source distribution or wheel for the current
    platform` ab: main/`v1.5.4` haben `.python-version` von 3.10 auf 3.13 angehoben, der
    custom flash-attn-Wheel-Index (`cu128_torch27`) liefert aber weiterhin nur `cp310`.
    `v1.5.0` ist der letzte Tag mit `.python-version=3.10` und funktioniert. **Nach jedem
    `--update-commit` den Build verifizieren** — Tags sind kein Garant gegen dieselbe
    Inkonsistenz, `update_image.sh` verfolgt nur die neueste *Tag*-Version, nicht main.
11. **Historie — eigener Control-Video-Schritt entfernt (2026-08-23).** Die ursprüngliche
    Version generierte Edge-Kontrollvideos selbst: erst per `cv2.Canny` (scheiterte an
    AV1-kodierten Quellvideos — `opencv-python-headless`s mitgeliefertes ffmpeg-Backend
    konnte sie nicht decodieren, lieferte aber still null Frames statt eines Fehlers), dann
    per System-`ffmpeg`/`edgedetect`-Filter (technisch funktionsfähig, aber unnötig
    komplex). Beim Debuggen des zweiten `--model`-Fehlers stellte sich heraus, dass
    `cosmos_transfer2/config.py::EdgeConfig` Kantenerkennung schon eingebaut hat
    ("generated on-the-fly ... using CannyEdge Model") — der ganze eigene Schritt entfällt
    seither, gesteuert nur noch über `AUGMENT_EDGE_THRESHOLD`.

## 8. Nachgelagerte QS (Follow-up, nicht Teil dieses Durchgangs)

- Geführtes CLI-Menü (`tools/menu/augment-*.spec`), analog zu `train-*.spec`.
- `tools/gen_docs.sh`-Drift-Check auf die neuen Skripte erweitern.
- Ein an [`measure_domain_gap.py`](../../Simulation/scripts/measure_domain_gap.py) angelehntes
  Werkzeug, das die frozen-SigLIP-Cosine-Distanz auch für augmentierte-vs-echte Bilder
  berechnet — erwartungsgemäß deutlich unter dem Sim-vs-Echt-Referenzwert 0,22 aus
  [domain-gap-analyse.md](../ergebnisse/domain-gap-analyse.md), da es sich um Echt-zu-Echt-
  Restyling handelt statt um Sim-Rendering.
