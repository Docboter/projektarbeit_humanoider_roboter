# Implementierungsplan — geführte CLI-Menüs für Training und Simulation

> **TL;DR:** Design- und Begründungsdokument für die geführten CLI-Menüs in Training und Simulation
> — inzwischen umgesetzt (Status-Banner unten). Erklärt Architektur, Parameter-Spezifikation als
> einzige Quelle und, in §12, die Abweichungen vom ursprünglichen Plan.

> **Status: umgesetzt am 2026-08-20** auf `training-luca-IKR-IS6.0`. Alle sieben Phasen
> stehen, mit zwei bewussten Abweichungen vom Plan (Phase 5 und §4.4) — siehe
> [§12 Umsetzungsstand](#12-umsetzungsstand). Der Plantext unten bleibt als
> Begründungsprotokoll erhalten; wo die Umsetzung abweicht, steht es dort.
> **Nachtrag 2026-08-21:** ein Einstiegspunkt [`./run.sh`](../../run.sh) über allen
> Launchern, Pfeiltasten-Bedienung, eine Gruppenebene für lange Aktionslisten und der
> Checkpoint als Grundfrage jedes Messlaufs —
> [§13](#13-nachtrag-2026-08-21--ein-einstiegspunkt-und-pfeiltasten).
>
> **Bedienung in einem Satz:** `./run.sh` starten — oder eines der Skripte direkt, ohne
> Parameter. Beides führt durch; `[←]` bzw. `[z]` führt jederzeit eine Ebene zurück. `MENU=0` bzw. `--no-menu` schaltet das ab, `MENU_ARROWS=0`
> nur die Pfeiltasten; jeder bisherige Aufruf funktioniert unverändert weiter.
>
> Idee aus der Projektabsprache vom 2026-08-20:
> *„Wenn man das Sim-Skript ohne Parameter startet, soll es die nötigen Werte in der CLI
> abfragen und ggf. erklären — dann muss man die Parameter nicht kennen."*
>
> Verwandt: [portabilitaet.md](../portabilitaet.md) (`.env.local`-Mechanik, auf der das Menü
> aufsetzt) · [env-vars.md](../training/env-vars.md) (die Tabelle, die später generiert werden
> soll) · [rl-anleitung.md](rl-anleitung.md) (die Bedienung, die das Menü abbilden muss)

---

## 0. Kurzfazit

Die Idee passt gut zum bestehenden Code, weil **jeder Einstiegspunkt ausschließlich über
Env-Vars konfiguriert wird** — das Menü muss also nichts umbauen, sondern nur Env-Vars füllen
und danach den unveränderten Pfad laufen lassen.

Der eigentliche Fallstrick ist ein anderer: Die Parameterbeschreibungen existieren heute schon
**viermal** (Shell-Default `${VAR:-…}`, `usage()`-Heredoc, [env-vars.md](../training/env-vars.md),
CLAUDE.md-Tabelle). Ein handgeschriebenes Menü wäre Kopie Nr. 5 und driftet garantiert
auseinander. Der Plan baut deshalb **eine deklarative Parameter-Spezifikation als einzige
Quelle**, aus der Menü, `usage()` und Doku-Tabellen erzeugt werden. Ziel ist, dass am Ende
*weniger* Zeilen im Repo stehen als vorher.

Drei Leitentscheidungen tragen alles Weitere:

1. **Das Menü erzeugt nur Env-Vars** und läuft vor dem Dispatch — kein zweiter Ausführungspfad.
2. **Menü nur, wenn wirklich ein Mensch davorsitzt** — SLURM, CI, vast.ai und die Container-
   Entrypoints dürfen nie an einem `read` hängen bleiben.
3. **Eine Spezifikation, drei Ausgaben** (Menü, Hilfe, Doku).

Aufwand: **1–2 Tage** für den tragenden Teil (Engine + `eval` + Trainings-Launcher),
**3–4 Tage** für den vollständigen Ausbau inklusive Doku-Generierung.

---

## 1. Befund am bestehenden Code

Ursprünglich erhoben auf `training-luca-IKR-IS6.0-GN1.7`; die Spalte „hier" ist auf
`training-luca-IKR-IS6.0` vor der Umsetzung nachgezählt worden.

| Befund | GN1.7 | hier |
|---|---|---|
| Aktionen in [`server_rl_run.sh`](../../Simulation/server_rl_run.sh) | 20 | **19** (`preflight … clean`, plus Alias `down`), Dispatch ab Zeile 1637 |
| Distinkte Env-Vars, die dasselbe Skript liest | 99 | **101** |
| Länge des `usage()`-Heredocs | ~200 Zeilen | **171 Zeilen** (1463–1634) |
| Gesamtlänge des Skripts | 1757 Zeilen | 1667 Zeilen / 91 KB |
| Weitere Host-Einstiegspunkte | `setup_and_train_DockerHub-pull.sh`, `setup_and_train_Container-build.sh`, `server_robocasa_ref_run.sh`, `update_image.sh`, `update_sim_image.sh`, `kisski_*.sh` | dieselben |
| Env-Vars im Trainings-Entrypoint | 20 | **19** (`entrypoint.sh`), plus die Feature-Schalter aus `run_finetuning*.sh` |
| **Davon vom Host-Launcher durchgereicht** | — | **6** — der Rest kam nie im Container an (siehe §12.1) |

Weitere für den Entwurf relevante Beobachtungen:

- **Alles ist env-getrieben.** Sowohl die Host-Launcher als auch die Container-Entrypoints
  lesen ausschließlich Env-Vars. Ein Menü, das nur exportiert, ist damit vollständig
  ausreichend — es braucht keinen einzigen neuen Parameter-Kanal.
- **Es gibt bereits Ansätze von Interaktivität**, aber handgestrickt und nur an zwei Stellen:
  [`setup_and_train_DockerHub-pull.sh`](../../Training/setup_and_train_DockerHub-pull.sh)
  fragt in Zeile ~152 nach `resume/destroy/abbrechen` und in Zeile ~165–181 nach
  `HF_TOKEN`/`WANDB_API_KEY`. Das ist der erste Anwendungsfall, den die Engine ablöst.
- **Keine TUI-Bibliothek verfügbar.** `whiptail`, `dialog`, `gum`, `fzf` sind auf dem
  Entwicklungsrechner nicht installiert und dürfen auf dem IKR-Server und dem
  KISSKI-Login-Node nicht vorausgesetzt werden. `jq` ist vorhanden, wird aber nicht gebraucht.
- **`.env.local` ist bereits die richtige Schicht** für alles Rechnerspezifische
  (siehe [portabilitaet.md](../portabilitaet.md)) — das Menü setzt darauf auf, statt sie zu
  ersetzen. **Beim Umsetzen zeigte sich allerdings:** nur `server_rl_run.sh` und
  `server_robocasa_ref_run.sh` lasen die Datei überhaupt. In den Trainings-Launchern galt
  die in `portabilitaet.md` beschriebene Vorrangregel gar nicht — ein dort hinterlegter
  `HF_TOKEN` wurde trotzdem abgefragt. Siehe §12.1.
- **`start_logging` spiegelt die gesamte Ausgabe** nach `$HOST_DATA_DIR/logs/<aktion>-<ts>.log`
  (Zeile 217 ff., `exec > >(tee -a …) 2>&1`). Das hat zwei Konsequenzen für das Menü —
  Reihenfolge und Geheimnisse, siehe §6.

---

## 2. Leitentscheidungen

### 2.1 Das Menü erzeugt nur Env-Vars

Das Menü läuft **vor** dem `case`-Dispatch, exportiert die ermittelten Werte und ist danach
fertig. `do_eval`, `do_rl` und alle übrigen Aktionsfunktionen bleiben byte-identisch.

Damit gibt es keinen „Menü-Modus" mit eigenem Verhalten, den man separat testen müsste, und
kein Risiko, dass ein interaktiver Lauf sich anders verhält als ein Skript-Lauf. Das ist die
Bedingung dafür, dass sich die Umstellung überhaupt lohnt: Ein Menü, das eigene
`docker run`-Aufrufe zusammenbaut, wäre ein zweiter Wartungspfad.

### 2.2 Menü nur bei echtem Terminal *und* fehlendem Argument

```bash
menu_enabled() {
  [[ "${MENU:-auto}" != 0 ]] || return 1        # harter Ausschalter
  [[ "${MENU:-auto}" == 1 ]] && return 0        # harter Einschalter (--menu)
  [[ -t 0 && -t 1 ]] || return 1                # kein Terminal -> nie fragen
  [[ -z "${SLURM_JOB_ID:-}" ]] || return 1      # SLURM-Batch
  [[ -z "${CI:-}" ]] || return 1
  [[ ! -f /.dockerenv ]] || return 1            # im Container niemals
  return 0
}
```

Der Container bleibt damit **vollständig autonom** — genau das ist die Voraussetzung dafür,
dass dasselbe Image weiter auf vast.ai und KISSKI läuft. Das Menü ist ausschließlich
Host-Werkzeug.

### 2.3 Eine Spezifikation, drei Ausgaben

Aus derselben Deklaration entstehen:

1. die **Menüfrage** (Kurztext, Default, Typ, Validierung, Langtext auf `?`),
2. die **`usage()`-Zeile** (Variablenname, Default, Kurztext),
3. die **Doku-Tabellenzeile** in [env-vars.md](../training/env-vars.md) (Langtext).

Ohne diesen Teil ist das Vorhaben nicht empfehlenswert — dann wäre das Menü nur eine weitere
Stelle, die veraltet.

### 2.4 Ausdrückliche Nicht-Ziele

- **Kein Vollbild-TUI.** Verträgt sich schlecht mit der `tee`-Log-Spiegelung, mit SSH-Pipes
  und mit dem Anzeigen des äquivalenten Ein-Zeilers. Reines `read`/`printf`.
- **Kein Menü im Container.** Entrypoints bleiben nicht-interaktiv.
- **Keine neue Konfigurationsdatei anstelle von `.env.local`.** Das Menü ergänzt sie.
- **Kein Ersatz für die Doku.** Der äquivalente Ein-Zeiler wird immer angezeigt, damit man
  beim dritten Mal ohne Menü auskommt.
- **Keine Python-Abhängigkeit.** Die Host-Launcher laufen, bevor irgendein venv garantiert ist;
  auf dem KISSKI-Login-Node kommt Python erst über `module load`.

---

## 3. Architektur

### 3.1 Dateien

Geplant war das hier; **tatsächlich entstanden** ist die rechte Fassung:

```text
tools/
├── lib_menu.sh              # Menü-Engine, reines Bash (>= 4.2), ~870 Zeilen
├── lib_env_local.sh         # ★ NEU: gemeinsames Laden der .env.local (§12.1)
├── gen_docs.sh              # ★ PRÜFT statt zu generieren (§12.2)
├── test_menu.sh             # ★ NEU: der Prüfplan aus §8, ausführbar (30 Prüfungen)
└── menu/
    ├── _order.spec          # ★ Reihenfolge der Gruppen (sonst alphabetisch, §12.3)
    ├── _order-train.spec
    ├── _order-kisski.spec
    ├── _common.spec         # was ALLE teilen (HF_TOKEN)
    ├── _common-sim.spec     # ★ je Prefix getrennt: LIVE* gehört nicht ins Training
    ├── _common-train.spec
    ├── _common-kisski.spec
    ├── sim-<aktion>.spec    # 19 Stück, je Aktion von server_rl_run.sh eine
    ├── train-{train,resume,destroy,interactive}.spec
    └── kisski-{train,openloop,rl,sim}.spec
```

Dazu ein neues Host-Skript [`Training/kisski_menu.sh`](../../Training/kisski_menu.sh):
der Login-Node-Weg aus §4.3. Bewusst eigenständig, weil `kisski_submit.sh` weiterhin
allein auf den Cluster kopierbar bleiben soll (`scp kisski_submit.sh …`) und deshalb
keine Abhängigkeit auf `tools/` bekommen darf.

**Warum `tools/` und nicht `Simulation/scripts/`:** Letzteres wird in die Images **kopiert**
(`COPY scripts/ /scripts/`), jede Änderung bräuchte dort einen Image-Rebuild. Die Host-Launcher
sind nicht im Image — Menüänderungen kosten so nie einen Build. Zusätzlich vermeidet ein
gemeinsames `tools/` die Doppelablage, die bei den *im Image* liegenden Bibliotheken durch die
Docker-`COPY`-Grenze erzwungen wird — für die Host-Launcher gilt diese Grenze nicht.

> **Branch-Hinweis:** Der Plan wurde auf `training-luca-IKR-IS6.0-GN1.7` erhoben. `GROOT_VERSION`
> ist dort die Laufzeit-Auswahl zwischen N1.6 und N1.7; auf `training-luca-IKR-IS6.0` existiert die
> Variable **nicht** — alle `GROOT_VERSION`-Nennungen unten entfallen hier ersatzlos, bis der
> N1.7-Pfad gemerged ist. Am Rest des Plans ändert das nichts.

### 3.2 Die Spezifikations-DSL

Eine Spec-Datei ist reines Bash: eine Folge von Aufrufen der Funktionen `action`, `group`,
`param`, `when` und `note`, die `lib_menu.sh` bereitstellt. Kein Parser nötig, und
Abhängigkeiten zwischen Fragen lassen sich direkt ausdrücken.

```bash
# tools/menu/sim-eval.spec
action eval "BC-Erfolgsrate in der Sim messen (Closed Loop)" \
       --group  "Messen" \
       --needs  setup \
       --hint   "Schritt 3 der Diagnosekette und der Nullpunkt jedes RL-Vergleichs"

group "Zugang"
param HF_TOKEN         secret  ""    basic \
      "HuggingFace-Token" \
      "Wird für den Checkpoint-Download gebraucht. Dauerhaft besser in .env.local ablegen — dann fragt das Menü hier nicht mehr."

group "Laufumfang"
param NUM_EPISODES     int     20    basic \
      "Anzahl Eval-Episoden" \
      "20 ist der Standard für vergleichbare Zahlen; 2 für einen Rauchtest." \
      --range 1:200

param EPISODE_LENGTH_S int     120   basic \
      "Zeitbudget je Episode (Sekunden)" \
      "0 bedeutet bis zu 9000 Steps — das sind Stunden. Die menschliche Demo dauert 39 s, das Dreifache davon sind 120 s." \
      --range 0:3600

group "Geschwindigkeit"
param SCENE_CAM        bool    1     advanced \
      "Übersichtskamera cam_scene rendern" \
      "0 spart eine von fünf Kameras je Step und ändert die MODELL-EINGABE NICHT — die Policy sieht cam_scene nie. Kostet nur MP4/Übersichtsbild."

param CAM_RES_SCALE    choice  1.0   advanced \
      "Kamera-Auflösung" \
      "Halbieren geht quadratisch in die Renderzeit ein, ändert aber die Modell-Eingabe. Nur zum Zuschauen, nicht für Messläufe." \
      --options "1.0:voll (Messlauf);0.5:halb (nur Zuschauen)"

param GROOT_INFERENCE_BACKEND choice eager advanced \
      "Inferenz-Backend" \
      "TensorRT braucht vorher 'optimize all'. Details: docs/simulation/inferenz-optimierung.md" \
      --options "eager:Standard;compile:torch.compile;tensorrt:vorher 'optimize all'"

when '[[ "${GROOT_INFERENCE_BACKEND}" == tensorrt ]]'
param GROOT_TRT_ENGINE_PATH path "" advanced \
      "TensorRT-Engine" \
      "Leer = per Checkpoint-Fingerprint automatisch finden."

param CAMERA_RENDER_EVERY_N int 1 expert \
      "Nur jedes n-te Frame rendern" \
      "Muss exakt dem EXECUTION_HORIZON entsprechen, sonst sieht die Policy veraltete Bilder."
```

**Typen:** `str` · `int` · `float` · `bool` · `choice` · `secret` · `path`
Jeder Typ bringt Validierung und Darstellung mit; `--range a:b` und `--options k:text;…` sind
typspezifische Zusätze.

**Stufen:**

| Stufe | Verhalten im Menü | Verhalten in Hilfe/Doku |
|---|---|---|
| `basic` | wird immer gefragt (Ziel: 3–5 Fragen je Aktion) | ja |
| `advanced` | nur hinter `[e] erweiterte Optionen` | ja |
| `expert` | wird nie gefragt, nur per Env setzbar | ja |

Die Stufeneinteilung ist die wichtigste einzelne Entwurfsentscheidung: `eval` liest heute rund
15 relevante Variablen. Alle 15 abzufragen wäre **schlechter** als der Status quo.

### 3.3 Öffentliche API von `lib_menu.sh`

| Funktion | Zweck |
|---|---|
| `menu_enabled` | Rückgabewert 0, wenn gefragt werden darf (§2.2) |
| `menu_pick_action <specdir>` | Aktionsliste anzeigen, gewählte Aktion **auf stdout** ausgeben |
| `menu_ask <action>` | Spec laden, fragen, Ergebnis exportieren |
| `menu_render_usage [<action>]` | Aktionsliste bzw. Parameterblock für `usage()` erzeugen |
| `menu_render_docs <action>` | Markdown-Tabellenzeilen für `docs/` erzeugen |
| `menu_replay_line <action>` | Den äquivalenten Ein-Zeiler bauen (Geheimnisse maskiert) |

**Implementierungsregeln, die nicht verhandelbar sind:**

- Die gesamte Menü-Oberfläche geht nach **stderr**, nur das Ergebnis von `menu_pick_action`
  nach stdout. Sonst frisst die Kommandosubstitution die Anzeige.
- Interne Helfer heißen `_menu_*`, damit sie die `log`/`ok`/`warn`/`err`-Helfer der beiden
  Launcher (die unterschiedlich definiert sind) nicht überschreiben.
- `read` liefert bei EOF einen Rückgabewert ≠ 0 und reißt unter `set -euo pipefail` das ganze
  Skript mit. Jeder Aufruf braucht `|| true` **plus** eine echte EOF-Behandlung, sonst stirbt
  `./skript < /dev/null` kommentarlos.
- Terminalbreite über `${COLUMNS:-$(tput cols 2>/dev/null || echo 80)}`.
- Mindestens Bash 4.2 (`${var,,}` wird schon heute benutzt); Version am Anfang prüfen.

### 3.4 Rangfolge der Werte

Die heutige Regel bleibt, das Menü schiebt sich nur ein:

```text
explizite Umgebungsvariable  >  Menü-Antwort  >  .env.local  >  Default im Skript
```

Eine **bereits gesetzte** Variable wird deshalb **nicht gefragt**, sondern als „vorgegeben"
angezeigt. Da `.env.local` in `server_rl_run.sh` bereits vor dem Menü mit `set -a` eingelesen
wird, fällt das automatisch heraus — mit dem angenehmen Nebeneffekt, dass **das Menü umso
stiller wird, je besser `.env.local` gepflegt ist**. Das lenkt Nutzer in die richtige
Dauer-Konfiguration, statt sie davon abzuhalten.

### 3.5 Zustand: Recall und Profile

```text
.menu/                               # gitignoriert
├── last/<skript>.<aktion>.env       # zuletzt benutzte Antworten
└── profiles/<name>.env              # benannte Profile (--profile rauchtest)
```

- Beim Start einer Aktion, für die eine `last`-Datei existiert:
  `[Enter] = letzte Konfiguration wiederholen`.
- Benannte Profile machen aus „man muss die Parameter kennen" ein
  „man wählt *Rauchtest* oder *Messlauf*".
- **Geheimnisse landen dort nie.** Statt des Werts wird ein Kommentar geschrieben
  (`# HF_TOKEN kam aus .env.local`).

---

## 4. Einbau je Einstiegspunkt

### 4.1 `Simulation/server_rl_run.sh` — 8 Zeilen

Heute (Zeile 1727 ff.):

```bash
# ── Dispatch ──────────────────────────────────────────────────────────────────
require_docker
ACTION="${1:-help}"
case "$ACTION" in
  preflight|setup|…) start_logging "$ACTION" ;;
esac
```

Danach:

```bash
# ── Dispatch ──────────────────────────────────────────────────────────────────
require_docker
source "$REPO_DIR/tools/lib_menu.sh"

ACTION="${1:-}"
if [[ -z "$ACTION" ]] && menu_enabled; then
    ACTION="$(menu_pick_action "$REPO_DIR/tools/menu" sim)" || { warn "Abgebrochen."; exit 0; }
fi
: "${ACTION:=help}"
menu_enabled && menu_ask "$ACTION"      # exportiert; fragt nichts bereits Gesetztes

case "$ACTION" in
  preflight|setup|…) start_logging "$ACTION" ;;
esac
```

**Wichtig: das Menü läuft vor `start_logging`.** Zwei Gründe:

1. `start_logging` setzt `exec > >(tee -a "$LOG_FILE") 2>&1`. Bash schreibt den `read -p`-Prompt
   nach **stderr**; durch die `tee`-Prozesssubstitution kann die Ausgabereihenfolge zwischen
   Prompt und Eingabe verrutschen — das Menü wirkt dann kaputt.
2. Der HF-Token darf nicht in die Log-Datei.

Direkt **nach** `start_logging` gibt das Menü dann die aufgelöste Konfiguration maskiert aus —
das ist die Provenienz-Information, die im Log stehen soll.

`usage()` wird zu einem Rumpf, der `menu_render_usage` aufruft. Die generierbaren Teile
(Aktionsliste, Parameter je Aktion) verschwinden aus dem Heredoc; die handgeschriebenen
Prosablöcke (Beispiele, „LIVE-Variante", „Tempo der Sim", GR00T-Versionshinweis) **bleiben** —
sie erklären Zusammenhänge, nicht einzelne Parameter.

### 4.2 `Training/setup_and_train_DockerHub-pull.sh`

Hier ersetzt das Menü die zwei handgestrickten Abfragen (Zeile ~152 `resume/destroy` und
~165–181 `HF_TOKEN`/`WANDB_API_KEY`) und ergänzt die Trainingsparameter
(`MAX_STEPS`, `GLOBAL_BATCH_SIZE`, `NUM_GPUS`, `TUNE_VISUAL`, `TRAIN_TEST_SPLIT`,
`USE_COTRAIN`, `USE_AUGMENTATION`; auf dem N1.7-Branch zusätzlich `GROOT_VERSION`).

Zusätzlicher Gewinn hier: Das Menü kann die **VRAM-Tabelle aus CLAUDE.md operativ machen** —
`nvidia-smi --query-gpu=memory.total` auslesen und `GLOBAL_BATCH_SIZE` passend vorschlagen
(24 GB → 2, 32 GB → 8, 80 GB → 64). Das ist genau die Art Wissen, die heute in einer Tabelle
steht, die man vorher gelesen haben muss.

Dieselbe Spec bedient `setup_and_train_Container-build.sh`; der Unterschied ist nur die
Bau-statt-Pull-Frage.

### 4.3 KISSKI (`Training/kisski_*.sh`, `Simulation/kisski_*.sh`)

Auf dem **Login-Node** baut das Menü die Absende-Zeile und zeigt sie an:

```bash
export HF_TOKEN=…  GLOBAL_BATCH_SIZE=32  MAX_STEPS=44000
sbatch Training/kisski_submit.sh
```

**Im Batch-Job selbst nie** — das erledigt bereits die `SLURM_JOB_ID`-Prüfung aus §2.2.
Zusätzlich sollte das Menü hier die Partitionswahl (`kisski` A100 80 GB / `kisski-h100`
94 GB) und die Walltime (max. 48 h) abfragen, weil das die zwei Werte sind, die man ohne
[kisski-hpc.md](../training/kisski-hpc.md) nicht rät.

### 4.4 Übrige Skripte

`server_robocasa_ref_run.sh`, `update_image.sh` und `update_sim_image.sh` haben wenige
Parameter und stehen hinten an. `update_*_image.sh` bekämen im Wesentlichen die Frage
„auch `:latest` pushen?" (`--push-latest`) — die ist heute leicht zu übersehen.

---

## 5. Bedienablauf (Sollzustand)

```text
$ ./Simulation/server_rl_run.sh

  Was möchtest du tun?

  Vorbereiten
   1) preflight    Image + GPU prüfen                     ✓ zuletzt heute 14:02
   2) setup        Checkpoint + USD von HF laden          ✓ liegt bereits vor
  Ansehen
   3) view         Szene ohne Modell/Checkpoint ansehen
  Messen
   4) eval         BC-Erfolgsrate in der Sim              ← Schritt 3 der Diagnosekette
   5) span         Fingerspanne auf echten Bildern
   6) gap          Domain-Gap real vs. sim                ! braucht vorher 'cams'
  …
  > 4

  eval — BC-Erfolgsrate in der Sim (Closed Loop)
  ────────────────────────────────────────────────────────────
  HF_TOKEN                hf_… (aus .env.local)

  Anzahl Eval-Episoden  (NUM_EPISODES)
     20 ist der Standard für vergleichbare Zahlen; 2 für einen Rauchtest.
     [20] >

  Zeitbudget je Episode in Sekunden  (EPISODE_LENGTH_S)
     [120] > ?
     ↳ 0 bedeutet bis zu 9000 Steps — das sind Stunden. Die menschliche Demo
       dauert 39 s, das Dreifache davon sind 120 s.
     [120] >

  ────────────────────────────────────────────────────────────
   1 NUM_EPISODES        20
   2 EPISODE_LENGTH_S    120
   3 SCENE_CAM           1          (erweitert)
   4 GROOT_INFERENCE_…   eager      (erweitert)

  [Enter] starten   [1-4] ändern   [e] erweiterte Optionen
  [b] nur Befehl zeigen   [p] als Profil sichern   [a] abbrechen

  Entspricht:
    NUM_EPISODES=20 EPISODE_LENGTH_S=120 ./Simulation/server_rl_run.sh eval
```

Zwei Details, die den Unterschied machen:

- **Die Zusammenfassung ist eine Bestätigungsseite mit Nummern zum Ändern**, kein linearer
  Fragebogen ohne Zurück. Sonst tippt man bei Frage 6 falsch und fängt von vorn an.
- **`?` als Antwort zeigt den Langtext.** Das ist die „ggf. erklärt"-Anforderung aus der
  ursprünglichen Idee, ohne dass jede Frage einen Absatz Erklärung mitschleppt.

### 5.1 Zustandsbewusstsein

Die Aktionsliste markiert, was schon erledigt ist — das kann eine statische Anleitung
prinzipiell nicht:

| Marker | Ermittlung |
|---|---|
| `setup` ✓ | Checkpoint-Verzeichnis unter `$HOST_DATA_DIR` vorhanden |
| `preflight` ✓ + Datum | jüngste `preflight-*.log` in `$HOST_DATA_DIR/logs/` |
| `gap` ! braucht `cams` | erwartete Kamera-PNGs fehlen |
| `eval` mit TensorRT ! | `optimize`-Artefakte unter `/data/optimized/` fehlen |

Bei einer Kette wie `preflight → setup → cams → gap → eval → render → rl` ist genau das der
Punkt, an dem Einsteiger hängenbleiben.

---

## 6. Fallstricke

| Falle | Umgang |
|---|---|
| **15 Fragen für `eval`** — schlimmer als heute | Stufen `basic`/`advanced`/`expert` (§3.2); gefragt werden 3–5, der Rest liegt hinter `[e]` |
| **Token im Log und in der History** | `start_logging` spiegelt *alles*. Also: Menü vor `start_logging`; `read -rsp` für `secret`; in Zusammenfassung, Ein-Zeiler und Recall-Datei maskiert (`hf_…`) |
| **`read` + `set -euo pipefail`** | `read` liefert bei EOF ≠ 0 und beendet das Skript. Jeder Aufruf `|| true` plus echte EOF-Behandlung |
| **Kommandosubstitution frisst die Anzeige** | Menü-Oberfläche nach stderr, nur das Ergebnis nach stdout |
| **Rangfolge kippt** | Bereits gesetzte Variablen werden angezeigt, nicht gefragt (§3.4) |
| **Specs werden selbst zur Doppelpflege** | Nur tragfähig mit generiertem `usage()` und generierten Doku-Tabellen (Phase 5). Ohne Phase 5 lieber ganz sein lassen |
| **Menü blockiert Automatisierung** | Harte Ausschalter `MENU=0`; jede Aktion muss weiterhin vollständig per Argument + Env aufrufbar sein. Das ist ein Testkriterium, kein Nebenaspekt (§8) |
| **Nicht-ASCII in der Anzeige** | Das Repo nutzt in Skripten bewusst ASCII-Umschreibungen (`ue`, `ae`). Für die Menü-Oberfläche sind Umlaute in Ordnung (UTF-8-Terminal), aber die Rahmenzeichen auf `─│✓!←` beschränken |

---

## 7. Umsetzungsreihenfolge

Bewusst inkrementell — nach jeder Phase ist das Repo benutzbar.

### Phase 1 — Engine + eine Aktion (Beweis) — ✓ erledigt
`tools/lib_menu.sh`, `tools/menu/sim-eval.spec`, Hook in `server_rl_run.sh`.
**Abnahme:** `./Simulation/server_rl_run.sh` ohne Argument führt durch `eval`;
`NUM_EPISODES=20 ./Simulation/server_rl_run.sh eval` verhält sich **unverändert**;
`./Simulation/server_rl_run.sh eval < /dev/null` bricht mit klarer Meldung ab statt zu hängen.

### Phase 2 — Trainings-Launcher — ✓ erledigt
`tools/menu/train-local.spec`, Hook in `setup_and_train_DockerHub-pull.sh`, Ablösung der
zwei handgestrickten Abfragen, VRAM-basierter Vorschlag für `GLOBAL_BATCH_SIZE`.
**Abnahme:** Ein Trainingsstart ist ohne Blick in [env-vars.md](../training/env-vars.md)
möglich; der Ein-Zeiler in der Zusammenfassung startet denselben Lauf.

### Phase 3 — Restliche Sim-Aktionen — ✓ erledigt
Specs für die übrigen 19 Aktionen, `_common.spec` für die geteilten Parameter
(`HF_TOKEN`, `LIVESTREAM*`, `LIVE_VIEW*`, `SHELL_ON_ERROR`; auf dem N1.7-Branch zusätzlich
`GROOT_VERSION`).
**Abnahme:** Jede Aktion aus dem Menü erreichbar; `usage()` und Menü nennen dieselben Defaults
(automatisch prüfbar, siehe §8).

### Phase 4 — Recall und Profile — ✓ erledigt
`.menu/last/`, `.menu/profiles/`, `--profile <name>`, `.gitignore`-Eintrag.
**Abnahme:** Zweiter Aufruf derselben Aktion ist mit einem `[Enter]` erledigt.

### Phase 5 — Generierung von Hilfe und Doku — ⚠ bewusst geändert, siehe §12.2
`menu_render_usage` ersetzt die generierbaren Teile des Heredocs;
`tools/gen_docs.sh` schreibt zwischen `<!-- BEGIN generated: … -->`-Marker in
[env-vars.md](../training/env-vars.md) und eine neue Sim-Variablen-Tabelle.
Prüfmodus `tools/gen_docs.sh --check` für einen Pre-Commit-Hook.
**Abnahme:** Ein geänderter Default in einer Spec taucht ohne weitere Handgriffe in `--help`
**und** in der Doku auf; `--check` schlägt bei veralteter Doku fehl.

### Phase 6 — KISSKI und Nebenskripte — ✓ KISSKI erledigt, Nebenskripte entfallen (§12.4)
Login-Node-Menü, das die `export …; sbatch …`-Zeile baut (Partition, Walltime, Batch-Size);
`update_image.sh` / `update_sim_image.sh` (`--push-latest`); `server_robocasa_ref_run.sh`.
**Abnahme:** Ein KISSKI-Job lässt sich ohne Blick in [kisski-hpc.md](../training/kisski-hpc.md)
absenden.

### Phase 7 — Doku-Pass — ✓ erledigt
[README.md](../../README.md), [docs/README.md](../README.md),
[anleitung.md](../training/anleitung.md), [rl-anleitung.md](rl-anleitung.md) und CLAUDE.md
auf „ohne Parameter starten führt durch" hinweisen. Die bestehenden Ein-Zeiler-Beispiele
**bleiben** — sie sind weiterhin der schnellere Weg für Geübte.

**Aufwand:** Phasen 1–2 zusammen 1–2 Tage (der tragende Teil). Phasen 3–7 weitere 2 Tage.

---

## 8. Test- und Validierungsplan

Alles ohne GPU und ohne Container prüfbar — das Menü ist reine Host-Logik.

| # | Prüfung | Erwartung |
|---|---|---|
| 1 | `MENU=0 ./Simulation/server_rl_run.sh eval` | verhält sich exakt wie heute |
| 2 | `./Simulation/server_rl_run.sh eval < /dev/null` | kein Hängen; klare Meldung |
| 3 | `SLURM_JOB_ID=1 ./…/server_rl_run.sh eval < /dev/null` | kein Menü |
| 4 | `printf '4\n\n\n\n' \| ./…/server_rl_run.sh` (Pipe, kein TTY) | kein Menü, Hilfe wie heute |
| 5 | `HF_TOKEN=x NUM_EPISODES=3 MENU=1 …` mit leeren Eingaben | fragt `HF_TOKEN`/`NUM_EPISODES` **nicht**, zeigt sie als „vorgegeben" |
| 6 | Log-Datei nach einem Menülauf durchsuchen | enthält den Token **nicht** |
| 7 | `.menu/last/*.env` durchsuchen | enthält den Token **nicht** |
| 8 | `tools/gen_docs.sh --check` nach Default-Änderung in einer Spec | schlägt fehl |
| 9 | Defaults aus Spec vs. `${VAR:-…}` im Skript | Abgleichskript; müssen übereinstimmen |
| 10 | `bash -n` über alle Specs und `lib_menu.sh` | syntaktisch sauber |
| 11 | Ungültige Eingaben (`NUM_EPISODES=abc`, `-1`, `9999`) | erneute Frage statt Absturz |
| 12 | Ctrl-C mitten in einer Frage | sauberer Abbruch, kein halbfertiger Container |

Prüfung 9 ist die wichtigste — sie ist die Versicherung gegen genau das Driften, das dieser
Plan verhindern soll. Solange Phase 5 nicht steht, muss sie manuell laufen.

---

## 9. Risiken

| Risiko | Einschätzung | Gegenmaßnahme |
|---|---|---|
| Spec-Dateien werden zur zusätzlichen Pflegelast | **hoch**, wenn Phase 5 ausbleibt | Phase 5 ist nicht optional; Prüfung 8+9 |
| `set -e` + `read` erzeugt sporadische, schwer zu findende Abbrüche | mittel | Regeln in §3.3, Tests 2/3/4/11/12 |
| Menü erscheint in einem Automatisierungspfad, an den niemand gedacht hat | mittel | Mehrfach abgesichert (TTY, SLURM, CI, `/.dockerenv`), plus `MENU=0` |
| 99 Env-Vars lassen sich nicht sinnvoll in 3–5 Fragen je Aktion pressen | mittel | Stufen; im Zweifel eher `advanced` als `basic` |
| Token gerät doch in ein Log | niedrig, Wirkung hoch | Menü vor `start_logging`, Maskierung an allen drei Ausgabestellen, Test 6+7 |
| Aufwand steht in keinem Verhältnis zur Restlaufzeit des Projekts | **abzuwägen** | Phase 1+2 liefern den Großteil des Nutzens; danach ist ein Stopp jederzeit möglich |

---

## 10. Offene Entscheidungen

1. **Nur Sim oder auch Training zuerst?** Der Plan beginnt mit `eval`, weil dort der
   Leidensdruck am größten ist (20 Aktionen, 99 Variablen). Der Trainings-Launcher ist der
   einfachere Beweis. Reihenfolge Phase 1/2 ist tauschbar.
2. **Wandert die Spec-Sprache später nach `.tsv`/JSON?** Bash-DSL ist am günstigsten, weil
   kein Parser nötig ist und `when` direkt funktioniert. Eine Datenform wäre nur nötig, wenn
   die Specs von Nicht-Bash-Werkzeugen gelesen werden sollen — derzeit nicht absehbar.
3. **Soll das Menü `.env.local` schreiben können?** Verlockend („Token dauerhaft merken?"),
   aber es schreibt dann Geheimnisse in eine Datei. Vorschlag: anbieten, aber nur mit
   ausdrücklicher Bestätigung und mit `chmod 600`.
4. **Generierte Tabellen auch in CLAUDE.md?** Eher nicht — die dortigen Tabellen sind bewusst
   kuratiert und kürzer als die Doku.

---

## 11. Entscheidungslog

| Datum | Entscheidung | Begründung |
|---|---|---|
| 2026-08-20 | Menü exportiert nur Env-Vars, kein eigener Ausführungspfad | Kein zweiter Wartungspfad; Aktionsfunktionen bleiben unverändert |
| 2026-08-20 | Reines Bash statt `whiptail`/`dialog`/`gum`/`fzf` | Keins davon ist auf den Zielrechnern vorausgesetzt; Vollbild-TUI verträgt sich nicht mit der `tee`-Log-Spiegelung |
| 2026-08-20 | Menü nur auf dem Host, nie im Container | Container muss autonom bleiben (vast.ai, KISSKI) |
| 2026-08-20 | `tools/` statt `*/scripts/` | Wird nicht ins Image kopiert → Änderungen ohne Rebuild, keine Doppelablage |
| 2026-08-20 | Spec-DSL als Bash-Funktionsaufrufe | Kein Parser nötig; `when`-Abhängigkeiten fallen direkt heraus |
| 2026-08-20 | Äquivalenter Ein-Zeiler wird immer angezeigt | Menü als Lernhilfe, nicht als Ersatz für das Wissen |
| 2026-08-20 | Doku-Generierung ist Pflicht-Phase, nicht Kür | Sonst ist das Menü die fünfte Kopie derselben Beschreibungen |

---

## 12. Umsetzungsstand

Umgesetzt am 2026-08-20. Der Plantext oben ist unverändert erhalten; hier steht, was
beim Bauen anders wurde und warum. Prüfstand: `tools/test_menu.sh` — **inzwischen 81
Prüfungen** (Stand bei Einführung 2026-08-20: 30; seither erweitert, s. §13), alle
bestanden, ohne GPU und ohne Container.

### 12.1 Zwei Defekte, die der Plan nicht kennen konnte

Beide fielen erst beim Verdrahten auf, und beide hätten das Trainings-Menü sinnlos
gemacht. Sie sind mitbehoben.

**Der Trainings-Launcher las `.env.local` überhaupt nicht.** Nur
[`server_rl_run.sh`](../../Simulation/server_rl_run.sh) und `server_robocasa_ref_run.sh`
hatten den Block; in [`setup_and_train_DockerHub-pull.sh`](../../Training/setup_and_train_DockerHub-pull.sh)
fehlte er ganz. Die in [portabilitaet.md](../portabilitaet.md) beschriebene Vorrangregel
galt dort also nie — ein in `.env.local` hinterlegter `HF_TOKEN` wurde trotzdem
abgefragt. Damit hätte auch §3.4 („das Menü wird stiller, je besser `.env.local` gepflegt
ist") auf der Trainingsseite nicht funktioniert. Die Mechanik steckt jetzt einmal in
[`tools/lib_env_local.sh`](../../tools/lib_env_local.sh) und wird von beiden Seiten
gesourct.

**Der Trainings-Launcher reichte 6 von 19 Env-Vars durch.**
[`entrypoint.sh`](../../Training/scripts/entrypoint.sh) liest 19 Variablen; die
`docker run`-Zeile setzte davon 6 (`HF_TOKEN`, `MAX_STEPS`, `GLOBAL_BATCH_SIZE`,
`NUM_GPUS`, `WANDB_PROJECT`, `WANDB_API_KEY`). `TUNE_VISUAL`, `USE_COTRAIN`,
`TRAIN_TEST_SPLIT`, `USE_AUGMENTATION`, `SKIP_*`, `SHELL_ON_ERROR` und `WANDB_MODE`
kamen im Container nie an: wer `TUNE_VISUAL=1 ./setup_and_train_DockerHub-pull.sh`
aufrief, bekam **still ein normales Training**. Ein Menü, das danach fragt und den Wert
dann verschluckt, wäre schlimmer als keins gewesen. Der Launcher reicht jetzt alle 19
durch — aber nur, wenn sie gesetzt sind, damit ein leeres `-e VAR=` nicht die
`ENV`-Defaults aus dem Dockerfile überschreibt.

Nebenbei fielen drei `read`-Aufrufe im selben Skript, die unter `set -euo pipefail` bei
EOF das Skript **kommentarlos beendeten** — genau der Fall aus §6.
`./setup_and_train_DockerHub-pull.sh < /dev/null` starb an der WandB-Abfrage. Zwei davon
ersetzt jetzt das Menü, die dritte (`resume`/`destroy`) prüft `[[ -t 0 ]]`, bevor sie
fragt.

### 12.2 Phase 5: Prüfen statt Generieren

**Geplant:** `usage()` und die Doku-Tabellen aus den Specs rendern.
**Gebaut:** [`tools/gen_docs.sh`](../../tools/gen_docs.sh) vergleicht drei Quellen —
Spec-Default, echtes `${VAR:-…}` im Skript (bzw. `ENV` im Dockerfile) und die Zeile in
der Doku-Tabelle — und schlägt bei Abweichung mit Rückgabewert 1 fehl.

Der Grund für den Tausch: Generieren hätte hier Information **vernichtet**. Der
`usage()`-Heredoc verwebt Prosa und Parameter und liest live `$HOST_DATA_DIR` und
`$IMAGE`; [env-vars.md](../training/env-vars.md) trägt je Zeile mehr Begründung, als
eine Spec je fassen wird (Lauf-Nummern, Verweise, Warnungen wie „gilt seit 2026-08-13
auch für `TUNE_VISUAL=1`"). Der Zweck von Phase 5 war nie die Generierung an sich,
sondern die Sicherheit gegen Drift — und die liefert der Abgleich vollständig, ohne eine
gewachsene Zeile anzufassen. Das ist Prüfung 9 des Plans, automatisiert.

Der Abgleich fand beim ersten Lauf **27 Abweichungen**. Die meisten waren falsche
Annahmen in den frisch geschriebenen Specs (der Abgleich hat sie also sofort erledigt,
wofür er da ist). Zwei sind **echte Inkonsistenzen im Repo**, die bewusst stehen
bleiben und hier nur festgehalten werden:

| Stelle | Wert | Bemerkung |
|---|---|---|
| `MAX_STEPS` | Launcher `30000`, `entrypoint.sh` `20000`, [env-vars.md](../training/env-vars.md) `20000` | Der Host-Launcher überstimmt den Container-Default. Nicht falsch, aber überraschend — wer über den Launcher startet, bekommt 30000, wer das Image direkt fährt, 20000. |
| `COTRAIN_MIX_RATIO` | Code `0.5`, Doku empfiehlt `0.25` | [run_finetuning_cotrain.sh](../../Training/scripts/run_finetuning_cotrain.sh) hat `0.5` als Default, während env-vars.md, CLAUDE.md und [co-training.md](../training/co-training.md) übereinstimmend `0.25` für den ersten Lauf empfehlen. Die Spec folgt dem Code (`0.5`) und nennt `0.25` im Erklärtext. **Zu entscheiden:** Default auf `0.25` ziehen oder die Empfehlung streichen. |

Zwei Ergänzungen der Spec-Sprache machen den Abgleich erst belastbar:

- `--override "<grund>"` — diese Aktion weicht **absichtlich** vom Skript-Default ab.
  `do_view` etwa setzt `SCENE_CAM=0` und `CAM_RES_SCALE=0.5` selbst, weil es nichts misst
  und deshalb sparen darf. Ohne diese Angabe meldete der Abgleich bekannte Wahrheiten als
  Fehler — und ein Prüfwerkzeug, das anmeckert, was stimmt, gewöhnt man sich ab zu lesen.
- `--default-from "<datei>"` — der wirksame Default liegt woanders (argparse in
  `rl_finetune.py`, `entrypoint_rl.sh`). Ein `secret` ist automatisch ausgenommen: ein
  Geheimnis hat per Definition keinen Default.

Was **nicht** stillschweigend durchgeht: Parameter ohne prüfbaren Default und ohne eine
dieser beiden Angaben meldet der Bericht als `UNGEPRUEFT`. Derzeit sind es null.

### 12.3 Kleinere Abweichungen

| Punkt | Plan | Umsetzung | Grund |
|---|---|---|---|
| **Positionsargumente** | nicht vorgesehen | `argpos` in der DSL; `set -- "$ACTION" "${MENU_ARGV[@]}"` nach dem Fragen | `optimize <phase>` und `webview stop` lesen `$2`. Ohne das hätte das Menü zwei der 19 Aktionen nicht vollständig bedienen können. |
| **Reihenfolge der Aktionsliste** | implizit | `_order.spec` mit `group_order`, `--rank` je Aktion | Specs werden alphabetisch geglobbt; die Liste stand in Dateinamen-Reihenfolge. Damit wäre gerade der Nutzen aus §5.1 weg — sie soll die Kette `preflight → setup → cams → gap → eval → layout → render → rl` zeigen. |
| **`_common.spec`** | eine Datei | zusätzlich `_common-<prefix>.spec` | Die `LIVE*`-Parameter gehören zur Sim, nicht ins Trainings-Menü. |
| **Container-Erkennung** | `/.dockerenv` | zusätzlich `APPTAINER_NAME`, `SINGULARITY_NAME`, `APPTAINER_CONTAINER`, `SLURM_JOBID`, `GITHUB_ACTIONS` | Auf KISSKI läuft derselbe Code unter Apptainer, ganz ohne `/.dockerenv`. Der Rest des Repos prüft beides (`run_finetuning.sh`), das Menü muss es auch. |
| **Ein-Zeiler** | Aktionsname anhängen | `--cli` je Aktion | Die Trainings-Launcher kennen Flags (`--resume`), keine Positionsargumente. Der angezeigte Ein-Zeiler wäre sonst nicht lauffähig gewesen — und er ist der halbe Zweck des Menüs. |
| **Schalter** | nur `MENU=0` | zusätzlich `--menu`, `--no-menu`, `--profile=<name>` | Über ein Flag stolpert man in `--help`; über eine Env-Var nicht. |
| **VRAM-Vorschlag** | Batch-Size vorschlagen | `--suggest` als allgemeiner Mechanismus | So hängt auch `NUM_GPUS` daran. Der **statische** Spec-Default bleibt die prüfbare Wahrheit; der Vorschlag ist nur das, was im Prompt steht. |

### 12.4 Was bewusst nicht gebaut wurde

- **`update_image.sh` / `update_sim_image.sh`** (§4.4). Der Plan nennt als Hauptgewinn die
  Frage „auch `:latest` pushen?" (`--push-latest`). Dieses Flag existiert auf
  `training-luca-IKR-IS6.0` **nicht** — es kam auf dem N1.7-Branch dazu. Übrig blieben
  vier selbsterklärende Flags (`--no-cache`, `--skip-push`, `--dry-run`,
  `--update-commit`); dafür lohnt ein Menü nicht. Nach einem Merge des N1.7-Pfads neu zu
  bewerten.
- **`server_robocasa_ref_run.sh`** (§4.4). Wenige Parameter, seltene Benutzung. Es liest
  seine `.env.local` weiterhin über den eigenen Block — der Umbau auf
  `lib_env_local.sh` wäre eine Verbesserung, gehört aber nicht in diesen Schritt.
- **`setup_and_train_Container-build.sh`.** Teilt sich die Spec mit der Pull-Fassung, ist
  aber noch nicht verdrahtet. Ein Einzeiler wie in der Pull-Fassung genügt dafür.

### 12.5 Bedienung

```bash
./Simulation/server_rl_run.sh                 # führt durch (Aktionsliste + Fragen)
./Simulation/server_rl_run.sh eval            # führt nur durch die Parameter von 'eval'
MENU=0 ./Simulation/server_rl_run.sh eval     # wie bisher, keine Rückfrage
./Simulation/server_rl_run.sh --profile=rauchtest eval

./Training/setup_and_train_DockerHub-pull.sh  # führt durch
./Training/kisski_menu.sh --dry-run           # baut die sbatch-Zeile, reicht nicht ein

./tools/gen_docs.sh                           # Abgleich Spec <-> Skript <-> Doku
./tools/gen_docs.sh --table sim               # Markdown-Tabelle aller Sim-Parameter
./tools/test_menu.sh                          # der Prüfplan aus §8 (aktuell 81 Prüfungen)
```

In der Fragerunde: `?` zeigt den Langtext, leere Eingabe nimmt den Default.
Auf der Bestätigungsseite: `[Enter]` startet, `[1-n]` ändert einen Wert,
`[e]` blendet die erweiterten Optionen ein, `[b]` zeigt nur den Befehl,
`[p]` sichert ein Profil, `[a]` bricht ab.

### 12.6 Wenn eine Spec geändert wird

1. Datei unter `tools/menu/` bearbeiten — reines Bash, `bash -n` prüft sie.
2. `./tools/gen_docs.sh` laufen lassen. Meldet er eine Abweichung, ist entweder die Spec
   falsch oder die andere Stelle — **nicht** die Meldung.
3. `./tools/test_menu.sh` laufen lassen.

Schritt 2 ist der Punkt, an dem dieses Vorhaben steht oder fällt: ohne ihn wären die
Specs die fünfte Kopie derselben Beschreibungen, und der Plan hätte in §9 mit dem
höchsten Risiko recht behalten.

---

## 13. Nachtrag 2026-08-21 — ein Einstiegspunkt und Pfeiltasten

Nachgereicht auf denselben Branch. Sechs Dinge, die der Plan nicht vorsah: die Ebene
**über** den Aktionen (§13.1), Pfeiltasten (§13.3), eine Gruppenebene **innerhalb**
einer Domäne (§13.4), den Rückweg über alle Ebenen (§13.5), die Einsicht, dass
KISSKI kein eigener Zweig ist, sondern ein Trainings-Ort (§13.6), und die Lücke, dass
kein Messlauf danach fragte, **welche Gewichte** er misst (§13.7). Prüfstand jetzt
**76 Prüfungen** statt 30.

### 13.1 `run.sh` — die Domänen-Ebene

Bisher musste man wissen, *welches Skript* man aufruft: `server_rl_run.sh`,
`setup_and_train_DockerHub-pull.sh` oder `kisski_menu.sh`. Das ist genau die Sorte
Vorwissen, die das Menü eigentlich abschaffen sollte — nur eine Ebene höher.

```bash
./run.sh              # fragt: Simulation / Training / KISSKI
./run.sh sim          # direkt in die Simulation
./run.sh sim eval     # ganz durch bis zu den Parametern von 'eval'
```

**Der Router wählt nur und übergibt per `exec`.** Er baut keinen `docker run`-Aufruf,
setzt keinen Trainingsparameter und kennt keine Aktionsliste — ab dem `exec` läuft der
unveränderte Launcher mit seinem eigenen Menü. Damit gilt §2.1 („kein zweiter
Ausführungspfad") eine Ebene höher unverändert weiter, und alles, was die Launcher
können, gilt automatisch mit: `MENU=0`, `--profile=<name>`, jede Env-Var, jedes Argument.

Der Ausbau war deshalb billig: [`lib_menu.sh`](../../tools/lib_menu.sh) war seit Phase 3
ohnehin über einen `<prefix>` parametrisiert (`sim` / `train` / `kisski`), inklusive
`_order-<prefix>.spec` und `_common-<prefix>.spec`. Es fehlte buchstäblich nur die Ebene
darüber.

Die Zuordnung Domäne → Launcher steht in
[`tools/menu/_domains.spec`](../../tools/menu/_domains.spec), nicht im Skript — dieselbe
Regel wie überall hier. `run.sh` enthält keine Liste.

### 13.2 Was der Router zusätzlich kann — und was bewusst nicht

**Kontexterkennung.** Die drei Domänen laufen an verschiedenen Orten: Sim und Training
brauchen Docker, KISSKI braucht `sbatch` (nur auf dem Login-Node). Fehlt das Kommando,
bleibt der Eintrag **sichtbar und gesperrt**, mit dem Grund daneben:

```text
  auf fedora:  docker ✓  sbatch ✗  apptainer ✗

   1) sim          Simulation      19 Aktionen
   2) train        Training         4 Aktionen
   3) kisski       KISSKI (HPC)    kein sbatch
```

Sichtbar-und-gesperrt statt versteckt: dass es den KISSKI-Weg *gibt*, soll man auch auf
dem Rechner erfahren, auf dem er gerade nicht geht. Die Aktionszahl wird aus den Specs
abgeleitet (`menu_list_actions`), nicht gepflegt — Prüfung 30 hält das fest.

**Keine Zustandsauswertung auf dem Startbildschirm.** Die `--state`-Marker der Aktionen
(`✓ liegt bereits vor`, `! braucht vorher cams`) wertet der Router **nicht** aus. Drei
von ihnen rufen `docker ps` (`train-resume`, `train-destroy`, `train-train`). Auf dem
Startbildschirm wäre das ein Aufruf, der bei hängendem Docker-Daemon das ganze Menü
einfriert — und zwar bevor überhaupt eine Domäne gewählt ist. Die Marker erscheinen
weiterhin, nur einen Schirm später in der Aktionsliste. Die Standortzeile oben nutzt aus
demselben Grund ausschließlich `command -v`, kein `docker info`, kein `nvidia-smi`.

> Der Kommentarkopf in `lib_menu.sh` verlangt für `--state` „reine
> Dateisystem-Prüfungen". Die drei `docker ps`-Ausdrücke halten das nicht ein. Sie sind
> mit `2>/dev/null` abgesichert und in der Aktionsliste vertretbar, aber es bleibt eine
> offene Kleinigkeit — nicht in diesem Schritt angefasst.

### 13.3 Pfeiltasten — eine Zugabe, kein Modus

Das Entscheidungslog (§11) verwirft `whiptail`/`dialog`/`gum`/`fzf`, und die Gründe
gelten unverändert: keins davon ist auf dem IKR-Server oder dem KISSKI-Login-Node
vorausgesetzt, und ein Vollbild-TUI verträgt sich nicht mit der `tee`-Spiegelung aus
`start_logging`. Pfeiltasten brauchen davon nichts — ANSI-Sequenzen und `read -rsn1`
genügen.

Umgesetzt als eine generische Auswahl (`menu_select`), die Domänenliste, Aktionsliste und
künftig weitere Listen gemeinsam benutzen. Zwei Regeln machen sie unkritisch:

1. **Die Zahleneingabe bleibt der Boden.** Jede Liste ist vollständig per Ziffer
   bedienbar, auch im Pfeiltasten-Modus (Prüfung 41).
2. **Sie schaltet sich selbst ab**, wenn kein echtes Terminal da ist, `TERM` fehlt oder
   `dumb` ist, oder `MENU_ARROWS=0` gesetzt wurde.

Regel 2 hat einen angenehmen Nebeneffekt: `tools/test_menu.sh` fährt sein Pseudoterminal
mit `TERM=dumb`, also laufen **alle 31 Alt-Prüfungen weiter über die Zahleneingabe** —
denselben Pfad wie vor der Änderung. Die Pfeiltasten bekamen eigene Prüfungen (35–41),
die den pty-Treiber um Rohbytes (`<RAW>\x1b[B`) und ein setzbares `TERM` erweitern.

Details, die beim Bauen Zeit gekostet haben und deshalb im Code kommentiert stehen:

| Punkt | Warum |
|---|---|
| Zeile erst als **Klartext** bauen, dann einfärben | Farbcodes zählen in `${#s}` mit. Färbt man zuerst, verrutscht die Breitenrechnung, die Zeile bricht um — und beim Neuzeichnen stimmt die Zeilenzahl nicht mehr, das Bild zerfranst |
| `\033[K` an **jedem** Zeilenende | Sonst bleiben Reste der vorigen, längeren Fassung stehen |
| Ziffern mit Timeout nachlesen | Nach einer `1` muss klar werden, ob noch eine `2` folgt (Eintrag 12) oder nicht (Eintrag 1) — ohne dass die Anzeige bis zum nächsten Tastendruck einfriert |
| `trap … INT` speichern und wiederherstellen | Der Cursor ist während der Auswahl versteckt. Ohne Handler bliebe er nach Ctrl-C unsichtbar; der Handler stellt ihn her, räumt sich weg und sendet das Signal erneut, damit Ctrl-C nicht verschluckt wird |
| Erklärung (`?`) erzwingt ein **frisches** Bild | Sie fügt Zeilen ein; ein Neuzeichnen an Ort und Stelle würde daneben landen |

### 13.4 Gruppenebene — Schachtelung mit Vorschau

Die Sim-Aktionsliste zeigte 19 Einträge in 7 Gruppen, gut 30 Zeilen. Sie ist jetzt
zweistufig — aber **nicht** einfach eingeklappt.

```text
  Simulation — Was moechtest du tun?

 › 1) Vorbereiten              preflight, setup, check                  3 Akt.
   2) Ansehen                  view, livecheck, webview                 3 Akt.
   3) Messen                   cams, gap, eval, grasp, span, latency    6 Akt.  !
   4) Co-Training vorbereiten  layoutcheck, layout, render              3 Akt.  !
   5) Trainieren               rl                                       1 Akt.
   6) Beschleunigen            optimize                                 1 Akt.
   7) Werkzeuge                shell, clean                             2 Akt.

  [↑↓] waehlen  [Enter] bestaetigen  [1-7] direkt  [?] erklaeren  [*] alle 19 Aktionen  [a] abbrechen
```

**Der Konflikt, den das lösen musste:** §12.3 hält fest, dass `_order.spec` genau dafür
existiert, die Kette `preflight → setup → cams → gap → eval → layout → render → rl`
sichtbar zu machen — sie läuft quer durch vier Gruppen. Reines Schachteln hätte sie
hinter Überschriften versteckt und damit den Nutzen aus §5.1 kassiert.

Drei Dinge halten sie sichtbar:

1. **Vorschau der Aktionsnamen** je Gruppe. Die Kette bleibt lesbar, in einem Siebtel
   der Zeilen.
2. **`[*]` schaltet auf die Gesamtliste** — das alte Verhalten, unverändert, einen
   Tastendruck entfernt.
3. **`?<nr>` erklärt eine ganze Gruppe**: Aktionsnamen mit Titeln und offenen
   Voraussetzungen, ohne hineinzugehen.

Das `!` in der rechten Spalte heißt: mindestens eine Aktion dieser Gruppe hat eine
offene Voraussetzung. Welche, steht eine Ebene tiefer — oben wäre es Rauschen.

**Geschachtelt wird nach Größe, nicht pauschal.** Bei 19 Aktionen in 7 Gruppen ist ein
Schirm voll; bei den 4 Aktionen von Training und KISSKI wäre eine Gruppenebene davor
reine Mehrarbeit. Schwelle: mehr als `MENU_NEST_MIN` (10) Aktionen **und** mindestens 3
Gruppen. `MENU_NEST=0` erzwingt flach, `MENU_NEST=1` erzwingt geschachtelt.

Zurück geht es mit `[←]` (bzw. `[z]` ohne Pfeiltasten). Die Zustandsmarker werden pro
Aufruf **einmal** ausgewertet und zwischengespeichert — beim Blättern zwischen den Ebenen
liefen sonst wiederholt Ausdrücke, unter denen `docker ps` ist (§13.2).

**Drei Prüfungen mussten dem folgen**, weil sich der Ablauf bewusst geändert hat: 7
(Kettenreihenfolge), 8 (`?<nr>`) und 14b (Auswahl plus Fragen). Sie prüfen jetzt beide
Ebenen — die alte Absicht steckt unverändert in der `MENU_NEST=0`-Variante daneben
(7b, 8, 14c). Dazu sieben neue (15a–15g) für die Schachtelung selbst. Stand: **57
Prüfungen**.

### 13.5 Der Rückweg — `[←]` führt immer eine Ebene höher

```text
   Untermenue  ──[←]──▶  Gruppenuebersicht  ──[←]──▶  Hauptmenue  ──[a]──▶  Ende
   (Messen)                (Simulation)                (Domaenen)
```

Ohne Pfeiltasten dieselbe Bewegung mit `[z]`. Die Fußzeile beschriftet die Taste nach
ihrem Ziel — `[←] zurueck` innerhalb einer Domäne, `[←] Hauptmenue` auf der obersten
Ebene.

**Warum das nicht trivial war:** `run.sh` übergab bisher per `exec`. Das ersetzt den
Prozess — mit ihm wäre das Hauptmenü weg gewesen, es gäbe schlicht nichts, wohin man
zurückkehren könnte. Die Übergabe hat jetzt zwei Wege:

| Fall | Übergabe | Warum |
|---|---|---|
| Menü an (Terminal, kein `MENU=0`) | Launcher als **Kindprozess** | Nur so überlebt `run.sh` und kann das Hauptmenü erneut zeigen |
| Menü aus (`MENU=0`, kein Terminal, Skriptaufruf) | **`exec`** | Kein Menü, kein Rückweg — und Rückgabewert wie Signale gehen unverändert durch |

Der Launcher meldet den Rückwunsch über den Rückgabewert **`MENU_RC_BACK` (97)**.
Bewusst eine hohe, sonst nirgends benutzte Zahl: die Launcher enden regulär mit 0, 1
oder 2. `run.sh` wertet sie ausschließlich aus, wenn es den Launcher selbst gestartet
hat, und `MENU_TOPLEVEL_BACK=1` schaltet das `[←]` der obersten Liste überhaupt erst
frei. **Beim direkten Aufruf** (`./Simulation/server_rl_run.sh`) bleibt die Taste
deshalb tot — ein Rücksprung ins Nichts würde das Skript kommentarlos beenden, und
genau das prüfen 15m/15n.

Zwei Feinheiten, die beim Bauen auffielen:

- **Auch eine per Argument gewählte Domäne hat einen Rückweg.** `./run.sh sim` bietet
  `[←] Hauptmenue` genau wie `./run.sh`. Die Asymmetrie „mit Argument kein Zurück"
  merkt sich niemand.
- **Beim Rücksprung fallen die restlichen Argumente weg.** Bei `./run.sh sim eval` →
  `[←]` → *Training* wäre `eval` an den Trainings-Launcher weitergereicht worden — es
  ist dort keine gültige Aktion. Sie gehörten zur alten Domäne und werden verworfen.

Sieben neue Prüfungen (15h–15n) decken das ab, inklusive der beiden Fälle, in denen
`[←]` **nichts** tun darf.

### 13.6 KISSKI gehört unter „Training" — die Sim kann der Cluster nicht

Die erste Fassung der Domänenliste stellte Simulation, Training und KISSKI gleichrangig
nebeneinander. Das mischte zwei Achsen: *was* man tut und *wo*. Der Befund dahinter:

| Aktion | Status auf KISSKI |
|---|---|
| `kisski-train` | läuft — der eigentliche Zweck des Clusters hier |
| `kisski-openloop` | läuft — reine Rechenarbeit, kein Rendering |
| `kisski-sim` | **tot.** Zielt auf die `jupyter`-Partition, weil nur deren Quadro RTX 5000 überhaupt RT-Cores hat. Die ist aber Turing und damit „eine Generation zu alt" ([fehlerbehebung.md](../fehlerbehebung.md)); die zugehörige Anleitung liegt längst in [`simulation/archiv/`](../simulation/archiv/kisski-desktop.md) |
| `kisski-rl` | **nicht einreichbar.** Vorlage mit `#SBATCH -p PLACEHOLDER_RTCORE_PARTITION`; der RT-Core-Guard bricht auf A100/H100 ab, und eine RT-Core-Partition gibt es dort nicht |

**KISSKI ist in diesem Projekt also ausschließlich Training plus Checkpoint-Auswertung.**
Die Simulation läuft auf dem IKR-Server. Zwei Konsequenzen:

**1. Die Domänenliste ist gruppiert, nicht geschachtelt.**

```text
  Simulation
   1) sim          auf dem IKR-Server (Docker, RT-Cores)      19 Aktionen
  Training
   2) train        auf diesem Rechner (Docker)                 4 Aktionen
   3) kisski       auf dem KISSKI-Cluster (sbatch)             kein sbatch
```

Ein echtes Untermenü unter „Training" wäre eine eigene Ebene für eine Ja/Nein-Wahl
gewesen — genau das, was §13.4 für kurze Listen ablehnt — und die beiden Zweige haben
verschiedene Launcher, was die Domänenebene architektonisch aufgeweicht hätte. Die
Gruppierung nutzt dieselbe `_MENU_SEL_HEAD`-Mechanik wie die Aktionsliste und kostet
keine Ebene.

Dafür beschreibt der Titel einer Domäne jetzt nur noch die **Variante** („auf diesem
Rechner (Docker)") — er ergibt ohne die Gruppe daneben keinen Satz mehr. Überschriften
tieferer Ebenen brauchen deshalb einen Kurznamen: `--label` („Training (KISSKI)"),
sonst stünde dort „auf dem KISSKI-Cluster (sbatch) — Was moechtest du tun?".

**2. `--blocked` für dauerhaft nicht lauffähige Aktionen.**

```text
  Nicht auf diesem Cluster
   3) sim         Sim-Eval einreichen         RTX 5000 zu alt -> IKR-Server
   4) rl          RL-Fine-tuning (FPO)        Vorlage, keine RT-Core-Partition
```

Bewusst nicht gelöscht: das Wissen *warum das hier nicht geht* soll dort stehen, wo die
Frage aufkommt — dieselbe Begründung wie beim Sperren einer ganzen Domäne (§13.2). `?`
liefert den Langtext samt Verweis auf den Weg, der funktioniert. Die Skripte bleiben
unangetastet im Repo.

Unterschied zum Domänen-Sperrgrund: `--needs` heißt „hier gerade nicht" (ortsabhängig,
per `command -v` geprüft), `--blocked` heißt „grundsätzlich nicht" (eine Eigenschaft des
Ziels, statisch in der Spec). Deshalb zählt die Domänenzeile nur die **lauffähigen**
Aktionen (`menu_list_actions --runnable`, „2 Aktionen" statt 4) — eine Zahl, die die
nächste Ebene nicht einlöst, wäre ein Versprechen zu viel. `menu_list_actions` **ohne**
den Schalter listet weiterhin alles: `gen_docs.sh` prüft darüber die Defaults, eine
stille Filterung wäre ein Loch in der Drift-Sicherung.

Eine Gruppe, in der nichts lauffähig ist, wird selbst gesperrt — hineingehen zu dürfen,
um dort nur Graues zu finden, wäre eine Sackgasse.

### 13.7 Welche Gewichte gemessen werden, gehört ins Menü

Beim ersten Durchspielen von `./run.sh → sim → Messen → eval` fiel auf, dass der Dialog
nach Token, Episodenzahl, Zeitbudget und Randomisierung fragt — aber mit keinem Wort
danach, **welcher Checkpoint** eigentlich gemessen wird. Der Befund:

- Nur [`sim-setup.spec`](../../tools/menu/sim-setup.spec) nannte `HF_CHECKPOINT_REPO`.
  Alle anderen Aktionen erbten davon nichts.
- `CHECKPOINT_PATH` stand in **keiner** Spec. Über das Menü war ein zweiter Checkpoint
  damit gar nicht erreichbar — obwohl genau dieser Vergleich der Punkt ist: Lauf 3
  zeigte eine U-Kurve über die Checkpoints, der beste lag bei 30000, der letzte war
  25 % schlechter ([lauf3-vision-split-auswertung.md](../ergebnisse/lauf3-vision-split-auswertung.md)).
- Zehn der 19 Sim-Aktionen rufen `ensure_checkpoint` **selbst** auf (`setup`, `check`,
  `rl`, `eval`, `optimize`, `cams`, `grasp`, `span`, `latency`, `render`). Das
  `--needs "setup"` bei `eval` ist nur ein Hinweis, keine Sperre — ohne vorheriges
  `setup` lud `eval` also stillschweigend das Default-Repo herunter, 10 GB, ungefragt.

Eine Erfolgsrate ohne die Angabe, welche Gewichte sie gemessen hat, ist nicht
vergleichbar. Also gehört der Checkpoint in den **Grunddialog**, nicht hinter `[e]`.

**Der Pfad ist der Hebel, nicht das Repo.** `ensure_checkpoint`
([`server_rl_run.sh`](../../Simulation/server_rl_run.sh)) prüft ausschließlich, ob
`$CHECKPOINT_PATH` existiert; liegt das Verzeichnis da, wird nichts nachgeladen und
`HF_CHECKPOINT_REPO` bleibt wirkungslos. Ein Wechsel des Trainingslaufs braucht deshalb
**beide** Felder — sonst misst der zweite Lauf den ersten Checkpoint. Genau das steht
jetzt als `note` über dem Dialog und im Langtext beider Parameter.

Umgesetzt nach dem Muster, das `_common.spec` für `HF_TOKEN` schon vorgibt:

| Ort | Stufe | Aktionen |
|---|---|---|
| [`_common-sim.spec`](../../tools/menu/_common-sim.spec), Gruppe „Gewichte" | `advanced` | alle Sim-Aktionen erben beide Felder |
| [`sim-eval.spec`](../../tools/menu/sim-eval.spec) | `CHECKPOINT_PATH` auf `basic` | der Messlauf fragt danach zuerst |
| [`sim-setup.spec`](../../tools/menu/sim-setup.spec) | beide erneut unter „Quelle" | Repo und Zielpfad stehen nebeneinander |
| `preflight`, `view`, `webview`, `livecheck`, `gap`, `layout`, `layoutcheck`, `shell`, `clean` | beide auf `expert` | dokumentiert, aber nie gefragt |

Das erneute Nennen in `sim-eval.spec` und `sim-setup.spec` ist kein Duplikat, sondern
die vorgesehene Mechanik: die spätere Fassung gewinnt und bestimmt auch die **Position**.
Ohne sie stünde `HF_CHECKPOINT_REPO` in der `[e]`-Liste ganz oben und der zugehörige
Pfad zehn Zeilen darunter.

**Der Zustandsmarker nennt den Checkpoint beim Namen.** Vorher: `! braucht vorher setup`.
Jetzt zeigt `eval` den tatsächlich vorliegenden Namen — `✓ lauf3` —, ausgewertet über
denselben billigen Dateisystem-Test wie bisher (§5.1: ein `docker inspect` an dieser
Stelle würde das Menü einfrieren).

Fünf neue Prüfungen (42–46) in [`test_menu.sh`](../../tools/test_menu.sh) halten das
fest, darunter die Gegenprobe, dass `view` weder Pfad noch Repo zeigt — der gemeinsame
Block darf für die neun gewichtslosen Aktionen keine Verschlechterung sein. Zählung
jetzt 76 statt 71.

**Nebenbefund beim Nachziehen der Tastenfolgen.** Die Prüfungen 10, 11 und 11b hatten je
eine Leerzeile zu viel: die letzte landete auf der Zusammenfassung und ist dort `[Enter]`
= **starten**. Auf dieser Maschine fiel das nie auf, weil kein Container hochkommt — auf
dem IKR-Server hätte der Prüfplan drei echte Eval-Läufe angestoßen. Behoben; damit fasst
außer dem absichtlichen Paar 19/20 (es prüft, dass der Token in keinem Log landet) keine
Prüfung mehr einen Container an.

### 13.8 Checkpoints vorschlagen — und wann ein Vorschlag überhaupt greifen darf

Nachtrag desselben Tages, ausgelöst von einer Frage aus der Praxis: „gibt es eine leichte
Möglichkeit, zwischen verschiedenen Checkpoints zu wechseln?" Der Mechanismus war da
(§13.7, `CHECKPOINT_PATH` als Grundfrage bei `eval`), aber die Frage war ein leeres Textfeld
mit einem Default darin. Der Pfad gilt **im Container**, die Verzeichnisse liegen unter
`HOST_DATA_DIR/checkpoints/` auf dem Host — man musste den Namen also woanders nachsehen und
abtippen.

Neu ist `_menu_suggest_checkpoint` in [`lib_menu.sh`](../../tools/lib_menu.sh): es listet die
Verzeichnisse, die tatsächlich da sind, schlägt den zuletzt geänderten **vollständigen** als
Vorgabe vor und nennt die übrigen darunter. Ein angeschnittener Download (kein
`*.safetensors`) wird getrennt als `UNVOLLSTAENDIG` gemeldet, aber nie vorgeschlagen —
dieselbe Bedingung, die `checkpoint_complete()` in `server_rl_run.sh` prüft, hier nur auf dem
Host und ohne Docker. Gibt es *nur* angeschnittene, kommt gar kein Vorschlag: ein kaputtes
Verzeichnis vorzuschlagen wäre schlechter als keine Hilfe.

**Der eigentliche Fund steckt eine Ebene tiefer.** Der `--suggest`-Mechanismus gab es seit
Phase 3, er hat aber praktisch nie gefeuert. Die Bedingung war:

```bash
if [[ -z "${!var:-}" && -n "${_MENU_P_SUGGEST[$var]:-}" ]]; then
```

„Variable leer" ist die falsche Frage. Die Launcher setzen ihre eigenen Defaults im
Konfigblock — `CHECKPOINT_PATH="${CHECKPOINT_PATH:-…}"` steht in `server_rl_run.sh` Zeile 129
—, und der läuft **lange vor** dem Menü. Damit war nie etwas leer, und ein Vorschlag erreichte
nur Variablen ohne Skript-Default. Der VRAM-Vorschlag beim Training funktionierte bloß, weil
`GLOBAL_BATCH_SIZE` zufällig zu dieser Sorte gehört.

Maßgeblich ist nicht „leer", sondern „stammt nicht vom Nutzer" — und genau das weiß die
Herkunftslogik aus §3.4 schon: `env_preset` kennt die Momentaufnahme von *vor* dem
Konfigblock, `env_local_provided` die `.env.local`-Herkunft. Die Bedingung heißt jetzt
sinngemäß „hat einen Vorschlag, ist noch nicht beantwortet, und der Wert kam nicht vom
Nutzer". Zwei Rückfallebenen bleiben:

- Fehlt die Herkunftsmaschinerie ganz — `kisski_menu.sh` läuft bewusst ohne
  `lib_env_local`, damit `kisski_submit.sh` allein scp-bar bleibt —, gilt weiter der alte,
  strenge Test. Lieber kein Vorschlag als einer, der eine bewusste Angabe überschreibt.
- `_MENU_P_ORIGIN` schützt die zweite Runde: wer einen Wert auf der Bestätigungsseite noch
  einmal ändert, bekommt seine eigene Antwort vorgelegt, nicht wieder den Vorschlag.

Prüfungen 47–51 decken die fünf Fälle ab: neuester gewinnt, die übrigen werden genannt, ein
angeschnittener wird gemeldet aber nicht vorgeschlagen, ohne Checkpoints bleibt der statische
Default stehen — und die Gegenprobe, dass ein vom Aufrufer gesetzter Pfad den Vorschlag
schlägt. Damit steht die Suite bei 81.

### 13.9 Nicht gebaut — und warum

**Eine flache Gesamtliste über alle Domänen** (27 Einträge, Domäne als
Gruppenüberschrift) wäre der nächste Schritt und ist von hier aus klein. Sie bleibt
offen: 27 Einträge auf einem Schirm sind ohne Tippfilter unübersichtlicher als zwei
saubere Ebenen. Sinnvoll wird sie erst, wenn die Kontexterkennung die Liste real
zusammenstreicht — oder mit `fzf`, das dann aber wieder eine Abhängigkeit wäre.

**`tools/check_tldr.sh` durchsuchte das Wurzelverzeichnis nicht.** Es prüfte nur
`Training/`, `Simulation/` und `tools/`; ein `run.sh` im Repo-Wurzelverzeichnis wäre
ungeprüft durchgerutscht — ausgerechnet der Einstiegspunkt. Mitbehoben, Zählung jetzt
151/151 statt 150/150.
