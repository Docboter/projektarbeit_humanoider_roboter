# Nächste Schritte

> **TL;DR:** Offene-Punkte-Liste des Projekts: aktueller Stand der Messkette, priorisierte
> nächste Schritte und offene Widersprüche zwischen den Ergebnisdokumenten. Bezugspunkt für
> die Prioritätensetzung, kein Ergebnisbericht selbst.

**Stand:** 2026-10-29, bei der Abgabe-Durchsicht 2026-11-09 unverändert bestätigt (Kurzstatus §0, nächste Schritte §2; Fassung davor 2026-10-15, 2026-10-14, 2026-10-13, 2026-10-11, 2026-08-14) · Grundlage: Ergebnis-Dokumente unter [`docs/ergebnisse/`](docs/ergebnisse/README.md),
[RL-Plan](docs/weiterfuehrend/reinforcement-learning-plan.md), Trainings-Launcher unter
[`Training/scripts/`](Training/scripts/).

---

## 0. Projektstatus in Kürze (Stand 2026-10-29)

Diese Zusammenfassung ist die **eine gepflegte Statusquelle** des Projekts; CLAUDE.md und README
verweisen hierher. Den N1.7-Detailstand pflegt
[groot-n17-migration.md, „Stand der Umsetzung“](docs/weiterfuehrend/groot-n17-migration.md#stand-der-umsetzung-2026-08-19).

- **Bester validierter N1.6-Checkpoint:** vision-v2 (Lauf 3), **Step 44.000**. Die Nachmessung
  auf 20 vollen Test-Episoden (2026-10-12) widerlegt die frühere U-Kurve (Optimum 30.000):
  Validierungs-MSE fällt monoton bis 44.000
  ([lauf4-n17-vergleich.md §5](docs/ergebnisse/lauf4-n17-vergleich.md#5-nachgeschärfter-vergleich-20-episoden--750-steps-volle-länge)).
- **GR00T N1.7** als paralleler Pfad (`GROOT_VERSION`): voller KISSKI-Lauf (44k Steps, 7 h 53 min)
  fertig, Open Loop am Ende gleichauf mit N1.6, lernt früher, trainiert 1,6× schneller
  ([Lauf 4](docs/ergebnisse/lauf4-n17-vergleich.md)). Closed Loop nach Abnahmeprotokoll
  (20 × 120 s) gelaufen: Läufe 81, 89, 93 je `lifted_clear` 0/20.
- **Neu 2026-10-29 — Co-Training v3 ausgewertet (erstes Ergebnis des Co-Training-Pfads):**
  Vier KISSKI-Trainings, 44 000 Steps, Mischung 0,15 gerendert: N1.6 und N1.7, je Render 5 cm
  und 4 cm (Läufe 73–76). Alle acht Checkpoints auf HF sind im Gate gemessen (Läufe 83, 86, 92):
  - **Sim-Lücke deutlich kleiner:** MSE Hände auf Sim-Bildern bei N1.6 halbiert (5 / 4 cm
    0,0115–0,0118 / 0,0104 gegen 0,0239 / 0,0221), bei N1.7 auf 18–30 % des Ausgangswerts.
  - **Realdomäne:** Das Echtbild bleibt bei N1.7 gleich (0,0024–0,0027) und wird bei N1.6 um
    50–60 % schlechter (0,0014 → 0,0021–0,0023). Die Sweep-Val-MSE liegt 3–9 % über Lauf 3.
  - **Closed Loop** (Szene Lauf 70, 20 × 120 s): Die co-trainierten Modelle berühren den Würfel
    in ~20/20 Episoden. **Erste echte Griffe der Policy** gab es nur bei N1.6: cube4 @44000 auf
    allen 60 Testlagen `lifted_clear` **2/60 = 3 %** (KI 0,4–11,5 %, Läufe 88/97), cube5 1/20
    (Lauf 84). Beide N1.7-Modelle 0/20 (Läufe 89, 93), `stacked` überall 0. Gegen 0/20 ohne
    Co-Training ist die Griffrate statistisch nicht zu trennen.
  - **Gegenproben:** Alte N1.6-Modelle ohne Co-Training bestehen die 80-%-Gate-Schwelle
    ebenfalls und öffnen die Hand ebenso weit, greifen aber nicht (Läufe 87, 90/91). Die
    Gate-besten Zwischenstände (@35000 / @40000) greifen nicht besser als @44000 (Läufe 94/95).
  - **Vorregistriertes primäres Gate bestanden** (nachgeholt 29.10., Lauf 96, 10 × 40 s,
    [co-training.md §5](docs/training/co-training.md#5-vorregistrierte-regel--was-den-lauf-zum-erfolg-oder-zum-ende-macht)):
    Spanne 1,53 rad = 73 % gegen 0,577 rad (Lauf 34) und 0,52 rad beim gleichen Rezept ohne
    Renderdaten (p = 5 · 10⁻⁶). Gegen das alte Lauf-1-Modell (1,21 rad) nur Tendenz (p = 0,07).
    Die Kontrolle `gap` ist dafür ungeeignet (generisches SigLIP, nicht der Encoder der Policy).
  - **Bester Sim-Kandidat:** `luca-mue/groot-g1dex3-cotrain-v3-cube4-ckpt44000` (N1.6, Lauf 74).
  Details: [co-training.md §3.5](docs/training/co-training.md#35-auswerten--zwei-messungen-nicht-eine),
  [Läufe 73–76](docs/ergebnisse/diagnose-chronik.md#läufe-7376-co-training-läufe-v3-auf-kisski),
  [Lauf 92](docs/ergebnisse/diagnose-chronik.md#lauf-92-visuelles-gate-der-restlichen-v3-checkpoints).
- **Closed Loop insgesamt:** Stand 2026-10-29 weiter **0 Erfolge** (`stacked`), aber erste
  Griffe (siehe Punkt darüber). Ältere Einordnung: 0 Erfolge (N1.6 Lauf 34: `lifted` 0/10, noch alte Geometrie; N1.7
  Läufe 54/55: 0/10). Die Sim-Geometrie ist seit 2026-08-24 korrigiert; die N1.6-Neumessung mit
  korrigierter Szene ergab `lifted_clear` 0/20 (Lauf 82).
- **Stand 2026-10-15 (überholt durch den Punkt 2026-10-29) — Co-Training v3 läuft auf KISSKI, Abnahme auf dem IKR vorbereitet:**
  Läufe `render_v3_cube5`/`render_v3_cube4` (N1.6, Mischung 0,15) eingereicht, Smoke wartet auf
  GPUs (Laufnummern 73–79 reserviert). Abnahme ist ein Befehl: `server_rl_run.sh visgate`.
  Die Gate-Messung war ungeseedet (MSE je Episode bis ×2); geseedete Ausgangswerte MSE Hände
  5 / 4 cm: **N1.6 @44000 0,0239 / 0,0221** (Maßstab, Schwelle ~0,019 / 0,018), @30000
  0,0214 / 0,0193, N1.7 @44000 0,0758 / 0,0666 (Lauf 80). Closed Loop N1.7 @44000 in der
  Szene aus Lauf 70: `lifted` 0/20 (Lauf 81); N1.6 @44000 `lifted_clear` 0/20, erreicht/berührt
  den Würfel aber in 16/17 von 20 Ep. (Lauf 82).
  Details: [Läufe 80/81](docs/ergebnisse/diagnose-chronik.md#läufe-8081-visuelles-gate-geseedet-closed-loop-n17-in-der-szene-aus-lauf-70).
- **Neu 2026-10-14 (abends) — zwei Blocker nachgewiesen, in dieser Reihenfolge:**
  1. *Registrierung Würfel↔Hand:* die Würfel liegen in der Sim je Seite ~3 cm zu weit innen;
     mit Korrektur hebt das Orakel in ~50 % statt ~0 % (Lauf 62). Ursache offen (kein einzelner
     Gelenk-Nullpunkt, keine mitwandernde Kopfkamera, Lauf 63); empirische Korrektur
     `layout_heldout_dy3.json`.
  2. *Visueller Domain-Gap:* gleicher Zustand, gleiche Aktion, nur Sim- statt Echtbild → Hand-
     Fehler ×13 (N1.6) bzw. ×36 (N1.7), so schlecht wie schwarze Bilder (Läufe 65/66). Die
     Policies tragen den Griff über die Bilder (Ablation). Sichtbare Ursachen: Tisch zu klein
     mit Boden-Gitter, weiße Arme, Handgelenkkamera-Pose (Lauf 67).
  Closed Loop auch mit korrigiertem Layout weiter 0/2 je Modell (Lauf 64, abgebrochen).
  Orakel auf dem korrigierten Layout: `lifted_clear` 6/20 = 30 % (KI 12–54 %, Lauf 69).
  Szene ans Echtbild angeglichen (breiter Tisch, graue Arme, Handgelenkkamera vor/tiefer, 85°):
  Open-Loop-Hand-Fehler auf Sim-Bildern −40 % (N1.6) bzw. −18 % (N1.7), bleibt aber ×8 bzw. ×29
  über Echtbild; N1.7 leidet vor allem unter den Handgelenkkameras (Läufe 70/71).
  Details: [Läufe 57–68](docs/ergebnisse/diagnose-chronik.md#läufe-6368-ursache-des-versatzes-bild-ablation-sim-bilder-im-open-loop).
- **Stand 2026-10-14 (mittags, überholt durch den Punkt darüber) — die Sim ist (noch) der Blocker, nicht das Modell:** Das Orakel-Replay
  (aufgezeichnete Demo-Aktionen statt Policy, Held-out-Layout, 30 Hz) hebt selbst kaum einen
  Würfel: `lifted_clear` 1/16 (KI 0,2–30 %), `stacked` 0/16. Die Hand erreicht und schiebt den
  Würfel jedes Mal, greift aber wenige Zentimeter daneben; Würfelgröße ausgeschlossen, Griff-
  stelle der echten Hand liegt 1–7 cm neben der Bild-Würfellage, seitlich spiegelbildlich je Hand
  ([Läufe 57–61](docs/ergebnisse/diagnose-chronik.md#läufe-5761-orakel-replay-würfelgröße-griffversatz-span-n16)).
  Bevor die Registrierung Würfel↔Hand stimmt, sind Closed-Loop-Modellzahlen gegen eine Schranke
  von ~0 gemessen. `span` N1.6 vision-v2 @30000 auf Test-Episoden: 90 % (N1.7 @44000: 94 %).
  Daumen-Gelenkgrenzen geprüft: kein Klemmen (Lauf 56).
- **Ursache des Closed-Loop-Versagens nicht identifiziert** (Stand 2026-10-13; teilweise
  überholt durch die Orakel-Befunde oben — mindestens die Würfellage relativ zur Hand ist ein
  Blocker). „Domain-Gap bestätigt“ ist ein
  starker Hinweis, kein Befund — Closed und Open Loop unterscheiden sich in mehreren Faktoren
  ([Review 10-13 §2.1](docs/archiv/projekt-review-2026-10-13.md#21-befunde)); billige trennende
  Experimente: [Review 10-13 §2.3](docs/archiv/projekt-review-2026-10-13.md#23-empfohlene-experimente-nach-kostennutzen).
- **Co-Training:** seit 2026-10-29 mit Ergebnis (siehe oben). `cotrain_synth_v1` (2026-09-07)
  war kein Co-Training-Lauf.
- **Cosmos-Augmentierung, RL:** Werkzeuge gebaut, **kein Ergebnis**. Cosmos lief einmal
  end-to-end (2026-09-23, IKR-Server): Testlauf mit den Quell-Episoden 0 und 120, je 3 Stilvarianten,
  6 Episoden / 6003 Frames in `luca-mue/g1-dex3-cosmos-augmented`; ein Trainingslauf damit steht aus. RL ist N1.6-only und erst sinnvoll
  ab `lifted` > 0. Mit `lifted_clear` 2/60 (N1.6 co-train cube4, Läufe 88/97) ist diese
  Voraussetzung erstmals knapp in Sicht, aber noch nicht belastbar.

## 1. Wo das Projekt steht

Die **Messkette ist geschlossen**. Der Reihe nach:

| Befund | Beleg |
|---|---|
| Die Sim-Harness ist nicht kaputt | [sim-bewertung.md](docs/ergebnisse/sim-bewertung.md) |
| Die GR00T-Inferenz-Pipeline ist extern bestätigt (47,7 % vs. 47,8 % erwartet) | [basismodell-referenz-eval.md](docs/ergebnisse/basismodell-referenz-eval.md) |
| Die Policy versucht in der Sim gar keinen Griff (0,39 rad gegen 2,09 rad = **19 %**) | Lauf 30, [diagnose-chronik.md](docs/ergebnisse/diagnose-chronik.md) |
| Auf **echten** Datensatz-Bildern kommandiert dieselbe Policy die volle Greifbewegung (Verhältnis **1,00**) | Lauf 32, `span`-Gate |
| ~~Die Checkpoint-Auswahl war bisher nicht nur unbelegt, sondern **falsch**: der letzte Checkpoint ist 25 % schlechter als der beste~~ **Widerlegt 2026-10-12:** mit 20 Episoden à 750 Steps ist der letzte Checkpoint (44.000) der beste, kein Overfitting. Die Checkpoint-Auswahl per Sweep bleibt trotzdem Pflicht | [lauf3-vision-split-auswertung.md](docs/ergebnisse/lauf3-vision-split-auswertung.md), [lauf4-n17-vergleich.md](docs/ergebnisse/lauf4-n17-vergleich.md) §5 |
| `TUNE_VISUAL` hebt die Sim-Fingerspanne von 20,5 % auf **27,6 %** (vollständige Trennung, p = 3,3 · 10⁻⁴) — und reicht trotzdem nicht: `lifted` 0/10 | Lauf 34, [diagnose-chronik.md](docs/ergebnisse/diagnose-chronik.md) |

Daraus folgt nach der vorregistrierten Regel: **Ursache ist wahrscheinlich der Domain-Gap**, nicht
„Griff nie gelernt" — als *starker Hinweis*, nicht als Beweis (siehe Vorbehalte unten). Die Kopfzahl des
Projekts bleibt **0 Erfolge im Closed-Loop** (`stacked` überall 0). *(Nachtrag 2026-10-30: die
früheren Griffzahlen 0/20 bzw. 0/10 aus Lauf 34 sind überholt durch das Co-Training v3 —
`lifted_clear` 2/60 = 3 %, Läufe 88/97, siehe §0.)*

### Was seit dem 2026-08-14 passiert ist (Stand 2026-10-11)

| Datum | Befund | Beleg |
|---|---|---|
| 08-17 bis 08-25 | **Würfellage aus den Realbildern** (Farbblob → Strahl → Würfelebene) ersetzt die Greifpunkt-Rekonstruktion; Würfelebene lag 2 cm zu hoch (korrigiert); **Gierwinkel** der Würfel gemessen (gleichverteilt 0–45°; Umstellung auf zufälligen Gierwinkel offen) | Läufe 35–37, 52, 53 in [diagnose-chronik.md](docs/ergebnisse/diagnose-chronik.md); [wuerfellage-rekonstruktion.md](docs/simulation/wuerfellage-rekonstruktion.md) |
| 08-24 | **Geometriefehler in der Sim gefunden und behoben:** der Vorzeichen-Flip der vier proximalen Fingergelenke war *falsch* (entfernt) — die Hand konnte in **keinem früheren Lauf** ganz schließen; Beckenhöhe 0,764 m; Basis x −0,057 m | Läufe 38–51 in [diagnose-chronik.md](docs/ergebnisse/diagnose-chronik.md) |
| 09-07 / 09-23 | Externer Sim-Datensatz `Cube_Stacking_synth` auf das echte Schema harmonisiert; v21 hatte vertauschte Fingerbefehle → v22 und `Cube_Stacking_synth_jointspace`; Presets in `Training/kisski_chain.sh` | [synth-datensatz.md](docs/training/synth-datensatz.md) |
| 09-07 | Lauf `cotrain_synth_v1` **abgebrochen** nach 138 s (W&B `7z81177r`) — laut W&B kein Co-Training (ein Datensatz, `tune_visual: false`, IKR-Server; Nachtrag 2026-10-13 in der Diagnose-Chronik) | [co-training.md](docs/training/co-training.md) |
| 10-11 | LaTeX-Projektarbeit überarbeitet (`latex/`, 28 Seiten Hauptteil); GR00T-N1.7-Migration lief auf einem separaten Branch (seit 10-13 gemergt, inzwischen gelöscht) | — |
| 10-12 / 10-13 | N1.7: voller KISSKI-Lauf (Lauf 4, 44k) und Sim-Läufe 54/55 (`lifted` 0/10); Lauf-3-Nachmessung 20 × 750 widerlegt die U-Kurve, bester Checkpoint 44.000 | [lauf4-n17-vergleich.md](docs/ergebnisse/lauf4-n17-vergleich.md), [Chronik 54/55](docs/ergebnisse/diagnose-chronik.md#läufe-5455-die-ersten-n17-checkpoints-in-der-sim) |
| 10-15 bis 10-29 | Szene ans Echtbild angeglichen, Gate geseedet (Läufe 70–72, 80); **Co-Training v3** auf KISSKI (Läufe 73–76), Gate und Closed Loop auf dem IKR (Läufe 83–95): Sim-Lücke halbiert, erste Griffe (N1.6 cube4 2/20), `stacked` 0 | [co-training.md §3.5](docs/training/co-training.md#35-auswerten--zwei-messungen-nicht-eine), [Chronik](docs/ergebnisse/diagnose-chronik.md) |

**Konsequenzen und Vorbehalte, die jede Zahl dieses Dokuments betreffen:**

1. **Alle Sim-Greifzahlen vor dem 2026-08-24 stehen unter Vorbehalt** (Läufe 29–34, inkl. 27,6 %
   Fingerspanne und `lifted` 0/10): die Hand konnte nicht voll schließen, der Roboter stand ~12 cm
   versetzt. Mit der korrigierten Szene gemessen waren zunächst nur die N1.7-Läufe 54/55
   (2026-10-12/13); *überholt:* die N1.6-Neumessung ist erfolgt (Lauf 82, `lifted_clear` 0/20).
2. **Der Renderdatensatz für das Co-Training musste neu erzeugt werden** (er entstand mit der
   fehlerhaften Geometrie) — *überholt:* erledigt, Render v3 (Lauf 72).
3. **Das `span`-Gate (Median 1,00) nutzte die Trainingsepisoden 0–4**, und 2,0944 rad ist zugleich das
   De-Normalisierungs-Maximum. Auswendiglernen ist damit nicht ausgeschlossen; „Domain-Gap bestätigt /
   (a2) ausgeschlossen" ist ein starker Hinweis, kein Beweis. Für N1.7 gibt es die Messung auf
   zurückgehaltenen Episoden inzwischen (Test 240/255/270/285/300: 90 % bzw. 94 %,
   [Läufe 54/55](docs/ergebnisse/diagnose-chronik.md#läufe-5455-die-ersten-n17-checkpoints-in-der-sim));
   für N1.6 @30000 inzwischen auch (Lauf 61: 90 %), @44000 steht aus.
4. **Trainingsparameter der Läufe 2/3 (W&B):** Peak-LR 1e-4, Warmup 0,10, 4 GPUs × Batch 8.

> **Ergänzung 2026-08-14 (abends):** Die letzte Zeile ist das Ergebnis des `span`-Gates aus
> Schritt 3 (§3, heute in der [Historie](docs/historie.md)) — und sie ist zweischneidig. Der Vision-Pfad **wirkt nachweislich**: drei von zehn
> Episoden kommandieren 60–80 % der Demonstration, ein Verhalten, das der alte Checkpoint über
> sieben Episoden nie gezeigt hat, und die Anhebung läuft mit der Fingerspanne mit
> (Spearman +0,62). Er **genügt aber nicht**: 27,6 % stehen gegen 100 % auf echten Bildern, und
> kein Würfel überschreitet die 2-cm-Schwelle. Ein weiterer Lauf derselben Art würde denselben
> Faktor-3,6-Rest übriglassen.

> **Ergänzung 2026-08-14:** Der letzte Punkt der Tabelle ist neu und relativiert die früheren
> Sim-Ergebnisse *methodisch*, nicht inhaltlich. Lauf 1 und Lauf 2 wurden mit dem jeweils
> **letzten** Checkpoint evaluiert, weil es keine Alternative gab. Lauf 3 zeigt erstmals, dass
> das systematisch der falsche Griff ist. Die 0/20 sind damit nicht erklärt — der Domain-Gap-
> Befund steht unverändert — aber jede künftige Sim-Zahl gehört an den **validierten**
> Checkpoint, nicht an den letzten.

> Alles, was seit Lauf 32 gebaut wurde — Livestream (Spur A), Latenz-Tooling, Render-Knöpfe —
> ist **Instrumentierung**. Es hat diese Zahl nicht bewegt und sollte in der Projektarbeit auch
> nicht als Fortschritt an ihr dargestellt werden.

---

## 2. Nächste Schritte — in dieser Reihenfolge (Stand 2026-10-29)

> **Abgabe:** Stand 2026-11-09: Ausarbeitung baut sauber (49 Seiten, davon 28 Hauptteil), Zahlen-Stichprobe erledigt; Abgabedatum 09.11.2026 eingetragen; Merge nach `main` erledigt; offen sind Fremdtest, Absatz-Test und Tag. (Der frühere Abgabeplan `docs/abgabe-plan.md` ist entfernt.)
>
> **Stand 2026-10-29 — nach dem Co-Training v3, vor allen älteren Punkten unten:**
> 1. ~~Griffrate belastbar machen~~ **erledigt 29.10. (Lauf 97):** C auf allen 60 Testlagen,
>    `lifted_clear` 2/60 = 3 % (KI 0,4–11,5 %). Griffe sind selten; der nächste Hebel muss die
>    Griffrate heben, nicht nur messen (Punkt 7).
> 2. ~~Vorregistriertes primäres Gate nachholen~~ **erledigt 29.10. (Lauf 96):** bestanden,
>    1,53 rad gegen 0,577 / 0,52 rad. `gap` gemessen, als Kontrolle ungeeignet.
> 3. **Robotertest:** cube4 @44000 gegen vision-v2 @44000 auf dem echten G1. Er entscheidet, ob
>    der Echtbild-Verlust von N1.6 (+50–60 % MSE Hände) tragbar ist. Falls nicht, ist N1.7 cube4
>    @44000 (`…-cube4-n17-ckpt44000`) die Alternative ohne Echtbild-Verlust, greift in der Sim
>    aber noch nicht.
> 4. **Gate-Schwelle nachschärfen:** Die 80 % von vision-v2 @44000 lassen auch nicht co-trainierte
>    Modelle durch (Läufe 87/90/91). Vorschlag: Schwelle relativ zum besten nicht co-trainierten
>    Modell derselben Generation (N1.6 ~0,017 / 0,016), Echtbild-Kriterium als feste Zahl.
> 5. **`lifted_clear` braucht den Videobefund:** Die Metrik zählt auch einen Würfel, der in der
>    Hand auf einem anderen liegt (Lauf 94). Ein Kontaktkriterium ohne Stütz-Würfel wäre die
>    saubere Lösung.
> 6. **Hänger im Closed Loop:** Am 25.10. und 29.10. hingen drei Läufe mitten in der Episode,
>    ohne Fehlermeldung und nicht an eine Lage gebunden (Lauf 97). Umgangen mit
>    [`tools/sim_cl_watchdog.sh`](tools/sim_cl_watchdog.sh); die Ursache ist offen.
> 7. **Griffrate heben:** Nach der vorregistrierten Regel heißt das primäre Gate „aufstocken“:
>    mehr Render-Episoden statt höherer Mischung (co-training.md §4.2). Alternativ RL auf C, das
>    mit 3 % Griffen erstmals ein, wenn auch schwaches, Startsignal hat
>    ([RL-Plan](docs/weiterfuehrend/reinforcement-learning-plan.md)).
>
> Die Punkte 00–3 unten sind für das Co-Training erledigt (00: Läufe 73–95; 3: Baselines
> Läufe 81/82).

> **Vorgezogen (2026-10-14), vor allen Punkten der Tabelle unten:**
> 00. **Stand 2026-10-15 mittags:** auf KISSKI eingereicht (Smoke wartet). Abnahme auf dem IKR
>    mit `server_rl_run.sh visgate` gegen die geseedeten Ausgangswerte aus Lauf 80 (Maßstab N1.6
>    @44000: 0,0239 / 0,0221, Schwelle ~0,019 / 0,018); erst darunter Closed Loop mit Szene
>    aus Lauf 70 gegen Lauf 81/82. Die Ausgangswerte aus Lauf 72 sind überholt durch Lauf 80.
>
>    **Stand 2026-10-15 früh — bereit für KISSKI:** Co-Training-Datensätze v3 (5 cm und 4 cm, je 187
>    Trainings-Ep., 29 890 Frames, Lauf 72) liegen auf HF (`luca-mue/g1-dex3-rendered-v3-cube5`,
>    `…-cube4`; damals privat, seit 2026-10-30 öffentlich). Presets in `Training/kisski_chain.sh`: `smoke_render_v3`, `render_v3_cube5`,
>    `render_v3_cube4`, `render_v3_cube5_n17` (Mischung 0,15). Abnahme danach auf dem IKR-Server mit
>    dem Prüfset `/data/cotrain/vis_gate_v3_cube{5,4}` (Trainings-Ep. 0/4/8/12, gleiche Szene).
> 0a. **(Läufe 69–71):** Parameter-Feinschliff der Szene ist ausgereizt (−18…−40 %, Abstand
>    zum Echtbild bleibt ×8/×29). Nächster Hebel ist die **Trainingsseite**: Co-Training mit
>    Sim-Renderbildern derselben Episoden (Render mit korrigierter Geometrie, korrigiertem Layout
>    und der besten Szene aus Lauf 70: `SIM_TABLE_SIZE="0.8 2.0"`, graue Arme,
>    `SIM_WRIST_EYE="0.02 0.0 0.08"`, `SIM_WRIST_TARGET="0.30 0.0 -0.08"`, `SIM_WRIST_HFOV=85`).
>    Abnahme vor jedem Closed Loop: die Open-Loop-Metrik aus Lauf 66 auf zurückgehaltenen
>    Sim-Renderbildern. Parallel offen: Ursache des 3-cm-Versatzes und die Parallaxe der
>    Kopfkamera (Hände im Echtbild ~2× größer bei gleicher Würfellage).
> 0. **(Läufe 63–68) den visuellen Gap gezielt schließen, mit billiger Messgröße** — Szenenteil
>    erledigt (Lauf 70), Ergebnis s. 0a.
>    Fitnessfunktion = MSE Hände im teacher-forced Open Loop auf Sim-Renderbildern (Lauf 66;
>    Ziel: Richtung Echtbild 0,002 statt 0,027/0,061), erst dann Closed Loop. Kandidaten der
>    Reihe nach: Tischgröße/-textur und Hintergrund (Wand statt Boden-Gitter), Armfarbe,
>    Handgelenkkamera-Pose, veraltetes Bild bei Step 0. Co-Training/`TUNE_VISUAL` erst, wenn
>    die Szene selbst nicht mehr offensichtlich abweicht.
> 1. **Registrierung Würfel↔Hand** — Versatz-Sweep **erledigt (Lauf 62)**: Würfel je Seite
>    ~3 cm nach außen → Orakel `lifted_clear` 23/48 = 48 % statt 0/24. Ursache weiter offen
>    (Lauf 63: kein einzelner Gelenk-Nullpunkt, Kopfkamera-Höhe widerlegt); bis dahin mit
>    `layout_heldout_dy3.json` messen.
> 2. Orakel-Hänger (Datensatz-Ep. 256, Halte-Phase) nachstellen — eine Eval darf nicht hängen.
> 3. Erst danach die Closed-Loop-Baselines N1.6 @44000 / N1.7 @44000 (20 Ep., 120 s, Held-out-Layout).
>    N1.6 @44000 liegt nur auf KISSKI → nach HF hochladen oder herkopieren.
>
> Der `span`-Schritt (Punkt 2 unten) ist für N1.6 auf zurückgehaltenen Episoden erledigt
> (Lauf 61, @30000: 90 %); @44000 steht aus.

**Tabelle Stand 2026-10-11** (Punkt 3 überholt durch den Orakel-Befund: die Closed-Loop-Neumessung
ist erst nach der Registrierung aussagekräftig):

| # | Schritt | Warum / Ausgang |
|---|---|---|
| 1 | **Open-Loop-MSE auf allen 61 Testepisoden**, volle Länge, getrennt nach Armen und Händen, mit trivialer Baseline (z. B. konstante Pose / letzter Zustand) | Der Sweep aus Lauf 3 nutzte nur 6 Episoden × 300 Steps; ohne Baseline ist die MSE nicht einzuordnen. Billig (`checkpoint_sweep.py`, `EVAL_NUM_TRAJ`/`EVAL_STEPS`) |
| 2 | **`span`-Gate auf zurückgehaltenen Testepisoden** (`SPAN_TRAJ_IDS` mit Test-Indices) — für N1.7 erledigt (Läufe 54/55: Test 90 % / 94 %); ✅ für N1.6 @30000 erledigt (Lauf 61: 90 %), @44000 offen | Schließt die Auswendiglern-Lücke des Lauf-32-Gates; trägt „Domain-Gap bestätigt“ vom Hinweis zum Befund |
| 3 | ✅ *erledigt (Lauf 82: N1.6 @44000, Szene aus Lauf 70, 20 × 120 s, `lifted_clear` 0/20; N1.7: Lauf 81)* **Closed-Loop-Neumessung mit der korrigierten Szene** (identisches Protokoll wie Lauf 34: 10 Episoden × 40 s, Checkpoint vision-v2-**44000** — der seit der Nachmessung 2026-10-12 validierte; bei Lauf 34 war es noch 30000), optional **Tauschtest** (echtes Bild + Sim-Zustand vs. Sim-Bild + echter Zustand) | Erst damit sind die Sim-Zahlen wieder belastbar; entscheidet, wie groß der Domain-Gap nach der Geometriekorrektur noch ist |
| 4 | ✅ *erledigt (Render v3 Lauf 72, Trainings Läufe 73–76, Auswertung Läufe 83–97)* **Renderdatensatz neu erzeugen → Co-Training-Lauf** (eigener Render und/oder Synth v22 / `Cube_Stacking_synth_jointspace`); Log des Abbruchs von `cotrain_synth_v1` aus dem Trainingscontainer des IKR-Servers holen; auf KISSKI zuerst `kisski_chain.sh smoke …` | Vorregistrierte Regeln in [co-training.md](docs/training/co-training.md) / [synth-datensatz.md](docs/training/synth-datensatz.md) |
| 5 | **RL (FPO)** erst, wenn die Policy gelegentlich einen Würfel hebt — seit Lauf 97 (2/60) schwaches Startsignal | Ohne Startsignal fehlt dem Reward der Gradient ([RL-Plan](docs/weiterfuehrend/reinforcement-learning-plan.md)) |
| 6 | **Echter Roboter** | setzt belastbare Sim- bzw. Open-Loop-Zahlen voraus |

Offene Entscheidung nebenbei: Umstellung der Sim auf zufälligen Würfel-Gierwinkel (gemessen: gleichverteilt
0–45°).

**Erledigt:** Der frühere Widerspruch zwischen RL-Plan und
[`lauf2-vision-auswertung.md`](docs/ergebnisse/lauf2-vision-auswertung.md) §6.1 (`TUNE_VISUAL` ja/nein) ist
aufgelöst — Lauf 2 hatte keinen Color-Jitter (`USE_AUGMENTATION` kam erst 2026-06-12, `c3a1cd6`), Lauf 3
erfüllt die Bedingung aus §6.2. Vorbehalt bleibt: Color-Jitter variiert weder Geometrie noch Textur noch
Rendering-Stil; `cam_left_wrist` liegt in der [Domain-Gap-Messung](docs/ergebnisse/domain-gap-analyse.md)
bei 0,36.

---

## 3. Bisherige Schritte

Der Verlauf der Schritte 1–5 (Checkpoint-Auswahl, Lauf 3, `span`-Gate, Co-Training, RL) steht seit
2026-11-09 in der [Historie](docs/historie.md#2026-08-13--2026-10-29--next-stepsmd-3-bisherige-schritte-ausgelagert-2026-11-09);
die Messwerte in der [Diagnose-Chronik](docs/ergebnisse/diagnose-chronik.md).

---

## 4. Nebenbei offen (Komfort, nicht Fortschritt)

| Punkt | Aufwand | Status |
|---|---|---|
| Hardware-Test der Live-Ansicht (Stand 2026-10-11 weiterhin offen; nicht belegt, ob inzwischen gelaufen): `./Simulation/server_rl_run.sh livecheck`, dann `… view` (Szene ohne Gewichte, braucht kein `HF_TOKEN`). Als Client wahlweise die native App oder `… webview` im Browser — letzteres ohne Installation | ~15 min | Gebaut 2026-08-13, `view`/`webview` 2026-08-17. Der Web-Viewer ist lokal end-to-end geprüft; der **WebRTC-Handshake lief nie auf Hardware**. Bleibt das Bild trotz offenem UDP schwarz: `RL_NETWORK_MODE=host` |
| `SCENE_CAM=0` / `RL_AA_MODE=Off` Wirkung auf Steps/s | ~15 min | Ungemessen (~15 % erwartet) |
| Doku-Abschnitt zur Latenz-Zerlegung | — | Erledigt: steht in [inferenz-optimierung.md](docs/simulation/inferenz-optimierung.md) |
| `tools/test_menu.sh` hinterlässt einen laufenden Container `groot-rl` mit `/data` im gelöschten Temp-Verzeichnis (2026-10-14 zweimal beobachtet und entfernt); außerdem scheitern 11 Tests, sobald eine `.env.local` mit `HF_TOKEN` existiert | ~30 min | Offen — der Test, der den Launcher wirklich startet, muss den Container selbst wieder entfernen bzw. `RL_CONTAINER` auf einen Wegwerfnamen setzen |

Nicht dringend, weil keiner davon die 0/20 bewegt.

---

## 5. Was ausdrücklich *kein* nächster Schritt ist

- **Inferenz-Optimierung.** Die Zerlegung `t(n) = 35,6 ms + n × 9,6 ms` steht in
  [inferenz-optimierung.md](docs/simulation/inferenz-optimierung.md), aber bei 74 ms gegen 267 ms Budget
  (28 % Auslastung) ist heute nichts zu tun. Das ist ein Planungswerkzeug für die Portierung auf
  Zielhardware, keine offene Baustelle.
- **`TUNE_VISUAL=1` ohne Augmentierung.** Das wäre die Wiederholung von Lauf 2 mit dem bereits
  gemessenen Ergebnis (Policy-Kollaps).
- **Qualitätsmessung der Denoising-Schritte über `open_loop_eval.py --denoising-steps`.** Die
  Option ist im Fork **tot** (deklariert, nirgends angewandt) — sie vergleicht zwei identische
  Läufe. Details und der korrekte Weg stehen in
  [inferenz-optimierung.md](docs/simulation/inferenz-optimierung.md).
