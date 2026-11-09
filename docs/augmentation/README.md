# Augmentation — Video-Augmentierung mit Cosmos-Transfer2.5

> **TL;DR:** Einstiegspunkt für die Cosmos-Transfer2.5-Video-Augmentierung: echte
> Trainingsvideos werden stilvariiert (Domain-Randomization), um zusätzliche Trainingsdaten
> zu erzeugen. Für die konkreten Befehle direkt zu [anleitung.md](anleitung.md).

Nimmt die echten G1/DEX3-Block-Stacking-Demonstrationsvideos und lässt sie durch NVIDIAs
Cosmos-Transfer2.5-Video-zu-Video-Modell laufen: gleiche Roboterbewegung, unterschiedliches
Licht/Hintergrund/Oberflächen-Textur. Läuft autonom in einem eigenen Docker-Container auf
einem dedizierten Server (A100/H100-Klasse, ~65 GB VRAM). Das Ergebnis ist ein
LeRobot-v2.1-Datensatz, der direkt als `COTRAIN_DATASET_PATH` in den bestehenden
Co-Training-Pfad des Trainings-Images eingehängt werden kann — siehe
[co-training.md](../training/co-training.md).

**Stand (2026-11-09): einmal durchgängig gelaufen, noch nicht im Training verwendet.** Am 2026-09-23 lief die Pipeline auf dem IKR-Server erstmals Ende-zu-Ende (nach dem Abbruch des ersten Versuchs am 2026-08-24 an `nvidia/Cosmos-Predict2.5-2B`). Ergebnis: HF-Datensatz `luca-mue/g1-dex3-cosmos-augmented`, ein **Testlauf** mit den Quell-Episoden 0 und 120, je 3 Stilvarianten (`AUGMENT_VARIANTS=3`), also 6 Episoden mit 6003 Frames (1173 bzw. 828 je Quelle, identisch zur Quelle); `augment_manifest.json` meldet alle 4 Kameras bei allen 6 Episoden als `ok`, `meta/modality.json` ist byte-identisch zur Quelle. Sichtprüfung: Pose und Layout bleiben erhalten, aber die stärkste Variante („warmes Licht, Holztisch") färbt trotz Prompt-Vorgabe „Roboter unverändert" die schwarzen Roboterhände weiß und einen Würfel grün. Offen: der volle Lauf über alle 240 Train-Episoden, ein Trainingslauf mit den Daten und die Dataset-Card auf HF (nur 25 Byte). Das Image liegt nicht auf Docker Hub, das Skript braucht einen Server mit ~65 GB VRAM. Details und offene Punkte: [anleitung.md §7](anleitung.md#7-offene-punkte).

## Einstieg

| Dokument | Wann lesen |
|---|---|
| [anleitung.md](anleitung.md) | **Zuerst.** Warum Cosmos-Transfer2.5 (statt cosmos-transfer1 oder der `robot_augmentation`-Variante), Build/Run-Schritte, Env-Var-Referenz, Einbindung ins Training, offene Punkte. |

## Verwandte Dokumente

- [../training/co-training.md](../training/co-training.md) — der Trainings-seitige Co-Training-Mechanismus (`USE_COTRAIN=1`), den dieser Datensatz nutzt
- [../ergebnisse/domain-gap-analyse.md](../ergebnisse/domain-gap-analyse.md) — Referenzmethodik (frozen-SigLIP Cosine-Distanz), mit der sich die Bildqualität der Augmentierung später einordnen lässt
