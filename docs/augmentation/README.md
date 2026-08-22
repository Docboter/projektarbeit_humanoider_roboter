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

## Einstieg

| Dokument | Wann lesen |
|---|---|
| [anleitung.md](anleitung.md) | **Zuerst.** Warum Cosmos-Transfer2.5 (statt cosmos-transfer1 oder der `robot_augmentation`-Variante), Build/Run-Schritte, Env-Var-Referenz, Einbindung ins Training, offene Punkte. |

## Verwandte Dokumente

- [../training/co-training.md](../training/co-training.md) — der Trainings-seitige Co-Training-Mechanismus (`USE_COTRAIN=1`), den dieser Datensatz nutzt
- [../ergebnisse/domain-gap-analyse.md](../ergebnisse/domain-gap-analyse.md) — Referenzmethodik (frozen-SigLIP Cosine-Distanz), mit der sich die Bildqualität der Augmentierung später einordnen lässt
