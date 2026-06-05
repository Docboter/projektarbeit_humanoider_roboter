<!-- markitdown: auto-converted from /home/luca/coding/projektarbeit/projektarbeit_humanoider_roboter/docs/ergebnisse/wandb-run-charts.html -->

# W&B-Verlauf · g1\_dex3\_blockstacking\_v1

Projekt `gr00t-g1-dex3` · Entity `projektarbeit_humanoider_roboter` · Run-ID `i6n1t613`
Node `ggpu177` (KISSKI) · 1× GPU · **Snapshot: 2026-06-03, ~08:39 UTC** — laufendes Training, nicht final.

**Lesehilfe:** Der **train/loss** ist ein Flow-Matching-/Diffusion-Loss — pro Schritt wird ein
zufälliger Diffusions-Timestep gezogen, daher die hohe Punkt-zu-Punkt-Streuung. Auf die
**geglättete Linie** (gleitender Mittelwert) achten, nicht auf Einzelwerte. Ein niedriger
Train-Loss bedeutet gute Anpassung an die *Trainingsverteilung* — er ist **kein**
Generalisierungs-/Task-Erfolgssignal (dafür fehlt die Eval, siehe
`wandb-run-auswertung.md` Abschnitt 3).

Daten via W&B-API (220 gesampelte Punkte über die bis dahin geloggte Historie + aktueller
Zusammenfassungspunkt). Erneuern: Werte im `DATA`-Array unten austauschen oder Datei
neu generieren lassen.