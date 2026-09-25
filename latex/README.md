# LaTeX — Projektarbeit (Ausarbeitung)

> [!NOTE]
> **Stand: 2026-08-26** (Abgabefassung: 35 Seiten Hauptteil / 58 Seiten gesamt, 11pt, baut
> sauber durch). In der Kompaktierungs-Runde vom 2026-08-26 wurde die Fassung von 45 auf
> 35 Seiten Hauptteil gebracht (Redundanz + Grundlagen raus, keine Substanzkürzung) und um
> vier neue Bausteine ergänzt: Beitragsübersicht/Eigenleistung (Kap. 1),
> Diagnose-Werkzeugkasten (Kap. 3), Checkpoint-U-Kurven-Diagramm (Kap. 4) und das neue
> **Kapitel 7 „Projektverlauf und Lessons Learned"** (Hindernisse, Lessons Learned,
> Arbeitsmittel-Wahl, KI-gestützte Entwicklung). **Noch offen:** persönliche
> Titelseiten-Daten in `preamble/settings.tex`, vollständige Namen der Teammitglieder in
> der Beitragstabelle (Kap. 1, TODO-Kommentar), Prüfung von `backmatter/erklaerung.tex`.

## Zweck

Schriftliche Ausarbeitung der Projektarbeit (Fine-Tuning von NVIDIA GR00T N1.6 auf dem
Unitree G1 + DEX3-Hand). Quelle des Inhalts ist die Projektdokumentation unter `docs/`;
diese Ausarbeitung fasst sie nicht zusammen, sondern erzählt den Arbeits- und
Erkenntnisprozess (inkl. des Domain-Gap-Befunds) und referenziert `docs/` an den
relevanten Stellen.

## Struktur

```
latex/
├── main.tex                 # Hauptdatei (scrreprt, KOMA-Script, lualatex)
├── preamble/
│   ├── packages.tex         # Pakete
│   └── settings.tex         # Einstellungen (Satzspiegel, Acronyms, minted, …)
├── frontmatter/
│   ├── titelseite.tex       # Titelseite        ← persönliche Daten offen
│   └── zusammenfassung.tex  # Zusammenfassung + englisches Abstract
├── chapters/
│   ├── 01_einleitung.tex                #  Motivation, Ziele, Beitragsübersicht/Eigenleistung
│   ├── 02_grundlagen.tex                #  VLA, Flow Matching, Domain-Gap (nur Projektspezifisches)
│   ├── 03_methoden.tex                  #  Infrastruktur, Datensatz, Training, Eval, Werkzeugkasten
│   ├── 04_ergebnisse.tex                #  Läufe 1–3 (inkl. U-Kurve), Diagnosekette, Geometrie
│   ├── 05_diskussion.tex                #  Einordnung, Messartefakte, Limitierungen
│   ├── 06_weiterfuehrende_arbeiten.tex  #  Lokomotion, Co-Training, Cosmos, RL
│   ├── 07_projektverlauf.tex            #  Hindernisse, Lessons Learned, Arbeitsmittel, KI-Einsatz
│   └── 08_fazit.tex                     #  Fazit und Ausblick (rein fachlich)
├── backmatter/
│   ├── anhang.tex           # Datensatz, SLURM, Software, Trainingsdynamik, Szene
│   └── erklaerung.tex       # Ehrenwörtliche Erklärung ← prüfen
├── figures/                 # Abbildungen
├── references.bib           # Literatur (18 Einträge, 16 zitiert)
├── Makefile / .latexmkrc    # Build-Konfiguration
└── main.pdf                 # zuletzt kompiliertes PDF (gitignored)
```

## Kompilieren

LuaLaTeX + Biber, `-shell-escape` für `minted`. Am einfachsten via `latexmk` (Konfiguration
liegt in `.latexmkrc`):

```bash
latexmk main.tex      # oder einfach:  make
```

Manuell:
```bash
lualatex -shell-escape main.tex
biber main
lualatex -shell-escape main.tex
lualatex -shell-escape main.tex
```

Hilfs- und Cache-Dateien (`*.aux`, `_minted/`, …) sowie das PDF sind über `.gitignore`
ausgeschlossen.

## Nächste Schritte (TODO)

- [x] Inhalt aller `chapters/*.tex` ausgearbeitet (01 Einleitung … 08 Fazit)
- [x] Kompaktierung auf Zielumfang 25–35 Seiten Hauptteil (2026-08-26: 35 S. bei 11pt)
- [ ] Vollständige Namen der Teammitglieder in der Beitragstabelle (Kap. 1) eintragen
- [x] Zusammenfassung / Abstract geschrieben (`frontmatter/zusammenfassung.tex`)
- [x] `references.bib` mit verifizierten Quellen befüllt (GR00T N1, SIMPLER, Flow Matching, π₀, π-RL, …)
- [x] Anhang an echte Datensatz-/SLURM-/Software-Fakten angepasst
- [x] Abgleich mit dem Projektstand vom 2026-08-25 (Lauf 3, `span`-Gate, Greif-Physik
      aufgelöst, geometrische Rekonstruktion, Widerruf des Vorzeichen-Fixes, Co-Training)
- [x] `latexmk` läuft lokal durch (LuaLaTeX + Biber sind installiert), keine undefinierten
      Referenzen, keine kritischen Overfull-Boxen
- [x] Verfasser eingetragen: Maximilian Berger, Matthias Späth, Luca Mücke
      (`\docAuthorA/B/C` in `preamble/settings.tex`; Titelblatt und ehrenwörtliche
      Erklärung geben alle drei einzeln aus)
- [ ] Restliche persönliche Daten in `preamble/settings.tex` (`\docMatrikelA/B/C`,
      `\docBetreuer`, `\docHochschule`, `\docFachbereich`, `\docStudiengang`,
      `\docAbgabedatum`)
- [ ] Optional: echte Abbildungen in `figures/` einbinden (statt der TikZ-Diagramme/W&B-Plot)
- [ ] Ehrenwörtliche Erklärung prüfen (`backmatter/erklaerung.tex`)

### Inhaltlich offen (hängt am Projekt, nicht am Dokument)

- Die simulationsseitigen Greifzahlen (0/20, Fingerspanne 19 % / 20,5 % / 27,6 %) sind mit
  korrigierter Geometrie **neu zu erheben** — siehe Abschnitt „Vorbehalt gegen alle
  simulationsseitigen Greifzahlen" in Kapitel 5.
- Sobald der Co-Training-Lauf gefahren ist, gehört sein Ergebnis in Kapitel 6/7.
