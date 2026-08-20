# GitHub-Wiki-Migration (Plan)

> **TL;DR:** Klärt, ob und wie sich die bestehende `docs/`-Doku ins GitHub-Wiki des Repos
> übertragen lässt. Ergebnis: technisch möglich, aber nicht per einfachem Kopieren — das Wiki ist
> ein zweites, separates Git-Repo mit eigener Verlinkungslogik. Reine Recherche, **nichts
> umgesetzt**.

## Ausgangsfrage

Lässt sich die vorhandene Dokumentation unter [`docs/`](.) leicht in den GitHub-Wiki-Tab des
Repos übertragen, statt sie dort separat zu pflegen?

## Kurzantwort

Ja, machbar — aber nicht mit einem einfachen "Ordner reinkopieren". Der Aufwand liegt im
Link-Rewriting, nicht im Kopieren selbst.

## Wie ein GitHub-Wiki technisch funktioniert

Ein GitHub-Wiki ist ein **eigenständiges, zweites Git-Repository** namens `<repo>.wiki.git`, das
entsteht, sobald der Wiki-Tab aktiviert und einmal über die Web-UI eine Seite angelegt wird. Jede
`.md`-Datei darin wird zu einer Wiki-Seite. Es lässt sich ganz normal klonen, committen und
pushen — kein Spezialtool nötig.

## Vorgehen (Schritte)

1. **Wiki aktivieren** — Repo-Settings → Features → Wikis, einmal eine Seite über die Web-UI
   anlegen (damit das Wiki-Git-Repo existiert).
2. **Klonen** — `git clone git@github.com:<user>/projektarbeit_humanoider_roboter.wiki.git`
3. **Dateien kopieren** — Inhalte aus `docs/` ins Wiki-Repo übertragen. Unterordner
   (`training/`, `simulation/`, `ergebnisse/`, `weiterfuehrend/`) werden vom Wiki-Repo technisch
   mitgetragen, tauchen aber nicht automatisch strukturiert in der Sidebar auf.
4. **Startseite + Navigation** — [`docs/README.md`](README.md) wird sinnvollerweise zu
   `Home.md`; zusätzlich eine `_Sidebar.md` anlegen, sonst zeigt GitHub nur eine flache
   A–Z-Liste aller Seitentitel.
5. **Commit & Push** ins Wiki-Repo — Seiten erscheinen sofort im Wiki-Tab.

## Blocker / worauf achten

- **Links außerhalb von `docs/` brechen.** Mehrere Docs verlinken relativ auf Dateien im
  Submodul `app/Groot-1.6/examples/G1_DEX3/` (siehe [`docs/README.md`](README.md#submodul-dokumentation-appgroot-16examplesg1_dex3)):
  `SETUP_DOCUMENTATION.md`, `FINETUNING_GUIDE.md`, `README.md`. Die liegen nicht im Wiki-Repo und
  wären dort unerreichbar — diese Links müssten auf volle GitHub-Blob-URLs
  (`https://github.com/<user>/<repo>/blob/main/app/...`) umgeschrieben werden.
- **Interne Docs-Links müssen umgeschrieben werden.** Wiki-Verlinkung läuft über Seitennamen
  (`[[Seiten Name]]` bzw. `[Text](Seiten-Name)`, ohne `.md`, Leerzeichen → Bindestriche), nicht
  über Dateipfade wie aktuell (`docs/training/anleitung.md`).
- **Kein automatischer Sync.** Das Wiki-Repo ist vom Hauptrepo getrennt. Änderungen an `docs/`
  landen nicht automatisch im Wiki — dafür bräuchte es eine GitHub Action, die bei Push nach
  `docs/` das Wiki-Repo mit-aktualisiert.
- **Bilder/Assets** (falls relativ referenziert) müssten mitkopiert oder verlinkt werden.

## Automatische Synchronisierung (docs/ → Wiki)

GitHub selbst bietet keinen nativen Sync zwischen einem Repo-Ordner und dem Wiki-Tab. Erreichbar
ist er nur **einseitig** (`docs/` → Wiki) über eine **GitHub Action**:

- Workflow, der bei jedem Push nach `docs/**` auf `main` auslöst, das Wiki-Repo
  (`<repo>.wiki.git`) auscheckt, die Dateien aus `docs/` reinkopiert (inkl. des oben
  beschriebenen Link-Rewritings als Skript-Schritt) und committet/pusht.
- **Voraussetzung:** Der Standard-`GITHUB_TOKEN` hat kein Schreibrecht auf das Wiki-Repo — dafür
  ist ein Personal Access Token mit `repo`-Scope als Repo-Secret nötig.
- **Trade-off:** Das Wiki wird dadurch zu einem reinen **generierten Spiegel**. Direkte
  Bearbeitungen im Wiki-UI gehen beim nächsten Sync-Lauf wieder verloren — eine echte
  Zweiwege-Synchronisation zwischen Wiki und `docs/` ist unüblich und deutlich aufwendiger.

## Alternative: GitHub Pages (z. B. MkDocs)

Falls es primär um eine hübsch navigierbare, öffentlich lesbare Doku-Seite geht statt um den
Wiki-Tab konkret: GitHub Pages mit z. B. MkDocs direkt aus `docs/` gebaut. Vorteil gegenüber dem
Wiki: relative Links und die vorhandene Ordnerstruktur bleiben größtenteils nutzbar, und es
entsteht keine zweite, manuell zu pflegende Kopie — die Seite wird bei jedem Push aus `docs/`
neu gebaut.

## Offen

Reine Recherche, keine Entscheidung getroffen. Optionen zur Wahl:

1. Wiki manuell pflegen, `docs/` bleibt Quelle der Wahrheit (kein Sync-Aufwand, aber Doppelpflege
   bei Änderungen).
2. Kleines Migrations-Skript (Wiki-Repo klonen, Dateien kopieren, Links per Skript umschreiben) —
   einmalig oder wiederholbar.
3. GitHub Pages/MkDocs statt Wiki — kein zweites Repo, automatisch synchron.

## Verwandte Dokumentation

- [Doku-Übersicht](../README.md) — globaler Navigations-Hub
- [Weiterführende Arbeiten](README.md) — Index dieser Sammlung
