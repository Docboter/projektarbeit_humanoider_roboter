#!/usr/bin/env bash
# TL;DR: Host-Werkzeug — fuehrt den 76-Pruefungen-Akzeptanztest fuer die CLI-Menues aus (§8).
# test_menu.sh — der Pruefplan aus docs/weiterfuehrend/cli-menuefuehrung.md §8.
#
# Alles ohne GPU und ohne Container pruefbar: das Menue ist reine Host-Logik. Die
# interaktiven Faelle laufen durch ein echtes Pseudoterminal (Python-Modul `pty`),
# weil `[[ -t 0 ]]` sonst nie wahr wird und genau der zu pruefende Pfad ausbliebe.
#
# Die wichtigste Pruefung ist die letzte (Defaults Spec vs. Skript vs. Doku) — sie ist
# die Versicherung gegen das Driften, das dieses Vorhaben ueberhaupt erst rechtfertigt.
# Sie steckt in tools/gen_docs.sh und wird hier nur aufgerufen.
#
# BEWUSST OHNE pipefail: fast jede Pruefung leitet ein Skript, das absichtlich mit einem
# Rueckgabewert != 0 endet (unbekannte Aktion, fehlende GPU), in ein grep. Mit pipefail
# ueberstimmte der Erzeuger das grep-Ergebnis und jede Pruefung schluege fehl.

set -u
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export RL_HOST_DATA_DIR="$TMP/data" MENU_STATE_DIR="$TMP/.menu"
SIM=./Simulation/server_rl_run.sh

PASS=0; FAIL=0
ok()  { printf '  \033[1;32m v \033[0m %s\n' "$*"; PASS=$((PASS+1)); }
bad() { printf '  \033[1;31m!! \033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
chk() { if eval "$2" >/dev/null 2>&1; then ok "$1"; else bad "$1"; fi; }
# Zustand zuruecksetzen. Eine Recall-Datei aus der vorigen Pruefung wuerde sonst die
# Frage "letzte Konfiguration wiederholen?" voranstellen und alle Eingaben verschieben —
# der haeufigste Weg, sich eine solche Testreihe selbst kaputtzumachen.
chki() { rm -rf "$MENU_STATE_DIR"; chk "$1" "$2"; }

# Treibt ein Kommando durch ein Pseudoterminal. $1 = Kommando, $2 = Zeilen, mit | getrennt,
# $3 = TERM (Vorgabe "dumb").
#
# Zwei Sonderformen einer Zeile:
#   <EOF>       sendet Ctrl-D statt einer Zeile — nur so laesst sich pruefen, was
#               passiert, wenn die Eingabe MITTEN in einer Frage endet.
#   <RAW>...    sendet die Bytes OHNE abschliessendes Newline, mit \x..-Dekodierung.
#               Das ist der einzige Weg, Pfeiltasten zu pruefen: sie sind Escape-
#               Sequenzen (\x1b[B), keine Zeilen.
#
# TERM=dumb ist die Vorgabe, weil die Pfeiltasten-Auswahl genau daran erkennt, dass
# sie sich heraushalten soll. Alle Alt-Pruefungen laufen so weiter ueber die
# Zahleneingabe — und pruefen damit denselben Pfad wie vor der Zugabe.
tty_run() {
  python3 - "$1" "${2:-}" "${3:-dumb}" <<'PYEOF'
import os, pty, sys, select, time
cmd = sys.argv[1]
lines = sys.argv[2].split('|') if len(sys.argv) > 2 and sys.argv[2] else []
term = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else 'dumb'
pid, fd = pty.fork()
if pid == 0:
    os.environ['COLUMNS'] = '100'; os.environ['TERM'] = term
    os.execv('/bin/bash', ['bash', '-c', cmd])
out = b''; i = 0; t0 = time.time()
while time.time() - t0 < 45:
    r, _, _ = select.select([fd], [], [], 0.4)
    if r:
        try: d = os.read(fd, 65536)
        except OSError: break
        if not d: break
        out += d
    else:
        if i < len(lines):
            ln = lines[i]
            if ln == '<EOF>':
                os.write(fd, b'\x04')
            elif ln.startswith('<RAW>'):
                os.write(fd, ln[5:].encode('latin-1', 'backslashreplace')
                                   .decode('unicode_escape').encode('latin-1'))
            else:
                os.write(fd, (ln + '\n').encode())
            i += 1
        elif i == len(lines): break
try: os.close(fd)
except OSError: pass
sys.stdout.write(out.decode('utf-8', 'replace'))
PYEOF
}

echo "Pruefplan gefuehrte CLI-Menues (§8 des Plans)"
echo ""
echo "Nicht-interaktive Pfade — muessen sich verhalten wie vor der Umstellung:"
chk  "1   MENU=0 <skript> help zeigt die Hilfe unveraendert" \
     'MENU=0 $SIM help 2>&1 | grep -q "Aktionen:"'
chk  "2   <skript> eval < /dev/null fragt nicht und haengt nicht" \
     '! timeout 40 $SIM eval < /dev/null 2>&1 | grep -q "Anzahl Eval-Episoden"'
chki "3   SLURM_JOB_ID gesetzt -> kein Menue (Batch-Job)" \
     '! SLURM_JOB_ID=1 tty_run "$SIM eval" "" | grep -q "Anzahl Eval-Episoden"'
chki "3b  APPTAINER_NAME gesetzt -> kein Menue (KISSKI-Container)" \
     '! APPTAINER_NAME=x tty_run "$SIM eval" "" | grep -q "Anzahl Eval-Episoden"'
chki "3c  CI gesetzt -> kein Menue" \
     '! CI=1 tty_run "$SIM eval" "" | grep -q "Anzahl Eval-Episoden"'
chki "3d  MENU=0 unterdrueckt das Menue auch am Terminal" \
     '! MENU=0 tty_run "$SIM eval" "" | grep -q "Anzahl Eval-Episoden"'
chk  "4   Pipe ohne TTY -> Hilfe wie frueher" \
     'printf "4\n\n" | timeout 30 $SIM 2>&1 | grep -q "Aktionen:"'
chk  "5   unbekannte Aktion -> Fehlermeldung" \
     'MENU=0 $SIM quatsch 2>&1 | grep -q "Unbekannte Aktion"'
echo ""
echo "Interaktive Pfade (echtes Pseudoterminal):"
chki "6   ohne Argument erscheint die Aktionsliste" \
     'tty_run "$SIM" "a" | grep -q "Was moechtest du tun"'
# Die Kettenreihenfolge ist der Zweck von _order.spec. Sie muss auf BEIDEN Ebenen
# gelten — sonst haette die Schachtelung genau das verschenkt, wofuer die Datei da ist.
chki "7   Kettenreihenfolge auf der Gruppenebene: Vorbereiten vor Messen" \
     '[ "$(tty_run "$SIM" "a" | grep -n ") Vorbereiten" | cut -d: -f1)" -lt "$(tty_run "$SIM" "a" | grep -n ") Messen" | cut -d: -f1)" ]'
chki "7b  ... und in der flachen Liste: preflight vor eval" \
     '[ "$(MENU_NEST=0 tty_run "$SIM" "a" | grep -n ") preflight" | cut -d: -f1)" -lt "$(MENU_NEST=0 tty_run "$SIM" "a" | grep -n ") eval" | cut -d: -f1)" ]'
chki "8   ?<nr> erklaert eine Aktion (flache Liste)" \
     'MENU_NEST=0 tty_run "$SIM" "?9|a" | grep -qi "diagnosekette"'
chki "8b  ?<nr> erklaert eine Gruppe samt Voraussetzungen" \
     'tty_run "$SIM" "?3|a" | grep -q "braucht vorher cams"'
chki "9   eval fragt hoechstens 5 Werte, nicht alle 21" \
     '[ "$(tty_run "$SIM eval" "t|||||b" | grep -cE "^  .+\(([A-Z_]+)\)")" -le 5 ]'
# Die Tastenfolgen zaehlen die Grundfragen von eval ab: HF_TOKEN, CHECKPOINT_PATH,
# NUM_EPISODES, EPISODE_LENGTH_S, DR_ENABLED, dann die Zusammenfassung. Eine Leerzeile
# zu viel landet dort als [Enter] und STARTET den Lauf — bei 10/11/11b war das lange so
# und fiel nur nicht auf, weil auf dieser Maschine kein Container hochkommt.
chki "10  ? bei einer Frage zeigt den Langtext" \
     'tty_run "$SIM eval" "t||?|20|||b" | grep -qi "rauchtest"'
chki "11  ungueltige Eingabe fragt erneut statt abzustuerzen" \
     'tty_run "$SIM eval" "t||abc|20|||b" | grep -q "ganze Zahl erwartet"'
chki "11b Wert ausserhalb des Bereichs wird abgewiesen" \
     'tty_run "$SIM eval" "t||9999|20|||b" | grep -q "ausserhalb von 1 bis 200"'
chki "12  [b] zeigt nur den Befehl und startet nichts" \
     'tty_run "$SIM eval" "t|||||b" | grep -q "Nicht gestartet"'
chki "13  [e] blendet die erweiterten Optionen ein" \
     'tty_run "$SIM eval" "t|||||e|b" | grep -q "GROOT_INFERENCE_BACKEND"'
chki "14  EOF mitten in einer Frage bricht sauber ab" \
     'tty_run "$SIM eval" "t|<EOF>" | grep -q "Eingabe abgebrochen"'
chki "14b Auswahl ueber Gruppe + Aktion UND anschliessendes Fragen" \
     'tty_run "$SIM" "3|3|hf_abc||5|120||b" | grep -q "NUM_EPISODES=5"'
chki "14c dasselbe ueber die flache Liste" \
     'MENU_NEST=0 tty_run "$SIM" "9|hf_abc||5|120||b" | grep -q "NUM_EPISODES=5"'
chki "15  Positionsargument (optimize <phase>) wird erfragt" \
     'tty_run "$SIM optimize" "t|build|b" | grep -q "optimize build"'
echo ""
echo "Schachtelung — Gruppenebene bei langen Listen, flach bei kurzen:"
chki "15a sim wird geschachtelt (19 Aktionen in 7 Gruppen)" \
     'tty_run "$SIM" "a" | grep -q "alle 19 Aktionen"'
chki "15b KISSKI bleibt flach (nur 4 Aktionen)" \
     '! tty_run "./Training/kisski_menu.sh --dry-run" "a" | grep -q "alle 4 Aktionen"'
chki "15c MENU_NEST=1 erzwingt die Gruppenebene auch dort" \
     'MENU_NEST=1 tty_run "./Training/kisski_menu.sh --dry-run" "a" | grep -q "alle 4 Aktionen"'
chki "15d MENU_NEST=0 erzwingt die flache Liste" \
     'MENU_NEST=0 tty_run "$SIM" "a" | grep -qE "\) preflight"'
chki "15e [*] schaltet aus der Gruppenebene auf die Gesamtliste" \
     'tty_run "$SIM" "*|a" | grep -qE "\) preflight"'
chki "15f [z] fuehrt aus dem Untermenue zurueck zur Gruppenebene" \
     '[ "$(tty_run "$SIM" "3|z|a" | grep -c "Was moechtest du tun")" -ge 2 ]'
chki "15g Untermenue nennt die Domaene und die Gruppe im Titel" \
     'tty_run "./run.sh" "1|3|a" | grep -q "Simulation · Messen"'
echo ""
echo "Rueckweg — jede Ebene fuehrt eine hoeher, bis zurueck ins Hauptmenue:"
chki "15h [z] auf der Gruppenebene fuehrt zurueck ins Hauptmenue" \
     '[ "$(tty_run "./run.sh" "1|z|a" | grep -c "Womit moechtest du arbeiten")" -ge 2 ]'
chki "15i drei Ebenen zurueck: Untermenue -> Gruppen -> Hauptmenue" \
     'o=$(tty_run "./run.sh" "1|3|z|z|a"); [ "$(echo "$o" | grep -c "Simulation · Messen")" -ge 1 ] && [ "$(echo "$o" | grep -c "Womit moechtest du arbeiten")" -ge 2 ]'
chki "15j auch die flache Liste kennt den Weg zurueck" \
     '[ "$(MENU_NEST=0 tty_run "./run.sh" "1|z|a" | grep -c "Womit moechtest du arbeiten")" -ge 2 ]'
chki "15k auch aus einer als Argument gewaehlten Domaene" \
     'tty_run "./run.sh sim" "z|a" | grep -q "Womit moechtest du arbeiten"'
chki "15l nach dem Rueckweg gilt eine andere Domaene: KISSKI ist waehlbar" \
     'tty_run "./run.sh" "1|z|3|a" | grep -q "kein sbatch"'
# Ohne run.sh gibt es keine Ebene darueber. Ein [←], das dort etwas taete, wuerde den
# Launcher kommentarlos beenden — deshalb muss die Taste dort tot sein.
chki "15m direkter Launcher-Aufruf bietet KEIN Hauptmenue an" \
     '! tty_run "$SIM" "a" | grep -q "Hauptmenue"'
chki "15n ... und [z] beendet ihn dort nicht versehentlich" \
     'tty_run "$SIM" "z|3|3|hf_x||2|10||b" | grep -q "NUM_EPISODES=2"'
echo ""
echo "Dauerhaft nicht lauffaehige Aktionen (KISSKI kann die Sim nicht):"
KM="./Training/kisski_menu.sh --dry-run"
chki "15o gesperrte Aktion nennt den Grund statt des Zustandsmarkers" \
     'tty_run "$KM" "a" | grep -E "\) sim" | grep -q "RTX 5000 zu alt"'
chki "15p ... und laesst sich nicht auswaehlen" \
     'o=$(tty_run "$KM" "3|a"); echo "$o" | grep -q "RTX 5000 zu alt" && ! echo "$o" | grep -q "Einreichen als"'
chki "15q ? erklaert, WARUM sie gesperrt ist" \
     'tty_run "$KM" "?4|a" | grep -q "nicht lauffaehig"'
chki "15r die lauffaehigen Aktionen bleiben waehlbar" \
     'tty_run "$KM" "2|||" | grep -q "kisski_open_loop_eval.sh"'
chki "15s gesperrte Gruppe steht zuletzt, nicht vor den lauffaehigen" \
     'o=$(tty_run "$KM" "a"); [ "$(echo "$o" | grep -n ") train" | cut -d: -f1)" -lt "$(echo "$o" | grep -n ") sim" | cut -d: -f1)" ]'
echo ""
echo "Vorrang und Geheimnisse:"
chki "16  gesetzte Variablen werden als vorgegeben angezeigt" \
     'HF_TOKEN=hf_x NUM_EPISODES=3 tty_run "$SIM eval" "|||b" | grep -q "vorgegeben"'
chki "17  gesetztes NUM_EPISODES wird nicht erneut gefragt" \
     '[ "$(HF_TOKEN=hf_x NUM_EPISODES=3 tty_run "$SIM eval" "|||b" | grep -c "Anzahl Eval-Episoden")" -le 1 ]'
chki "18  Token erscheint maskiert" \
     'tty_run "$SIM eval" "hf_GEHEIM123|||||b" | grep -q "hf_…"'
chki "18b Token erscheint NICHT im Klartext" \
     '! tty_run "$SIM eval" "hf_GEHEIM123|||||b" | grep -q "GEHEIM123"'
rm -rf "$MENU_STATE_DIR" "$TMP/data/logs"
tty_run "$SIM eval" "hf_GEHEIM123||2|10|||" >/dev/null 2>&1
chk  "19  Token steht in keiner Log-Datei" \
     '! grep -rq "hf_GEHEIM123" "$TMP/data/logs" 2>/dev/null'
chk  "20  Token steht in keiner Recall-Datei" \
     '! grep -rq "hf_GEHEIM123" "$MENU_STATE_DIR" 2>/dev/null'
chk  "21  Recall-Datei existiert und ist mode 600" \
     '[ "$(stat -c %a "$MENU_STATE_DIR/last/sim.eval.env")" = 600 ]'
echo ""
echo "Einstiegspunkt ./run.sh — waehlt nur die Domaene und uebergibt per exec:"
chk  "26  --help nennt alle Domaenen aus _domains.spec" \
     './run.sh --help 2>&1 | grep -q "Simulation/server_rl_run.sh" && ./run.sh --help 2>&1 | grep -q "kisski_menu.sh"'
chk  "27  unbekannte Domaene -> Fehler statt Menue" \
     './run.sh quatsch < /dev/null 2>&1 | grep -q "Unbekannte Domaene"'
chk  "28  ohne Domaene und ohne TTY -> klare Meldung, kein Haengen" \
     '! timeout 20 ./run.sh < /dev/null 2>&1 | grep -q "Womit moechtest du arbeiten"'
chki "29  mit TTY erscheint die Domaenenliste" \
     'tty_run "./run.sh" "a" | grep -q "Womit moechtest du arbeiten"'
chki "30  Aktionszahl wird aus den Specs abgeleitet, nicht gepflegt" \
     'tty_run "./run.sh" "a" | grep -E "1\) sim" | grep -qE "[0-9]+ Aktionen"'
# sbatch fehlt auf jedem Rechner ausser dem Login-Node — der KISSKI-Eintrag muss dort
# sichtbar bleiben und gesperrt sein, nicht verschwinden.
# Der Cluster kann die Simulation nicht (keine RT-Cores), also ist KISSKI dort ein
# TRAININGS-Weg und steht unter derselben Ueberschrift wie das lokale Training.
chki "30b Domaenenliste ist gruppiert: KISSKI steht unter \"Training\"" \
     'o=$(tty_run "./run.sh" "a"); [ "$(echo "$o" | grep -n "Training" | head -1 | cut -d: -f1)" -lt "$(echo "$o" | grep -n ") kisski" | cut -d: -f1)" ]'
mkdir -p "$TMP/bin"; printf '#!/bin/sh\nexit 0\n' > "$TMP/bin/sbatch"; chmod +x "$TMP/bin/sbatch"
chki "30c mit sbatch zaehlt KISSKI nur die LAUFFAEHIGEN Aktionen (2 statt 4)" \
     'PATH="$TMP/bin:$PATH" tty_run "./run.sh" "a" | grep -E "\) kisski" | grep -q "2 Aktionen"'
chki "31  fehlendes Kommando sperrt die Domaene sichtbar (mit Grund)" \
     'if command -v sbatch >/dev/null 2>&1; then true; else tty_run "./run.sh" "a" | grep -E "\) kisski" | grep -q "kein sbatch"; fi'
chki "32  gesperrte Domaene laesst sich nicht waehlen" \
     'if command -v sbatch >/dev/null 2>&1; then true; else tty_run "./run.sh" "3|a" | grep -q "kein sbatch"; fi'
chki "33  Domaenenwahl fuehrt in die Aktionsliste des Launchers" \
     'tty_run "./run.sh" "1|a" | grep -q "Was moechtest du tun"'
chk  "34  ./run.sh sim <aktion> reicht unveraendert durch" \
     'MENU=0 ./run.sh sim help < /dev/null 2>&1 | grep -q "Aktionen:"'
echo ""
echo "Pfeiltasten — Zugabe, die sich bei TERM=dumb heraushaelt:"
chki "35  TERM=dumb -> Zahleneingabe wie bisher, keine Pfeiltasten-Zeile" \
     'tty_run "./run.sh" "a" dumb | grep -q "erklaert einen Eintrag" && ! tty_run "./run.sh" "a" dumb | grep -q "waehlen"'
chki "36  TERM=xterm -> Pfeiltasten-Fusszeile erscheint" \
     'tty_run "./run.sh" "<RAW>a" xterm | grep -q "\[Enter\] bestaetigen"'
chki "37  Pfeil ab bewegt die Hervorhebung auf den zweiten Eintrag" \
     'tty_run "./run.sh" "<RAW>\x1b[B|<RAW>a" xterm | grep -qE "\[7m.*2\) train"'
chki "38  MENU_ARROWS=0 schaltet sie auch am Terminal ab" \
     'MENU_ARROWS=0 tty_run "./run.sh" "a" xterm | grep -q "erklaert einen Eintrag"'
# Die eigentliche Auswahl gegen eine EIGENE Domaenenliste pruefen, deren Launcher
# folgenlos sind. Sonst wuerde der Test den Trainings-Launcher wirklich starten —
# und ein Pruefplan, der nebenbei einen Container anfasst, ist keiner.
mkdir -p "$TMP/spec"
cat > "$TMP/spec/_domains.spec" <<'SPECEOF'
# TL;DR: Nur fuer tools/test_menu.sh — zwei folgenlose Domaenen zum Pruefen der Auswahl.
domain alpha "Erste"  --rank 10 --launcher "tools/check_tldr.sh"
domain beta  "Zweite" --rank 20 --launcher "tools/gen_docs.sh"
SPECEOF
# Geprueft wird die Zeile, die run.sh vor dem exec ausgibt. Sie NICHT mit "==> " davor
# greppen: zwischen Pfeil und Pfad steht ein Farbcode. Stattdessen beide Richtungen
# pruefen — der gewaehlte Launcher muss auftauchen und der andere fehlen, sonst besteht
# der Test auch dann, wenn die Auswahl gar nicht bewegt wurde.
chki "39  Enter ohne Bewegung waehlt den ersten Eintrag" \
     'o=$(MENU_SPEC_DIR=$TMP/spec tty_run "./run.sh" "<RAW>\r" xterm); echo "$o" | grep -q "tools/check_tldr.sh" && ! echo "$o" | grep -q "tools/gen_docs.sh"'
chki "40  Pfeil ab + Enter waehlt tatsaechlich den zweiten Eintrag" \
     'o=$(MENU_SPEC_DIR=$TMP/spec tty_run "./run.sh" "<RAW>\x1b[B|<RAW>\r" xterm); echo "$o" | grep -q "tools/gen_docs.sh" && ! echo "$o" | grep -q "tools/check_tldr.sh"'
chki "41  Zifferneingabe waehlt auch im Pfeiltasten-Modus" \
     'o=$(MENU_SPEC_DIR=$TMP/spec tty_run "./run.sh" "<RAW>2|<RAW>\r" xterm); echo "$o" | grep -q "tools/gen_docs.sh" && ! echo "$o" | grep -q "tools/check_tldr.sh"'
echo ""
echo "Gewichte — WELCHER Checkpoint gemessen wird, muss im Menue stehen:"
# Bis 2026-08-21 nannte nur sim-setup.spec den Checkpoint. Jede andere Aktion ruft
# ensure_checkpoint aber selbst auf und hat den Default im Zweifel stillschweigend
# geladen — eine Erfolgsrate ohne die Angabe, welche Gewichte sie gemessen hat, ist
# nicht vergleichbar. Deshalb ist der Pfad bei 'eval' eine Grundfrage.
chki "42  eval fragt den Checkpoint-Pfad im Grunddialog, nicht erst hinter [e]" \
     'tty_run "$SIM eval" "t|||||b" | grep -q "(CHECKPOINT_PATH)"'
chki "43  ... und die Antwort landet in der Befehlszeile" \
     'tty_run "$SIM eval" "t|/data/checkpoints/lauf3||||b" | grep -q "CHECKPOINT_PATH=/data/checkpoints/lauf3"'
# Das Repo ist der schwaechere Hebel (ensure_checkpoint prueft nur, ob der PFAD schon
# existiert), gehoert aber erreichbar zu sein — hinter [e] reicht.
chki "44  HF_CHECKPOINT_REPO ist bei eval unter [e] erreichbar" \
     'tty_run "$SIM eval" "t|||||e|b" | grep -q "HF_CHECKPOINT_REPO"'
# Gegenprobe: Aktionen ohne Gewichte duerfen davon nichts sehen — sonst waere der
# gemeinsame Block aus _common-sim.spec eine Verschlechterung fuer neun Aktionen.
chki "45  'view' fragt weder Pfad noch Repo — auch nicht unter [e]" \
     '! tty_run "$SIM view" "||e|b" | grep -qE "\(CHECKPOINT_PATH\)|\(HF_CHECKPOINT_REPO\)"'
mkdir -p "$TMP/data/checkpoints/lauf3"
chki "46  der Zustandsmarker nennt den vorliegenden Checkpoint beim Namen" \
     'CHECKPOINT_PATH=/data/checkpoints/lauf3 MENU_NEST=0 tty_run "$SIM" "a" | grep -E "\) eval" | grep -q "lauf3"'
rmdir "$TMP/data/checkpoints/lauf3" 2>/dev/null

# Vorschlagsliste: der Pfad in der Frage gilt im Container, die Verzeichnisse liegen auf
# dem Host — ohne Uebersetzung tippt man den Namen blind ab. Aufbau der Probe: zwei
# brauchbare Checkpoints und ein angeschnittener Download, der NEUER ist als beide.
mkdir -p "$TMP/data/checkpoints/alt-3000" "$TMP/data/checkpoints/neu-30000" \
         "$TMP/data/checkpoints/abgebrochen"
touch "$TMP/data/checkpoints/alt-3000/model.safetensors" \
      "$TMP/data/checkpoints/neu-30000/model.safetensors" \
      "$TMP/data/checkpoints/abgebrochen/config.json"
touch "$TMP/data/checkpoints/neu-30000"    # juengste Aenderungszeit -> soll gewinnen
chki "47  der Vorschlag setzt den neuesten vorhandenen Checkpoint als Vorgabe" \
     'tty_run "$SIM eval" "t|||||b" | grep -q "\[/data/checkpoints/neu-30000\]"'
chki "48  ... und nennt die uebrigen, damit ein Vergleichslauf abschreibbar ist" \
     'tty_run "$SIM eval" "t|||||b" | grep -q "alt-3000"'
# Der angeschnittene Ordner ist der juengste — er darf trotzdem NICHT die Vorgabe werden.
# Genau diese Verwechslung liess `check` am 2026-08-21 erst im Isaac-Sim-Aufbau sterben.
chki "49  ein angeschnittener Download wird gemeldet, aber nicht vorgeschlagen" \
     'o=$(tty_run "$SIM eval" "t|||||b"); echo "$o" | grep -q "UNVOLLSTAENDIG" \
      && ! echo "$o" | grep -q "\[/data/checkpoints/abgebrochen\]"'
rm -rf "$TMP/data/checkpoints"
chki "50  ohne jeden Checkpoint bleibt der statische Default stehen" \
     'tty_run "$SIM eval" "t|||||b" | grep -q "\[/data/checkpoints/groot-g1dex3-checkpoint\]"'
# Die Gegenprobe zur erweiterten Vorschlagsregel: ein Vorschlag darf einen Wert, den der
# Aufrufer selbst mitgegeben hat, unter keinen Umstaenden ueberschreiben. Der Fixture-Ordner
# ist dabei absichtlich vorhanden — sonst prueft der Test nur, dass gar nichts vorlag.
mkdir -p "$TMP/data/checkpoints/neu-30000"
touch "$TMP/data/checkpoints/neu-30000/model.safetensors"
chki "51  ein vom Aufrufer gesetzter Pfad schlaegt den Vorschlag" \
     'CHECKPOINT_PATH=/data/checkpoints/meiner tty_run "$SIM eval" "t|||||b" \
      | grep -q "CHECKPOINT_PATH */data/checkpoints/meiner"'
rm -rf "$TMP/data/checkpoints"

echo ""
echo "Struktur und Abgleich:"
chk  "22  bash -n ueber Bibliotheken, Werkzeug und alle Specs" \
     'for f in tools/lib_menu.sh tools/lib_env_local.sh tools/gen_docs.sh tools/test_menu.sh tools/menu/*.spec; do bash -n "$f" || exit 1; done'
chk  "23  jede Aktion von server_rl_run.sh hat eine Spec" \
     'for a in $(sed -n "/^case .\\\$ACTION. in$/,/^esac$/p" $SIM | grep -oE "^  [a-z|]+\)" | tr -d " )" | tr "|" "\n" | grep -vE "^(help|down)$"); do [ -f "tools/menu/sim-$a.spec" ] || exit 1; done'
chk  "24  Trainings-Launcher reicht die Feature-Schalter durch" \
     'MENU=0 TUNE_VISUAL=1 HF_TOKEN=hf_x ./Training/setup_and_train_DockerHub-pull.sh --dry-run --skip-pull < /dev/null 2>&1 | grep -q -- "-e TUNE_VISUAL=1"'
chk  "25  Abgleich Spec <-> Skript <-> Doku (Pruefung 9 des Plans)" \
     './tools/gen_docs.sh --quiet'
echo ""
printf 'Ergebnis: %d bestanden, %d fehlgeschlagen\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
