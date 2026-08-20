#!/usr/bin/env bash
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

# Treibt ein Kommando durch ein Pseudoterminal. $1 = Kommando, $2 = Zeilen, mit | getrennt.
# Die Sonderzeile <EOF> sendet Ctrl-D statt einer Zeile — nur so laesst sich pruefen, was
# passiert, wenn die Eingabe MITTEN in einer Frage endet.
tty_run() {
  python3 - "$1" "${2:-}" <<'PYEOF'
import os, pty, sys, select, time
cmd = sys.argv[1]
lines = sys.argv[2].split('|') if len(sys.argv) > 2 and sys.argv[2] else []
pid, fd = pty.fork()
if pid == 0:
    os.environ['COLUMNS'] = '100'; os.environ['TERM'] = 'dumb'
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
            os.write(fd, b'\x04' if ln == '<EOF>' else (ln + '\n').encode())
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
chki "7   Liste in Kettenreihenfolge: preflight vor eval" \
     '[ "$(tty_run "$SIM" "a" | grep -n ") preflight" | cut -d: -f1)" -lt "$(tty_run "$SIM" "a" | grep -n ") eval" | cut -d: -f1)" ]'
chki "8   ?<nr> erklaert eine Aktion" \
     'tty_run "$SIM" "?9|a" | grep -qi "diagnosekette"'
chki "9   eval fragt hoechstens 5 Werte, nicht alle 15" \
     '[ "$(tty_run "$SIM eval" "t||||b" | grep -cE "^  .+\(([A-Z_]+)\)")" -le 5 ]'
chki "10  ? bei einer Frage zeigt den Langtext" \
     'tty_run "$SIM eval" "t|?|20||||b" | grep -qi "rauchtest"'
chki "11  ungueltige Eingabe fragt erneut statt abzustuerzen" \
     'tty_run "$SIM eval" "t|abc|20||||b" | grep -q "ganze Zahl erwartet"'
chki "11b Wert ausserhalb des Bereichs wird abgewiesen" \
     'tty_run "$SIM eval" "t|9999|20||||b" | grep -q "ausserhalb von 1 bis 200"'
chki "12  [b] zeigt nur den Befehl und startet nichts" \
     'tty_run "$SIM eval" "t||||b" | grep -q "Nicht gestartet"'
chki "13  [e] blendet die erweiterten Optionen ein" \
     'tty_run "$SIM eval" "t||||e|b" | grep -q "GROOT_INFERENCE_BACKEND"'
chki "14  EOF mitten in einer Frage bricht sauber ab" \
     'tty_run "$SIM eval" "t|<EOF>" | grep -q "Eingabe abgebrochen"'
chki "14b Auswahl aus der Liste UND anschliessendes Fragen" \
     'tty_run "$SIM" "9|hf_abc|5|120||b" | grep -q "NUM_EPISODES=5"'
chki "15  Positionsargument (optimize <phase>) wird erfragt" \
     'tty_run "$SIM optimize" "t|build|b" | grep -q "optimize build"'
echo ""
echo "Vorrang und Geheimnisse:"
chki "16  gesetzte Variablen werden als vorgegeben angezeigt" \
     'HF_TOKEN=hf_x NUM_EPISODES=3 tty_run "$SIM eval" "|||b" | grep -q "vorgegeben"'
chki "17  gesetztes NUM_EPISODES wird nicht erneut gefragt" \
     '[ "$(HF_TOKEN=hf_x NUM_EPISODES=3 tty_run "$SIM eval" "|||b" | grep -c "Anzahl Eval-Episoden")" -le 1 ]'
chki "18  Token erscheint maskiert" \
     'tty_run "$SIM eval" "hf_GEHEIM123||||b" | grep -q "hf_…"'
chki "18b Token erscheint NICHT im Klartext" \
     '! tty_run "$SIM eval" "hf_GEHEIM123||||b" | grep -q "GEHEIM123"'
rm -rf "$MENU_STATE_DIR" "$TMP/data/logs"
tty_run "$SIM eval" "hf_GEHEIM123|2|10|||" >/dev/null 2>&1
chk  "19  Token steht in keiner Log-Datei" \
     '! grep -rq "hf_GEHEIM123" "$TMP/data/logs" 2>/dev/null'
chk  "20  Token steht in keiner Recall-Datei" \
     '! grep -rq "hf_GEHEIM123" "$MENU_STATE_DIR" 2>/dev/null'
chk  "21  Recall-Datei existiert und ist mode 600" \
     '[ "$(stat -c %a "$MENU_STATE_DIR/last/sim.eval.env")" = 600 ]'
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
