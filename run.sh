#!/usr/bin/env bash
# TL;DR: Der eine Einstiegspunkt — fragt Simulation/Training/KISSKI ab und uebergibt an den passenden Launcher.
#   Ohne Argument fuehrt es durch; ./run.sh sim eval reicht direkt durch. Doku: docs/weiterfuehrend/cli-menuefuehrung.md
#
# run.sh — die Ebene ueber den Host-Launchern.
#
# GRUNDIDEE: Dieses Skript WAEHLT NUR und uebergibt dann an den unveraenderten
# Launcher. Es baut keinen `docker run`-Aufruf zusammen, setzt keine Trainingsparameter
# und kennt keine Aktionsliste — das alles passiert danach, im Zielskript, mit dessen
# eigenem Menue. Damit entsteht kein zweiter Wartungspfad; es ist dieselbe
# Leitentscheidung wie beim Menue selbst (Plan §2.1).
#
# UEBERGABE, zwei Wege:
#   * ohne Menue (MENU=0, kein Terminal, Aktion als Argument):  `exec` — dieser Prozess
#     wird ersetzt, Rueckgabewert und Signale gehen unveraendert durch.
#   * mit Menue:  der Launcher laeuft als KIND. Nur so gibt es einen Rueckweg: seine
#     oberste Aktionsliste bietet dann [←] an und endet mit MENU_RC_BACK (97), worauf
#     hier wieder das Hauptmenue erscheint.
#
# Praktische Folge: alles, was die Launcher koennen, gilt hier unveraendert weiter —
# MENU=0, --profile=<name>, jede Env-Var, jedes Argument. run.sh reicht sie durch.
#
# NUTZUNG:
#   ./run.sh                    # fragt: Simulation / Training / KISSKI, dann weiter
#   ./run.sh sim                # direkt in die Simulation (deren Aktionsliste)
#   ./run.sh sim eval           # ganz durch bis zu den Parametern von 'eval'
#   ./run.sh train              # lokales Training
#   ./run.sh kisski train       # Job auf dem Cluster einreichen
#   MENU=0 ./run.sh sim eval    # ohne jede Rueckfrage, wie der direkte Aufruf
#
# Im Menue fuehrt [←] (bzw. [z]) immer eine Ebene zurueck — aus einer Aktionsgruppe in
# die Gruppenuebersicht, von dort ins Hauptmenue. Auch aus einer per Argument gewaehlten
# Domaene ("./run.sh sim"), damit der Rueckweg nicht davon abhaengt, wie man hereinkam.
#
# Die Zuordnung Domaene -> Launcher steht in tools/menu/_domains.spec, nicht hier.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MENU_SPEC_DIR="${MENU_SPEC_DIR:-$REPO_DIR/tools/menu}"   # gleicher Name wie in server_rl_run.sh
SPEC_DIR="$MENU_SPEC_DIR"

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m ! \033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; }

# shellcheck source=tools/lib_env_local.sh
source "$REPO_DIR/tools/lib_env_local.sh"; env_local_load "$REPO_DIR"
# shellcheck source=tools/lib_menu.sh
source "$REPO_DIR/tools/lib_menu.sh"

usage() {
  awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print; next } exit }' "$0"
  echo "Domaenen:"
  menu_load_domains "$SPEC_DIR" >/dev/null 2>&1 || true
  local d
  for d in $(menu_domain_list); do
    printf '  %-10s %-18s %-40s -> %s\n' "$d" "$(menu_domain_label "$d")" \
      "${_MENU_D_TITLE[$d]}" "$(menu_domain_launcher "$d")"
  done
}

DOMAIN=""
case "${1:-}" in
  -h|--help|help) usage; exit 0 ;;
  --no-menu) MENU=0; export MENU; shift ;;
  --menu)    MENU=1; export MENU; shift ;;
esac
[[ $# -gt 0 ]] && { DOMAIN="$1"; shift; }

menu_load_domains "$SPEC_DIR" || { err "tools/menu/_domains.spec fehlt oder ist leer."; exit 1; }

# Eine Domaene, die es nicht gibt, ist ein Tippfehler — nicht der Anlass, das Menue
# aufzuklappen. Sonst verschluckt ein falsch geschriebenes Argument die Absicht.
if [[ -n "$DOMAIN" ]] && ! menu_domain_known "$DOMAIN"; then
  err "Unbekannte Domaene: '$DOMAIN'"
  err "  Verfuegbar: $(menu_domain_list | tr '\n' ' ')"
  exit 2
fi

# Standortbestimmung ueber der Liste. Absichtlich nur `command -v` — kein `docker info`,
# kein `nvidia-smi`: der Startbildschirm darf nie auf einen Daemon warten, der gerade
# nicht antwortet.
show_context() {
  local c ctx=""
  for c in docker sbatch apptainer; do
    if command -v "$c" >/dev/null 2>&1; then ctx+="$c ✓  "; else ctx+="$c ✗  "; fi
  done
  printf '\n  \033[2m%s\033[0m\n' "auf $(hostname -s 2>/dev/null || echo '?'):  ${ctx% }" >&2
}

start_domain() {
  local domain="$1"; shift
  local launcher target
  launcher="$(menu_domain_launcher "$domain")"
  [[ -n "$launcher" ]] || { err "Domaene '$domain' hat keinen --launcher in _domains.spec."; exit 1; }
  target="$REPO_DIR/$launcher"
  [[ -f "$target" ]] || { err "Launcher fehlt: $launcher"; exit 1; }

  # Der Launcher setzt damit "Simulation — Was moechtest du tun?" statt nur "Was
  # moechtest du tun?" ueber seine Liste. Nach zwei Ebenen weiss man sonst nicht mehr
  # sicher, wo man gelandet ist. Beim direkten Aufruf ohne run.sh bleibt es leer.
  export MENU_DOMAIN_LABEL="$(menu_domain_label "$domain")"

  # Der aequivalente Aufruf, damit man beim naechsten Mal ohne run.sh auskommt — dieselbe
  # Rolle wie der Ein-Zeiler auf der Bestaetigungsseite des Menues.
  log "$launcher ${*:-}"

  if (( INTERACTIVE_PICK )); then
    # NICHT exec: der Launcher muss ein Kind bleiben, sonst gaebe es kein Zurueck —
    # exec ersetzt diesen Prozess und mit ihm das Hauptmenue. Dafuer darf die oberste
    # Aktionsliste jetzt [←] anbieten und mit MENU_RC_BACK hierher zurueckkehren.
    export MENU_TOPLEVEL_BACK=1
    local rc=0
    if [[ -x "$target" ]]; then "$target" "$@" || rc=$?; else bash "$target" "$@" || rc=$?; fi
    return "$rc"
  fi

  # Domaene kam als Argument — es gibt kein Hauptmenue, in das man zurueckkehren
  # koennte. Also der billigste Weg: diesen Prozess ersetzen.
  unset MENU_TOPLEVEL_BACK
  if [[ -x "$target" ]]; then exec "$target" "$@"; else exec bash "$target" "$@"; fi
}

# Ohne Menue gibt es kein Hauptmenue, in das man zurueckkehren koennte — dann ist der
# exec-Weg richtig und die Rueckgabe geht unveraendert durch.
INTERACTIVE_PICK=0
menu_enabled && INTERACTIVE_PICK=1

if [[ -z "$DOMAIN" ]] && (( ! INTERACTIVE_PICK )); then
  err "Keine Domaene angegeben und kein Terminal zum Fragen."
  err "  Verfuegbar: $(menu_domain_list | tr '\n' ' ')"
  err "  Beispiel:   MENU=0 $0 sim eval"
  exit 2
fi

# Als Argument uebergebene Domaene zuerst starten. Auch von dort fuehrt [←] ins
# Hauptmenue — sonst haette "./run.sh sim" kein Zurueck, "./run.sh" aber schon, und
# genau solche Asymmetrien merkt sich niemand.
if [[ -n "$DOMAIN" ]]; then
  RC=0; start_domain "$DOMAIN" "$@" || RC=$?
  (( RC == MENU_RC_BACK )) || exit "$RC"
  # Ab hier steht die Domaene NICHT mehr fest. Die restlichen Argumente gehoerten zu
  # ihr ("./run.sh sim eval") und waeren an einer anderen Domaene sinnlos bis falsch —
  # 'eval' ist keine Trainings-Aktion. Also fallen lassen.
  set --
  DOMAIN=""
fi

while :; do
  show_context
  DOMAIN="$(menu_pick_domain "$SPEC_DIR")" || { echo >&2; warn "Abgebrochen."; exit 0; }
  RC=0; start_domain "$DOMAIN" "$@" || RC=$?
  # Der Launcher meldet "der Nutzer will zurueck ins Hauptmenue" — kein Fehler.
  (( RC == MENU_RC_BACK )) || exit "$RC"
done
