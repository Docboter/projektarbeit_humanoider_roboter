#!/usr/bin/env bash
# lib_menu.sh — gefuehrte CLI-Menues fuer die Host-Launcher dieses Repos.
#
# Plan und Begruendung: docs/weiterfuehrend/cli-menuefuehrung.md
#
# GRUNDIDEE: Das Menue erzeugt NUR Umgebungsvariablen und laeuft VOR dem Dispatch.
# Danach laeuft der unveraenderte Pfad. Es gibt keinen zweiten Ausfuehrungsweg, keine
# eigenen `docker run`-Aufrufe, keinen "Menue-Modus" mit abweichendem Verhalten.
#
# HOST-WERKZEUG. Diese Datei liegt bewusst NICHT unter */scripts/ — jene Verzeichnisse
# werden per `COPY scripts/ /scripts/` in die Images gebacken, jede Aenderung braeuchte
# dort einen Image-Rebuild. tools/ ist nie im Image; die Container-Entrypoints bleiben
# nicht-interaktiv und damit autonom (vast.ai, KISSKI).
#
# HAUSREGELN, die nicht verhandelbar sind:
#   * Die gesamte Oberflaeche geht nach stderr. Nur das Ergebnis von menu_pick_action
#     geht nach stdout — sonst frisst die Kommandosubstitution die Anzeige.
#   * Alle internen Helfer heissen _menu_*. Die Launcher definieren log/ok/warn/err
#     jeweils UNTERSCHIEDLICH; diese Datei fasst sie nicht an.
#   * Jedes `read` bekommt eine echte EOF-Behandlung. Unter `set -euo pipefail` reisst
#     ein Rueckgabewert != 0 sonst das ganze Skript mit, und `./skript < /dev/null`
#     stirbt kommentarlos.
#   * Reines Bash. Kein whiptail/dialog/gum/fzf (auf den Zielrechnern nicht vorhanden),
#     kein Python (auf dem KISSKI-Login-Node erst nach `module load`), kein jq.

[[ -n "${_MENU_LIB_LOADED:-}" ]] && return 0
_MENU_LIB_LOADED=1

if (( BASH_VERSINFO[0] < 4 )); then
  printf 'lib_menu.sh braucht Bash >= 4.2 (gefunden: %s).\n' "${BASH_VERSION}" >&2
  return 1 2>/dev/null || exit 1
fi

# ── Darstellung ───────────────────────────────────────────────────────────────
# Rahmenzeichen bewusst auf ─ │ ✓ ! ← beschraenkt (siehe Plan §6): das Repo schreibt in
# Skripten sonst ASCII-Umschreibungen, die Menue-Oberflaeche darf UTF-8 nutzen.
if [[ -t 2 ]]; then
  _MENU_C_HEAD=$'\033[1;36m'; _MENU_C_DIM=$'\033[2m';  _MENU_C_OK=$'\033[1;32m'
  _MENU_C_WARN=$'\033[1;33m'; _MENU_C_KEY=$'\033[1m';  _MENU_C_OFF=$'\033[0m'
else
  _MENU_C_HEAD=''; _MENU_C_DIM=''; _MENU_C_OK=''; _MENU_C_WARN=''; _MENU_C_KEY=''; _MENU_C_OFF=''
fi

# Auf <breite> ZEICHEN auffuellen. printf rechnet in Bytes — bei UTF-8 (Umlaute,
# Gedankenstrich, das … der Maskierung) verrutschen damit alle Spalten dahinter.
_menu_pad() {
  local t="$1" w="$2"
  local n=${#t}
  (( n >= w )) && { printf '%s' "$t"; return; }
  printf '%s%*s' "$t" $(( w - n )) ''
}

_menu_cols() { echo "${COLUMNS:-$(tput cols 2>/dev/null || echo 80)}"; }
_menu_out()  { printf '%s\n' "$*" >&2; }
_menu_rule() {
  local w; w=$(_menu_cols); (( w > 76 )) && w=76
  printf '%s%s%s\n' "$_MENU_C_DIM" "$(printf '─%.0s' $(seq 1 "$w"))" "$_MENU_C_OFF" >&2
}

# Bricht <text> auf die Terminalbreite um und rueckt jede Zeile mit <indent> ein.
_menu_wrap() {
  local indent="$1"; shift
  local w; w=$(_menu_cols); (( w > 78 )) && w=78
  local width=$(( w - ${#indent} )); (( width < 30 )) && width=30
  printf '%s\n' "$*" | fold -s -w "$width" | sed "s/^/$indent/" >&2
}

# ── read mit echter EOF-Behandlung ────────────────────────────────────────────
# Rueckgabewert 0 = Eingabe (ggf. leer) in _MENU_REPLY.  1 = EOF / kein Terminal mehr.
_MENU_REPLY=""
_menu_read() {
  local prompt="$1" silent="${2:-0}"
  _MENU_REPLY=""
  printf '%s' "$prompt" >&2
  local rc=0
  if [[ "$silent" == 1 ]]; then
    IFS= read -rs _MENU_REPLY || rc=$?
    printf '\n' >&2
  else
    IFS= read -r _MENU_REPLY || rc=$?
  fi
  if (( rc != 0 )); then
    _MENU_REPLY=""
    printf '\n' >&2
    return 1
  fi
  return 0
}

# Gemeinsamer Abbruch bei EOF — eine klare Meldung statt eines stummen Todes.
_menu_eof_abort() {
  _menu_out ""
  _menu_out "${_MENU_C_WARN} ! ${_MENU_C_OFF}Eingabe abgebrochen (EOF/kein Terminal)."
  _menu_out "   Nicht-interaktiv aufrufen:  MENU=0 <skript> <aktion>   (oder Werte per Env setzen)"
  return 1
}

# ── Darf ueberhaupt gefragt werden? (Plan §2.2) ───────────────────────────────
# Mehrfach abgesichert. Jede einzelne Bedingung reicht aus, um zu schweigen — ein
# Automatisierungspfad, an den niemand gedacht hat, faellt so eher auf die stille Seite.
menu_enabled() {
  [[ "${MENU:-auto}" != 0 ]] || return 1          # harter Ausschalter (MENU=0 / --no-menu)
  [[ "${MENU:-auto}" != 1 ]] || return 0          # harter Einschalter (MENU=1 / --menu)
  [[ -t 0 && -t 2 ]]         || return 1          # kein Terminal -> nie fragen
  [[ -z "${SLURM_JOB_ID:-}"     ]] || return 1    # SLURM-Batch
  [[ -z "${SLURM_JOBID:-}"      ]] || return 1
  [[ -z "${CI:-}"               ]] || return 1
  [[ -z "${GITHUB_ACTIONS:-}"   ]] || return 1
  [[ ! -f /.dockerenv           ]] || return 1    # im Container niemals
  # Apptainer/Singularity: auf KISSKI laeuft derselbe Code ohne /.dockerenv. Der Rest des
  # Repos prueft beides (run_finetuning.sh), das Menue muss es auch.
  [[ -z "${APPTAINER_NAME:-}"   ]] || return 1
  [[ -z "${SINGULARITY_NAME:-}" ]] || return 1
  [[ -z "${APPTAINER_CONTAINER:-}" ]] || return 1
  return 0
}

# ── Spezifikations-DSL ────────────────────────────────────────────────────────
# Eine .spec-Datei ist reines Bash: eine Folge von Aufrufen der Funktionen unten.
# Kein Parser noetig, und Abhaengigkeiten zwischen Fragen (`when`) fallen direkt heraus.

declare -A _MENU_A_TITLE _MENU_A_GROUP _MENU_A_HINT _MENU_A_NEEDS _MENU_A_STATE _MENU_A_FILE
declare -A _MENU_A_RANK _MENU_A_CLI
declare -a _MENU_A_ORDER=()

# Die Specs werden alphabetisch eingelesen (sim-cams vor sim-preflight). Ohne eine
# ausdrueckliche Ordnung stuende die Liste damit in Dateinamen-Reihenfolge — genau der
# Nutzen der Aktionsliste ginge verloren, denn sie soll die KETTE abbilden:
# preflight -> setup -> cams -> gap -> eval -> layout -> render -> rl.
declare -a MENU_GROUP_ORDER=()
group_order() { MENU_GROUP_ORDER=("$@"); }

declare -A _MENU_P_TYPE _MENU_P_DEFAULT _MENU_P_LEVEL _MENU_P_SHORT _MENU_P_LONG
declare -A _MENU_P_RANGE _MENU_P_OPTIONS _MENU_P_GROUP _MENU_P_WHEN _MENU_P_ORIGIN
declare -A _MENU_P_SOURCE _MENU_P_SUGGEST _MENU_P_OVERRIDE
declare -a _MENU_P_ORDER=()

declare -a _MENU_ARG_ORDER=()
declare -A _MENU_ARG_DEFAULT _MENU_ARG_SHORT _MENU_ARG_LONG _MENU_ARG_OPTIONS

_MENU_CUR_GROUP=""
_MENU_PENDING_WHEN=""
_MENU_ACTION=""
declare -a _MENU_NOTES=()

# action <name> "<titel>" [--group G] [--hint H] [--needs A] [--state 'shell-ausdruck']
#   --state wird beim Anzeigen der Aktionsliste ausgewertet und darf einen Marker auf
#   stdout schreiben ("✓ liegt bereits vor"). Er MUSS billig sein: reine Dateisystem-
#   Pruefungen. Ein `docker inspect`, das haengt, wuerde das Menue einfrieren.
action() {
  local name="$1" title="$2"; shift 2
  local group="Weiteres" hint="" needs="" state="" rank=50 cli="__SELF__"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --group) group="$2"; shift 2 ;;
      --rank)  rank="$2";  shift 2 ;;
      --cli)   cli="$2";   shift 2 ;;
      --hint)  hint="$2";  shift 2 ;;
      --needs) needs="$2"; shift 2 ;;
      --state) state="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  _MENU_A_ORDER+=("$name")
  _MENU_A_TITLE["$name"]="$title"
  _MENU_A_GROUP["$name"]="$group"
  _MENU_A_HINT["$name"]="$hint"
  _MENU_A_NEEDS["$name"]="$needs"
  _MENU_A_STATE["$name"]="$state"
  _MENU_A_RANK["$name"]="$rank"
  [[ "$cli" == "__SELF__" ]] && cli="$name"
  _MENU_A_CLI["$name"]="$cli"
  _MENU_ACTION="$name"
}

group() { _MENU_CUR_GROUP="$1"; }
note()  { _MENU_NOTES+=("$1"); }

# when '<bash-ausdruck>'  — gilt fuer den NAECHSTEN param.
when() { _MENU_PENDING_WHEN="$1"; }

# param <VAR> <typ> <default> <stufe> "<kurz>" ["<lang>"] [--range a:b] [--options "k:t;k:t"]
#   typ:    str | int | float | bool | choice | secret | path
#   stufe:  basic (immer gefragt) | advanced (hinter [e]) | expert (nie gefragt, nur Doku)
param() {
  local var="$1" type="$2" def="$3" level="$4" short="$5"; shift 5
  local long=""
  if [[ $# -gt 0 && "$1" != --* ]]; then long="$1"; shift; fi
  local range="" options="" src="" sug="" ovr="" cond="$_MENU_PENDING_WHEN"
  _MENU_PENDING_WHEN=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --range)   range="$2";   shift 2 ;;
      --options) options="$2"; shift 2 ;;
      --when)    cond="$2";    shift 2 ;;
      # --default-from sagt, WO der wirksame Default steht, wenn er nicht als
      # ${VAR:-…} im Host-Skript liegt (z. B. weil das Skript den Wert nur
      # durchreicht und argparse in rl_finetune.py entscheidet). gen_docs.sh --check
      # vergleicht dann gegen diese Quelle statt gegen das Host-Skript.
      --default-from) src="$2"; shift 2 ;;
      --suggest) sug="$2"; shift 2 ;;
      --override) ovr="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  # Ein Parameter, den _common.spec und eine Aktions-Spec beide nennen, wird nicht
  # doppelt gefragt — die aktionsspezifische Fassung gewinnt (sie kommt spaeter) und
  # bestimmt auch die Position. Sonst stuenden die geteilten Parameter aus _common.spec
  # immer vorn, obwohl die aktionseigenen die wichtigeren sind.
  if [[ -n "${_MENU_P_TYPE[$var]:-}" ]]; then
    local -a _keep=(); local _o
    for _o in "${_MENU_P_ORDER[@]}"; do [[ "$_o" == "$var" ]] || _keep+=("$_o"); done
    _MENU_P_ORDER=("${_keep[@]}")
  fi
  _MENU_P_ORDER+=("$var")
  _MENU_P_TYPE["$var"]="$type";      _MENU_P_DEFAULT["$var"]="$def"
  _MENU_P_LEVEL["$var"]="$level";    _MENU_P_SHORT["$var"]="$short"
  _MENU_P_LONG["$var"]="$long";      _MENU_P_RANGE["$var"]="$range"
  _MENU_P_OPTIONS["$var"]="$options"; _MENU_P_GROUP["$var"]="$_MENU_CUR_GROUP"
  _MENU_P_WHEN["$var"]="$cond"; _MENU_P_SOURCE["$var"]="$src"
  _MENU_P_SUGGEST["$var"]="$sug"; _MENU_P_OVERRIDE["$var"]="$ovr"
}

# argpos <n> <default> "<kurz>" ["<lang>"] [--options "k:t;k:t"]
#   Positionsargument der Kommandozeile, NICHT Env-Var: `optimize all`, `webview stop`.
#   Ohne das koennte das Menue diese beiden Aktionen nicht vollstaendig bedienen — der
#   urspruengliche Plan hatte sie uebersehen.
argpos() {
  local idx="$1" def="$2" short="$3"; shift 3
  local long=""
  if [[ $# -gt 0 && "$1" != --* ]]; then long="$1"; shift; fi
  local options=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --options) options="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  _MENU_ARG_ORDER+=("$idx")
  _MENU_ARG_DEFAULT["$idx"]="$def"; _MENU_ARG_SHORT["$idx"]="$short"
  _MENU_ARG_LONG["$idx"]="$long";   _MENU_ARG_OPTIONS["$idx"]="$options"
}

# ── Laden ─────────────────────────────────────────────────────────────────────
_menu_reset() {
  local v
  for v in _MENU_A_TITLE _MENU_A_GROUP _MENU_A_HINT _MENU_A_NEEDS _MENU_A_STATE _MENU_A_FILE \
           _MENU_A_RANK _MENU_A_CLI \
           _MENU_P_TYPE _MENU_P_DEFAULT _MENU_P_LEVEL _MENU_P_SHORT _MENU_P_LONG \
           _MENU_P_RANGE _MENU_P_OPTIONS _MENU_P_GROUP _MENU_P_WHEN _MENU_P_ORIGIN \
           _MENU_P_SOURCE _MENU_P_SUGGEST _MENU_P_OVERRIDE \
           _MENU_ARG_DEFAULT _MENU_ARG_SHORT _MENU_ARG_LONG _MENU_ARG_OPTIONS; do
    unset "$v"; declare -gA "$v"
  done
  _MENU_A_ORDER=(); _MENU_P_ORDER=(); _MENU_ARG_ORDER=(); _MENU_NOTES=()
  _MENU_CUR_GROUP=""; _MENU_PENDING_WHEN=""; _MENU_ACTION=""
}

# Nur die `action`-Zeilen einlesen — fuer die Aktionsliste. param/group/... sind dabei
# wirkungslos, damit das Laden aller Specs billig bleibt.
_menu_load_actions() {
  local specdir="$1" prefix="$2" f
  _menu_reset
  param() { :; }; group() { :; }; when() { :; }; note() { :; }; argpos() { :; }
  # group_order bleibt aktiv — es gehoert zur Aktionsliste.
  local _o
  for _o in "_order.spec" "_order-$prefix.spec"; do
    [[ -f "$specdir/$_o" ]] && source "$specdir/$_o"
  done
  for f in "$specdir/$prefix-"*.spec; do
    [[ -f "$f" ]] || continue
    # shellcheck source=/dev/null
    source "$f"
    [[ -n "$_MENU_ACTION" ]] && _MENU_A_FILE["$_MENU_ACTION"]="$f"
  done
  unset -f param group when note argpos
  _menu_restore_dsl
}

# Nach dem Aktions-Scan die echten DSL-Funktionen wiederherstellen. Sie stehen weiter
# oben in dieser Datei; ein erneutes `source` der Bibliothek waere durch den Guard
# wirkungslos, deshalb wird die Datei hier gezielt in einer Subshell-freien Form neu
# eingelesen. Billiger und robuster: die Definitionen sind in _MENU_DSL_SRC gesichert.
_menu_restore_dsl() {
  eval "$_MENU_DSL_SRC"
}
_MENU_DSL_SRC="$(declare -f param group when note argpos)"

# Vollstaendig laden: _common.spec zuerst, dann die Aktions-Spec (sie darf ueberschreiben).
_menu_load_action() {
  local specdir="$1" prefix="$2" act="$3"
  _menu_reset
  local c
  for c in "_common.spec" "_common-$prefix.spec"; do
    [[ -f "$specdir/$c" ]] || continue
    _MENU_ACTION=""; _MENU_CUR_GROUP=""
    # shellcheck source=/dev/null
    source "$specdir/$c"
  done
  local f="$specdir/$prefix-$act.spec"
  [[ -f "$f" ]] || return 1
  _MENU_CUR_GROUP=""
  # shellcheck source=/dev/null
  source "$f"
  return 0
}

# ── Aktionsliste (Plan §5) ────────────────────────────────────────────────────
# Gibt die gewaehlte Aktion auf STDOUT aus; alles andere geht nach stderr.
menu_pick_action() {
  local specdir="$1" prefix="$2"
  _menu_load_actions "$specdir" "$prefix"
  (( ${#_MENU_A_ORDER[@]} )) || { _menu_out "Keine Spezifikationen unter $specdir gefunden."; return 1; }

  # Gruppen: erst die in MENU_GROUP_ORDER genannten (Kettenreihenfolge), dann der Rest
  # in Reihenfolge des ersten Auftretens — eine neue Spec faellt so hinten heraus statt
  # unsichtbar zu werden.
  local -a groups=(); local -A seen=(); local a g have
  for g in ${MENU_GROUP_ORDER[@]+"${MENU_GROUP_ORDER[@]}"}; do
    have=0
    for a in "${_MENU_A_ORDER[@]}"; do [[ "${_MENU_A_GROUP[$a]}" == "$g" ]] && have=1; done
    (( have )) && { groups+=("$g"); seen["$g"]=1; }
  done
  for a in "${_MENU_A_ORDER[@]}"; do
    g="${_MENU_A_GROUP[$a]}"
    [[ -n "${seen[$g]:-}" ]] || { groups+=("$g"); seen["$g"]=1; }
  done

  # Aktionen innerhalb einer Gruppe nach --rank sortieren.
  local -a sorted=()
  while IFS=$'\t' read -r _ a; do sorted+=("$a"); done < <(
    for a in "${_MENU_A_ORDER[@]}"; do printf '%s\t%s\n' "${_MENU_A_RANK[$a]:-50}" "$a"; done | sort -n -s -k1,1
  )
  _MENU_A_ORDER=("${sorted[@]}")

  local -a idx_action=(); local n=0
  _menu_out ""
  _menu_out "  ${_MENU_C_HEAD}Was moechtest du tun?${_MENU_C_OFF}"
  for g in "${groups[@]}"; do
    _menu_out ""
    _menu_out "  ${_MENU_C_DIM}$g${_MENU_C_OFF}"
    for a in "${_MENU_A_ORDER[@]}"; do
      [[ "${_MENU_A_GROUP[$a]}" == "$g" ]] || continue
      n=$((n+1)); idx_action[$n]="$a"
      local marker=""
      if [[ -n "${_MENU_A_STATE[$a]}" ]]; then
        # Fehler im Zustands-Ausdruck duerfen das Menue nie umbringen.
        marker="$(eval "${_MENU_A_STATE[$a]}" 2>/dev/null || true)"
      fi
      printf '   %2d) %s %s %s\n' \
        "$n" "$(_menu_pad "$a" 11)" "$(_menu_pad "${_MENU_A_TITLE[$a]:0:44}" 44)" \
        "${_MENU_C_DIM}${marker}${_MENU_C_OFF}" >&2
    done
  done
  _menu_out ""
  _menu_out "  ${_MENU_C_DIM}[a] abbrechen   [?] <nr> erklaert eine Aktion${_MENU_C_OFF}"

  while :; do
    _menu_read "  > " || { _menu_eof_abort; return 1; }
    local ans="$_MENU_REPLY"
    case "$ans" in
      a|A|q|Q|"") [[ -z "$ans" ]] && continue; return 1 ;;
      \?*)
        local want="${ans#\?}"; want="${want// /}"
        if [[ "$want" =~ ^[0-9]+$ ]] && (( want >= 1 && want <= n )); then
          local t="${idx_action[$want]}"
          _menu_out ""
          _menu_out "  ${_MENU_C_KEY}$t${_MENU_C_OFF} — ${_MENU_A_TITLE[$t]}"
          [[ -n "${_MENU_A_HINT[$t]}" ]] && _menu_wrap "     " "${_MENU_A_HINT[$t]}"
          [[ -n "${_MENU_A_NEEDS[$t]}" ]] && _menu_out "     ${_MENU_C_WARN}!${_MENU_C_OFF} setzt voraus: ${_MENU_A_NEEDS[$t]}"
          _menu_out ""
        else
          _menu_out "  ${_MENU_C_WARN}!${_MENU_C_OFF} '?<nr>' mit einer Nummer aus der Liste."
        fi
        continue ;;
    esac
    if [[ "$ans" =~ ^[0-9]+$ ]] && (( ans >= 1 && ans <= n )); then
      printf '%s\n' "${idx_action[$ans]}"
      return 0
    fi
    # Auch der Aktionsname selbst wird akzeptiert.
    for a in "${_MENU_A_ORDER[@]}"; do
      if [[ "$a" == "$ans" ]]; then printf '%s\n' "$a"; return 0; fi
    done
    _menu_out "  ${_MENU_C_WARN}!${_MENU_C_OFF} Bitte eine Zahl von 1 bis $n (oder 'a')."
  done
}

# ── Validierung je Typ ────────────────────────────────────────────────────────
# Rueckgabewert 0 = gueltig. Bei 1 steht die Begruendung in _MENU_VERR.
_MENU_VERR=""
_menu_validate() {
  local var="$1" val="$2"
  local type="${_MENU_P_TYPE[$var]}" range="${_MENU_P_RANGE[$var]}" opts="${_MENU_P_OPTIONS[$var]}"
  _MENU_VERR=""
  case "$type" in
    int)
      [[ "$val" =~ ^-?[0-9]+$ ]] || { _MENU_VERR="ganze Zahl erwartet"; return 1; } ;;
    float)
      [[ "$val" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || { _MENU_VERR="Zahl erwartet (z. B. 0.5)"; return 1; } ;;
    bool)
      [[ "$val" == 0 || "$val" == 1 ]] || { _MENU_VERR="0 oder 1"; return 1; } ;;
    choice)
      local o k found=0
      IFS=';' read -ra o <<< "$opts"
      for k in "${o[@]}"; do [[ "${k%%:*}" == "$val" ]] && found=1; done
      (( found )) || { _MENU_VERR="einer von: $(_menu_option_keys "$var")"; return 1; } ;;
    path|str|secret) : ;;
  esac
  if [[ -n "$range" && ( "$type" == int || "$type" == float ) ]]; then
    local lo="${range%%:*}" hi="${range##*:}"
    # Vergleich ueber awk, damit float mitspielt.
    if [[ "$(awk -v v="$val" -v lo="$lo" -v hi="$hi" 'BEGIN{print (v<lo||v>hi)?"1":"0"}')" == 1 ]]; then
      _MENU_VERR="ausserhalb von $lo bis $hi"; return 1
    fi
  fi
  return 0
}

_menu_option_keys() {
  local o k out=""; IFS=';' read -ra o <<< "${_MENU_P_OPTIONS[$1]}"
  for k in "${o[@]}"; do out+="${out:+, }${k%%:*}"; done
  printf '%s' "$out"
}

# Beschriftung eines choice-Wertes ("eager" -> "Standard").
_menu_option_label() {
  local o k; IFS=';' read -ra o <<< "${_MENU_P_OPTIONS[$1]}"
  for k in "${o[@]}"; do [[ "${k%%:*}" == "$2" ]] && { printf '%s' "${k#*:}"; return; }; done
  printf ''
}

# Ja/Nein-Eingaben freundlich auf 0/1 abbilden — deutsch und englisch.
_menu_normalize() {
  local var="$1" val="$2"
  if [[ "${_MENU_P_TYPE[$var]}" == bool ]]; then
    case "${val,,}" in
      j|ja|y|yes|an|ein|true|1) val=1 ;;
      n|nein|no|aus|off|false|0) val=0 ;;
    esac
  fi
  printf '%s' "$val"
}

# ── Eine Frage stellen ────────────────────────────────────────────────────────
# Setzt die Variable per `export`. Rueckgabewert 1 nur bei EOF/Abbruch.
_menu_ask_one() {
  local var="$1"
  local type="${_MENU_P_TYPE[$var]}"
  local cur="${!var:-${_MENU_P_DEFAULT[$var]}}" why=""
  # Vorschlag aus der Umgebung (z. B. VRAM) schlaegt den statischen Default — aber nur,
  # wenn der Nutzer nichts gesetzt hat und der Ausdruck wirklich etwas liefert.
  if [[ -z "${!var:-}" && -n "${_MENU_P_SUGGEST[$var]:-}" ]]; then
    local sv; sv="$(eval "${_MENU_P_SUGGEST[$var]}" 2>/dev/null || true)"
    if [[ -n "$sv" ]]; then
      why="${sv#*|}"; sv="${sv%%|*}"
      [[ "$why" == "$sv" ]] && why=""
      cur="$sv"
    fi
  fi

  _menu_out ""
  _menu_out "  ${_MENU_C_KEY}${_MENU_P_SHORT[$var]}${_MENU_C_OFF}  ${_MENU_C_DIM}($var)${_MENU_C_OFF}"
  if [[ "$type" == choice ]]; then
    local o k; IFS=';' read -ra o <<< "${_MENU_P_OPTIONS[$var]}"
    for k in "${o[@]}"; do
      printf '     %-10s %s\n' "${k%%:*}" "${_MENU_C_DIM}${k#*:}${_MENU_C_OFF}" >&2
    done
  fi

  if [[ -n "$why" ]]; then
    local _first=1 _l
    while IFS= read -r _l; do
      if (( _first )); then
        printf '     %s~%s %s\n' "$_MENU_C_OK" "$_MENU_C_OFF" "$_l" >&2; _first=0
      else
        printf '       %s\n' "$_l" >&2
      fi
    done < <(printf '%s\n' "$why" | fold -s -w 62)
  fi

  local shown=0
  while :; do
    local prompt
    if [[ "$type" == secret ]]; then
      prompt="     ${_MENU_C_DIM}[verdeckt]${_MENU_C_OFF} > "
      _menu_read "$prompt" 1 || return 1
    else
      prompt="     [${cur}] > "
      _menu_read "$prompt" || return 1
    fi
    local val="$_MENU_REPLY"

    # '?' zeigt den Langtext — die "ggf. erklaeren"-Anforderung, ohne dass jede Frage
    # dauerhaft einen Absatz mitschleppt.
    if [[ "$val" == "?" ]]; then
      if (( shown == 0 )) && [[ -n "${_MENU_P_LONG[$var]}" ]]; then
        _menu_wrap "     ${_MENU_C_DIM}| ${_MENU_C_OFF}" "${_MENU_P_LONG[$var]}"
        shown=1
      elif [[ -z "${_MENU_P_LONG[$var]}" ]]; then
        _menu_out "     ${_MENU_C_DIM}| (keine weitere Erklaerung hinterlegt)${_MENU_C_OFF}"
      fi
      continue
    fi

    [[ -z "$val" ]] && val="$cur"
    [[ "$type" == secret && -z "$val" ]] && { export "$var="; return 0; }
    val="$(_menu_normalize "$var" "$val")"
    if _menu_validate "$var" "$val"; then
      export "$var=$val"
      _MENU_P_ORIGIN["$var"]="menue"
      return 0
    fi
    _menu_out "     ${_MENU_C_WARN}!${_MENU_C_OFF} $_MENU_VERR"
  done
}

# Gilt die when-Bedingung eines Parameters gerade?
_menu_cond_ok() {
  local c="${_MENU_P_WHEN[$1]}"
  [[ -z "$c" ]] && return 0
  eval "$c" 2>/dev/null
}

# ── Herkunft eines Wertes (Plan §3.4) ─────────────────────────────────────────
# Vorrang:  explizite Umgebungsvariable  >  .env.local  >  Menue-Antwort  >  Default
# Eine bereits gesetzte Variable wird deshalb NICHT gefragt, sondern als "vorgegeben"
# angezeigt. Das ist der Mechanismus, der das Menue umso stiller macht, je besser
# .env.local gepflegt ist — und der Nutzer damit in die richtige Dauerkonfiguration
# lenkt, statt sie davon abzuhalten. Ueber die Bestaetigungsseite laesst sich trotzdem
# jeder Wert noch aendern.
_menu_origin() {
  local var="$1"
  if declare -F env_preset >/dev/null && env_preset "$var" && [[ -n "${!var:-}" ]]; then
    printf 'umgebung'; return
  fi
  if declare -F env_local_provided >/dev/null && env_local_provided "$var"; then
    printf 'envlocal'; return
  fi
  printf '%s' "${_MENU_P_ORIGIN[$var]:-default}"
}

_menu_origin_label() {
  case "$1" in
    umgebung) printf 'vorgegeben' ;;
    envlocal) printf 'aus .env.local' ;;
    menue)    printf '' ;;
    *)        printf '' ;;
  esac
}

_menu_pregiven() {
  local o; o="$(_menu_origin "$1")"
  [[ "$o" == umgebung || "$o" == envlocal ]]
}

# Geheimnisse werden an ALLEN drei Ausgabestellen maskiert: Zusammenfassung,
# Ein-Zeiler und Recall-Datei. (Plan §6)
_menu_mask() {
  local var="$1" val="$2"
  [[ "${_MENU_P_TYPE[$var]:-str}" == secret ]] || { printf '%s' "$val"; return; }
  [[ -n "$val" ]] || { printf '%s' ''; return; }
  printf '%s…' "${val:0:3}"
}

# Wird dieser Parameter gerade angezeigt/gefragt?
_menu_visible() {
  local var="$1"
  local lvl="${_MENU_P_LEVEL[$var]}"
  [[ "$lvl" == expert ]] && return 1
  [[ "$lvl" == advanced && "${_MENU_SHOW_ADVANCED:-0}" != 1 ]] && return 1
  _menu_cond_ok "$var" || return 1
  return 0
}

# ── Hauptfunktion ─────────────────────────────────────────────────────────────
# menu_ask <specdir> <prefix> <aktion>
#   Fragt, exportiert und fuellt MENU_ARGV mit den Positionsargumenten.
#   Rueckgabewert 1 = abgebrochen (der Aufrufer soll dann beenden).
declare -a MENU_ARGV=()
MENU_ACTION_ASKED=""

menu_ask() {
  local specdir="$1" prefix="$2" act="$3"
  MENU_ARGV=(); MENU_ACTION_ASKED=""
  _menu_load_action "$specdir" "$prefix" "$act" || return 0   # keine Spec = nichts zu fragen
  MENU_ACTION_ASKED="$act"
  _MENU_SHOW_ADVANCED=0

  _menu_out ""
  _menu_out "  ${_MENU_C_HEAD}$act${_MENU_C_OFF} — ${_MENU_A_TITLE[$act]:-}"
  _menu_rule
  local nt
  for nt in "${_MENU_NOTES[@]}"; do _menu_wrap "  " "$nt"; done

  # Vorgegebene Werte zuerst zeigen — das erklaert, warum sie nicht gefragt werden.
  local var shown_pre=0
  for var in "${_MENU_P_ORDER[@]}"; do
    [[ "${_MENU_P_LEVEL[$var]}" == expert ]] && continue
    _menu_pregiven "$var" || continue
    _menu_cond_ok "$var" || continue
    printf '  %s %s %s\n' "$(_menu_pad "$var" 24)" "$(_menu_pad "$(_menu_mask "$var" "${!var}")" 14)" \
      "${_MENU_C_DIM}($(_menu_origin_label "$(_menu_origin "$var")"))${_MENU_C_OFF}" >&2
    shown_pre=1
  done
  (( shown_pre )) && _menu_out ""

  # Recall: die letzte Konfiguration derselben Aktion wiederholen.
  _menu_offer_recall "$prefix" "$act" || return 1

  # Erstdurchlauf: alle sichtbaren, nicht vorgegebenen Parameter der Reihe nach.
  local last_group=""
  for var in "${_MENU_P_ORDER[@]}"; do
    _menu_visible "$var" || continue
    _menu_pregiven "$var" && continue
    [[ -n "${_MENU_P_ORIGIN[$var]:-}" ]] && continue     # kam aus Recall/Profil
    if [[ "${_MENU_P_GROUP[$var]}" != "$last_group" && -n "${_MENU_P_GROUP[$var]}" ]]; then
      last_group="${_MENU_P_GROUP[$var]}"
      _menu_out ""
      _menu_out "  ${_MENU_C_DIM}${last_group}${_MENU_C_OFF}"
    fi
    _menu_ask_one "$var" || { _menu_eof_abort; return 1; }
  done

  # Positionsargumente (optimize <phase>, webview stop)
  local i
  for i in "${_MENU_ARG_ORDER[@]}"; do
    _menu_ask_argpos "$i" || { _menu_eof_abort; return 1; }
  done

  _menu_confirm_loop "$prefix" "$act"
}

# ── Positionsargument fragen ──────────────────────────────────────────────────
declare -A _MENU_ARGVAL=()
_menu_ask_argpos() {
  local i="$1" cur="${_MENU_ARG_DEFAULT[$i]}"
  _menu_out ""
  _menu_out "  ${_MENU_C_KEY}${_MENU_ARG_SHORT[$i]}${_MENU_C_OFF}  ${_MENU_C_DIM}(Argument \$$i)${_MENU_C_OFF}"
  local o k
  if [[ -n "${_MENU_ARG_OPTIONS[$i]}" ]]; then
    IFS=';' read -ra o <<< "${_MENU_ARG_OPTIONS[$i]}"
    for k in "${o[@]}"; do printf '     %-10s %s\n' "${k%%:*}" "${_MENU_C_DIM}${k#*:}${_MENU_C_OFF}" >&2; done
  fi
  while :; do
    _menu_read "     [${cur}] > " || return 1
    local val="$_MENU_REPLY"
    if [[ "$val" == "?" ]]; then
      [[ -n "${_MENU_ARG_LONG[$i]}" ]] && _menu_wrap "     ${_MENU_C_DIM}| ${_MENU_C_OFF}" "${_MENU_ARG_LONG[$i]}"
      continue
    fi
    [[ -z "$val" ]] && val="$cur"
    if [[ -n "${_MENU_ARG_OPTIONS[$i]}" ]]; then
      local found=0
      IFS=';' read -ra o <<< "${_MENU_ARG_OPTIONS[$i]}"
      for k in "${o[@]}"; do [[ "${k%%:*}" == "$val" ]] && found=1; done
      (( found )) || { _menu_out "     ${_MENU_C_WARN}!${_MENU_C_OFF} unbekannt"; continue; }
    fi
    _MENU_ARGVAL["$i"]="$val"
    return 0
  done
}

_menu_collect_argv() {
  MENU_ARGV=()
  local i
  for i in "${_MENU_ARG_ORDER[@]}"; do MENU_ARGV+=("${_MENU_ARGVAL[$i]:-${_MENU_ARG_DEFAULT[$i]}}"); done
}

# ── Bestaetigungsseite ────────────────────────────────────────────────────────
# Bewusst KEIN linearer Fragebogen ohne Zurueck: wer bei Frage 6 falsch tippt, soll
# nicht von vorn anfangen muessen.
_menu_confirm_loop() {
  local prefix="$1" act="$2"
  while :; do
    local -a idx=(); local n=0 var
    _menu_out ""
    _menu_rule
    for var in "${_MENU_P_ORDER[@]}"; do
      _menu_visible "$var" || continue
      n=$((n+1)); idx[$n]="$var"
      local extra=""
      [[ "${_MENU_P_LEVEL[$var]}" == advanced ]] && extra="(erweitert)"
      local o; o="$(_menu_origin "$var")"
      [[ "$o" == umgebung || "$o" == envlocal ]] && extra="$(_menu_origin_label "$o")"
      local lbl=""
      [[ "${_MENU_P_TYPE[$var]}" == choice ]] && lbl="$(_menu_option_label "$var" "${!var:-}")"
      printf '  %2d %s %s %s\n' "$n" "$(_menu_pad "$var" 22)" \
        "$(_menu_pad "$(_menu_mask "$var" "${!var:-${_MENU_P_DEFAULT[$var]}}")" 14)" \
        "${_MENU_C_DIM}${lbl:+$lbl }${extra}${_MENU_C_OFF}" >&2
    done
    local i
    for i in "${_MENU_ARG_ORDER[@]}"; do
      printf '  %2s %s %s\n' "\$$i" "$(_menu_pad "${_MENU_ARG_SHORT[$i]}" 22)" \
        "${_MENU_ARGVAL[$i]:-${_MENU_ARG_DEFAULT[$i]}}" >&2
    done
    _menu_out ""
    local adv_label="[e] erweiterte Optionen"
    [[ "${_MENU_SHOW_ADVANCED}" == 1 ]] && adv_label="[e] erweiterte ausblenden"
    _menu_out "  ${_MENU_C_KEY}[Enter]${_MENU_C_OFF} starten   ${_MENU_C_KEY}[1-$n]${_MENU_C_OFF} aendern   ${_MENU_C_KEY}$adv_label${_MENU_C_OFF}"
    _menu_out "  ${_MENU_C_DIM}[b] nur Befehl zeigen   [p] als Profil sichern   [a] abbrechen${_MENU_C_OFF}"
    _menu_out ""
    _menu_out "  ${_MENU_C_DIM}Entspricht:${_MENU_C_OFF}"
    _menu_out "    $(menu_replay_line "$act")"

    _menu_read "  > " || { _menu_eof_abort; return 1; }
    local ans="$_MENU_REPLY"
    case "$ans" in
      "")  _menu_collect_argv; _menu_save_recall "$prefix" "$act"; return 0 ;;
      a|A) _menu_out "  Abgebrochen."; return 1 ;;
      e|E) _MENU_SHOW_ADVANCED=$(( 1 - _MENU_SHOW_ADVANCED )); continue ;;
      b|B) _menu_collect_argv
           _menu_out ""
           _menu_out "  $(menu_replay_line "$act")"
           _menu_out ""
           _menu_out "  ${_MENU_C_DIM}Nicht gestartet. Zeile kopieren und selbst ausfuehren.${_MENU_C_OFF}"
           return 1 ;;
      p|P) _menu_save_profile "$act"; continue ;;
      \$*) local ai="${ans#\$}"
           [[ -n "${_MENU_ARG_DEFAULT[$ai]:-}" ]] && { _menu_ask_argpos "$ai" || return 1; }
           continue ;;
    esac
    if [[ "$ans" =~ ^[0-9]+$ ]] && (( ans >= 1 && ans <= n )); then
      _menu_ask_one "${idx[$ans]}" || { _menu_eof_abort; return 1; }
    else
      _menu_out "  ${_MENU_C_WARN}!${_MENU_C_OFF} Bitte Enter, eine Zahl 1-$n, oder e/b/p/a."
    fi
  done
}

# ── Der aequivalente Ein-Zeiler ───────────────────────────────────────────────
# Wird IMMER angezeigt. Das Menue ist Lernhilfe, nicht Ersatz fuer das Wissen — beim
# dritten Mal soll man ohne auskommen.
menu_replay_line() {
  local act="$1" out="" var
  for var in "${_MENU_P_ORDER[@]}"; do
    [[ "${_MENU_P_LEVEL[$var]}" == expert ]] && continue
    _menu_cond_ok "$var" || continue
    local v="${!var:-}"
    [[ -z "$v" ]] && continue
    [[ "$v" == "${_MENU_P_DEFAULT[$var]}" ]] && continue      # Defaults weglassen
    out+="$var=$(_menu_mask "$var" "$v") "
  done
  local i argstr=""
  for i in "${_MENU_ARG_ORDER[@]}"; do argstr+=" ${_MENU_ARGVAL[$i]:-${_MENU_ARG_DEFAULT[$i]}}"; done
  local cli="${_MENU_A_CLI[$act]-$act}"
  printf '%s%s%s%s' "$out" "${_MENU_LAUNCHER:-$0}" "${cli:+ $cli}" "$argstr"
}

# Fuer die Log-Datei: dieselbe Aufloesung, aber vollstaendig und mit Herkunft.
# Wird NACH start_logging aufgerufen — deshalb maskiert.
menu_summary_for_log() {
  [[ -n "$MENU_ACTION_ASKED" ]] || return 0
  local var
  printf 'Menue-Konfiguration fuer "%s":\n' "$MENU_ACTION_ASKED"
  for var in "${_MENU_P_ORDER[@]}"; do
    _menu_cond_ok "$var" || continue
    local v="${!var:-}"
    [[ -z "$v" ]] && continue
    printf '  %-26s %-16s %s\n' "$var" "$(_menu_mask "$var" "$v")" "$(_menu_origin "$var")"
  done
}

# ── VRAM-Vorschlag fuer GLOBAL_BATCH_SIZE ─────────────────────────────────────
# Die Tabelle in CLAUDE.md ist heute Wissen, das man vorher gelesen haben muss. Hier
# wird sie ausgewertet. Gibt "<wert>|<begruendung>" aus, oder nichts, wenn keine GPU
# sichtbar ist — dann bleibt der statische Spec-Default stehen.
_menu_suggest_batch_size() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local mib bs
  mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  [[ "$mib" =~ ^[0-9]+$ ]] || return 0
  local gb=$(( mib / 1024 ))
  if   (( gb >= 90 )); then bs=128
  elif (( gb >= 75 )); then bs=64
  elif (( gb >= 30 )); then bs=8
  elif (( gb >= 20 )); then bs=2
  else                      bs=1
  fi
  printf '%s|%s GB VRAM erkannt — Richtwert aus der Tabelle in CLAUDE.md. Bei OOM zuerst hier herunter.' "$bs" "$gb"
  if (( gb < 40 )); then
    printf ' NVIDIA nennt fuer ein volles Fine-tuning mindestens 40 GB; darunter laeuft es, aber sehr langsam.'
  fi
}

# ── Zustand: Recall und Profile (Plan §3.5) ───────────────────────────────────
#   .menu/last/<prefix>.<aktion>.env    zuletzt benutzte Antworten
#   .menu/profiles/<name>.env           benannte Profile (--profile rauchtest)
# Geheimnisse landen dort NIE — statt des Werts wird ein Kommentar geschrieben.
_menu_state_dir() { printf '%s' "${MENU_STATE_DIR:-${REPO_DIR:-.}/.menu}"; }

_menu_write_env_file() {
  local file="$1"; shift
  local dir; dir="$(dirname "$file")"
  mkdir -p "$dir" 2>/dev/null || return 1
  chmod 700 "$dir" "$(_menu_state_dir)" 2>/dev/null || true
  {
    printf '# Automatisch geschrieben von tools/lib_menu.sh — von Hand editierbar.\n'
    printf '# Geheimnisse werden hier bewusst NICHT abgelegt.\n'
    local var
    for var in "${_MENU_P_ORDER[@]}"; do
      local v="${!var:-}"
      [[ -z "$v" ]] && continue
      if [[ "${_MENU_P_TYPE[$var]}" == secret ]]; then
        printf '# %s wurde nicht gespeichert (Geheimnis) — via .env.local oder Env setzen.\n' "$var"
        continue
      fi
      printf '%s=%q\n' "$var" "$v"
    done
    local i
    for i in "${_MENU_ARG_ORDER[@]}"; do
      printf '_MENU_ARGVAL_%s=%q\n' "$i" "${_MENU_ARGVAL[$i]:-${_MENU_ARG_DEFAULT[$i]}}"
    done
  } > "$file"
  chmod 600 "$file" 2>/dev/null || true
}

_menu_read_env_file() {
  local file="$1" line key val
  [[ -f "$file" ]] || return 1
  while IFS= read -r line; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "$line" ]] && continue
    key="${line%%=*}"; val="${line#*=}"
    if [[ "$key" == _MENU_ARGVAL_* ]]; then
      eval "_MENU_ARGVAL[${key#_MENU_ARGVAL_}]=$val"
      continue
    fi
    # Nur Parameter uebernehmen, die diese Aktion ueberhaupt kennt.
    [[ -n "${_MENU_P_TYPE[$key]:-}" ]] || continue
    _menu_pregiven "$key" && continue          # Umgebung/.env.local behalten Vorrang
    eval "export $key=$val"
    _MENU_P_ORIGIN["$key"]="menue"
  done < "$file"
  return 0
}

_menu_offer_recall() {
  local prefix="$1" act="$2"
  local last; last="$(_menu_state_dir)/last/$prefix.$act.env"

  # --profile <name> hat Vorrang und fragt nicht nach.
  if [[ -n "${MENU_PROFILE:-}" ]]; then
    local pf; pf="$(_menu_state_dir)/profiles/$MENU_PROFILE.env"
    if _menu_read_env_file "$pf"; then
      _menu_out "  ${_MENU_C_OK}✓${_MENU_C_OFF} Profil '$MENU_PROFILE' geladen."
      return 0
    fi
    _menu_out "  ${_MENU_C_WARN}!${_MENU_C_OFF} Profil '$MENU_PROFILE' nicht gefunden ($pf) — frage normal."
  fi

  [[ -f "$last" ]] || return 0
  local ts; ts="$(date -r "$last" '+%d.%m. %H:%M' 2>/dev/null || echo '?')"
  _menu_out "  ${_MENU_C_DIM}Letzte Konfiguration vom $ts liegt vor.${_MENU_C_OFF}"
  _menu_read "  [Enter] wiederholen, [n] neu beantworten > " || { _menu_eof_abort; return 1; }
  case "${_MENU_REPLY,,}" in
    n|neu) return 0 ;;
    a)     return 1 ;;
    *)     _menu_read_env_file "$last" && _menu_out "  ${_MENU_C_OK}✓${_MENU_C_OFF} uebernommen." ;;
  esac
  return 0
}

_menu_save_recall() {
  _menu_write_env_file "$(_menu_state_dir)/last/$1.$2.env" || true
}

_menu_save_profile() {
  _menu_read "  Profilname > " || return 1
  local name="${_MENU_REPLY// /_}"
  [[ -n "$name" ]] || { _menu_out "  ${_MENU_C_WARN}!${_MENU_C_OFF} Kein Name — nichts gesichert."; return 0; }
  _menu_collect_argv
  local f; f="$(_menu_state_dir)/profiles/$name.env"
  if _menu_write_env_file "$f"; then
    _menu_out "  ${_MENU_C_OK}✓${_MENU_C_OFF} Profil gesichert. Naechstes Mal:  MENU_PROFILE=$name $0 $MENU_ACTION_ASKED"
  fi
}

# ── Ausgaben fuer Hilfe und Doku (Plan §2.3) ──────────────────────────────────
# menu_render_docs <specdir> <prefix> <aktion>
#   Markdown-Tabellenzeilen. Quelle fuer die generierten Bloecke in docs/.
menu_render_docs() {
  local specdir="$1" prefix="$2" act="$3"
  _menu_load_action "$specdir" "$prefix" "$act" || return 1
  local var
  for var in "${_MENU_P_ORDER[@]}"; do
    local d="${_MENU_P_DEFAULT[$var]}"
    [[ -z "$d" ]] && d="—" || d="\`$d\`"
    local txt="${_MENU_P_LONG[$var]:-${_MENU_P_SHORT[$var]}}"
    txt="${txt//|/\\|}"
    printf '| `%s` | %s | %s | %s |\n' "$var" "$d" "${_MENU_P_LEVEL[$var]}" "$txt"
  done
}

# menu_spec_defaults <specdir> <prefix> <aktion>
#   je Zeile:  VAR <US> default <US> default-from <US> override <US> typ   (US = 0x1f)
#   TAB waere hier falsch: er ist IFS-Whitespace, `read` wuerde zwei aufeinanderfolgende
#   Trenner zu einem zusammenziehen und ein leeres Feld verschoebe alle weiteren Spalten.
# Grundlage des Abgleichs gegen die echten ${VAR:-…} in den Skripten (gen_docs.sh --check).
menu_spec_defaults() {
  local specdir="$1" prefix="$2" act="$3"
  _menu_load_action "$specdir" "$prefix" "$act" || return 1
  local var
  for var in "${_MENU_P_ORDER[@]}"; do
    printf '%s\x1f%s\x1f%s\x1f%s\x1f%s\n' "$var" "${_MENU_P_DEFAULT[$var]}" \
      "${_MENU_P_SOURCE[$var]:-}" "${_MENU_P_OVERRIDE[$var]:-}" "${_MENU_P_TYPE[$var]}"
  done
}

menu_list_actions() {
  _menu_load_actions "$1" "$2"
  local a; for a in "${_MENU_A_ORDER[@]}"; do printf '%s\n' "$a"; done
}
