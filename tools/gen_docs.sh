#!/usr/bin/env bash
# TL;DR: Host-Werkzeug — prueft Parameter-Defaults in Spec, Skript und Doku gegeneinander (Drift-Check).
# gen_docs.sh — die Versicherung gegen auseinanderdriftende Parameterbeschreibungen.
#
# HINTERGRUND (docs/weiterfuehrend/cli-menuefuehrung.md §0): Jeder Parameter ist heute
# an bis zu vier Stellen beschrieben — als Shell-Default ${VAR:-…}, im usage()-Heredoc,
# in docs/training/env-vars.md und in einer Tabelle in CLAUDE.md. Die Menue-Specs waeren
# Kopie Nummer fuenf. Ohne Gegenmassnahme driftet das garantiert auseinander.
#
# Der Plan sah vor, usage() und die Doku-Tabellen aus den Specs zu GENERIEREN. Beim
# Umsetzen zeigte sich, dass das ein schlechter Tausch ist: der usage()-Heredoc in
# server_rl_run.sh verwebt Prosa und Parameter und liest live $HOST_DATA_DIR und $IMAGE;
# env-vars.md enthaelt je Zeile mehr Begruendung, als eine Spec je tragen wird
# (Lauf-Nummern, Verweise, Warnungen). Generieren wuerde diese Information vernichten.
#
# Deshalb PRUEFT dieses Werkzeug, statt zu erzeugen. Es vergleicht drei Quellen:
#     1. den Default in der .spec
#     2. den echten ${VAR:-…} im Skript (bzw. ENV im Dockerfile)
#     3. den Wert in der Doku-Tabelle
# und meldet jede Abweichung. Das ist Pruefung 9 des Plans, automatisiert — es verhindert
# dasselbe Driften, ohne eine einzige gewachsene Zeile anzufassen.
#
# NUTZUNG:
#   tools/gen_docs.sh            # Bericht, Rueckgabewert 1 bei Abweichung  (fuer CI/Hook)
#   tools/gen_docs.sh --table sim    # Markdown-Tabelle der Sim-Parameter auf stdout
#   tools/gen_docs.sh --table train  # dito fuer das Training
#   tools/gen_docs.sh --quiet    # nur Abweichungen, kein "ok"-Rauschen

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR
source "$REPO_DIR/tools/lib_menu.sh"
SPEC_DIR="$REPO_DIR/tools/menu"

MODE=check; QUIET=0; TABLE_PREFIX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)  MODE=check; shift ;;
    --quiet)  QUIET=1; shift ;;
    --table)  MODE=table; TABLE_PREFIX="${2:-sim}"; shift 2 ;;
    -h|--help) awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print; next } if (/^[[:space:]]*$/) { print ""; next } exit }' "$0"; exit 0 ;;
    *) echo "Unbekannt: $1" >&2; exit 2 ;;
  esac
done

# Wo der wirksame Default eines Parameters steht — je Prefix.
sim_sources=(
  "Simulation/server_rl_run.sh"
  "Simulation/scripts/entrypoint_sim.sh"
  "Simulation/scripts/entrypoint_rl.sh"
  "Simulation/scripts/lib_livestream.sh"
)
train_sources=(
  "Training/setup_and_train_DockerHub-pull.sh"
  "Training/scripts/entrypoint.sh"
  "Training/scripts/run_finetuning.sh"
  "Training/scripts/run_finetuning_vision.sh"
  "Training/scripts/run_finetuning_cotrain.sh"
  "Training/scripts/lib_split.sh"
)
DOCKERFILES=("Training/Dockerfile")
DOC_FILES=("docs/training/env-vars.md" "CLAUDE.md")

# ── Default aus den Skripten ziehen ───────────────────────────────────────────
# Gibt den Wert aus, oder nichts, wenn der Parameter dort nicht als Default vorkommt.
script_default() {
  local var="$1"; shift
  local f hit
  for f in "$@"; do
    [[ -f "$REPO_DIR/$f" ]] || continue
    hit="$(grep -hoE "\\\$\{$var:-[^}]*\}" "$REPO_DIR/$f" 2>/dev/null | head -1)"
    if [[ -n "$hit" ]]; then
      hit="${hit#\$\{$var:-}"; hit="${hit%\}}"
      # ${VAR:-} ohne Wert = Durchreichen, kein Default. Weitersuchen.
      [[ -z "$hit" ]] && continue
      printf '%s' "$hit"; return 0
    fi
  done
  # Dockerfile-ENV zaehlt genauso: dort stehen die Defaults des autonomen Containers.
  for f in "${DOCKERFILES[@]}"; do
    [[ -f "$REPO_DIR/$f" ]] || continue
    hit="$(grep -hoE "^ENV +$var=[^ ]*" "$REPO_DIR/$f" 2>/dev/null | head -1)"
    if [[ -n "$hit" ]]; then
      hit="${hit#ENV *}"; hit="${hit#$var=}"; hit="${hit%\"}"; hit="${hit#\"}"
      printf '%s' "$hit"; return 0
    fi
  done
  return 1
}

# ── Default aus einer Doku-Tabelle ziehen ─────────────────────────────────────
doc_default() {
  local var="$1" f row val
  for f in "${DOC_FILES[@]}"; do
    [[ -f "$REPO_DIR/$f" ]] || continue
    row="$(grep -m1 -E "^\| \`$var\` \|" "$REPO_DIR/$f" 2>/dev/null)" || continue
    [[ -n "$row" ]] || continue
    val="$(printf '%s' "$row" | awk -F'|' '{print $3}' | sed -e 's/^ *//' -e 's/ *$//')"
    printf '%s\t%s' "$val" "$f"; return 0
  done
  return 1
}

# Zahlen zusaetzlich numerisch vergleichen: 1e-5 und 0.00001 sind derselbe Wert, und
# ein Werkzeug, das Schreibweisen anmeckert, gewoehnt man sich ab zu lesen.
same_number() {
  [[ "$1" =~ ^-?[0-9.]+([eE][-+]?[0-9]+)?$ && "$2" =~ ^-?[0-9.]+([eE][-+]?[0-9]+)?$ ]] || return 1
  [[ "$(awk -v a="$1" -v b="$2" 'BEGIN{print (a==b)?"y":"n"}')" == y ]]
}

norm() {
  local v="$1"
  v="${v//\\\"/\"}"                       # \" -> "
  v="${v#\"}"; v="${v%\"}"; v="${v#\'}"; v="${v%\'}"
  printf '%s' "$v"
}

DRIFT=0; CHECKED=0; UNVERIFIABLE=0
declare -a NOTES=()

check_prefix() {
  local prefix="$1"; shift
  local -a sources=("$@")
  local act var
  while IFS= read -r act; do
    [[ -n "$act" ]] || continue
    while IFS=$'\x1f' read -r var spec_def src_hint override ptype; do
      [[ -n "$var" ]] || continue
      CHECKED=$((CHECKED+1))
      local sd
      if sd="$(script_default "$var" "${sources[@]}")"; then
        if [[ "$(norm "$sd")" != "$(norm "$spec_def")" ]] && ! same_number "$sd" "$spec_def"; then
          if [[ -n "$override" ]]; then
            NOTES+=("BEWUSST     $prefix/$act  $var: Spec='$spec_def' statt '$sd' — $override")
          else
            DRIFT=$((DRIFT+1))
            NOTES+=("ABWEICHUNG  $prefix/$act  $var: Spec='$spec_def'  Skript='$sd'")
          fi
        fi
      elif [[ -n "$src_hint" || -n "$override" || "$ptype" == secret ]]; then
        # Default liegt bewusst woanders (--default-from, z. B. argparse), oder die
        # Aktion weicht ausdruecklich ab (--override). Ein Geheimnis (secret) hat per
        # Definition keinen Default. Alle drei sind Aussagen, kein Loch.
        :
      else
        UNVERIFIABLE=$((UNVERIFIABLE+1))
        NOTES+=("UNGEPRUEFT  $prefix/$act  $var: kein \${VAR:-…} in den Quelldateien gefunden")
      fi

      local dd df
      if IFS=$'\t' read -r dd df < <(doc_default "$var"); then
        dd="$(printf '%s' "$dd" | sed -e 's/^`//' -e 's/`$//')"
        case "$dd" in
          "—"|"*(leer)*"|"*auto*"|"") : ;;   # bewusst unbestimmt in der Doku
          *)
            if [[ "$(norm "$dd")" != "$(norm "$spec_def")" ]] && ! same_number "$dd" "$spec_def"; then
              if [[ -n "$override" ]]; then
                NOTES+=("BEWUSST     $prefix/$act  $var: Spec='$spec_def' statt '$dd' ($df) — $override")
              else
                DRIFT=$((DRIFT+1))
                NOTES+=("ABWEICHUNG  $prefix/$act  $var: Spec='$spec_def'  $df='$dd'")
              fi
            fi ;;
        esac
      fi
    done < <(menu_spec_defaults "$SPEC_DIR" "$prefix" "$act")
  done < <(menu_list_actions "$SPEC_DIR" "$prefix")
}

if [[ "$MODE" == table ]]; then
  printf '| Variable | Default | Stufe | Bedeutung |\n|---|---|---|---|\n'
  while IFS= read -r a; do
    [[ -n "$a" ]] || continue
    printf '\n**Aktion `%s`**\n\n' "$a"
    menu_render_docs "$SPEC_DIR" "$TABLE_PREFIX" "$a"
  done < <(menu_list_actions "$SPEC_DIR" "$TABLE_PREFIX")
  exit 0
fi

check_prefix sim   "${sim_sources[@]}"
check_prefix train "${train_sources[@]}"

# Doppelte Meldungen zusammenfassen: ein Parameter aus _common.spec taucht in jeder
# Aktion auf und wuerde den Bericht sonst zwanzigfach fluten.
printf '%s\n' ${NOTES[@]+"${NOTES[@]}"} \
  | sed -E 's#^(ABWEICHUNG|UNGEPRUEFT|BEWUSST) +[a-z]+/[a-z]+  #\1  #' \
  | sort -u > /tmp/.gen_docs_notes.$$
UNIQUE=$(grep -c . /tmp/.gen_docs_notes.$$ 2>/dev/null); UNIQUE=${UNIQUE:-0}

if (( QUIET == 0 )); then
  echo "Abgleich Spec <-> Skript <-> Doku"
  echo "  geprueft:            $CHECKED Parametervorkommen"
  echo "  eindeutige Befunde:  $UNIQUE"
  echo ""
fi
grep -E '^ABWEICHUNG' /tmp/.gen_docs_notes.$$ | sed 's/^/  /'
if (( QUIET == 0 )); then
  grep -E '^BEWUSST'    /tmp/.gen_docs_notes.$$ | sed 's/^/  /'
  grep -E '^UNGEPRUEFT' /tmp/.gen_docs_notes.$$ | sed 's/^/  /'
fi
ABW=$(grep -cE '^ABWEICHUNG' /tmp/.gen_docs_notes.$$ 2>/dev/null); ABW=${ABW:-0}
rm -f /tmp/.gen_docs_notes.$$

echo ""
if (( ABW > 0 )); then
  echo "FEHLGESCHLAGEN: $ABW Abweichung(en). Entweder die Spec oder die andere Stelle anpassen."
  exit 1
fi
echo "In Ordnung — keine Abweichung zwischen Specs, Skripten und Doku-Tabellen."
exit 0
