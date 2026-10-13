#!/usr/bin/env bash
# TL;DR: Prueft, dass jeder GR00T-Commit-Pin in den Dockerfiles gleich dem Submodul-Stand von app/Groot-1.6 bzw. app/Groot-1.7 ist (CI + lokal).
#
# ci_check_pins.sh — Host-Werkzeug, nie im Image.
#
# Die Images klonen den Fork selbst und checken einen fest eingetragenen Commit aus. Das
# Submodul im Repo ist der Stand, gegen den entwickelt wird. Laufen beide auseinander, testet
# man lokal anderen Code als im Image laeuft. Dieses Skript vergleicht:
#
#   ARG GROOT16_COMMIT=<sha>         -> app/Groot-1.6
#   ARG GROOT_FORK_COMMIT=<sha>      -> app/Groot-1.6  (Augmentation kopiert Dateien aus dem N1.6-Fork)
#   ARG GROOT17_COMMIT=<sha>         -> app/Groot-1.7
#   git -C /app/Groot-1.6 checkout <sha>   (nacktes Literal, z.B. Simulation/Dockerfile.standalone)
#   git -C /app/Groot-1.7 checkout <sha>
#
# Der Submodul-SHA kommt aus dem Git-Index (gitlink), nicht aus dem ausgecheckten Submodul —
# darum braucht CI kein `submodules: true`. Mit --ref <rev> stattdessen aus einem Commit
# (z.B. --ref HEAD). Im CI-Checkout sind Index und HEAD gleich.
#
# Pflicht-Pins: Training/Dockerfile und Simulation/Dockerfile.standalone muessen fuer beide
# Generationen einen Pin tragen, sonst schlaegt der Check fehl (schuetzt davor, dass eine
# Umbenennung den Check still leerlaufen laesst).
#
# Exit: 0 = alles gleich, 1 = Abweichung oder Pflicht-Pin fehlt, 2 = Aufruffehler.
#
# tools/gen_docs.sh vergleicht nur Augmentation- gegen Training-Pin (N1.6), nicht gegen das
# Submodul und nicht N1.7 — dieser Check ist die Obermenge.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref) REF="${2:?--ref braucht eine Revision}"; shift 2 ;;
    -h|--help) awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print; next } exit }' "$0"; exit 0 ;;
    *) echo "Unbekannt: $1" >&2; exit 2 ;;
  esac
done

submodule_sha() {
  local path="$1" line
  if [[ -n "$REF" ]]; then
    line="$(git -C "$REPO_DIR" ls-tree "$REF" -- "$path")"
    [[ "$line" == 160000\ commit\ * ]] || { echo ""; return; }
    awk '{print $3}' <<<"$line"
  else
    line="$(git -C "$REPO_DIR" ls-files -s -- "$path")"
    [[ "$line" == 160000\ * ]] || { echo ""; return; }
    awk '{print $2}' <<<"$line"
  fi
}

declare -A SUB
SUB[1.6]="$(submodule_sha app/Groot-1.6)"
SUB[1.7]="$(submodule_sha app/Groot-1.7)"
for v in 1.6 1.7; do
  if [[ -z "${SUB[$v]}" ]]; then
    echo "FEHLER: app/Groot-$v ist kein Submodul-Eintrag ${REF:+in $REF }im Index." >&2
    exit 1
  fi
  echo "Submodul app/Groot-$v = ${SUB[$v]}"
done

FAIL=0
declare -A SEEN   # "<datei>|<version>" -> 1

check() {  # check <datei> <zeile> <version> <sha> <quelle>
  local file="$1" lineno="$2" ver="$3" sha="$4" what="$5"
  SEEN["$file|$ver"]=1
  if [[ "$sha" == "${SUB[$ver]}" ]]; then
    printf '  ok          %s:%s  %s = %s\n' "$file" "$lineno" "$what" "${sha:0:12}"
  else
    printf '  ABWEICHUNG  %s:%s  %s = %s, Submodul app/Groot-%s = %s\n' \
      "$file" "$lineno" "$what" "${sha:0:12}" "$ver" "${SUB[$ver]:0:12}"
    FAIL=1
  fi
}

mapfile -t DOCKERFILES < <(cd "$REPO_DIR" && find . -path ./app -prune -o -path ./data -prune \
  -o -path ./.git -prune -o -path ./.venv -prune -o -type f -name 'Dockerfile*' -print \
  | sed 's#^\./##' | sort)

for f in "${DOCKERFILES[@]}"; do
  n=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    n=$((n + 1))
    if [[ "$line" =~ ^[[:space:]]*ARG[[:space:]]+(GROOT16_COMMIT|GROOT17_COMMIT|GROOT_FORK_COMMIT)=\"?([0-9a-f]{40}) ]]; then
      name="${BASH_REMATCH[1]}" sha="${BASH_REMATCH[2]}"
      case "$name" in GROOT17_COMMIT) ver=1.7 ;; *) ver=1.6 ;; esac
      check "$f" "$n" "$ver" "$sha" "ARG $name"
    elif [[ "$line" =~ /app/Groot-(1\.[67])[[:space:]]+checkout[[:space:]]+\"?([0-9a-f]{40}) ]]; then
      check "$f" "$n" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "checkout-Literal"
    fi
  done < "$REPO_DIR/$f"
done

for req in "Training/Dockerfile|1.6" "Training/Dockerfile|1.7" \
           "Simulation/Dockerfile.standalone|1.6" "Simulation/Dockerfile.standalone|1.7"; do
  if [[ -z "${SEEN[$req]:-}" ]]; then
    echo "  FEHLT       ${req%|*}: kein N${req#*|}-Pin gefunden (ARG GROOT*_COMMIT=<sha40> oder checkout <sha40>)"
    FAIL=1
  fi
done

echo ""
if (( FAIL )); then
  echo "FEHLGESCHLAGEN: Dockerfile-Pins und Submodule stimmen nicht ueberein."
  echo "Entweder den Pin im Dockerfile anpassen oder das Submodul (git add app/Groot-1.x) nachziehen."
  exit 1
fi
echo "In Ordnung — alle GR00T-Pins gleich dem Submodul-Stand."
