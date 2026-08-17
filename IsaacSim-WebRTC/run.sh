#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ISAAC_SIM_IMAGE="${ISAAC_SIM_IMAGE:-lucam03/projekt-humanoider-roboter-sim-vastai:latest}"
export ISAACSIM_HOST="${ISAACSIM_HOST:-192.168.20.230}"
export ISAACSIM_SIGNAL_PORT="${ISAACSIM_SIGNAL_PORT:-49200}"
export ISAACSIM_STREAM_PORT="${ISAACSIM_STREAM_PORT:-48100}"
export WEB_VIEWER_PORT="${WEB_VIEWER_PORT:-8211}"
export GPU_DEVICE="${GPU_DEVICE:-1}"

compose() {
  docker compose --project-directory "$SCRIPT_DIR" -f "$SCRIPT_DIR/compose.yaml" "$@"
}

die() {
  echo "FEHLER: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' ist nicht installiert."
}

port_busy() {
  local protocol="$1" port="$2"
  if [[ "$protocol" == "tcp" ]]; then
    [[ -n "$(ss -H -ltn "sport = :$port")" ]]
  else
    [[ -n "$(ss -H -lun "sport = :$port")" ]]
  fi
}

preflight() {
  require_command docker
  require_command ip
  require_command nvidia-smi
  require_command ss

  docker info >/dev/null 2>&1 || die "Docker ist nicht erreichbar."
  docker compose version >/dev/null 2>&1 || die "Docker Compose fehlt."
  docker image inspect "$ISAAC_SIM_IMAGE" >/dev/null 2>&1 \
    || die "Image fehlt: $ISAAC_SIM_IMAGE"
  ip -o -4 addr show | awk '{print $4}' | cut -d/ -f1 | grep -Fxq "$ISAACSIM_HOST" \
    || die "ISAACSIM_HOST=$ISAACSIM_HOST ist kein IPv4-Interface dieses Servers."
  nvidia-smi -i "$GPU_DEVICE" >/dev/null 2>&1 \
    || die "GPU_DEVICE=$GPU_DEVICE existiert nicht oder ist nicht erreichbar."
  compose config --quiet

  if [[ -z "$(compose ps --all --quiet 2>/dev/null)" ]]; then
    port_busy tcp "$WEB_VIEWER_PORT" \
      && die "Port $WEB_VIEWER_PORT/tcp ist bereits belegt."
    port_busy tcp "$ISAACSIM_SIGNAL_PORT" \
      && die "Port $ISAACSIM_SIGNAL_PORT/tcp ist bereits belegt."
    port_busy udp "$ISAACSIM_STREAM_PORT" \
      && die "Port $ISAACSIM_STREAM_PORT/udp ist bereits belegt."
  fi

  docker run --rm --gpus "device=$GPU_DEVICE" --entrypoint bash "$ISAAC_SIM_IMAGE" \
    -lc 'ldconfig -p | grep -q libnvidia-encode.so.1' \
    || die "NVENC ist im Container nicht verfügbar."
}

show_ports() {
  local protocol port state
  while read -r protocol port; do
    state="frei"
    port_busy "$protocol" "$port" && state="belegt"
    printf '  %-5s %-5s %s\n' "$protocol" "$port" "$state"
  done <<EOF
tcp $WEB_VIEWER_PORT
tcp $ISAACSIM_SIGNAL_PORT
udp $ISAACSIM_STREAM_PORT
EOF
}

start() {
  preflight
  compose up --detach --build
  compose ps
  echo
  echo "Browser: http://$ISAACSIM_HOST:$WEB_VIEWER_PORT/"
  echo "Isaac Sim braucht beim ersten Start mehrere Minuten."
  echo "Bereitschaft prüfen: $0 logs"
  echo
  echo "Die Firewall wird nicht verändert. Über WireGuard erreichbar sein müssen:"
  echo "  $WEB_VIEWER_PORT/tcp, $ISAACSIM_SIGNAL_PORT/tcp, $ISAACSIM_STREAM_PORT/udp"
}

status() {
  compose ps
  echo
  nvidia-smi -i "$GPU_DEVICE" \
    --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
  echo
  show_ports
}

case "${1:-}" in
  start)  start ;;
  status) status ;;
  logs)   compose logs --follow --tail=200 ;;
  stop)   compose down ;;
  *)
    echo "Verwendung: $0 {start|status|logs|stop}"
    exit 2
    ;;
esac
