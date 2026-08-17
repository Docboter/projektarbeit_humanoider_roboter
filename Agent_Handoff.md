# Agent Handoff

## Endziel

Das Projekt trainiert und evaluiert NVIDIA GR00T N1.6 für einen Unitree G1 mit DEX3-Hand.
Der aktuelle Infrastruktur-Meilenstein ist zusätzlich eine möglichst kleine, unabhängige
Möglichkeit, die Default-Oberfläche von Isaac Sim auf dem IKR-GPU-Server zu starten und vom
lokalen Linux-PC über WireGuard in Chromium anzusehen.

## IKR-WebRTC-Setup

- Server: `192.168.20.230`, Benutzer `mspaeth`.
- SSH benötigt `KexAlgorithms=sntrup761x25519-sha512@openssh.com`.
- Branch: `codex/ikr-isaacsim-browser`, Basis `training-luca-IKR-IS6.0`.
- Bedienung: `IsaacSim-WebRTC/run.sh {start|status|logs|stop}`.
- Browser: `http://192.168.20.230:8211/` in Chromium/Chrome.
- Image: `lucam03/projekt-humanoider-roboter-sim-vastai:latest` mit Isaac Sim 6.0.1 RC.
- Startpfad: `/isaac-sim/runheadless.sh -v`; Lucas GR00T-Entrypoint wird überschrieben.
- Physische GPU: 1. Im Container ist sie die einzige sichtbare GPU und wird dort als GPU 0
  angezeigt.
- Eigene Ports: `8211/tcp`, `49200/tcp`, `48100/udp`.
- Netzwerk: Docker Host-Modus; `192.168.20.230` ist im rootless Container sichtbar.
- Cache: Compose-Volume `mspaeth-isaacsim-browser_isaac-data`; `stop` behält es.
- Browser-Client wird mit `Simulation/Dockerfile.webviewer` für Host und Ports gebaut.

## Sicherheits- und Betriebsgrenzen

- Bestehende Container `groot-rl`, `groot-webview` und `mspaeth-groot-webrtc-test` niemals
  stoppen, ersetzen oder umkonfigurieren. Sie benutzen die Standardports 8210/49100/47998.
- Keine globalen Docker-Bereinigungen und niemals `docker system/image/volume prune`.
- Firewallregeln nur prüfen, nicht automatisch ändern. Fehlt UDP 48100 über WireGuard, muss
  ein IKR-Admin die Regel gezielt ergänzen.
- Secrets liegen ausschließlich in `/home/mspaeth/.config/projektarbeit/server.env`; niemals
  anzeigen, protokollieren oder committen.
- Der Server-Hauptcheckout enthält unversionierte Daten unter `data/RL/`. Für diese Branch
  deshalb einen separaten Worktree unter `~/projektarbeit_humanoider_roboter-webrtc` nutzen.

## Initialisierung auf einer neuen Maschine

1. WireGuard-Verbindung zum IKR-Netz einrichten.
2. SSH-KEX in `~/.ssh/config` hinterlegen und den öffentlichen Schlüssel mit `ssh-copy-id`
   installieren; die genauen Befehle stehen im Root-README.
3. Docker, Docker Compose, NVIDIA Container Toolkit und das oben genannte Image auf dem
   Server bereitstellen.
4. Branch in einen separaten Worktree auschecken und `./IsaacSim-WebRTC/run.sh start`
   ausführen.
5. Im Browser die URL öffnen; für WebRTC müssen alle drei eigenen Ports erreichbar sein.

## Verifikation

Stand 2026-08-17 vor dem abschließenden Laufzeittest:

- Server: zwei NVIDIA RTX PRO 6000 Blackwell, Treiber 610.43.02.
- Docker 29.7.2 und Docker Compose 5.4.0.
- Das Luca-Image ist lokal vorhanden; `/isaac-sim/runheadless.sh` und NVENC sind verfügbar.
- Die WireGuard-IP ist im rootless Host-Netzwerk sichtbar.
- Die alternativen Ports waren bei der Vorprüfung frei.
- Laufzeit-, Browser- und Wiederanlauftest sind noch auszuführen; Ergebnis danach hier
  ergänzen.
