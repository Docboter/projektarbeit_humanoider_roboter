# TL;DR: Die Domaenenliste fuer ./run.sh — welcher Launcher hinter Simulation/Training/KISSKI steckt.
# _domains.spec — die Ebene UEBER den Aktionen, Grundlage von ./run.sh.
#
# Einzige Quelle fuer die Zuordnung Domaene -> Launcher. run.sh selbst kennt keine
# Liste; wer einen vierten Einstiegspunkt ergaenzt, tut es hier und sonst nirgends.
#
#   --group    Ueberschrift in der Liste. Sie traegt die eigentliche Aussage dieser
#              Datei: es gibt ZWEI Taetigkeiten, aber DREI Orte — Training laeuft
#              lokal ODER auf KISSKI, die Simulation nur auf dem IKR-Server.
#   --prefix   waehlt die Aktions-Specs (sim-*.spec, train-*.spec, kisski-*.spec)
#              und liefert die angezeigte Zahl der LAUFFAEHIGEN Aktionen.
#   --needs    Kommandos, ohne die die Domaene auf diesem Rechner nicht laeuft.
#              Der Eintrag bleibt sichtbar und wird nur gesperrt — dass es den
#              KISSKI-Weg gibt, soll man auch auf dem Laptop erfahren.
#
# WARUM KISSKI UNTER "TRAINING" STEHT: Der Cluster kann die Simulation nicht. Isaac
# Sim braucht RT-Cores; die Rechenpartitionen (A100/H100) haben keine, und die einzige
# GPU mit RT-Cores — die Quadro RTX 5000 der jupyter-Partition — ist Turing und damit
# eine Generation zu alt (docs/fehlerbehebung.md). Die Sim laeuft deshalb auf dem
# IKR-Server. Auf KISSKI bleiben Training und Checkpoint-Auswertung; die beiden
# Sim-Aktionen dort sind als --blocked markiert und stehen nur noch als Begruendung da.

# Die Reihenfolge macht --rank; die Ueberschriften folgen ihr. Ein group_order wie in
# _order.spec gibt es hier bewusst nicht — es waere wirkungslos, weil menu_pick_domain
# nach --rank sortiert.

domain sim "auf dem IKR-Server (Docker, RT-Cores)" \
  --rank 10 \
  --group "Simulation" \
  --label "Simulation" \
  --prefix sim \
  --launcher "Simulation/server_rl_run.sh" \
  --needs docker \
  --hint "Closed-Loop-Eval, RL-Feintuning, Rendern des Co-Training-Datensatzes und die Diagnosekette preflight -> setup -> cams -> gap -> eval. Braucht Docker und eine GPU mit RT-Cores (L40, RTX 4090, A6000). A100/H100 haben keine — deshalb geht das auf KISSKI nicht."

domain train "auf diesem Rechner (Docker)" \
  --rank 20 \
  --group "Training" \
  --label "Training (lokal)" \
  --prefix train \
  --launcher "Training/setup_and_train_dockerhub_pull.sh" \
  --needs docker \
  --hint "Behaviour-Cloning-Feintuning von GR00T im Container auf diesem Rechner: Download, Konvertierung und Training in einem Lauf. Der Container ist langlebig — kein --rm, keine Volume-Mounts."

domain kisski "auf dem KISSKI-Cluster (sbatch)" \
  --rank 30 \
  --group "Training" \
  --label "Training (KISSKI)" \
  --prefix kisski \
  --launcher "Training/kisski_menu.sh" \
  --needs sbatch \
  --hint "Dasselbe Training als SLURM-Job auf dem GWDG-Cluster, plus die Open-Loop-Checkpoint-Auswertung. Laeuft nur auf dem Login-Node glogin-gpu.hpc.gwdg.de, weil es sbatch braucht. Partitionen: kisski (A100 80 GB) und kisski-h100 (94 GB), max. 48 h. Die Simulation kann der Cluster nicht — dafuer der IKR-Server."
