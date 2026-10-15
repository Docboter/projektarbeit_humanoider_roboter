# TL;DR: Menue-Parameter der Aktion visgate (Open Loop auf Sim-Pruefsets, Lauf 66/72) von server_rl_run.sh.
action visgate "Visuelles Gate: Open Loop auf Sim-Pruefsets" \
  --rank 51 \
  --group "Messen" \
  --hint  "Abnahme eines co-trainierten Checkpoints vor jedem Closed Loop: gleicher Zustand, gleiche Aktion, Sim- statt Echtbild. Tabelle MSE Haende gegen die Ausgangswerte aus Lauf 72. Kein Isaac Sim."

group "Pruefsets"
param VISGATE_SETS str "vis_gate_v3_cube5 vis_gate_v3_cube4" basic \
  "Pruefsets unter /data/cotrain" \
  "Durch Leerzeichen getrennt. Erzeugt in Lauf 72 (Trainings-Ep. 0/4/8/12, ohne DR)."
param VISGATE_TAG str "" basic \
  "Kennung der Messung" \
  "Teil der JSON-Namen. Leer = Ordnername des Checkpoints." \
  --override "leer heisst: do_visgate nimmt den Ordnernamen von CHECKPOINT_PATH"
param VISGATE_STEPS int 600 advanced \
  "Steps je Episode" \
  "600 = volle Laenge der Pruefsets; nur so mit Lauf 72 vergleichbar."
param VISGATE_REAL_TRAJ_IDS str "0 4 8 12" expert \
  "Echte Vergleichs-Episoden" \
  "Positionen im echten Datensatz, passend zu den Pruefset-Episoden."
param VISGATE_GATE_TRAJ_IDS str "0 1 2 3" expert \
  "Positionen im Pruefset" \
  "Der Loader zaehlt ab 0, nicht nach episode_index."
param VISGATE_SEED int 0 advanced \
  "Seed des Diffusions-Rauschens" \
  "Ohne festen Seed streut die MSE Haende derselben Episode bis ~x2. Fuer die Streuung mehrere Seeds fahren."
