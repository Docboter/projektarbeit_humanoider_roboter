# TL;DR: Reihenfolge der Aktionsgruppen im KISSKI-Menue (Training, Auswerten, Gesperrtes zuletzt).
# Die Sim-Gruppe gibt es hier nicht mehr: der Cluster hat keine GPU, die Isaac Sim
# tragen kann (A100/H100 ohne RT-Cores, die RTX 5000 der jupyter-Partition zu alt).
# Was davon uebrig ist, steht unter "Nicht auf diesem Cluster" — gesperrt, mit Grund,
# und bewusst ganz unten.
group_order "Training" "Auswerten" "Nicht auf diesem Cluster"
