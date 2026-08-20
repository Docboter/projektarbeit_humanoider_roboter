# TL;DR: Menue-Parameter der Aktion grasp (Greif-Physik, Open-Loop) von server_rl_run.sh.
action grasp "Greif-Physik isoliert (Open-Loop-Replay)" \
  --rank 40 \
  --group "Messen" \
  --hint  "Spielt Dataset-Aktionen ab, ganz ohne Modell. Beantwortet die eine Frage, die vor jeder Policy-Diskussion steht: kann ein Wuerfel ueberhaupt angehoben werden?"

group "Betriebsart"
param GRASP_MODE choice test basic \
  "Wie der Wuerfel platziert wird" \
  "'hold' misst die Greif-Physik OHNE jede Platzierungs-Annahme und ist damit die schaerfere Aussage: sitzt der Wuerfel zwischen den Fingerkuppen und faellt trotzdem, liegt es an der Physik, nicht am geschaetzten Greifpunkt." \
  --options "test:Wuerfel an den geschaetzten Greifpunkten;hold:Wuerfel im Moment des Zugreifens zwischen die Fingerspitzen gesetzt"

group "Live-Ansicht"
param LIVESTREAM choice 0 basic \
  "Isaac-Sim-Viewport streamen" \
  "Gerade bei 'grasp' lohnt es: man will die Greifpose mit eigenen Augen sehen, statt sie aus Zahlen zu erschliessen." \
  --options "0:aus;2:an, privates Netz/VPN;1:an, oeffentliches Netz"
