// TL;DR: Inhalt der Präsentation: 16 Stationen mit Folientext (≤ ~25 Wörter), Sprechernotizen,
// Zeitplan, Quellen und 3D-Zustand je Schritt. Jede Zahl ist in presentation/zahlen.md belegt.
(function () {
  'use strict';

  const META = {
    title: 'Fine-Tuning von NVIDIA GR00T für humanoide Roboter-Manipulation',
    sub: 'Unitree G1 mit DEX3-Händen · Würfel stapeln',
    authors: 'Maximilian Berger · Matthias Späth · Luca Mücke',
    advisor: 'Betreuer: Prof. Dr. Matthias Lorenzen',
    school: 'Hochschule für angewandte Wissenschaften Kempten',
    date: '', // z. B. '24.11.2026' – leer = nicht angezeigt
  };

  // Abstand der beiden Welten (m) und Kurzschreibweisen für Kameras
  const S1 = 2.6; // nach der Spaltung
  const S2 = 1.9; // nach dem Co-Training
  const S3 = 1.35; // Fazit
  const cam = (pos, target, fov, shift) => ({ pos, target, fov, shift });
  const img = (k) => (window.PRES_IMAGES || {})[k] || '';

  const stations = [
    {
      id: 'titel',
      kicker: 'Projektarbeit',
      layout: 'title',
      time: 45,
      steps: 1,
      html: () => `
        <img class="logo" src="${img('logo')}" alt="Logo Hochschule Kempten">
        <h1 class="title">${META.title}</h1>
        <p class="subtitle">${META.sub}</p>
        <p class="authors">${META.authors}</p>
        <p class="meta">${META.advisor}${META.date ? ' · ' + META.date : ''}</p>`,
      state: () => ({ orbit: { c: [0, 0.95, 0.35], r: 3.3, h: 1.85, speed: 0.07, phase: 0.6, shift: 0.16 } }),
      source: '',
      notes: `(0:45) Begrüßung, wer wir sind. Thema: ein vortrainiertes Robotermodell, NVIDIA GR00T, auf einen humanoiden Roboter anpassen, den Unitree G1 mit DEX3-Händen.
Roter Faden heute: die Reise eines einzigen Griffs – vom Datensatz über das Training bis in die Simulation.
Vorweg ehrlich: Am Ende steht kein Turm. Aber wir können zeigen, warum nicht, und was der erste Schritt dahin war.`,
    },
    {
      id: 'aufgabe',
      capture: 4.4, // s Simulationszeit bis zum Standbild (Export)
      kicker: 'Die Aufgabe',
      time: 60,
      steps: 1,
      html: () => `
        <h2>Ein Modell, das sieht, liest und handelt</h2>
        <p class="lede">Kamerabilder und ein Satz Sprache hinein, Arm- und Handbefehle heraus. Ziel: Der G1 stapelt drei Würfel, ohne programmierte Bewegung.</p>
        <p class="tag">Animation des Ziels, keine Messung</p>
        <div class="flow">
          <span class="chip real">4 Kamerabilder</span><span class="plus">+</span><span class="chip">Aufgabe in Sprache</span>
          <span class="arrow">→</span><span class="chip strong">GR00T (VLA)</span><span class="arrow">→</span><span class="chip">Gelenkbefehle</span>
        </div>`,
      state: () => ({
        cam: cam([0.9, 1.55, 2.3], [0, 1.0, 0.35], 38, 0.18),
        anim: { real: 'stack' },
        restart: true,
      }),
      source: 'Kap. 1 der Ausarbeitung · Animation zeigt das Ziel, keine Messung',
      notes: `(1:00) VLA heißt Vision-Language-Action: Das Modell bekommt Kamerabilder und eine Aufgabenbeschreibung und gibt direkt Gelenkbefehle aus.
GR00T ist von NVIDIA auf vielen Robotern vortrainiert; wir passen es mit wenigen hundert Demonstrationen an Roboter und Aufgabe an.
Wichtig: Was Sie hier sehen, ist eine Animation des Ziels, keine Messung.
Ein echter G1 stand uns nicht zur Verfügung. Bewertet wird deshalb auf aufgezeichneten Episoden und in einer Simulation – diese Lücke zwischen Training und Bewertung zieht sich durch das ganze Projekt.`,
    },
    {
      id: 'bausteine',
      kicker: 'Die Bausteine',
      time: 75,
      steps: 1,
      html: () => `
        <h2>301 Demonstrationen, zwei Modelle, vier Kameras</h2>
        <p class="lede">Ein öffentlicher Datensatz echter Teleoperation, GR00T in zwei Generationen und vier Kamerabilder je Zeitschritt.</p>
        <div class="facts">
          <div class="fact real"><b>301</b><span>Episoden · Split 240 Training / 61 Test</span></div>
          <div class="fact"><b>N1.6 · N1.7</b><span>zwei GR00T-Generationen, Checkpoints nicht austauschbar</span></div>
          <div class="fact real"><b>4</b><span>Kameras: 2 am Kopf, 2 an den Handgelenken</span></div>
        </div>`,
      state: () => ({
        cam: cam([0.15, 1.4, 2.75], [0, 1.3, 0.25], 38, 0.2),
        show: ['monitors'],
      }),
      source: 'Datensatz unitreerobotics/G1_Dex3_BlockStacking_Dataset · Kap. 2',
      notes: `(1:15) Der Datensatz kommt von Unitree selbst: 301 per Teleoperation aufgezeichnete Episoden, in denen der G1 Würfel stapelt.
Seit Lauf 3 trennen wir nach Episoden: 240 fürs Training, 61 bleiben für die Bewertung zurück.
Die Bilder an den Monitoren sind echte Kamerabilder aus dem Datensatz – genau das sieht das Modell.
N1.6 ist unser Standard; N1.7 läuft als paralleler Pfad mit neuem Sprach-Bild-Rückgrat. (Optional Sprecherwechsel: die N1.7-Migration hat Matthias Späth gemacht.)`,
    },
    {
      id: 'pipeline',
      kicker: 'Die Pipeline',
      time: 90,
      steps: 3,
      html: (k) => `
        <h2>Ein Weg vom Datensatz in die Simulation</h2>
        <p class="lede">Ein Docker-Image erledigt Download, Konvertierung und Training, gesteuert nur über Umgebungsvariablen. Es läuft lokal, auf vast.ai und auf KISSKI.</p>
        <ol class="steps">
          <li class="${k >= 0 ? 'on' : ''}"><b>Daten</b> Hugging Face, LeRobot v3.0 → v2.1</li>
          <li class="${k >= 1 ? 'on' : ''}"><b>Training</b> Container, Checkpoint auf Hugging Face</li>
          <li class="${k >= 2 ? 'on sim' : ''}"><b>Bewertung</b> Closed Loop in Isaac Lab</li>
        </ol>`,
      state: (k) => ({
        pipe: [0.3, 0.75, 1][k],
        cam: [
          cam([-3.0, 2.9, 6.4], [-3.5, 0.5, 2.0], 38, 0.2),
          cam([0.6, 3.3, 8.4], [0.4, 0.5, 2.9], 38, 0.2),
          cam([5.0, 2.7, 5.4], [4.5, 0.6, 1.0], 38, 0.27),
        ][k],
        camDur: 2.6,
      }),
      source: 'docs/schnittstellen.md · Kap. 3 · latex/figures/projektkarte',
      notes: `(1:30) Der Weg auf dem Boden ist unsere Pipeline. Bernstein steht ab jetzt für echte Daten, Cyan für die Simulation.
Schritt 1: Download des Datensatzes und Konvertierung vom neuen LeRobot-Format v3.0 ins Format v2.1, das GR00T erwartet.
Schritt 2: Training in einem einzigen Container, konfiguriert nur über Umgebungsvariablen. Dasselbe Image läuft lokal, auf gemieteten GPUs bei vast.ai und auf dem KISSKI-Cluster der GWDG über Apptainer. Checkpoints landen auf Hugging Face.
Schritt 3: Die Bewertung im geschlossenen Regelkreis in Isaac Lab, auf dem IKR-Server – dafür braucht es GPUs mit RT-Kernen, die der Cluster nicht hat.`,
    },
    {
      id: 'training',
      kicker: 'Training',
      time: 75,
      steps: 1,
      panel: 'right',
      html: () => `
        <h2>Auf echten Bildern: beide gleichauf</h2>
        <p class="lede">Bestes Val-MSE: N1.6 0,00482 (Schritt 44 000), N1.7 0,00512 (35 000), nicht signifikant verschieden. N1.7 trainiert 1,6× schneller.</p>
        <figure class="panel right wide">${PresCharts.valMse()}
          <figcaption>Open Loop auf zurückgehaltenen Episoden · n = 20 Test-Episoden × 750 Schritte · Lauf 3 (N1.6), Lauf 4 (N1.7)</figcaption>
        </figure>`,
      state: () => ({ cam: cam([-3.0, 2.3, 4.2], [-1.1, 0.9, 0.2]), pipe: 1, pipeDim: true }),
      source: 'val_mse_lauf3_lauf4.csv · docs/ergebnisse/lauf4-n17-vergleich.md §5',
      notes: `(1:15) Gemessen wird Open Loop: Das Modell sieht eine echte, zurückgehaltene Episode und sagt die nächsten Aktionen voraus; wir vergleichen mit dem, was der Mensch tatsächlich getan hat.
N1.6 wird bis Schritt 44 000 monoton besser, bestes Ergebnis 0,00482. Eine frühere Messung auf nur 6 Episoden hatte eine U-Kurve gezeigt – die Nachmessung mit 20 Episoden hat das widerlegt.
N1.7 lernt früher (bei 10 000 und 20 000 Schritten signifikant besser), landet am Ende aber gleichauf, 0,00512, Unterschied nicht signifikant.
Auf vier GPUs brauchte N1.7 7 h 53 min statt 12 h 27 min. Kurz: Das Training funktioniert. Die Frage ist, ob das Modell im Regelkreis greift.`,
    },
    {
      id: 'sim',
      kicker: 'Ab in die Simulation',
      time: 60,
      steps: 3,
      layout: 'top',
      html: (k) => `
        <h2>Gleiche Policy, andere Welt</h2>
        <p class="lede">Ohne Roboter bewerten wir in Isaac Lab: Die Policy sieht jetzt gerenderte Bilder und steuert den simulierten G1.</p>
        ${k >= 1 ? compare(k) : ''}`,
      state: (k) => ({ sep: S1, labels: k === 0, seam: true, cam: cam([0, 2.0, 5.2], [0, 0.75, 0.3]) }),
      source: 'latex/figures/sim/*_cam_left_high.png · Szene aus Lauf 70',
      interactive: true,
      notes: `(1:00) Jetzt spaltet sich die Welt: links die echte, aus der die Trainingsdaten stammen, rechts die Simulation, in der wir bewerten.
Im geschlossenen Regelkreis bekommt die Policy die gerenderten Kamerabilder und den Gelenkzustand, schickt Aktionen, die Simulation rechnet weiter.
[Klick] Mit dem Regler sieht man den Unterschied derselben Kopfkamera: echt gegen simuliert.
[Klick] Die angeglichene Szene aus Lauf 70: breiterer Tisch, graue Arme, korrigierte Handgelenkkamera. Näher dran, aber nicht gleich – das wird gleich wichtig.`,
    },
    {
      id: 'null',
      capture: 3.6, // s Simulationszeit bis zum Standbild (Export)
      kicker: 'Ergebnis im Regelkreis',
      time: 60,
      steps: 1,
      html: () => `
        <h2>Die Hand kommt hin, greift aber nicht</h2>
        <div class="bigstats">
          <div><b class="sim">16/20</b><span>Würfel erreicht</span></div>
          <div><b class="sim">0/20</b><span>Würfel angehoben</span></div>
        </div>
        <p class="lede small">N1.6 @44 000 ohne Co-Training, 20 Episoden à 120 s. N1.7: 7/20 erreicht, 0/20 angehoben.</p>`,
      state: () => ({
        sep: S1,
        cam: cam([1.75, 1.45, 1.75], [1.15, 0.98, 0.33], 38, 0.2),
        anim: { sim: 'miss' },
        restart: true,
      }),
      source: 'closed_loop.csv · Läufe 81, 82',
      notes: `(1:00) Das ist das ehrliche Ergebnis ohne weitere Maßnahmen: In 16 von 20 Episoden erreicht die Hand den Würfel, angehoben wird er nie. N1.7 ist noch schwächer.
Die Animation zeigt das typische Bild: hinfahren, Hand kaum schließen, Würfel anschubsen.
Ist die Simulation kaputt? Nein: Mit dem Basismodell auf dem Referenz-Benchmark RoboCasa messen wir 47,7 % gegen 47,8 % erwartet. Die Pipeline stimmt.
Also Detektivarbeit: Wir haben zwei Ursachen gefunden und gemessen.`,
    },
    {
      id: 'registrierung',
      capture: 4.0, // s Simulationszeit bis zum Standbild (Export)
      kicker: 'Blocker 1 · Registrierung',
      time: 90,
      steps: 2,
      html: (k) => `
        <h2>Die Würfel liegen 3 cm daneben</h2>
        <p class="lede">In der Sim liegen die Würfel je Seite ~3 cm zu weit innen. Korrigiert hebt die abgespielte Demo 6/20 = 30 % statt 0/24.</p>
        ${k >= 1 ? `<figure class="panel bottomleft heat">${PresCharts.heatmap()}
          <figcaption>Demo-Wiedergabe („Orakel“), angehoben (lifted_clear) von 8 Episoden je Versatz · Lauf 62</figcaption></figure>` : ''}`,
      state: () => ({
        sep: S1,
        cam: cam([1.3, 1.45, 1.32], [1.3, 0.9, 0.36], 30, 0.22),
        anim: { sim: 'shift' },
        show: ['shift'],
        restart: true,
      }),
      source: 'orakel_versatz_lauf62.csv · Läufe 57–62, 69 (KI 12–54 %)',
      notes: `(1:30) Erster Test ohne Modell: Wir spielen die aufgezeichneten Bewegungen eines Menschen einfach ab – das „Orakel“. Selbst das hebt kaum etwas, 1 von 16.
Die Hand greift jedes Mal ein paar Zentimeter daneben. Also haben wir die Würfel systematisch verschoben.
[Klick] Ergebnis des Sweeps: ohne Seitenkorrektur 0 von 24, mit 3 cm nach außen 5 von 8. Auf allen 20 Testlagen danach 6 von 20, also 30 %, Konfidenzintervall 12 bis 54 %.
Die Ursache des Versatzes ist offen; die Korrektur stimmt nur im Mittel. Aber erst ab hier misst der Regelkreis überhaupt etwas.`,
    },
    {
      id: 'domain-gap',
      kicker: 'Blocker 2 · Domain-Gap',
      time: 75,
      steps: 2,
      html: (k) => `
        <h2>Die Bilder sind dem Modell fremd</h2>
        <p class="lede">Gleicher Zustand, gleiche Aktion, nur Sim- statt Echtbild: Der Hand-Fehler steigt auf das 13-Fache (N1.6) bzw. 36-Fache (N1.7).</p>
        <p class="note ${k >= 1 ? 'on' : 'off'}">Szene angeglichen (Lauf 70): −40 % bzw. −18 %, bleibt aber ×8 bzw. ×29.</p>
        <div class="legend"><span class="sw real"></span>Echtbild = 1 <span class="sw sim"></span>Sim-Bild</div>`,
      state: (k) => ({
        sep: S1,
        bars: k ? [8, 29] : [13, 36],
        cam: cam([2.6, 1.5, 3.6], [2.2, 0.75, 0.4], 38, 0.15),
      }),
      source: 'Läufe 65/66 (N1.6 @30 000, N1.7 @44 000), Lauf 70 · Open Loop, MSE Hände',
      notes: `(1:15) Zweiter Test: Wir nehmen exakt denselben Zustand und dieselbe Aktion aus dem Datensatz, tauschen aber nur das Bild – echt gegen gerendert.
Auf Sim-Bildern ist der Fehler der Handbewegungen 13-mal so groß bei N1.6, 36-mal bei N1.7 – etwa so schlecht wie mit schwarzen Bildern. Die Policies stützen den Griff also auf die Bilder.
Fußnote: Die ×13 gelten für N1.6 @30 000; mit dem Gate-Protokoll ist @44 000 bei ×17, nur bedingt vergleichbar.
[Klick] Wir haben die Szene angeglichen: breiterer Tisch, graue Arme, neue Handgelenkkamera. Das bringt −40 % bzw. −18 %, aber es bleibt ×8 bzw. ×29. Am Szenenbau allein ist das nicht zu lösen.`,
    },
    {
      id: 'cotraining',
      kicker: 'Lösungsansatz',
      time: 90,
      steps: 1,
      panel: 'right',
      html: () => `
        <h2>Co-Training: die Sim mitlernen</h2>
        <p class="lede">15 % gerenderte Sim-Bilder derselben Episoden ins Training gemischt. Der Sim-Fehler halbiert sich (N1.6) bzw. sinkt auf 18–30 % (N1.7).</p>
        <p class="note on">Preis: N1.6 auf Echtbildern +50–60 %.</p>
        <figure class="panel right wide">${PresCharts.gate()}
          <figcaption>Visuelles Gate · Mittel über 3 Seeds · alle Modelle @44 000 · Läufe 80, 83, 92</figcaption>
        </figure>`,
      state: () => ({ sep: S2, cam: cam([-1.2, 2.0, 4.4], [0.1, 0.9, 0.2]) }),
      source: 'gate_mse_haende.csv · Läufe 73–76, 80, 83, 92',
      notes: `(1:30) Wenn das Modell die Sim-Bilder nicht kennt, zeigen wir sie ihm: Wir rendern dieselben Trainings-Episoden in der Simulation, je 187 Episoden, und mischen sie mit 15 % ins Training.
Vier Trainings auf KISSKI, je 44 000 Schritte: N1.6 und N1.7, je mit 5- und 4-cm-Würfeln.
Vor jedem Closed Loop prüfen wir im „visuellen Gate“ den Hand-Fehler auf zurückgehaltenen Sim-Bildern. N1.6 halbiert ihn, N1.7 fällt auf 18 bis 30 % des Ausgangswerts.
Der Preis: N1.6 wird auf Echtbildern um 50 bis 60 % schlechter, N1.7 nicht. Ob das am echten Roboter stört, kann nur ein Robotertest zeigen.
Die beiden Welten rücken zusammen – aber sie berühren sich nicht.`,
    },
    {
      id: 'griffe',
      capture: 5.6, // s Simulationszeit bis zum Standbild (Export)
      kicker: 'Erste Griffe',
      time: 90,
      steps: 1,
      html: () => `
        <h2>Die Policy greift zum ersten Mal</h2>
        <div class="bigstats">
          <div><b class="sim">2/60</b><span>angehoben = 3 % (KI 0,4–11,5 %)</span></div>
          <div><b class="sim">73 %</b><span>der Demo-Fingerspanne (p = 5·10⁻⁶)</span></div>
        </div>
        <p class="lede small">N1.6 mit Co-Training (cube4 @44 000) auf allen 60 Testlagen. Gegen 0/20 ohne Co-Training statistisch nicht trennbar.</p>`,
      state: () => ({
        sep: S2,
        show: ['griff'],
        anim: { sim: 'grasp' },
        cam: cam([1.7, 1.45, 3.0], [1.45, 1.1, 0.4], 38, 0.17),
        restart: true,
      }),
      source: 'closed_loop.csv · Läufe 88, 96, 97 · Standbild Lauf 88, Ep. 5',
      notes: `(1:30) Das ist der Moment, auf den die ganze Reise hinausläuft: Das Bild rechts ist ein echtes Standbild aus der Simulation, Lauf 88, Episode 5 – die linke Hand hält den roten Würfel in der Luft. Die Animation daneben ist nachgestellt.
Auf allen 60 Testlagen: 2 echte Griffe, also 3 %, Konfidenzintervall 0,4 bis 11,5 %.
Deutlicher ist die Handöffnung: Die Finger öffnen auf 73 % der Spanne aus den Demonstrationen, 1,53 rad gegen 0,52 rad beim gleichen Rezept ohne Renderdaten, p = 5 · 10⁻⁶.
Aber ehrlich: 2 von 60 gegen 0 von 20 ist statistisch nicht zu trennen. Es ist ein Startsignal, kein Durchbruch.`,
    },
    {
      id: 'stufen',
      kicker: 'Wo es hängt',
      time: 60,
      steps: 1,
      html: () => `
        <h2>Berühren fast immer, greifen selten, stapeln nie</h2>
        <p class="lede small st12">Jede Stufe wird einzeln geprüft, deshalb kann „berührt“ über „erreicht“ liegen.</p>
        <div class="legend st12"><div><span class="sw grey"></span>ohne Co-Training (n = 20, 5-cm-Würfel)</div><div><span class="sw sim"></span>mit Co-Training (n = 60, 4-cm-Würfel)</div></div>`,
      state: () => ({
        sep: S2,
        show: ['columns'],
        cam: cam([0.0, 1.55, 4.9], [0, 0.42, 1.6], 38, 0.2),
      }),
      source: 'closed_loop.csv · Lauf 82 (vision-v2 @44 000) und Läufe 88+97 (cube4 @44 000) · 120 s je Episode',
      notes: `(1:00) Die Stufen im Überblick: erreicht heißt Fingerkuppe höchstens 4 cm vom Würfel, berührt heißt Würfel mehr als 1 cm verschoben, angehoben heißt Unterseite über 2 cm in der Luft mit Hand in der Nähe, gestapelt ist der fertige Turm.
Mit Co-Training wird fast jeder Würfel berührt, aber nur selten gehoben – und gestapelt nie.
Achtung bei den Säulen: Die Bedingungen unterscheiden sich in Würfelgröße und Episodenzahl; vergleichbar sind nur die großen Unterschiede.`,
    },
    {
      id: 'cosmos',
      kicker: 'Ausblick · Cosmos',
      time: 60,
      steps: 2,
      html: (k) => `
        <h2>Echte Videos neu einfärben</h2>
        <p class="lede">Cosmos-Transfer2.5 ändert Licht, Hintergrund und Oberflächen echter Videos; die Bewegung bleibt. Die Pipeline lief einmal komplett, ein Training damit steht aus.</p>
        <p class="note on">${k ? 'Variante: warmes Licht, Holztisch' : 'Original aus dem Datensatz'}</p>`,
      state: (k) => ({
        sep: S2,
        show: ['cosmos'],
        cosmos: k,
        tableWood: k === 1,
        cam: cam([-0.45, 1.5, 2.9], [-0.6, 1.25, 0.3], 38, 0.2),
      }),
      source: 'latex/figures/cosmos/ep000_f450_*.png · Kap. 6 · Teilprojekt Maximilian Berger',
      notes: `(1:00) Zweite Quelle zusätzlicher Bilder, Teilprojekt von Maximilian Berger (optional Sprecherwechsel).
Statt Sim-Bilder zu rendern, nimmt Cosmos die echten Videos und ändert nur das Aussehen. Kanten aus dem Original halten Roboter, Würfel und Tisch fest, ein Textprompt beschreibt den neuen Stil.
[Klick] Weil die Bewegung identisch bleibt, passen die aufgezeichneten Aktionen unverändert. Das ist Domain Randomization auf echten Daten.
Stand: einmal end-to-end gelaufen, 2 Quell-Episoden × 3 Varianten, rund 65 GB Grafikspeicher. Ein Training damit steht aus.`,
    },
    {
      id: 'grenzen',
      kicker: 'Grenzen und nächste Schritte',
      time: 75,
      steps: 1,
      html: () => `
        <h2>Kein Turm, kein Robotertest</h2>
        <p class="lede">Gestapelt: 0. Alles stammt aus der Simulation. Griffraten aus 20–60 Episoden sind kaum trennbar.</p>
        <ol class="next">
          <li><b>Robotertest</b> Co-Training-Modell gegen vision-v2 am echten G1</li>
          <li><b>Griffrate heben</b> mehr Render-Episoden oder RL</li>
          <li><b>Metriken schärfen</b> Gate-Schwelle, Kontaktkriterium</li>
        </ol>`,
      state: () => ({ sep: S2, labels: true, cam: cam([0.2, 2.2, 5.4], [0, 0.85, 0.3], 38, 0.2) }),
      source: 'next-steps.md §2 (Punkte 3, 4, 5, 7) · Kap. 7',
      notes: `(1:15) Die Grenzen, als Erkenntnis: Kein Würfel wurde gestapelt. Alle Aussagen zur Aufgabe stammen aus der Simulation. Ohne Treffer in 20 Episoden reicht das 95-%-Intervall bis 17 % – kleine Griffraten sind nicht unterscheidbar.
Dazu Schwächen der Metriken: „angehoben“ zählt auch einen Würfel, der nur auf einem anderen liegt; Treffer bestätigen wir im Video. Die Gate-Schwelle lassen auch alte Modelle durch.
Nächste Schritte in dieser Reihenfolge: erstens der Robotertest, er entscheidet, ob der Echtbild-Verlust von N1.6 tragbar ist. Zweitens die Griffrate heben – nach unserer vorab festgelegten Regel mit mehr Render-Episoden, alternativ RL, das jetzt erstmals ein Startsignal hat. Drittens Gate und Erfolgskriterium nachschärfen.`,
    },
    {
      id: 'fazit',
      kicker: 'Fazit',
      time: 45,
      steps: 1,
      layout: 'center',
      html: () => `
        <h2 class="big">Das Training gelingt. Die Lücke liegt zwischen Trainingsdaten und Bewertung – und sie ist jetzt vermessen.</h2>`,
      state: () => ({ sep: S3, seam: true, cam: cam([0, 2.2, 6.0], [0, 0.6, 0]) }),
      source: 'Kap. 9 der Ausarbeitung (Fazit)',
      notes: `(0:45) Wenn Sie einen Satz mitnehmen: Das Fine-Tuning funktioniert reproduzierbar, auf echten Bildern folgt das Modell den Demonstrationen.
Gescheitert ist es an der Lücke zwischen Daten und Bewertung – zwei Ursachen, Würfellage und Bildstil, sind gemessen, und Co-Training verkleinert die zweite.
Rückblickend wären beide Blocker vermeidbar gewesen, wenn Trainingsdaten und Bewertung aus derselben Umgebung gestammt hätten. Danke!`,
    },
    {
      id: 'credits',
      kicker: 'Quellen und Lizenzen',
      time: 30,
      steps: 1,
      layout: 'credits',
      html: () => `
        <h2>Quellen, Lizenzen, KI-Nutzung</h2>
        <div class="credits">
          <div><h3>Modelle</h3><p>NVIDIA Isaac GR00T N1.6 und N1.7 (Fork lucam06/Isaac-GR00T) · NVIDIA Cosmos-Transfer2.5 · Lizenzen laut Modellkarten</p></div>
          <div><h3>Daten</h3><p>unitreerobotics/G1_Dex3_BlockStacking_Dataset (Apache-2.0) · eigene Renderdaten luca-mue/g1-dex3-rendered-v3-*</p></div>
          <div><h3>Simulation und Rechnen</h3><p>NVIDIA Isaac Lab / Isaac Sim · KISSKI-Cluster der GWDG · IKR-Server · vast.ai</p></div>
          <div><h3>Diese Präsentation</h3><p>Robotermodell: unitreerobotics/unitree_ros (BSD-3-Clause) · three.js (MIT) · Schrift Inter (SIL OFL 1.1)</p></div>
          <div><h3>KI-Nutzung</h3><p>Code, Auswertungen, Dokumentation und diese Präsentation entstanden mit Claude Code (Anthropic). Jede Zahl ist mit Quelle belegt: presentation/zahlen.md</p></div>
          <div><h3>Repository</h3><p>github.com/Docboter/projektarbeit_humanoider_roboter</p></div>
        </div>`,
      state: () => ({ sep: S3, orbit: { c: [0, 0.9, 0.2], r: 5.2, h: 2.6, speed: 0.04, phase: 0.2, shift: 0 } }),
      source: '',
      notes: `(0:30) Quellen und Lizenzen. Fragen gern – Details zu jeder Zahl stehen mit Laufnummer im Repository.`,
    },
  ];

  // Vorher/Nachher-Regler (Station 6)
  function compare(k) {
    const right = k >= 2 ? 'sim70_left_high' : 'sim_left_high';
    const lab = k >= 2 ? 'Sim, angeglichen (Lauf 70)' : 'Sim, erste Szene';
    return `<figure class="panel right compare interactive" data-compare>
      <div class="cmp">
        <img class="cmp-a" src="${img('real_left_high')}" alt="Echtes Kamerabild">
        <div class="cmp-b"><img src="${img(right)}" alt="Simuliertes Kamerabild"></div>
        <div class="cmp-h"><span></span></div>
        <span class="cmp-l real">Echt</span><span class="cmp-l sim">${lab}</span>
      </div>
      <figcaption>Kopfkamera links · Regler ziehen</figcaption></figure>`;
  }

  // Der Pfad ist nur in Station 4 voll und in Station 5 gedimmt zu sehen
  stations.forEach((s) => {
    const st = s.state;
    s.state = (k) => Object.assign({ pipe: -1 }, st(k));
  });

  window.PresStations = { META, stations };
})();
