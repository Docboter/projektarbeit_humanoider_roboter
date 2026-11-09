// TL;DR: Die eine 3D-Szene der Präsentation: Real- und Sim-Zwilling (G1 + Tisch + Würfel),
// Pipeline-Pfad, Monitore, Balken, Säulen, Cosmos-Schirm; Zustände je Station, Kamerafahrten.
(function () {
  'use strict';
  const T = THREE;
  const COL = {
    amber: 0xf5a524,
    cyan: 0x22d3ee,
    bg: 0x0a0d12,
    grey: 0x6b7a8c,
    ink: 0xe8edf2,
  };
  const CUBE = 0.05;
  const TABLE_TOP = 0.87;
  const CUBE_Y = TABLE_TOP + CUBE / 2;
  const CUBE_HOME = {
    yellow: [0.15, CUBE_Y, 0.33],
    green: [0.0, CUBE_Y, 0.4],
    red: [-0.15, CUBE_Y, 0.33],
  };
  const CUBE_COLOR = { yellow: 0xe8c547, green: 0x4cbf6a, red: 0xd9473f };
  const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
  const easeOut = (u) => 1 - Math.pow(1 - u, 3);
  const v3 = (a) => new T.Vector3(a[0], a[1], a[2]);

  // ---------- Hilfen: Texturen, Beschriftungen ----------
  function canvasTex(w, h, draw) {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    draw(c.getContext('2d'), w, h);
    const t = new T.CanvasTexture(c);
    t.colorSpace = T.SRGBColorSpace;
    return t;
  }

  function radialTex(rgb, inner) {
    return canvasTex(256, 256, (g, w, h) => {
      const gr = g.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
      gr.addColorStop(0, `rgba(${rgb},${inner})`);
      gr.addColorStop(0.45, `rgba(${rgb},${inner * 0.35})`);
      gr.addColorStop(1, `rgba(${rgb},0)`);
      g.fillStyle = gr;
      g.fillRect(0, 0, w, h);
    });
  }

  function hex(c) {
    return '#' + c.toString(16).padStart(6, '0');
  }

  /** Text-Sprite; lines: string oder [string, ...]; erste Zeile fett. */
  function makeLabel(lines, o) {
    o = Object.assign({ color: '#e8edf2', sub: '#9aa7b4', size: 0.12, bg: null, align: 'center' }, o);
    const sprite = new T.Sprite(new T.SpriteMaterial({ transparent: true, depthWrite: false }));
    sprite.userData.opts = o;
    sprite.setText = function (ls) {
      ls = Array.isArray(ls) ? ls : [ls];
      const px = 96;
      const font0 = `600 ${px}px Inter, system-ui, sans-serif`;
      const font1 = `400 ${Math.round(px * 0.62)}px Inter, system-ui, sans-serif`;
      const meas = document.createElement('canvas').getContext('2d');
      let wmax = 0;
      ls.forEach((l, i) => {
        meas.font = i === 0 ? font0 : font1;
        wmax = Math.max(wmax, meas.measureText(l).width);
      });
      const pad = px * 0.45;
      const lh = [px * 1.15].concat(ls.slice(1).map(() => px * 0.8));
      const W = Math.ceil(wmax + pad * 2);
      const H = Math.ceil(lh.reduce((a, b) => a + b, 0) + pad * 1.2);
      const tex = canvasTex(W, H, (g) => {
        if (o.bg) {
          g.fillStyle = o.bg;
          const r = px * 0.3;
          g.beginPath();
          g.roundRect(0, 0, W, H, r);
          g.fill();
        }
        g.textBaseline = 'top';
        g.textAlign = o.align;
        const x = o.align === 'center' ? W / 2 : pad;
        let y = pad * 0.6;
        ls.forEach((l, i) => {
          g.font = i === 0 ? font0 : font1;
          g.fillStyle = i === 0 ? o.color : o.sub;
          g.fillText(l, x, y);
          y += lh[i];
        });
      });
      tex.anisotropy = 4;
      if (sprite.material.map) sprite.material.map.dispose();
      sprite.material.map = tex;
      sprite.material.needsUpdate = true;
      const hWorld = o.size * (H / (px * 1.15));
      sprite.scale.set((hWorld * W) / H, hWorld, 1);
    };
    sprite.setText(lines);
    sprite.renderOrder = 10;
    return sprite;
  }

  // ---------- Tweens ----------
  const tweens = new Map();
  function tween(key, from, to, dur, fn, easing) {
    if (dur <= 0) {
      fn(to);
      tweens.delete(key);
      return;
    }
    tweens.set(key, { t: 0, from, to, dur, fn, easing: easing || ease });
  }
  function updateTweens(dt) {
    for (const [k, tw] of tweens) {
      tw.t = Math.min(1, tw.t + dt / tw.dur);
      const u = tw.easing(tw.t);
      tw.fn(tw.from + (tw.to - tw.from) * u);
      if (tw.t >= 1) tweens.delete(k);
    }
  }

  // ---------- Zwilling: Roboter + Tisch + Würfel ----------
  function createRig(kind, restPose) {
    const sim = kind === 'sim';
    const group = new T.Group();
    const body = new T.MeshStandardMaterial(
      sim
        ? { color: 0xe9eef2, roughness: 0.85, metalness: 0.0, emissive: COL.cyan, emissiveIntensity: 0.04 }
        : { color: 0xa9afb8, roughness: 0.42, metalness: 0.45 }
    );
    const dark = new T.MeshStandardMaterial({ color: sim ? 0x2a3038 : 0x1a1d22, roughness: 0.55, metalness: 0.2 });
    const robot = PresRobot.createRobot({ body, dark });
    const holder = new T.Group();
    holder.rotation.y = -Math.PI / 2; // Roboter schaut nach +z (zur Kamera)
    holder.add(robot.root);
    group.add(holder);
    robot.root.position.y = 0.8;
    group.updateMatrixWorld(true);
    robot.root.position.y -= robot.lowestPoint();

    // Tisch: Oberkante 0,87 m wie in scene_geometry.py; Mitte 0,557 m vor dem Becken.
    const tableMat = new T.MeshStandardMaterial({ color: sim ? 0xf3f6f8 : 0xe6e2da, roughness: 0.8 });
    const legMat = new T.MeshStandardMaterial({ color: 0x2b3039, roughness: 0.7 });
    const table = new T.Group();
    const top = new T.Mesh(new T.BoxGeometry(1.2, 0.04, 0.8), tableMat);
    top.position.set(0, TABLE_TOP - 0.02, 0.557);
    table.add(top);
    const legGeo = new T.BoxGeometry(0.04, TABLE_TOP - 0.04, 0.04);
    for (const [x, z] of [[-0.56, 0.2], [0.56, 0.2], [-0.56, 0.92], [0.56, 0.92]]) {
      const l = new T.Mesh(legGeo, legMat);
      l.position.set(x, (TABLE_TOP - 0.04) / 2, z);
      table.add(l);
    }
    group.add(table);

    const cubes = {};
    const cubeGeo = new T.BoxGeometry(CUBE, CUBE, CUBE);
    for (const k in CUBE_HOME) {
      const m = new T.Mesh(cubeGeo, new T.MeshStandardMaterial({ color: CUBE_COLOR[k], roughness: 0.5 }));
      m.position.fromArray(CUBE_HOME[k]);
      m.name = k;
      group.add(m);
      cubes[k] = m;
    }
    // Bodenschein und Schatten
    const glow = new T.Mesh(
      new T.PlaneGeometry(3.4, 3.4),
      new T.MeshBasicMaterial({
        map: radialTex(sim ? '34,211,238' : '245,165,36', 0.55),
        transparent: true,
        depthWrite: false,
        blending: T.AdditiveBlending,
        opacity: 0,
      })
    );
    glow.rotation.x = -Math.PI / 2;
    glow.position.set(0, 0.004, 0.3);
    group.add(glow);
    const shadow = new T.Mesh(
      new T.PlaneGeometry(2.0, 2.0),
      new T.MeshBasicMaterial({ map: radialTex('0,0,0', 0.75), transparent: true, depthWrite: false })
    );
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.set(0, 0.006, 0.35);
    group.add(shadow);

    const label = makeLabel(sim ? ['SIMULATION', 'Isaac Lab'] : ['ECHTE WELT', 'Datensatz, Teleoperation'], {
      color: hex(sim ? COL.cyan : COL.amber),
      size: 0.13,
    });
    label.position.set(0, 0.16, 1.35);
    label.material.opacity = 0;
    group.add(label);

    if (restPose) robot.setPose(restPose);
    const rig = { kind, group, robot, cubes, table, tableMat, glow, label, body, dark };
    rig.anim = new Animator(rig);
    return rig;
  }

  function computeRestPose() {
    // Ellbogen unten, Unterarme nach vorn, Hände knapp über dem Tisch (Teleoperations-Startpose).
    const tmp = PresRobot.createRobot({ body: new T.MeshBasicMaterial(), dark: new T.MeshBasicMaterial() });
    for (const sd of ['left', 'right']) {
      tmp.set(sd + '_shoulder_pitch_joint', -0.3);
      tmp.grip(sd, 0.08);
    }
    return tmp.getPose();
  }

  // ---------- Animationsskripte (Roboter) ----------
  class Animator {
    constructor(rig) {
      this.rig = rig;
      this.queue = [];
      this.cur = null;
      this.script = null;
      this.speed = 1;
      this.held = {};
    }
    play(name) {
      this.scriptName = name;
      this.script = SCRIPTS[name] || null;
      this.reset();
      this.queue = this.script ? this.script.steps.slice() : [];
      this.cur = null;
    }
    reset() {
      const r = this.rig;
      for (const k in r.cubes) {
        const c = r.cubes[k];
        if (c.parent !== r.group) r.group.attach(c);
        c.position.fromArray(CUBE_HOME[k]);
        c.quaternion.identity();
      }
      this.held = {};
      if (World.restPose) r.robot.setPose(World.restPose);
    }
    begin(a) {
      const r = this.rig;
      a = Object.assign({}, a);
      a.t = 0;
      if (a.a === 'reach') {
        r.group.updateMatrixWorld(true);
        a.from = r.group.worldToLocal(r.robot.palmPoint(a.side));
        let goal = a.cube ? r.cubes[a.cube].getWorldPosition(new T.Vector3()) : null;
        if (goal) goal = r.group.worldToLocal(goal);
        else goal = v3(a.to);
        if (a.off) goal.add(v3(a.off));
        if (a.carry) {
          // Zielpunkt gilt für den Würfel; die Hand hält ihren aktuellen Abstand.
          const c = r.cubes[a.carry].getWorldPosition(new T.Vector3());
          const p = r.robot.palmPoint(a.side);
          goal.add(r.group.worldToLocal(p).sub(r.group.worldToLocal(c)));
        }
        a.goal = goal;
        a.upV = v3(a.up || [0, 1, 0]);
      } else if (a.a === 'grip') {
        a.g0 = this.gripState(a.side);
      } else if (a.a === 'rest' || a.a === 'pose') {
        a.p0 = r.robot.getPose();
      } else if (a.a === 'grab') {
        r.group.updateMatrixWorld(true);
        r.robot.links[a.side + '_hand_palm_link'].attach(r.cubes[a.cube]);
        this.held[a.cube] = a.side;
      } else if (a.a === 'release') {
        const c = r.cubes[a.cube];
        r.group.attach(c);
        delete this.held[a.cube];
        a.p0 = c.position.clone();
        a.q0 = c.quaternion.clone();
        a.p1 = a.place ? v3(a.place) : a.p0.clone();
      } else if (a.a === 'nudge') {
        a.p0 = r.cubes[a.cube].position.clone();
        a.p1 = a.p0.clone().add(v3(a.by));
      } else if (a.a === 'reset') {
        this.reset();
      }
      return a;
    }
    gripState(side) {
      const j = this.rig.robot.joints[side + '_hand_index_1_joint'];
      return Math.abs(j.angle) / 1.0;
    }
    update(dt) {
      if (!this.script) return;
      if (!this.cur) {
        if (!this.queue.length) {
          if (this.script.loop) this.queue = this.script.steps.slice();
          else return;
        }
        this.cur = this.begin(this.queue.shift());
      }
      const a = this.cur;
      const r = this.rig;
      a.t += dt * this.speed;
      const d = a.d || 0;
      const u = d > 0 ? Math.min(1, a.t / d) : 1;
      const e = ease(u);
      if (a.a === 'reach') {
        const p = a.from.clone().lerp(a.goal, e);
        r.group.updateMatrixWorld(true);
        r.robot.solveIK(a.side, r.group.localToWorld(p), a.upV, 3, a.w === undefined ? 0.25 : a.w);
      } else if (a.a === 'grip') {
        r.robot.grip(a.side, a.g0 + (a.to - a.g0) * e);
      } else if (a.a === 'rest') {
        const p1 = World.restPose;
        for (const k in p1) {
          if (a.side && k.indexOf(a.side) !== 0) continue;
          r.robot.set(k, a.p0[k] + (p1[k] - a.p0[k]) * e);
        }
      } else if (a.a === 'pose') {
        for (const k in a.pose) r.robot.set(k, a.p0[k] + (a.pose[k] - a.p0[k]) * e);
      } else if (a.a === 'release') {
        const c = r.cubes[a.cube];
        c.position.copy(a.p0).lerp(a.p1, e);
        c.quaternion.copy(a.q0).slerp(new T.Quaternion(), e);
      } else if (a.a === 'nudge') {
        r.cubes[a.cube].position.copy(a.p0).lerp(a.p1, e);
      }
      if (u >= 1) this.cur = null;
    }
  }

  const up = [0, 1, 0];
  // Arme zurückgenommen, damit der Blick auf die Würfel frei ist (Station 8)
  const ARMS_BACK = {
    left_shoulder_pitch_joint: 0.35, right_shoulder_pitch_joint: 0.35,
    left_shoulder_roll_joint: 0.25, right_shoulder_roll_joint: -0.25,
    left_elbow_joint: 0.9, right_elbow_joint: 0.9,
  };
  const SCRIPTS = {
    // Station 2: das Ziel, als Animation (keine Messung).
    stack: {
      loop: true,
      steps: [
        { a: 'reset' },
        { a: 'wait', d: 0.8 },
        { a: 'reach', side: 'left', cube: 'yellow', off: [0, 0.09, 0], up, d: 1.1 },
        { a: 'reach', side: 'left', cube: 'yellow', off: [0, 0.005, 0], up, d: 0.6 },
        { a: 'grip', side: 'left', to: 0.62, d: 0.45 },
        { a: 'grab', side: 'left', cube: 'yellow' },
        { a: 'reach', side: 'left', carry: 'yellow', to: [0.06, CUBE_Y + 0.13, 0.38], up, d: 0.9 },
        { a: 'reach', side: 'left', carry: 'yellow', to: [0.0, CUBE_Y + CUBE + 0.004, 0.4], up, d: 0.8 },
        { a: 'release', cube: 'yellow', place: [0.0, CUBE_Y + CUBE, 0.4], d: 0.2 },
        { a: 'grip', side: 'left', to: 0.08, d: 0.35 },
        { a: 'reach', side: 'left', to: [0.12, CUBE_Y + 0.15, 0.3], up, d: 0.6 },
        { a: 'rest', side: 'left', d: 0.8 },
        { a: 'reach', side: 'right', cube: 'red', off: [0, 0.09, 0], up, d: 1.0 },
        { a: 'reach', side: 'right', cube: 'red', off: [0, 0.005, 0], up, d: 0.6 },
        { a: 'grip', side: 'right', to: 0.62, d: 0.45 },
        { a: 'grab', side: 'right', cube: 'red' },
        { a: 'reach', side: 'right', carry: 'red', to: [-0.06, CUBE_Y + 0.2, 0.38], up, d: 0.9 },
        { a: 'reach', side: 'right', carry: 'red', to: [0.0, CUBE_Y + 2 * CUBE + 0.004, 0.4], up, d: 0.8 },
        { a: 'release', cube: 'red', place: [0.0, CUBE_Y + 2 * CUBE, 0.4], d: 0.2 },
        { a: 'grip', side: 'right', to: 0.08, d: 0.35 },
        { a: 'reach', side: 'right', to: [-0.12, CUBE_Y + 0.2, 0.3], up, d: 0.6 },
        { a: 'rest', side: 'right', d: 0.8 },
        { a: 'wait', d: 2.5 },
      ],
    },
    // Station 7: Hand erreicht den Würfel, schließt kaum, schiebt ihn an.
    miss: {
      loop: true,
      steps: [
        { a: 'reset' },
        { a: 'wait', d: 0.6 },
        { a: 'reach', side: 'right', cube: 'red', off: [0.0, 0.08, -0.02], up, d: 1.3 },
        { a: 'reach', side: 'right', cube: 'red', off: [0.02, 0.02, -0.035], up, d: 0.7 },
        { a: 'grip', side: 'right', to: 0.22, d: 0.5 },
        { a: 'nudge', cube: 'red', by: [-0.012, 0, 0.018], d: 0.6 },
        { a: 'grip', side: 'right', to: 0.08, d: 0.5 },
        { a: 'reach', side: 'right', cube: 'red', off: [0.02, 0.07, -0.04], up, d: 0.9 },
        { a: 'wait', d: 0.4 },
        { a: 'reach', side: 'right', cube: 'red', off: [0.025, 0.02, -0.035], up, d: 0.8 },
        { a: 'grip', side: 'right', to: 0.2, d: 0.5 },
        { a: 'grip', side: 'right', to: 0.08, d: 0.5 },
        { a: 'rest', side: 'right', d: 1.0 },
        { a: 'wait', d: 1.5 },
      ],
    },
    // Station 8: Würfel 3 cm nach außen (Orakel-Korrektur).
    shift: {
      loop: false,
      steps: [
        { a: 'reset' },
        { a: 'pose', pose: ARMS_BACK, d: 0.01 },
        { a: 'wait', d: 0.9 },
        { a: 'nudge', cube: 'yellow', by: [0.03, 0, 0], d: 1.4 },
        { a: 'nudge', cube: 'red', by: [-0.03, 0, 0], d: 1.4 },
      ],
    },
    // Station 11: der erste Griff (Animation; Beleg ist das Standbild aus Lauf 88).
    grasp: {
      loop: true,
      steps: [
        { a: 'reset' },
        { a: 'wait', d: 0.6 },
        { a: 'reach', side: 'right', cube: 'red', off: [0, 0.09, 0], up, d: 1.2 },
        { a: 'reach', side: 'right', cube: 'red', off: [0, 0.005, 0], up, d: 0.7 },
        { a: 'grip', side: 'right', to: 0.62, d: 0.5 },
        { a: 'grab', side: 'right', cube: 'red' },
        { a: 'reach', side: 'right', carry: 'red', to: [-0.13, CUBE_Y + 0.12, 0.31], up, d: 1.4 },
        { a: 'wait', d: 3.0 },
        { a: 'reach', side: 'right', carry: 'red', to: [-0.15, CUBE_Y + 0.003, 0.33], up, d: 1.2 },
        { a: 'release', cube: 'red', place: CUBE_HOME.red, d: 0.2 },
        { a: 'grip', side: 'right', to: 0.08, d: 0.4 },
        { a: 'rest', side: 'right', d: 1.0 },
        { a: 'wait', d: 1.0 },
      ],
    },
    idle: { loop: false, steps: [{ a: 'reset' }] },
  };

  // ---------- Szene ----------
  const World = {
    COL,
    restPose: null,
    sep: 0,
    sepTarget: 0,
    ready: false,
  };

  World.init = function (canvas, opts) {
    opts = opts || {};
    World.opts = opts;
    const renderer = new T.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    renderer.toneMapping = T.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.setClearColor(COL.bg, 1);
    World.renderer = renderer;

    const scene = new T.Scene();
    scene.fog = new T.FogExp2(COL.bg, 0.075);
    World.scene = scene;
    const camera = new T.PerspectiveCamera(38, 16 / 9, 0.05, 80);
    World.camera = camera;

    scene.add(new T.HemisphereLight(0xdfe8f2, 0x1a2230, 1.25));
    const key = new T.DirectionalLight(0xffffff, 1.6);
    key.position.set(1.5, 4, 4);
    scene.add(key);
    const rimA = new T.DirectionalLight(COL.amber, 1.4);
    rimA.position.set(-4, 2.5, -2.5);
    scene.add(rimA);
    const rimC = new T.DirectionalLight(COL.cyan, 1.4);
    rimC.position.set(4, 2.5, -2.5);
    scene.add(rimC);

    // Boden: Gitter + Farbhälften
    const grid = new T.GridHelper(40, 80, 0x2a3a4c, 0x18222e);
    grid.material.transparent = true;
    grid.material.opacity = 0.8;
    scene.add(grid);
    const halves = new T.Mesh(
      new T.PlaneGeometry(24, 16),
      new T.MeshBasicMaterial({
        map: canvasTex(512, 8, (g, w) => {
          const gr = g.createLinearGradient(0, 0, w, 0);
          gr.addColorStop(0, 'rgba(245,165,36,0)');
          gr.addColorStop(0.3, 'rgba(245,165,36,0.16)');
          gr.addColorStop(0.497, 'rgba(245,165,36,0.05)');
          gr.addColorStop(0.503, 'rgba(34,211,238,0.05)');
          gr.addColorStop(0.7, 'rgba(34,211,238,0.16)');
          gr.addColorStop(1, 'rgba(34,211,238,0)');
          g.fillStyle = gr;
          g.fillRect(0, 0, w, 8);
        }),
        transparent: true,
        depthWrite: false,
        blending: T.AdditiveBlending,
        opacity: 0,
      })
    );
    halves.rotation.x = -Math.PI / 2;
    halves.position.y = 0.002;
    scene.add(halves);
    World.halves = halves;

    // Naht zwischen den Welten
    const seam = new T.Mesh(
      new T.PlaneGeometry(0.03, 3.2),
      new T.MeshBasicMaterial({
        map: canvasTex(4, 256, (g, w, h) => {
          const gr = g.createLinearGradient(0, h, 0, 0);
          gr.addColorStop(0, 'rgba(255,255,255,0.9)');
          gr.addColorStop(1, 'rgba(255,255,255,0)');
          g.fillStyle = gr;
          g.fillRect(0, 0, w, h);
        }),
        transparent: true,
        depthWrite: false,
        blending: T.AdditiveBlending,
        opacity: 0,
      })
    );
    seam.position.set(0, 1.6, 0.3);
    scene.add(seam);
    const seamFloor = new T.Mesh(
      new T.PlaneGeometry(0.02, 12),
      new T.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, depthWrite: false })
    );
    seamFloor.rotation.x = -Math.PI / 2;
    seamFloor.position.set(0, 0.003, 0);
    scene.add(seamFloor);
    World.seam = [seam, seamFloor];

    World.restPose = computeRestPose();
    World.real = createRig('real', World.restPose);
    World.sim = createRig('sim', World.restPose);
    scene.add(World.real.group, World.sim.group);
    World.sim.group.visible = false;

    buildMonitors();
    buildPipeline();
    buildBars();
    buildColumns();
    buildScreens();
    buildShiftMarkers();

    World.camPos = new T.Vector3(3, 2, 4);
    World.camTgt = new T.Vector3(0, 1, 0);
    World.ready = true;
  };

  // ---------- Bausteine der Stationen ----------
  function framedImage(key, w, color) {
    const img = new Image();
    const tex = new T.Texture(img);
    tex.colorSpace = T.SRGBColorSpace;
    img.onload = () => {
      tex.needsUpdate = true;
    };
    img.src = window.PRES_IMAGES[key];
    const g = new T.Group();
    const frame = new T.Mesh(
      new T.PlaneGeometry(w + 0.02, w * 0.75 + 0.02),
      new T.MeshBasicMaterial({ color, transparent: true })
    );
    const pic = new T.Mesh(new T.PlaneGeometry(w, w * 0.75), new T.MeshBasicMaterial({ map: tex, transparent: true, toneMapped: false }));
    pic.position.z = 0.002;
    g.add(frame, pic);
    g.userData.mats = [frame.material, pic.material];
    g.userData.tex = tex;
    return g;
  }

  function setOpacity(obj, o) {
    obj.visible = o > 0.001;
    obj.traverse((n) => {
      if (n.material) {
        n.material.transparent = true;
        n.material.opacity = o * (n.userData.baseOpacity !== undefined ? n.userData.baseOpacity : 1);
      }
    });
  }

  function buildMonitors() {
    // Vier Kamerabilder des Datensatzes, an Kopf und Handgelenken des echten Zwillings.
    const defs = [
      ['cam_left_high', 'Kopf links', [0.3, 1.62, 0.12], [0.05, 1.27, 0.08]],
      ['cam_right_high', 'Kopf rechts', [-0.3, 1.62, 0.12], [-0.05, 1.27, 0.08]],
      ['cam_left_wrist', 'Handgelenk links', [0.55, 1.08, 0.32], [0.24, 0.98, 0.25]],
      ['cam_right_wrist', 'Handgelenk rechts', [-0.55, 1.08, 0.32], [-0.24, 0.98, 0.25]],
    ];
    const g = new T.Group();
    for (const [key, text, pos, src] of defs) {
      const m = framedImage(key, 0.32, COL.amber);
      m.position.fromArray(pos);
      m.rotation.y = pos[0] > 0 ? 0.25 : -0.25;
      const lab = makeLabel(text, { color: '#f5c46b', size: 0.035 });
      lab.position.set(0, -0.15, 0.01);
      m.add(lab);
      g.add(m);
      const line = new T.Line(
        new T.BufferGeometry().setFromPoints([v3(pos).add(new T.Vector3(0, -0.12, 0)), v3(src)]),
        new T.LineBasicMaterial({ color: COL.amber, transparent: true, opacity: 0.6 })
      );
      line.userData.baseOpacity = 0.6;
      g.add(line);
    }
    World.real.group.add(g);
    World.monitors = g;
    setOpacity(g, 0);
  }

  function buildPipeline() {
    const pts = [
      [-4.6, 1.2], [-3.0, 2.5], [-1.0, 3.0], [1.0, 2.95], [2.9, 2.35], [4.3, 1.1],
    ].map(([x, z]) => new T.Vector3(x, 0.012, z));
    const curve = new T.CatmullRomCurve3(pts);
    const N = 240;
    const pos = [], colr = [], idx = [];
    const ca = new T.Color(COL.amber), cc = new T.Color(COL.cyan), tmp = new T.Color();
    for (let i = 0; i <= N; i++) {
      const t = i / N;
      const p = curve.getPointAt(t);
      const tg = curve.getTangentAt(t);
      const n = new T.Vector3(-tg.z, 0, tg.x).normalize().multiplyScalar(0.07);
      pos.push(p.x + n.x, p.y, p.z + n.z, p.x - n.x, p.y, p.z - n.z);
      tmp.copy(ca).lerp(cc, t);
      colr.push(tmp.r, tmp.g, tmp.b, tmp.r, tmp.g, tmp.b);
      if (i < N) {
        const k = i * 2;
        idx.push(k, k + 1, k + 2, k + 1, k + 3, k + 2);
      }
    }
    const geo = new T.BufferGeometry();
    geo.setAttribute('position', new T.Float32BufferAttribute(pos, 3));
    geo.setAttribute('color', new T.Float32BufferAttribute(colr, 3));
    geo.setIndex(idx);
    const ribbon = new T.Mesh(
      geo,
      new T.MeshBasicMaterial({ vertexColors: true, transparent: true, side: T.DoubleSide, depthWrite: false, toneMapped: false })
    );
    const g = new T.Group();
    g.add(ribbon);
    const nodes = [
      [0.04, ['Download', 'Hugging Face · LeRobot v3.0']],
      [0.25, ['Konvertierung', 'v3.0 → v2.1']],
      [0.48, ['Fine-Tuning im Docker', 'lokal · vast.ai · KISSKI (Apptainer)']],
      [0.72, ['Checkpoint', 'Hugging Face']],
      [0.96, ['Closed-Loop-Eval', 'Isaac Lab · IKR-Server']],
    ];
    World.pipeNodes = [];
    for (const [t, text] of nodes) {
      const p = curve.getPointAt(t);
      tmp.copy(ca).lerp(cc, t);
      const n = new T.Group();
      n.position.copy(p);
      const post = new T.Mesh(
        new T.CylinderGeometry(0.05, 0.05, 0.7, 12),
        new T.MeshBasicMaterial({ color: tmp.clone(), toneMapped: false, transparent: true })
      );
      post.position.y = 0.35;
      const ring = new T.Mesh(
        new T.RingGeometry(0.16, 0.2, 32),
        new T.MeshBasicMaterial({ color: tmp.clone(), toneMapped: false, transparent: true, side: T.DoubleSide })
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.014;
      const lab = makeLabel(text, { color: '#' + tmp.getHexString(), size: 0.14 });
      lab.position.y = 1.0;
      n.add(post, ring, lab);
      g.add(n);
      World.pipeNodes.push({ group: n, t });
    }
    // Portal in die Simulation
    const portal = new T.Mesh(
      new T.TorusGeometry(0.62, 0.025, 8, 64),
      new T.MeshBasicMaterial({ color: COL.cyan, toneMapped: false, transparent: true })
    );
    portal.position.set(4.75, 0.9, 0.4);
    portal.rotation.y = -0.9;
    g.add(portal);
    World.scene.add(g);
    World.pipeline = { group: g, ribbon, N, reveal: 0, portal };
    setPipeReveal(0);
    setOpacity(g, 0);
  }

  function setPipeReveal(r) {
    const P = World.pipeline;
    P.reveal = r;
    P.ribbon.geometry.setDrawRange(0, Math.floor(P.N * r) * 6);
    for (const n of World.pipeNodes) {
      const on = r >= n.t - 0.01;
      n.group.scale.setScalar(on ? 1 : 0.0001);
    }
    P.portal.scale.setScalar(r > 0.97 ? 1 : 0.0001);
  }

  function bar(color, w) {
    const m = new T.Mesh(
      new T.BoxGeometry(w, 1, w),
      new T.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35, roughness: 0.5, transparent: true })
    );
    m.geometry.translate(0, 0.5, 0);
    return m;
  }

  // Station 9: Hand-Fehler Sim-Bild / Echtbild (Läufe 65/66, Lauf 70)
  const BAR_UNIT = 0.034;
  function buildBars() {
    const g = new T.Group();
    const defs = [['N1.6', 13], ['N1.7', 36]];
    World.bars = [];
    defs.forEach(([name, f], i) => {
      const x = i * 0.55;
      const real = bar(COL.amber, 0.16);
      real.position.set(x - 0.1, 0, 0);
      const sim = bar(COL.cyan, 0.16);
      sim.position.set(x + 0.1, 0, 0);
      const lab = makeLabel('×' + f, { color: hex(COL.cyan), size: 0.12 });
      const labR = makeLabel('1', { color: hex(COL.amber), size: 0.08 });
      const base = makeLabel(name, { color: '#e8edf2', size: 0.09 });
      base.position.set(x, -0.09, 0.15);
      g.add(real, sim, lab, labR, base);
      World.bars.push({ real, sim, lab, labR, x, f });
    });
    g.position.set(0.95, 0, 0.35);
    World.sim.group.add(g);
    World.barGroup = g;
    setBarValues([0, 0], 0);
    setOpacity(g, 0);
  }

  function setBarValues(fs, realOn) {
    World.bars.forEach((b, i) => {
      const f = Math.max(fs[i], 0.0001);
      b.sim.scale.y = f * BAR_UNIT;
      b.real.scale.y = Math.max(realOn, 0.0001) * BAR_UNIT;
      b.lab.position.set(b.x + 0.1, f * BAR_UNIT + 0.09, 0);
      b.labR.position.set(b.x - 0.1, realOn * BAR_UNIT + 0.07, 0);
    });
  }

  // Station 12: Stufen ohne vs. mit Co-Training (closed_loop.csv, Läufe 82 und 88+97)
  function buildColumns() {
    const g = new T.Group();
    World.columns = [];
    const stages = ['erreicht', 'berührt', 'angehoben', 'gestapelt'];
    stages.forEach((s, i) => {
      const x = (i - 1.5) * 0.5;
      const a = bar(0x5f7186, 0.15);
      a.position.set(x - 0.1, 0, 0);
      const b = bar(COL.cyan, 0.15);
      b.position.set(x + 0.1, 0, 0);
      const la = makeLabel('', { color: '#aab6c3', size: 0.05 });
      const lb = makeLabel('', { color: hex(COL.cyan), size: 0.05 });
      const base = makeLabel(s, { color: '#e8edf2', size: 0.065 });
      base.position.set(x, 0.035, 0.2);
      g.add(a, b, la, lb, base);
      World.columns.push({ a, b, la, lb, x });
    });
    g.position.set(0, 0, 1.6);
    World.scene.add(g);
    World.columnGroup = g;
    setColumns(null, 0);
    setOpacity(g, 0);
  }

  function setColumns(data, u) {
    World.columns.forEach((c, i) => {
      const fa = data ? data.a[i] / data.na : 0;
      const fb = data ? data.b[i] / data.nb : 0;
      const H = 0.85;
      c.a.scale.y = Math.max(fa * H * u, 0.004);
      c.b.scale.y = Math.max(fb * H * u, 0.004);
      if (data) {
        c.la.setText(`${data.a[i]}/${data.na}`);
        c.lb.setText(`${data.b[i]}/${data.nb}`);
      }
      c.la.position.set(c.x - 0.1, fa * H * u + 0.05, 0);
      c.lb.position.set(c.x + 0.1, fb * H * u + 0.05, 0);
    });
  }

  function buildScreens() {
    // Station 11: Standbild des ersten Griffs am Sim-Zwilling
    const griff = framedImage('griff_crop', 0.78, COL.cyan);
    griff.position.set(1.05, 1.32, 0.35);
    griff.rotation.y = -0.35;
    const lab = makeLabel(['Lauf 88 · Ep. 5 · t = 80 s', 'Standbild aus der Simulation'], { color: hex(COL.cyan), size: 0.05 });
    lab.position.set(0, -0.36, 0.01);
    griff.add(lab);
    World.sim.group.add(griff);
    World.griffScreen = griff;
    setOpacity(griff, 0);

    // Station 13: Cosmos – Original und Variante, Überblendung im Shader (ein Draw-Call)
    const load = (k) => {
      const img = new Image();
      const t = new T.Texture(img);
      t.colorSpace = T.SRGBColorSpace;
      img.onload = () => (t.needsUpdate = true);
      img.src = window.PRES_IMAGES[k];
      return t;
    };
    const mat = new T.ShaderMaterial({
      transparent: true,
      uniforms: { a: { value: load('cosmos_orig') }, b: { value: load('cosmos_var') }, mixv: { value: 0 }, opacity: { value: 1 } },
      vertexShader: 'varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
      fragmentShader:
        'uniform sampler2D a; uniform sampler2D b; uniform float mixv; uniform float opacity; varying vec2 vUv;' +
        'void main(){ vec4 c=mix(texture2D(a,vUv),texture2D(b,vUv),mixv); gl_FragColor=vec4(c.rgb,opacity);\n' +
        '#include <colorspace_fragment>\n}',
    });
    const cos = new T.Group();
    const frame = new T.Mesh(new T.PlaneGeometry(0.82, 0.62), new T.MeshBasicMaterial({ color: COL.amber, transparent: true }));
    const pic = new T.Mesh(new T.PlaneGeometry(0.8, 0.6), mat);
    pic.position.z = 0.002;
    cos.add(frame, pic);
    cos.position.set(0.72, 1.5, 0.3);
    cos.rotation.y = -0.15;
    const cl = makeLabel('Original', { color: '#f5c46b', size: 0.045 });
    cl.position.set(0, -0.37, 0.01);
    cos.add(cl);
    World.real.group.add(cos);
    World.cosmos = { group: cos, mat, label: cl, frame };
    setCosmosOpacity(0);
  }

  function setCosmosOpacity(o) {
    const c = World.cosmos;
    c.group.visible = o > 0.001;
    c.mat.uniforms.opacity.value = o;
    c.frame.material.opacity = o;
    c.label.material.opacity = o;
  }

  function buildShiftMarkers() {
    // Station 8: Geisterwürfel an der alten Lage + Beschriftung „3 cm“
    const g = new T.Group();
    const mk = (x) => {
      const m = new T.Mesh(
        new T.BoxGeometry(CUBE, CUBE, CUBE),
        new T.MeshBasicMaterial({ color: COL.cyan, transparent: true, opacity: 0.25, depthWrite: false })
      );
      m.userData.baseOpacity = 0.25;
      m.position.set(x, CUBE_Y, 0.33);
      const e = new T.LineSegments(
        new T.EdgesGeometry(m.geometry),
        new T.LineBasicMaterial({ color: COL.cyan, transparent: true })
      );
      m.add(e);
      return m;
    };
    g.add(mk(0.15), mk(-0.15));
    const l1 = makeLabel('3 cm →', { color: hex(COL.cyan), size: 0.028 });
    l1.position.set(0.17, CUBE_Y + 0.055, 0.36);
    const l2 = makeLabel('← 3 cm', { color: hex(COL.cyan), size: 0.028 });
    l2.position.set(-0.17, CUBE_Y + 0.055, 0.36);
    for (const l of [l1, l2]) l.material.depthTest = false;
    g.add(l1, l2);
    World.sim.group.add(g);
    World.shiftMarkers = g;
    setOpacity(g, 0);
  }

  // ---------- Zustände ----------
  const fades = new Map(); // key -> aktuelle Deckkraft
  function fadeTo(key, obj, to, dur, setter) {
    const from = fades.has(key) ? fades.get(key) : 0;
    if (from === to && dur > 0) return;
    tween('fade:' + key, from, to, dur, (v) => {
      fades.set(key, v);
      (setter || ((x) => setOpacity(obj, x)))(v);
    });
  }

  const COLUMN_DATA = {
    a: [16, 17, 0, 0], na: 20, // Lauf 82: N1.6 vision-v2 @44000, ohne Co-Training
    b: [58, 60, 2, 0], nb: 60, // Läufe 88+97: N1.6 cube4 @44000, mit Co-Training
  };
  World.COLUMN_DATA = COLUMN_DATA;

  /**
   * Zustand einer Station anwenden. s: { sep, show: [..], anim: {real, sim}, bars, cosmos,
   * tableWood, cam: {pos, target}, orbit }. instant: ohne Übergang (Sprung, reduzierte Bewegung).
   */
  World.apply = function (s, instant) {
    const D = instant ? 0 : 1.6;
    const show = new Set(s.show || []);
    // Trennung der Welten
    const sep = s.sep || 0;
    if (sep > 0) World.sim.group.visible = true;
    tween('sep', World.sep, sep, instant ? 0 : 2.2, (v) => {
      World.sep = v;
      World.real.group.position.x = -v / 2;
      World.sim.group.position.x = v / 2;
      World.sim.group.visible = v > 0.02;
    });
    const split = sep > 0 ? 1 : 0;
    fadeTo('halves', null, split, D, (v) => (World.halves.material.opacity = v));
    fadeTo('seam', null, split && s.seam ? 1 : 0, D, (v) => (World.seam[0].material.opacity = v * 0.9));
    fadeTo('seamFloor', null, split, D, (v) => (World.seam[1].material.opacity = v * 0.35));
    fadeTo('glow', null, split, D, (v) => {
      World.real.glow.material.opacity = v;
      World.sim.glow.material.opacity = v;
    });
    fadeTo('labels', null, split && s.labels ? 1 : 0, D, (v) => {
      World.real.label.material.opacity = v;
      World.sim.label.material.opacity = v;
    });

    fadeTo('monitors', World.monitors, show.has('monitors') ? 1 : 0, D);
    fadeTo('shift', World.shiftMarkers, show.has('shift') ? 1 : 0, D);
    fadeTo('griff', World.griffScreen, show.has('griff') ? 1 : 0, D);
    fadeTo('cosmos', null, show.has('cosmos') ? 1 : 0, D, setCosmosOpacity);
    const cm = s.cosmos || 0;
    tween('cosmosMix', World.cosmos.mat.uniforms.mixv.value, cm, instant ? 0 : 1.4, (v) => {
      World.cosmos.mat.uniforms.mixv.value = v;
    });
    World.cosmos.label.setText(cm > 0.5 ? 'Cosmos-Variante (Licht, Holztisch)' : 'Original (Datensatz)');
    const wood = new T.Color(0x9a6a42), white = new T.Color(0xe6e2da);
    tween('wood', World.real.tableMat.userData.w || 0, s.tableWood ? 1 : 0, instant ? 0 : 1.4, (v) => {
      World.real.tableMat.userData.w = v;
      World.real.tableMat.color.copy(white).lerp(wood, v);
    });

    // Pipeline
    const pipe = s.pipe === undefined ? -1 : s.pipe; // -1 aus, 0..1 sichtbar bis, 'dim'
    const nodesOn = pipe !== -1 && !s.pipeDim;
    World.pipeNodes.forEach((n) => (n.group.visible = nodesOn));
    World.pipeline.portal.visible = nodesOn;
    const ribbonMax = s.pipeDim ? 0.3 : 1;
    fadeTo('pipe', World.pipeline.group, pipe === -1 ? 0 : ribbonMax, D, (v) => {
      World.pipeline.group.visible = v > 0.001;
      World.pipeline.ribbon.material.opacity = v;
      World.pipeNodes.forEach((n) => n.group.traverse((o) => o.material && (o.material.opacity = Math.min(1, v))));
      World.pipeline.portal.material.opacity = Math.min(1, v);
    });
    if (pipe >= 0) {
      tween('pipeReveal', World.pipeline.reveal, pipe, instant ? 0 : 2.0, setPipeReveal, easeOut);
    }

    // Balken (Domain-Gap)
    const bars = s.bars || null;
    fadeTo('bars', World.barGroup, bars ? 1 : 0, D);
    const cur = World.bars.map((b) => b.sim.scale.y / BAR_UNIT);
    const target = bars || [0, 0];
    World.bars.forEach((b, i) => b.lab.setText('×' + (bars ? bars[i] : b.f)));
    tween('barsGrow', 0, 1, instant ? 0 : 1.8, (u) => {
      setBarValues(cur.map((c, i) => c + (target[i] - c) * u), bars ? 1 : 0);
    }, easeOut);

    // Säulen
    fadeTo('columns', World.columnGroup, show.has('columns') ? 1 : 0, D);
    if (show.has('columns')) tween('colGrow', 0, 1, instant ? 0 : 1.8, (u) => setColumns(COLUMN_DATA, u), easeOut);

    // Animationen
    const anim = s.anim || {};
    for (const k of ['real', 'sim']) {
      const name = anim[k] || 'idle';
      const r = World[k];
      if (r.anim.scriptName !== name || s.restart) r.anim.play(name);
    }

    // Kamera
    World.orbit = s.orbit || null;
    if (s.cam) {
      const p0 = World.camPos.clone(), t0 = World.camTgt.clone();
      const p1 = v3(s.cam.pos), t1 = v3(s.cam.target);
      World.camFov = s.cam.fov || 38;
      const sh0 = World.shift || 0, sh1 = s.cam.shift || 0;
      tween('cam', 0, 1, instant ? 0 : s.camDur || 2.2, (u) => {
        World.camPos.copy(p0).lerp(p1, u);
        World.camTgt.copy(t0).lerp(t1, u);
        World.shift = sh0 + (sh1 - sh0) * u;
      });
    } else if (s.orbit) {
      const sh0 = World.shift || 0, sh1 = s.orbit.shift || 0;
      tween('cam', 0, 1, instant ? 0 : 1.5, (u) => (World.shift = sh0 + (sh1 - sh0) * u));
    }
  };

  World.isSettled = function () {
    return tweens.size === 0;
  };

  let time = 0;
  World.update = function (dt, skipRender) {
    time += dt;
    updateTweens(dt);
    World.real.anim.update(dt);
    if (World.sim.group.visible) World.sim.anim.update(dt);
    const cam = World.camera;
    if (World.orbit && !World.opts.still) {
      const o = World.orbit;
      const a = o.phase + time * o.speed;
      cam.position.set(Math.sin(a) * o.r + o.c[0], o.h, Math.cos(a) * o.r + o.c[2]);
      cam.lookAt(o.c[0], o.c[1], o.c[2]);
    } else if (World.orbit) {
      const o = World.orbit;
      cam.position.set(Math.sin(o.phase) * o.r + o.c[0], o.h, Math.cos(o.phase) * o.r + o.c[2]);
      cam.lookAt(o.c[0], o.c[1], o.c[2]);
    } else {
      cam.position.copy(World.camPos);
      if (!World.opts.still) {
        // ruhiges Atmen der Kamera
        cam.position.x += Math.sin(time * 0.31) * 0.025;
        cam.position.y += Math.sin(time * 0.23) * 0.015;
      }
      cam.lookAt(World.camTgt);
    }
    // Motiv seitlich versetzen (Text links): Bildausschnitt verschieben statt Kamera drehen
    const sh = World.shift || 0;
    if (Math.abs(sh) > 0.0005) cam.setViewOffset(1920, 1080, -sh * 1920, 0, 1920, 1080);
    else if (cam.view) cam.clearViewOffset();
    if (cam.fov !== (World.camFov || 38)) {
      cam.fov += ((World.camFov || 38) - cam.fov) * Math.min(1, dt * 3);
      cam.updateProjectionMatrix();
    }
    if (!skipRender) World.renderer.render(World.scene, cam);
  };

  World.resize = function (pr) {
    World.renderer.setPixelRatio(pr);
    World.renderer.setSize(1920, 1080, false);
    World.camera.aspect = 16 / 9;
    World.camera.updateProjectionMatrix();
  };

  window.PresWorld = World;
})();
