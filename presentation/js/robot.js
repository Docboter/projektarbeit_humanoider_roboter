// TL;DR: Baut den Unitree G1 (+DEX3) aus assets/robot.js als Three.js-Gelenkbaum, mit Gelenk-
// winkeln, Handschluss und einer kleinen CCD-IK für die Arme. Kein Loader, keine Module.
(function () {
  'use strict';
  const T = THREE;
  let geomCache = null;

  function decode(b64, Ctor) {
    const bin = atob(b64);
    const buf = new ArrayBuffer(bin.length);
    const u8 = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Ctor(buf);
  }

  function geometries(data) {
    if (geomCache) return geomCache;
    geomCache = data.meshes.map((m) => {
      const q = decode(m.pos, Uint16Array);
      const pos = new Float32Array(q.length);
      for (let i = 0; i < q.length; i++) pos[i] = m.min[i % 3] + q[i] * m.scale[i % 3];
      const g = new T.BufferGeometry();
      g.setAttribute('position', new T.BufferAttribute(pos, 3));
      g.setIndex(new T.BufferAttribute(decode(m.idx, m.i32 ? Uint32Array : Uint16Array), 1));
      g.computeVertexNormals();
      g.computeBoundingSphere();
      return g;
    });
    return geomCache;
  }

  function rpyQuat(rpy) {
    // URDF: fixe Achsen, Reihenfolge roll (x), pitch (y), yaw (z) -> Euler 'ZYX'
    return new T.Quaternion().setFromEuler(new T.Euler(rpy[0], rpy[1], rpy[2], 'ZYX'));
  }

  const DARK = /hand_(palm|thumb|index|middle)|wrist_yaw/;

  /**
   * opts.body / opts.dark: Materialien für Gehäuse und Hände.
   * Rückgabe: { root, joints, set(name, a), grip(side, g), solveIK(side, target, iters), ... }
   */
  function createRobot(opts) {
    const data = window.G1_ROBOT;
    const geos = geometries(data);
    const links = {};
    for (const name in data.links) {
      const o = new T.Object3D();
      o.name = name;
      const l = data.links[name];
      if (l.mesh !== undefined) {
        const mat = DARK.test(name) ? opts.dark : opts.body;
        const mesh = new T.Mesh(geos[l.mesh], mat);
        mesh.position.fromArray(l.xyz);
        mesh.quaternion.copy(rpyQuat(l.rpy));
        mesh.matrixAutoUpdate = false;
        mesh.updateMatrix();
        o.add(mesh);
      }
      links[name] = o;
    }
    const joints = {};
    for (const j of data.joints) {
      const frame = new T.Object3D();
      frame.position.fromArray(j.xyz);
      frame.quaternion.copy(rpyQuat(j.rpy));
      links[j.parent].add(frame);
      frame.add(links[j.child]);
      joints[j.name] = {
        name: j.name,
        type: j.type,
        frame,
        link: links[j.child],
        axis: new T.Vector3().fromArray(j.axis).normalize(),
        lower: j.lower,
        upper: j.upper,
        angle: 0,
      };
    }
    // URDF ist z-oben, x vorne; das Wurzelobjekt dreht in Three.js (y oben).
    const urdf = new T.Group();
    urdf.rotation.x = -Math.PI / 2;
    urdf.add(links[data.root]);
    const root = new T.Group();
    root.add(urdf);

    function set(name, a) {
      const j = joints[name];
      if (!j || j.type === 'fixed') return;
      a = Math.min(j.upper, Math.max(j.lower, a));
      j.angle = a;
      j.link.quaternion.setFromAxisAngle(j.axis, a);
    }
    function setPose(pose) {
      for (const k in pose) set(k, pose[k]);
    }

    // Handschluss g in [0,1]; Vorzeichen aus den URDF-Grenzen (links/rechts gespiegelt).
    function grip(side, g) {
      const p = side + '_hand_';
      const s = side === 'left' ? 1 : -1;
      set(p + 'index_0_joint', -s * 1.05 * g);
      set(p + 'index_1_joint', -s * 1.0 * g);
      set(p + 'middle_0_joint', -s * 1.05 * g);
      set(p + 'middle_1_joint', -s * 1.0 * g);
      set(p + 'thumb_0_joint', 0);
      set(p + 'thumb_1_joint', s * 0.35 * g);
      set(p + 'thumb_2_joint', s * 1.0 * g);
    }

    const ARM = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll',
      'wrist_pitch', 'wrist_yaw'];
    // Effektor-Punkte im Handflächen-Frame: Greifmitte und ein Punkt auf dem Handrücken.
    // Fingerspitzen zeigen +x, die Handfläche links nach -y, rechts nach +y.
    const effLocal = {
      left: [new T.Vector3(0.085, -0.035, 0), new T.Vector3(0.06, 0.07, 0)],
      right: [new T.Vector3(0.085, 0.035, 0), new T.Vector3(0.06, -0.07, 0)],
    };
    const _w = new T.Vector3(), _p = new T.Vector3(), _a = new T.Vector3(), _b = new T.Vector3();
    const _axis = new T.Vector3(), _q = new T.Quaternion(), _c = new T.Vector3();

    function effWorld(side, i, out) {
      const palm = links[side + '_hand_palm_link'];
      return out.copy(effLocal[side][i]).applyMatrix4(palm.matrixWorld);
    }

    /** Mehrpunkt-CCD: target = Greifpunkt (Welt), up = Richtung, in die der Handrücken zeigt. */
    function solveIK(side, target, up, iters, weightUp) {
      const wUp = weightUp === undefined ? 0.35 : weightUp;
      const tUp = _c.copy(target).addScaledVector(up, 0.08);
      const targets = [target, tUp.clone()];
      const weights = [1, wUp];
      root.updateMatrixWorld(true);
      for (let it = 0; it < (iters || 6); it++) {
        for (let k = ARM.length - 1; k >= 0; k--) {
          const j = joints[side + '_' + ARM[k] + '_joint'];
          j.frame.getWorldPosition(_p);
          _axis.copy(j.axis).applyQuaternion(j.frame.getWorldQuaternion(_q)).normalize();
          let sx = 0, sy = 0;
          for (let e = 0; e < 2; e++) {
            effWorld(side, e, _w).sub(_p);
            _a.copy(_w).addScaledVector(_axis, -_w.dot(_axis));
            _b.copy(targets[e]).sub(_p);
            _b.addScaledVector(_axis, -_b.dot(_axis));
            sx += weights[e] * _a.dot(_b);
            sy += weights[e] * _axis.dot(_w.copy(_a).cross(_b));
          }
          const d = Math.atan2(sy, sx);
          if (!isFinite(d)) continue;
          set(j.name, j.angle + d * 0.8);
          j.link.updateMatrixWorld(true);
        }
      }
      return effWorld(side, 0, new T.Vector3()).distanceTo(target);
    }

    function palmPoint(side) {
      root.updateMatrixWorld(true);
      return effWorld(side, 0, new T.Vector3());
    }

    function lowestPoint() {
      root.updateMatrixWorld(true);
      const box = new T.Box3().setFromObject(root);
      return box.min.y;
    }

    function getPose() {
      const p = {};
      for (const k in joints) if (joints[k].type !== 'fixed') p[k] = joints[k].angle;
      return p;
    }

    return { root, joints, links, set, setPose, getPose, grip, solveIK, palmPoint, lowestPoint };
  }

  window.PresRobot = { createRobot };
})();
