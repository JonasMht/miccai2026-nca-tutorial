// The planner, run in the browser. The NCA itself executes as WebGL2 shaders on
// the viewer's GPU: an edit reruns the K-step rollout in a few milliseconds,
// with no round trip to Python.
//
// State layout on the GPU: C channels packed four per RGBA32F tile, tiles side
// by side, so a 16-channel state is one 512x128 texture. A 3x3 convolution is
// then, per output tile, 9 taps x (input tiles) 4x4 matrix-vector products.
// Weights arrive pre-packed in exactly that order (see webplanner.py).
//
// `P` (the payload) and `root` (the container) are injected by webplanner.py.

(function (root, P) {
  "use strict";
  const N = P.grid, DX = P.dx_mm, VIEW = 512, S = VIEW / N;

  // ---------------------------------------------------------------- decoding
  function bytes(b64) {
    const s = atob(b64), u = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    return u;
  }
  const f32 = (b64) => new Float32Array(bytes(b64).buffer);
  function bits(b64) {  // np.packbits, most significant bit first
    const p = bytes(b64), u = new Uint8Array(p.length * 8);
    for (let i = 0; i < u.length; i++) u[i] = (p[i >> 3] >> (7 - (i & 7))) & 1;
    return u;
  }


  // ---------------------------------------------------------------- layout
  // Explicit colours throughout: the card must read the same in a light and a
  // dark notebook theme.
  const MAXN = P.max_needles;
  root.innerHTML = `
  <div class="np-card">
    <div class="np-left">
      <div class="np-stage">
        <canvas class="np-gl" width="${VIEW}" height="${VIEW}"></canvas>
        <canvas class="np-ui" width="${VIEW}" height="${VIEW}"></canvas>
      </div>
      <div class="np-legend">
        <span><i style="background:#f06a14"></i>necrosis</span>
        <span><i style="background:#33d6ff"></i>vessels, predicted</span>
        <span><i class="ring" style="border-color:#ffb81a"></i>vessels, true</span>
        <span><i style="background:#39d98a"></i>radiating slot</span>
      </div>
      <div class="np-hint">drag the <b>white</b> dot to move a needle, the <b>grey</b> dot to turn it ·
        double-click to place the second needle · scroll to change power</div>
    </div>
    <div class="np-side">
      <div class="np-grid">
        <span>model</span><select class="np-model"></select>
        <span>patient</span><select class="np-slice"></select>
      </div>
      <div class="np-h">needles <span class="np-dim">(up to ${MAXN})</span></div>
      <div class="np-needles"></div>
      <div class="np-h">result</div>
      <div class="np-read"></div>
      <div class="np-h">view</div>
      <div class="np-grid">
        <span>show</span><select class="np-view"></select>
      </div>
      <label class="np-check"><input class="np-truth" type="checkbox" checked> outline the true vessels</label>
      <div class="np-row">
        <button class="np-replay" title="clear the state and watch the ${""}automaton grow the answer">&#9654; replay growth</button>
        <button class="np-erase" title="wipe part of the state with the mouse; the automaton repairs it">eraser</button>
      </div>
      <label class="np-check" title="off: every edit restarts the rollout from the seed. on: the state is carried across the edit, which shows what persistence training taught it">
        <input class="np-keep" type="checkbox"> carry the state across edits <span class="np-dim">(experiment)</span></label>
    </div>
  </div>
  <style>
    .np-card{display:flex;gap:22px;flex-wrap:wrap;padding:16px;border-radius:12px;background:#15171c;
      color:#e9ecef;font:13px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;max-width:900px}
    .np-card *{box-sizing:border-box}
    .np-stage{position:relative;width:${VIEW}px;height:${VIEW}px;max-width:100%}
    .np-stage canvas{position:absolute;left:0;top:0;border-radius:8px;width:100%}
    .np-ui{cursor:crosshair}
    .np-legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:9px;color:#ced4da;font-size:12px}
    .np-legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
    .np-legend i.ring{background:transparent;border:2px solid}
    .np-hint{color:#8d949c;font-size:11.5px;margin-top:4px;max-width:${VIEW}px}
    .np-side{width:310px;display:flex;flex-direction:column;gap:9px}
    .np-grid{display:grid;grid-template-columns:62px 1fr;gap:6px 8px;align-items:center}
    .np-grid span{color:#adb5bd}
    .np-card select{background:#23262d;color:#e9ecef;border:1px solid #3a3f48;border-radius:6px;padding:4px 6px;font:inherit}
    .np-h{margin-top:6px;font-weight:600;color:#f8f9fa;border-bottom:1px solid #2c3038;padding-bottom:3px}
    .np-dim{color:#8d949c;font-weight:400}
    .np-needles{display:flex;flex-direction:column;gap:6px}
    .np-n{display:grid;grid-template-columns:22px 1fr 58px 24px;align-items:center;gap:6px;padding:5px 6px;
      border-radius:8px;background:#1d2026;border:1px solid transparent;cursor:pointer}
    .np-n.sel{border-color:#5c7cfa}
    .np-n .dot{width:18px;height:18px;border-radius:50%;background:#fff;color:#15171c;font-weight:700;
      font-size:11px;display:flex;align-items:center;justify-content:center}
    .np-n .w{text-align:right;font-variant-numeric:tabular-nums}
    .np-n .w.hot{color:#ff8787}
    .np-n input[type=range]{width:100%;accent-color:#f06a14}
    .np-card button{background:#23262d;color:#e9ecef;border:1px solid #3a3f48;border-radius:6px;
      padding:5px 10px;cursor:pointer;font:inherit}
    .np-card button:hover{background:#2c3038}
    .np-card button.on{background:#5c3a14;border-color:#f06a14}
    .np-n button{padding:0;width:22px;height:22px;line-height:1}
    .np-add{border-style:dashed !important;color:#adb5bd !important}
    .np-row{display:flex;gap:8px;flex-wrap:wrap}
    .np-check{display:flex;gap:7px;align-items:center;color:#ced4da;cursor:pointer}
    .np-read{background:#1d2026;border-radius:8px;padding:9px 11px;font-variant-numeric:tabular-nums;line-height:1.6}
    .np-read .big{font-size:22px;font-weight:700;color:#fff}
    .np-read .note{color:#ff8787;font-size:12px}
    .np-fail{padding:14px;background:#2b1a17;color:#ffc9c9;border:1px solid #7a3b2e;border-radius:8px}
  </style>`;
  const $ = (c) => root.querySelector(c);
  const glCanvas = $(".np-gl"), ui = $(".np-ui"), ctx = ui.getContext("2d");

  const gl = glCanvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true });
  if (!gl || !gl.getExtension("EXT_color_buffer_float")) {
    root.innerHTML = `<div class="np-fail">This browser has no WebGL2 with float
      render targets, so the live planner cannot run here. Everything else in the
      notebook works; the <code>AblationStudio</code> cell below is the
      slider-based fallback.</div>`;
    return;
  }

  // ---------------------------------------------------------------- GL helpers
  function shader(src) {
    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, `#version 300 es
      in vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }`);
    gl.compileShader(vs);
    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, "#version 300 es\nprecision highp float;\nprecision highp int;\n"
                        + "precision highp sampler2D;\n" + src);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS))
      throw new Error(gl.getShaderInfoLog(fs));
    const prog = gl.createProgram();
    gl.attachShader(prog, vs); gl.attachShader(prog, fs);
    gl.bindAttribLocation(prog, 0, "p");
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(prog));
    const loc = {};
    return {
      prog,
      use(uniforms) {
        gl.useProgram(prog);
        let unit = 0;
        for (const [k, v] of Object.entries(uniforms)) {
          if (!(k in loc)) loc[k] = gl.getUniformLocation(prog, k);
          if (loc[k] === null) continue;
          if (v && v.tex) {
            gl.activeTexture(gl.TEXTURE0 + unit);
            gl.bindTexture(gl.TEXTURE_2D, v.tex);
            gl.uniform1i(loc[k], unit++);
          } else if (Array.isArray(v)) gl[`uniform${v.length}f`](loc[k], ...v);
          else if (Number.isInteger(v) && !k.startsWith("f_")) gl.uniform1i(loc[k], v);
          else gl.uniform1f(loc[k], v);
        }
      },
    };
  }

  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

  function texture(w, h, data = null) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, data);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER])
      gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T])
      gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { tex, fbo, w, h };
  }
  // A packed weight array becomes a texture 1024 texels wide.
  function weights(arr) {
    const n = arr.length / 4, w = Math.min(1024, n), h = Math.ceil(n / w);
    const pad = new Float32Array(w * h * 4);
    pad.set(arr);
    return texture(w, h, pad);
  }
  function draw(target, prog, uniforms) {
    prog.use(uniforms);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fbo : null);
    if (target) gl.viewport(0, 0, target.w, target.h);
    else gl.viewport(0, 0, VIEW, VIEW);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // ---------------------------------------------------------------- shaders
  // One 3x3 convolution: out tile t = sum over in tiles and taps of W * in.
  const CONV = `
    uniform sampler2D uIn, uW, uB;
    uniform int uInTiles, uN, uWW, uRelu, uBias;
    out vec4 o;
    vec4 W(int i) { return texelFetch(uW, ivec2(i % uWW, i / uWW), 0); }
    void main() {
      ivec2 q = ivec2(gl_FragCoord.xy);
      int t = q.x / uN, x = q.x - t * uN, y = q.y;
      vec4 acc = uBias == 1 ? texelFetch(uB, ivec2(t, 0), 0) : vec4(0.0);
      for (int it = 0; it < uInTiles; ++it)
        for (int ky = 0; ky < 3; ++ky) {
          int yy = y + ky - 1;
          if (yy < 0 || yy >= uN) continue;
          for (int kx = 0; kx < 3; ++kx) {
            int xx = x + kx - 1;
            if (xx < 0 || xx >= uN) continue;
            vec4 v = texelFetch(uIn, ivec2(it * uN + xx, yy), 0);
            int b = ((t * uInTiles + it) * 9 + ky * 3 + kx) * 4;
            acc += vec4(dot(W(b), v), dot(W(b + 1), v), dot(W(b + 2), v), dot(W(b + 3), v));
          }
        }
      o = uRelu == 1 ? max(acc, 0.0) : acc;
    }`;
  // The residual update: stochastic firing per cell, soft clamp, and on the last
  // sub-model of a step the environment channels (2, 3, 4) are written back.
  const UPDATE = `
    uniform sampler2D uX, uD, uInput, uHold;
    uniform int uN, uRestore, uHoldOn;
    uniform float f_fire;
    uniform uint uSeed;
    out vec4 o;
    uint pcg(uint v) {
      uint s = v * 747796405u + 2891336453u;
      uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
      return (w >> 22u) ^ w;
    }
    void main() {
      ivec2 q = ivec2(gl_FragCoord.xy);
      int t = q.x / uN, x = q.x - t * uN, y = q.y;
      // top 24 bits: exact in a float, so r < 1 always holds
      float r = float(pcg(uint(x) ^ pcg(uint(y) ^ pcg(uSeed))) >> 8) / 16777216.0;
      float fire = r < f_fire ? 1.0 : 0.0;
      vec4 v = 4.0 * tanh((texelFetch(uX, q, 0) + fire * texelFetch(uD, q, 0)) / 4.0);
      if (uRestore == 1) {
        vec4 e = texelFetch(uInput, ivec2(x, y), 0);
        if (t == 0) { v.z = e.r; v.w = e.g; }
        if (t == 1) { v.x = e.b; }
        if (t == 0 && uHoldOn == 1) v.y = texelFetch(uHold, ivec2(x, y), 0).y;   // vessels, fixed
      }
      o = v;
    }`;
  // Seed: zeros everywhere but the environment.
  const SEED = `
    uniform sampler2D uInput;
    uniform int uN;
    out vec4 o;
    void main() {
      ivec2 q = ivec2(gl_FragCoord.xy);
      int t = q.x / uN, x = q.x - t * uN;
      vec4 e = texelFetch(uInput, ivec2(x, q.y), 0);
      o = t == 0 ? vec4(0.0, 0.0, e.r, e.g) : (t == 1 ? vec4(e.b, 0.0, 0.0, 0.0) : vec4(0.0));
    }`;
  // Eraser: zero the state inside a disk. The environment comes back next step.
  const ERASE = `
    uniform sampler2D uX;
    uniform int uN;
    uniform vec3 uDisk;
    out vec4 o;
    void main() {
      ivec2 q = ivec2(gl_FragCoord.xy);
      int t = q.x / uN, x = q.x - t * uN;
      vec2 d = vec2(float(x), float(q.y)) + 0.5 - uDisk.xy;
      o = dot(d, d) < uDisk.z * uDisk.z ? vec4(0.0) : texelFetch(uX, q, 0);
    }`;
  // Display: CT, then necrosis ramp or vessels or one raw channel on top.
  const DISPLAY = `
    uniform sampler2D uX, uCT, uTruth;
    uniform int uN, uView, uChan, uTruthOn;
    uniform float f_vthr;
    out vec4 o;
    float chan(ivec2 c, int k) {
      vec4 v = texelFetch(uX, ivec2((k / 4) * uN + c.x, c.y), 0);
      int j = k - (k / 4) * 4;
      return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w;
    }
    vec2 ans(ivec2 c) {
      c = clamp(c, ivec2(0), ivec2(uN - 1));
      return 1.0 / (1.0 + exp(-vec2(chan(c, 0), chan(c, 1))));
    }
    // bilinear, by hand: float textures are not filterable everywhere
    vec2 ansAt(vec2 g) {
      vec2 f = g - 0.5, i = floor(f), w = f - i;
      ivec2 c = ivec2(i);
      return mix(mix(ans(c), ans(c + ivec2(1, 0)), w.x),
                 mix(ans(c + ivec2(0, 1)), ans(c + ivec2(1, 1)), w.x), w.y);
    }
    float truthAt(vec2 g) {
      vec2 f = g - 0.5, i = floor(f), w = f - i;
      ivec2 c = ivec2(i), m = ivec2(uN - 1);
      #define T(d) texelFetch(uTruth, clamp(c + d, ivec2(0), m), 0).r
      return mix(mix(T(ivec2(0, 0)), T(ivec2(1, 0)), w.x), mix(T(ivec2(0, 1)), T(ivec2(1, 1)), w.x), w.y);
    }
    vec3 necrosis(float p) {
      vec3 a = vec3(0.99, 0.80, 0.25), b = vec3(0.94, 0.42, 0.08), c = vec3(0.62, 0.05, 0.10);
      return p < 0.55 ? mix(a, b, (p - 0.1) / 0.45) : mix(b, c, (p - 0.55) / 0.45);
    }
    void main() {
      vec2 g = vec2(gl_FragCoord.x, float(${VIEW}) - gl_FragCoord.y) / float(${S});
      ivec2 c = ivec2(g);
      vec4 ct = texelFetch(uCT, c, 0);
      float hu = ct.r * 2000.0 - 1000.0;
      float grey = clamp((hu - 20.0) / 200.0, 0.0, 1.0) * (ct.g > 0.5 ? 1.0 : 0.45);
      vec3 col = vec3(grey);
      vec2 a = ansAt(g);
      if (uView == 0 || uView == 1) {
        float p = a.x;
        float al = smoothstep(0.1, 0.3, p) * 0.85;
        col = mix(col, necrosis(p), al);
        float edge = 1.0 - smoothstep(0.0, max(1.5 * fwidth(p), 1e-3), abs(p - 0.5));
        col = mix(col, vec3(1.0, 0.95, 0.85), edge * 0.9);
      }
      if (uView == 0 || uView == 2) {
        float v = a.y;
        float line = 1.0 - smoothstep(0.0, max(2.2 * fwidth(v), 1e-3), abs(v - f_vthr));
        col = mix(col, vec3(0.2, 0.85, 1.0), v > f_vthr ? 0.4 : 0.0);
        col = mix(col, vec3(0.55, 0.95, 1.0), line);          // outline on top, visible over a lesion
      }
      if (uView == 3) {
        float v = clamp(chan(c, uChan) / 4.0, -1.0, 1.0);
        vec3 d = v > 0.0 ? mix(vec3(1.0), vec3(0.80, 0.15, 0.12), v)
                         : mix(vec3(1.0), vec3(0.13, 0.35, 0.80), -v);
        col = mix(col, d, 0.85);
      }
      if (uTruthOn == 1 && uView != 3) {
        float t = truthAt(g);
        float line = 1.0 - smoothstep(0.0, max(1.2 * fwidth(t), 1e-3), abs(t - 0.5));
        col = mix(col, vec3(1.0, 0.72, 0.05), line * 0.9);
      }
      o = vec4(col, 1.0);
    }`;
  const P_CONV = shader(CONV), P_UPDATE = shader(UPDATE), P_SEED = shader(SEED),
        P_ERASE = shader(ERASE), P_DISPLAY = shader(DISPLAY);

  // ---------------------------------------------------------------- models
  const models = P.models.map((m) => {
    const T = Math.ceil(m.channels / 4), HT = Math.ceil(m.hidden / 4);
    const flat = f32(m.weights);
    let off = 0;
    const take = (n) => flat.subarray(off, (off += n));
    const subs = [];
    for (let s = 0; s < m.n_sub_models; s++) {
      subs.push({
        w1: weights(take(HT * T * 9 * 16)),
        b1: weights(take(HT * 4)),
        w2: weights(take(T * HT * 9 * 16)),
      });
    }
    return { ...m, T, HT, subs };
  });

  // ---------------------------------------------------------------- slices
  const slices = P.slices.map((s) => {
    const ct = bytes(s.ct), organ = bits(s.organ), vessel = bits(s.vessel);
    const img = new Float32Array(N * N * 4), truth = new Float32Array(N * N * 4);
    for (let i = 0; i < N * N; i++) {
      img[4 * i] = ct[i] / 255; img[4 * i + 1] = organ[i];
      truth[4 * i] = vessel[i];
    }
    return { ...s, ct, organ, vessel, ctTex: texture(N, N, img), truthTex: texture(N, N, truth) };
  });

  // ---------------------------------------------------------------- state
  let model = models[0], slice = slices[0];
  let X, Y, H;
  const input = texture(N, N), ctOnly = texture(N, N);
  let anatomy = null, anatomyValid = false;    // phase 1 result, per patient and model
  let needles = [], selected = 0, stepCount = 0, target = 0, frameSeed = 1;
  let erasing = false, carry = 0, viewMode = 0, viewChan = 0;
  const ANIM_STEPS_PER_S = 5;         // replay and repair are slowed down to be watched

  function allocState() {
    for (const t of [X, Y, H]) if (t) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fbo); }
    if (anatomy) { gl.deleteTexture(anatomy.tex); gl.deleteFramebuffer(anatomy.fbo); }
    X = texture(N * model.T, N); Y = texture(N * model.T, N); anatomy = texture(N * model.T, N);
    anatomyValid = false;
    H = texture(N * model.HT, N);
  }

  // -- painting the plan, same geometry as channels.paint_plan
  function segDist(px, py, ax, ay, bx, by) {
    const abx = bx - ax, aby = by - ay, L2 = abx * abx + aby * aby;
    let t = L2 < 1e-9 ? 0 : ((px - ax) * abx + (py - ay) * aby) / L2;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (ax + t * abx), py - (ay + t * aby));
  }
  function base(n) {
    const u = [Math.cos(n.angle), Math.sin(n.angle)], fov = N * DX;
    let t = 120;
    for (const [o, d] of [[n.x, u[0]], [n.y, u[1]]])
      if (Math.abs(d) > 1e-9) t = Math.min(t, ((d > 0 ? fov : 0) - o) / d);
    t = Math.max(t, 25);
    return [n.x + u[0] * t, n.y + u[1] * t];
  }
  function slotEnds(n) {
    const u = [Math.cos(n.angle), Math.sin(n.angle)], d = P.device;
    return [[n.x + u[0] * d.slot_start_mm, n.y + u[1] * d.slot_start_mm],
            [n.x + u[0] * d.slot_end_mm, n.y + u[1] * d.slot_end_mm]];
  }
  function paint() {
    const data = new Float32Array(N * N * 4);
    const r = Math.max(0.5 * P.device.diameter_mm, 0.5 * DX);
    for (let i = 0; i < N * N; i++)
      data[4 * i] = P.mask_ct ? (slice.organ[i] ? slice.ct[i] / 255 : 0) : slice.ct[i] / 255;
    for (const n of needles) {
      const [bx, by] = base(n), [[s0x, s0y], [s1x, s1y]] = slotEnds(n);
      const act = n.power / P.device.max_power_w;
      let best = Infinity, bestI = -1, any = false;
      const dslot = new Float32Array(N * N);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const i = y * N + x, px = (x + 0.5) * DX, py = (y + 0.5) * DX;
        if (segDist(px, py, n.x, n.y, bx, by) <= r) data[4 * i + 1] = 1;
        const ds = segDist(px, py, s0x, s0y, s1x, s1y);
        dslot[i] = ds;
        if (ds <= r) { data[4 * i + 2] = Math.max(data[4 * i + 2], act); any = true; }
        if (ds < best) { best = ds; bestI = i; }
      }
      if (!any) for (let i = 0; i < N * N; i++)
        if (dslot[i] <= best + 1e-6) data[4 * i + 2] = Math.max(data[4 * i + 2], act);
    }
    gl.bindTexture(gl.TEXTURE_2D, input.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, gl.RGBA, gl.FLOAT, data);
    const ct = new Float32Array(N * N * 4);
    for (let i = 0; i < N * N; i++) ct[4 * i] = data[4 * i];
    gl.bindTexture(gl.TEXTURE_2D, ctOnly.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, gl.RGBA, gl.FLOAT, ct);
  }

  // ---------------------------------------------------------------- the NCA
  // Phase 1 (models with anatomy_steps > 0): the CT alone, no needle, so the
  // vessels cannot depend on the plan. Phase 2: the planned steps, vessels held.
  function copyInto(dst, src) {           // the eraser shader with a zero radius is a copy
    draw(dst, P_ERASE, { uX: src, uN: N, uDisk: [0, 0, 0] });
  }
  function ensureAnatomy() {
    if (anatomyValid || !model.anatomy_steps) return;
    draw(X, P_SEED, { uInput: ctOnly, uN: N });
    for (let i = 0; i < model.anatomy_steps; i++) step(ctOnly, false);
    copyInto(anatomy, X);
    anatomyValid = true;
  }
  function reseed() {
    if (model.anatomy_steps) { ensureAnatomy(); copyInto(X, anatomy); }
    else draw(X, P_SEED, { uInput: input, uN: N });
    stepCount = 0;
  }
  let scratch = null;
  function scratchFor(like) {
    if (!scratch || scratch.w !== like.w) scratch = texture(like.w, like.h);
    return scratch;
  }
  const seedLoc = gl.getUniformLocation(P_UPDATE.prog, "uSeed");
  function step(env = input, hold = model.anatomy_steps > 0) {
    const last = model.subs.length - 1;
    model.subs.forEach((sub, s) => {
      draw(H, P_CONV, { uIn: X, uW: sub.w1, uB: sub.b1, uInTiles: model.T, uN: N,
                        uWW: sub.w1.w, uRelu: 1, uBias: 1 });
      draw(Y, P_CONV, { uIn: H, uW: sub.w2, uB: sub.b1, uInTiles: model.HT, uN: N,
                        uWW: sub.w2.w, uRelu: 0, uBias: 0 });
      const out = scratchFor(X);
      P_UPDATE.use({ uX: X, uD: Y, uInput: env, uHold: anatomy, uN: N, uRestore: s === last ? 1 : 0,
                     uHoldOn: hold ? 1 : 0, f_fire: model.fire_rate });
      gl.uniform1ui(seedLoc, Math.imul(frameSeed++, 2654435761) >>> 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, out.fbo);
      gl.viewport(0, 0, out.w, out.h);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      [X, scratch] = [scratch, X];
    });
    stepCount++;
  }

  // ---------------------------------------------------------------- readout
  const readBuf = new Float32Array(N * N * 4);
  function readAnswers() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, X.fbo);
    gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, readBuf);       // tile 0: the two answers (logits)
    let area = 0, tp = 0, fp = 0, fn = 0;
    const thr = model.vessel_threshold;
    for (let i = 0; i < N * N; i++) {
      if (readBuf[4 * i] > 0) area++;                          // sigmoid > 0.5
      const p = readBuf[4 * i + 1] > Math.log(thr / (1 - thr)) && slice.organ[i], t = slice.vessel[i];
      if (p && t) tp++; else if (p) fp++; else if (t) fn++;
    }
    return { area: area * DX * DX / 100, f1: 2 * tp / Math.max(1, 2 * tp + fp + fn) };
  }

  // ---------------------------------------------------------------- drawing
  function render() {
    // during phase 1 the necrosis channel is only scratch: show the vessels alone
    const view = phase1 > 0 && viewMode < 2 ? 2 : viewMode;
    draw(null, P_DISPLAY, { uX: X, uCT: slice.ctTex, uTruth: slice.truthTex, uN: N,
                            uView: view, uChan: viewChan,
                            uTruthOn: $(".np-truth").checked ? 1 : 0, f_vthr: model.vessel_threshold });
    ctx.clearRect(0, 0, VIEW, VIEW);
    const px = (x, y) => [x / DX * S, y / DX * S];
    needles.forEach((n, i) => {
      const [bx, by] = base(n), [s0, s1] = slotEnds(n);
      const [tx, ty] = px(n.x, n.y), [qx, qy] = px(bx, by), h = handleOf(n);
      ctx.lineCap = "round";
      ctx.strokeStyle = i === selected ? "rgba(255,255,255,0.95)" : "rgba(220,220,228,0.65)";
      ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(tx, ty); ctx.stroke();
      ctx.strokeStyle = "#39d98a"; ctx.lineWidth = 5; ctx.beginPath();
      ctx.moveTo(...px(...s0)); ctx.lineTo(...px(...s1)); ctx.stroke();
      ctx.fillStyle = "#9aa3ad"; ctx.beginPath(); ctx.arc(h[0], h[1], 7, 0, 7); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(tx, ty, 8, 0, 7); ctx.fill();
      ctx.strokeStyle = i === selected ? "#5c7cfa" : "#333"; ctx.lineWidth = 2; ctx.stroke();
      ctx.font = "bold 11px system-ui"; ctx.fillStyle = "#15171c"; ctx.textAlign = "center";
      ctx.fillText(String(i + 1), tx, ty + 4);
      ctx.textAlign = "left"; ctx.font = "bold 12px system-ui"; ctx.fillStyle = "#fff";
      ctx.fillText(`${n.power} W`, tx + 12, ty - 9);
    });
    if (erasing && mouse) {
      ctx.strokeStyle = "rgba(255,120,120,0.9)"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(mouse[0], mouse[1], ERASE_R * S, 0, 7); ctx.stroke();
    }
  }
  function handleOf(n) {  // the rotation handle, 45 mm up the shaft
    const d = 45, u = [Math.cos(n.angle), Math.sin(n.angle)];
    return [(n.x + u[0] * d) / DX * S, (n.y + u[1] * d) / DX * S];
  }

  // ---------------------------------------------------------------- controls
  function needlePanel() {
    const box = $(".np-needles");
    box.innerHTML = needles.map((n, i) => `
      <div class="np-n${i === selected ? " sel" : ""}" data-i="${i}">
        <span class="dot">${i + 1}</span>
        <input type="range" min="0" max="${P.power_max_w}" step="5" value="${n.power}">
        <span class="w${n.power > P.device.max_power_w ? " hot" : ""}">${n.power} W</span>
        <button title="remove this needle">&#10005;</button>
      </div>`).join("")
      + (needles.length < MAXN ? `<button class="np-add">+ add a needle</button>` : "");
    box.querySelectorAll(".np-n").forEach((row) => {
      const i = +row.dataset.i;
      row.onmousedown = () => { if (selected !== i) { selected = i; needlePanel(); } };
      row.querySelector("input").oninput = (e) => {
        needles[i].power = +e.target.value;
        const w = row.querySelector(".w");
        w.textContent = `${needles[i].power} W`;
        w.classList.toggle("hot", needles[i].power > P.device.max_power_w);
        changed(false);
      };
      row.querySelector("button").onclick = (e) => {
        e.stopPropagation();
        needles.splice(i, 1); selected = Math.max(0, Math.min(selected, needles.length - 1));
        changed();
      };
    });
    const add = box.querySelector(".np-add");
    if (add) add.onclick = () => addNeedle();
  }
  function addNeedle(x, y) {
    if (needles.length >= MAXN) return;
    const a = needles[0];
    if (x === undefined) {                // beside the first needle, or the default spot
      x = a ? a.x + 18 : slice.tip[0]; y = a ? a.y + 6 : slice.tip[1];
    }
    needles.push({ x, y, angle: -Math.PI / 2, power: 75 });
    selected = needles.length - 1;
    changed();
  }
  let dirty = false;
  function changed(rebuildPanel = true) { paint(); if (rebuildPanel) needlePanel(); dirty = true; }

  function loadSlice(i) {
    slice = slices[i];
    anatomyValid = false;
    needles = [{ x: slice.tip[0], y: slice.tip[1], angle: slice.angle, power: 75 }];
    selected = 0;
    changed();
  }
  function loadModel(i) {
    model = models[i]; allocState(); scratch = null;
    const sel = $(".np-view"), keep = sel.value;
    sel.innerHTML = `<option value="0">necrosis and vessels</option>
      <option value="1">necrosis only</option><option value="2">vessels only</option>`
      + Array.from({ length: model.channels }, (_, k) =>
        `<option value="c${k}">state channel ${k}${model.roles[k] ? " (" + model.roles[k] + ")" : " (scratch)"}</option>`).join("");
    sel.value = [...sel.options].some((o) => o.value === keep) ? keep : "0";
    setView(sel.value);
    changed();
  }
  function setView(v) {
    if (v[0] === "c") { viewMode = 3; viewChan = +v.slice(1); }
    else viewMode = +v;
  }

  $(".np-model").innerHTML = models.map((m, i) => `<option value="${i}">${m.name}</option>`).join("");
  $(".np-slice").innerHTML = slices.map((s, i) => `<option value="${i}">${s.name}</option>`).join("");
  $(".np-model").onchange = (e) => loadModel(+e.target.value);
  $(".np-slice").onchange = (e) => loadSlice(+e.target.value);
  $(".np-view").onchange = (e) => setView(e.target.value);
  let phase1 = 0;                         // phase-1 steps still to animate during a replay
  $(".np-replay").onclick = () => {
    carry = 0;
    if (model.anatomy_steps) {
      draw(X, P_SEED, { uInput: ctOnly, uN: N });
      phase1 = model.anatomy_steps; stepCount = 0; target = model.steps;
    } else { reseed(); target = model.steps; }
  };
  $(".np-erase").onclick = (e) => { erasing = !erasing; e.target.classList.toggle("on", erasing); };

  // -- mouse
  const ERASE_R = 7;
  let drag = null, mouse = null;
  const toMM = (e) => {
    const r = ui.getBoundingClientRect();
    const cx = (e.clientX - r.left) * VIEW / r.width, cy = (e.clientY - r.top) * VIEW / r.height;
    return { cx, cy, x: cx / S * DX, y: cy / S * DX };
  };
  function eraseAt(m) {
    draw(scratchFor(X), P_ERASE, { uX: X, uN: N, uDisk: [m.cx / S, m.cy / S, ERASE_R] });
    [X, scratch] = [scratch, X];
    target = stepCount + model.steps;      // then let it repair, slowly enough to watch
  }
  function hit(m) {
    for (let i = needles.length - 1; i >= 0; i--) {
      const n = needles[i], h = handleOf(n);
      if (Math.hypot(m.cx - n.x / DX * S, m.cy - n.y / DX * S) < 12) return { i, move: true };
      if (Math.hypot(m.cx - h[0], m.cy - h[1]) < 12) return { i, move: false };
    }
    return null;
  }
  ui.addEventListener("mousedown", (e) => {
    const m = toMM(e);
    if (erasing) { drag = { erase: true }; eraseAt(m); return; }
    drag = hit(m);
    if (drag) { selected = drag.i; needlePanel(); }
  });
  window.addEventListener("mousemove", (e) => {
    const m = toMM(e);
    mouse = [m.cx, m.cy];
    if (!drag) {
      if (!erasing) { const h = hit(m); ui.style.cursor = h ? (h.move ? "move" : "grab") : "crosshair"; }
      return;
    }
    if (drag.erase) { eraseAt(m); return; }
    const n = needles[drag.i], fov = N * DX;
    if (drag.move) {
      n.x = Math.max(4, Math.min(fov - 4, m.x)); n.y = Math.max(4, Math.min(fov - 4, m.y));
    } else n.angle = Math.atan2(m.y - n.y, m.x - n.x);
    changed(false);
  });
  window.addEventListener("mouseup", () => { drag = null; });
  ui.addEventListener("mouseleave", () => { mouse = null; });
  ui.addEventListener("dblclick", (e) => {
    if (erasing) return;
    const m = toMM(e);
    addNeedle(m.x, m.y);
  });
  ui.addEventListener("wheel", (e) => {
    const n = needles[selected];
    if (!n) return;
    e.preventDefault();
    n.power = Math.max(0, Math.min(P.power_max_w, n.power + (e.deltaY < 0 ? 5 : -5)));
    changed();
  }, { passive: false });

  // ---------------------------------------------------------------- main loop
  // The automaton is run for exactly K steps (chosen on validation) and then
  // stops. An edit jumps straight to the K-step answer; replay and repair are
  // played out at a watchable speed.
  const read = $(".np-read");
  let last = performance.now(), alive = true, lastRead = { area: 0, f1: 0 }, readDue = true;
  function frame(now) {
    if (!alive) return;
    if (!root.isConnected) { alive = false; return; }
    if (root.offsetParent === null) { last = now; requestAnimationFrame(frame); return; }  // hidden
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (dirty) {
      if (!$(".np-keep").checked) reseed();
      target = stepCount + model.steps;
      while (stepCount < target) step();
      dirty = false; readDue = true;
    }
    if (phase1 > 0 || stepCount < target) {
      carry += ANIM_STEPS_PER_S * dt;
      while (carry >= 1 && (phase1 > 0 || stepCount < target)) {
        if (phase1 > 0) {
          step(ctOnly, false);
          if (--phase1 === 0) { copyInto(anatomy, X); anatomyValid = true; stepCount = 0; }
        } else step();
        carry -= 1; readDue = true;
      }
    } else carry = 0;
    render();
    if (readDue) { lastRead = readAnswers(); readDue = false; }
    const done = stepCount >= target;
    const hot = needles.some((n) => n.power > P.device.max_power_w);
    read.innerHTML = `
      <div>necrosis <span class="big">${phase1 > 0 ? "…" : lastRead.area.toFixed(1) + " cm²"}</span></div>
      <div>vessel F1 on this slice <b>${lastRead.f1.toFixed(2)}</b>
        <span class="np-dim">· best HU threshold ${slice.hu_f1.toFixed(2)}</span></div>
      <div class="np-dim">${phase1 > 0 ? `reading the anatomy: step ${model.anatomy_steps - phase1} of ${model.anatomy_steps} …`
        : done ? (model.anatomy_steps ? `${model.anatomy_steps} anatomy steps, then ${stepCount} heating steps`
                                      : `${stepCount} steps`) + ` (K = ${model.steps})`
               : `heating: step ${stepCount} of ${target} …`} · ${model.params.toLocaleString()} parameters</div>`
      + (hot ? `<div class="note">above ${P.device.max_power_w} W: outside what the model was trained on</div>` : "")
      + (needles.length === 0 ? `<div class="note">no needle: double-click on the liver</div>` : "");
    requestAnimationFrame(frame);
  }

  // ---------------------------------------------------------------- test hook
  // Used by tests/test_webplanner.py through a headless browser: run k steps
  // with a given fire rate and hand back the full state.
  root.nca = {
    run(k, fire) {
      const f = model.fire_rate;
      if (fire !== undefined) model.fire_rate = fire;
      for (let i = 0; i < k; i++) step();
      target = stepCount;
      model.fire_rate = f;
      const out = new Float32Array(N * model.T * N * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, X.fbo);
      gl.readPixels(0, 0, N * model.T, N, gl.RGBA, gl.FLOAT, out);
      return Array.from(out);
    },
    input() {
      const out = new Float32Array(N * N * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, input.fbo);
      gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, out);
      return Array.from(out);
    },
    setNeedles(list) { needles = list; paint(); anatomyValid = false; reseed(); dirty = false; target = 0; },
    pause() { dirty = false; target = stepCount; },
    state() { return { stepCount, target, needles: needles.length }; },
    reseed,
  };

  loadModel(0);
  loadSlice(0);
  requestAnimationFrame(frame);
})(ROOT, PAYLOAD);
