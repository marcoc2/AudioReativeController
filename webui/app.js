"use strict";
// ARC Studio: the page draws, Python decides. Every edit is a call to core/web (pywebview's
// js_api in its window, or POST /api in a browser) and the page redraws from the answer.
// The page never works out the music: bars, notes, the montage and every knob's value over
// the song come from core/. What Python does on its own (the loading steps, the montage, the
// engine) comes as events.

const TOKEN = new URLSearchParams(location.search).get("t") || "";
let inWebview = false;

async function call(name, ...args) {
  if (inWebview) return await window.pywebview.api[name](...args);
  const r = await fetch(`/api/${name}`, { method: "POST", headers: { "X-ARC-Token": TOKEN, "Content-Type": "application/json" }, body: JSON.stringify(args) });
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return r.json();
}
// a call whose error is told (toast and log) and answered as null
async function ask(name, ...args) {
  try {
    const r = await call(name, ...args);
    if (r && r.error) { toast(r.error, true); return null; }
    return r;
  } catch (e) { toast(`${name}: ${e.message}`, true); log("err", `${name}: ${e.message}`); return null; }
}

// ---------------------------------------------------------------------------
// State (what Python last said)
// ---------------------------------------------------------------------------
let S = null;               // the project as opened: song, tracks, catalog, clips
let members = [];           // the lanes' members
let pins = {};
let montage = [];           // [{t0, t1, name, url}]
let undoN = 0, redoN = 0;
let dirty = false, savedAt = "";
let B = [0, 1];             // where each bar starts, then where the last one ends
const NB = () => B.length - 1;
let catalogByKey = new Map();
let sel = null;             // {type: "member", id, k} | {type: "clip", bar}
let loadGen = 0;            // the opening the loading events belong to
let gpuBusy = [];           // Python's work on the GPU now (the live preview waits for it)
let steps = [];             // the loading steps, as Python tells them

// the panels in windows of their own (detachPanel): their documents are searched too
const outWins = {};            // "inspector" | "subpanel" -> its window
const outSecs = {};            // and the panel itself (it comes back even when its window just went)
const outDocs = () => Object.values(outWins).map(w => { try { return w.closed ? null : w.document; } catch (e) { return null; } }).filter(Boolean);
const $ = id => document.getElementById(id) || outDocs().map(d => d.getElementById(id)).find(Boolean) || null;
const $q = sel => document.querySelector(sel) || outDocs().map(d => d.querySelector(sel)).find(Boolean) || null;
let menuDoc = document;        // where the pointer last was: menus and notes open there
function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "value") el.value = v;
    else if (k === "checked") el.checked = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) el.append(c.nodeType ? c : document.createTextNode(String(c)));
  return el;
}
// an element's children, the null ones (a part not shown) left out
function fill(el, ...kids) { el.replaceChildren(...kids.filter(k => k != null && k !== false)); }
const fmt = v => { if (!isFinite(v)) return "0"; const s = Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3); return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s; };
const fmtTime = t => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
const idOf = ref => JSON.stringify(ref);
const stem = name => String(name).replace(/\.[^.]+$/, "");
let toastTimer = 0;
function toast(msg, bad = false) {
  const doc = menuDoc && menuDoc.defaultView && !menuDoc.defaultView.closed ? menuDoc : document;
  let el = doc.querySelector(".toast");
  if (!el) { el = h("div", { class: "toast", role: "status" }); doc.body.append(el); }
  el.textContent = msg; el.hidden = false; el.classList.toggle("bad", bad);
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { el.hidden = true; }, bad ? 4500 : 2200);
}

// the log: every line, filtered by kind; a traceback folds under the error before it
const LOG_MAX = 2000;
let logFilter = "all", errorsUnseen = 0;
const LOG_KINDS = { all: null, err: ["err", "trace"], warn: ["warn"], io: ["in", "out"], sys: ["sys"] };
function shownKind(kind) { const ks = LOG_KINDS[logFilter]; return !ks || ks.includes(kind); }
function log(kind, text) {
  const el = $("log");
  let row;
  if (kind === "trace") {
    const lines = String(text).trim().split("\n");
    row = h("details", { class: "trace" }, h("summary", { text: lines[lines.length - 1] || "traceback" }), h("pre", { text }));
  } else row = h("div", { class: kind, text: (kind === "out" ? "→ " : kind === "in" ? "← " : "  ") + text });
  row.dataset.kind = kind;
  row.hidden = !shownKind(kind);
  const atEnd = $("dockLog").scrollTop + $("dockLog").clientHeight >= $("dockLog").scrollHeight - 8;
  el.append(row);
  while (el.children.length > LOG_MAX) el.firstChild.remove();
  if (atEnd) $("dockLog").scrollTop = $("dockLog").scrollHeight;
  if (kind === "err" && $("dockLog").hidden) { errorsUnseen++; showErrBadge(); }
}
function showErrBadge() { $("errBadge").hidden = !errorsUnseen; $("errBadge").textContent = errorsUnseen; }
function setLogFilter(k) {
  logFilter = k;
  document.querySelectorAll("#logFilter button").forEach(b => b.setAttribute("aria-pressed", b.dataset.kind === k));
  for (const row of $("log").children) row.hidden = !shownKind(row.dataset.kind);
  $("dockLog").scrollTop = $("dockLog").scrollHeight;
}

// what Python tells the page (evaluate_js in the web view; polled in a browser)
const ON = {
  log: e => log(e.kind, e.text),
  load: e => onLoadStep(e),
  opened: st => onOpened(st),
  state: st => { if (S) later(() => apply(st)); },
  montage: segs => { montage = segs || []; streamEdited(); later(() => { renderTimeline(); if (sel && sel.type === "montage") renderInspector(); }); },
  knobs: () => { knobs.clear(); if (sel && sel.type === "member") renderInspector(); },
  stems: e => { stemWaves = (e && e.list) || []; if (S) later(renderTimeline); },
  covers: got => { for (const [clip, cover] of Object.entries(got || {})) thumbs.set(clip, cover); scheduleThumbRedraw(); },
  lengths: got => { Object.assign(clipLens, got || {}); scheduleThumbRedraw(); },
  dropped: e => onDropped(e),
  song: e => { if (S && e.url && e.url !== audioUrl) loadAudio(e.url); },
  engine: e => onEngine(e),
  proxies: got => { Object.assign(proxyOf, got || {}); },
  final: j => showFinal(j),
  demo: e => { if (entryKey === e.key) showEntryPanel(); },
  repaint: j => { if (j && j.settings) renderRepaint(); else showRepaint(j); },
  gpu: e => {
    gpuBusy = e.busy || [];
    $("gpuNote").hidden = !gpuBusy.length || view !== "sketch";
    $("gpuNote").textContent = gpuBusy.length ? `The GPU is busy (${gpuBusy.join(", ")}): the sketch waits` : "";
  },
};
window.arc = { event: (fn, data) => { try { if (ON[fn]) ON[fn](data); } catch (e) { console.error(fn, e); } } };
// what comes in the middle of a drag waits for it to end
let pendingAfterDrag = [];
function later(fn) { if (drag || rsel) pendingAfterDrag.push(fn); else fn(); }
let eventSeq = 0;
async function pollEvents() {            // one ask at a time: an event is never told twice
  for (; ;) {
    try {
      const r = await fetch(`/events?since=${eventSeq}`, { headers: { "X-ARC-Token": TOKEN } });
      for (const e of await r.json()) if (e.seq > eventSeq) { eventSeq = e.seq; window.arc.event(e.fn, e.data); }
    } catch (err) { /* the server went away */ }
    await new Promise(res => setTimeout(res, 250));
  }
}

// ---------------------------------------------------------------------------
// Bars and time
// ---------------------------------------------------------------------------
let ppb = 34;
function barAt(t) {             // 1-based
  let lo = 0, hi = NB() - 1, r = 0;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (B[m] <= t + 1e-6) { r = m; lo = m + 1; } else hi = m - 1; }
  return r + 1;
}
function tToX(t) { const i = barAt(t) - 1; const f = (t - B[i]) / (B[i + 1] - B[i]); return (i + Math.max(0, Math.min(1, f))) * ppb; }
function xToT(x) { const fb = x / ppb, i = Math.max(0, Math.min(NB() - 1, Math.floor(fb))); return B[i] + (fb - i) * (B[i + 1] - B[i]); }
const xToBar = x => Math.max(1, Math.min(NB(), Math.floor(x / ppb) + 1));

// ---------------------------------------------------------------------------
// Transport: the song in WebAudio (its clock is the clock)
// ---------------------------------------------------------------------------
let actx = null, audioBuf = null, src = null, playing = false, startCtx = 0, startPos = 0, startPerf = 0, pausedAt = 0;
let loop = null;               // [first bar, last bar]
let peaks = null, peaksHz = 200;
let stemWaves = [];            // [{name, hz, values}]: the stems with sound (core/web/load.py stem_waves)
let holding = false;            // playing, but the song waits for the engine's frames
function now() {
  if (!playing || holding) return pausedAt;
  if (audioBuf && actx) return Math.max(startPos, startPos + (actx.currentTime - startCtx));
  return startPos + (performance.now() / 1000 - startPerf);
}
function startSource() {
  startPos = pausedAt; startPerf = performance.now() / 1000;
  if (audioBuf && actx) {
    const how = hearing();                       // the song, or what the Lab's feature listens to
    src = actx.createBufferSource(); src.buffer = how.buf;
    let out = src;
    if (how.lo != null) for (const [type, f] of [["highpass", how.lo], ["highpass", how.lo], ["lowpass", how.hi], ["lowpass", how.hi]]) {
      const b = actx.createBiquadFilter(); b.type = type; b.frequency.value = f; out.connect(b); out = b;
    }
    out.connect(actx.destination);
    startCtx = actx.currentTime + 0.03;
    src.start(startCtx, Math.min(pausedAt, how.buf.duration - 0.01));
  }
}
function stopSource() { if (src) { try { src.stop(); } catch (e) { } src.disconnect(); src = null; } }
async function setPlaying(on) {
  if (on === playing) return;
  if (on) {
    if (actx && actx.state === "suspended") await actx.resume();
    playing = true;
    if (view === "live" && !streamHas(pausedAt)) { holding = true; streamAsk(pausedAt); }     // its frames first
    else startSource();
  } else { pausedAt = now(); playing = false; holding = false; stopSource(); }
  const pv = $("player").querySelector("video");              // the song plays: the clip's own sound goes off
  if (pv) pv.muted = playing || !clipSound;
  $("playIcon").innerHTML = playing ? '<path d="M3 2h3.5v12H3zM9.5 2H13v12H9.5z"/>' : '<path d="M4 2l10 6-10 6z"/>';
  $("bPlay").classList.toggle("on", playing);
}
function seek(t) {
  t = Math.max(0, Math.min((S ? S.song.duration : 1) - 0.05, t));
  if (playing && !holding) { stopSource(); pausedAt = t; startSource(); } else pausedAt = t;
  if (view === "live") streamAsk(t);
}
let audioUrl = null;
async function loadAudio(url) {
  audioUrl = url;
  const gen = loadGen;
  try {
    if (!actx) actx = new AudioContext();
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.arrayBuffer();
    const buf = await actx.decodeAudioData(data);
    if (gen !== loadGen || url !== audioUrl) return;          // another project opened meanwhile
    audioBuf = buf;
    const ch = audioBuf.getChannelData(0), n = Math.ceil(audioBuf.duration * peaksHz), step = audioBuf.sampleRate / peaksHz;
    peaks = new Float32Array(n);
    for (let i = 0; i < n; i++) { let m = 0; const a = Math.floor(i * step), b = Math.min(ch.length, Math.floor((i + 1) * step)); for (let j = a; j < b; j += 4) { const v = Math.abs(ch[j]); if (v > m) m = v; } peaks[i] = m; }
    if (playing) { stopSource(); pausedAt = now(); startSource(); }
    renderTimeline();
    const msg = `decoded: ${audioBuf.duration.toFixed(1)} s, ${audioBuf.sampleRate} Hz`;
    log("sys", `the song ${msg}`);
    call("page_step", "song", "done", msg).catch(() => { });
  } catch (e) {
    if (gen !== loadGen) return;
    log("err", `the song could not be decoded here: ${e.message}`);
    call("page_step", "song", "failed", `the page could not decode it: ${e.message}`).catch(() => { });
  }
}

// ---------------------------------------------------------------------------
// Knobs: values over the song, from Python (core.video.layers._knob)
// ---------------------------------------------------------------------------
const knobs = new Map();      // YAML text -> {values} | {error} | "pending"
function knobData(text) {
  if (!knobs.has(text)) {
    knobs.set(text, "pending");
    const gen = loadGen;
    call("knob", text).then(r => {
      if (gen !== loadGen || knobs.get(text) !== "pending") return;      // asked for another project, or cleared
      knobs.set(text, r); if (sel && sel.type === "member") renderInspector();
    }).catch(e => knobs.set(text, { error: e.message }));
  }
  return knobs.get(text);
}
function knobAt(text, t) {
  const k = knobData(text);
  if (!k || k === "pending" || !k.values) return null;
  const i = Math.max(0, Math.min(k.values.length - 1, Math.round(t * S.knob_hz)));
  return k.values[i];
}
function paramAt(p, t) {
  if (p.knob != null) { const v = knobAt(p.knob, t); return v == null ? p.default : v; }
  return p.v;
}

// ---------------------------------------------------------------------------
// WebGL2: the montage, BG clips and one-pass shaders, live
// ---------------------------------------------------------------------------
const canvas = $("gl");
const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false });
let W = 640, H = 360, TEX = {}, blackTex = null, frameNo = 0;
const VS = `#version 300 es
void main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;
// what core/shadertoy.py declares, in GLSL ES
const HEAD = `#version 300 es
precision highp float;
precision highp int;
uniform vec3 iResolution;
uniform float iTime, iTimeDelta, iFrameRate, iSampleRate;
uniform int iFrame;
uniform vec4 iMouse, iDate;
uniform vec3 iChannelResolution[4];
uniform float iChannelTime[4];
uniform sampler2D iChannel0, iChannel1, iChannel2, iChannel3;
out vec4 arc_out;
`;
const FOOT = `
void main() { vec4 c = vec4(0.0, 0.0, 0.0, 1.0); mainImage(c, gl_FragCoord.xy); arc_out = vec4(c.rgb, 1.0); }`;
// our own passes that carry an alpha (the characters' layer)
const FOOT_ALPHA = `
void main() { vec4 c = vec4(0.0); mainImage(c, gl_FragCoord.xy); arc_out = c; }`;
const OWN_ALPHA = {
  // a character: the clip (its alpha the cutout) put at u_rect (0..1 of the frame, y down), over what is there
  place: `uniform vec4 u_rect; uniform float u_flip; uniform float u_opacity;
void mainImage(out vec4 o, in vec2 fc) {
  vec2 uv = fc / iResolution.xy;
  vec4 below = texture(iChannel0, uv);
  vec2 p = vec2(uv.x, 1.0 - uv.y);
  vec2 q = (p - u_rect.xy) / max(u_rect.zw - u_rect.xy, vec2(1e-6));
  if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) { o = below; return; }
  if (u_flip > 0.5) q.x = 1.0 - q.x;
  vec4 c = texture(iChannel1, q);
  float a = c.a * u_opacity;
  o = vec4(mix(below.rgb, c.rgb, a), a + below.a * (1.0 - a));
}`,
  // an FX's picture with the alpha it was given (an FX on the characters keeps their figures)
  keepalpha: `void mainImage(out vec4 o, in vec2 fc) { vec2 uv = fc / iResolution.xy; o = vec4(texture(iChannel0, uv).rgb, texture(iChannel1, uv).a); }`,
};
const OWN = {
  copy: `void mainImage(out vec4 o, in vec2 fc) { o = vec4(texture(iChannel0, fc / iResolution.xy).rgb, 1.0); }`,
  // a video frame, cover or contain, top row first
  fit: `uniform float u_aspect; uniform float u_cover;
void mainImage(out vec4 o, in vec2 fc) {
  vec2 uv = fc / iResolution.xy; float A = iResolution.x / iResolution.y;
  bool wide = u_aspect > A;
  if (u_cover > 0.5) { if (wide) uv.x = (uv.x - 0.5) * A / u_aspect + 0.5; else uv.y = (uv.y - 0.5) * u_aspect / A + 0.5; }
  else { if (wide) uv.y = (uv.y - 0.5) * u_aspect / A + 0.5; else uv.x = (uv.x - 0.5) * A / u_aspect + 0.5; }
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { o = vec4(0.0, 0.0, 0.0, 1.0); return; }
  o = vec4(texture(iChannel0, vec2(uv.x, 1.0 - uv.y)).rgb, 1.0);
}`,
  // the characters' layer over the picture, by its alpha
  over: `void mainImage(out vec4 o, in vec2 fc) { vec2 uv = fc / iResolution.xy; vec4 b = texture(iChannel1, uv); o = vec4(mix(texture(iChannel0, uv).rgb, b.rgb, b.a), 1.0); }`,
  comp: `uniform float u_mode; uniform float u_opacity;
void mainImage(out vec4 o, in vec2 fc) {
  vec2 uv = fc / iResolution.xy;
  vec3 a = texture(iChannel0, uv).rgb, b = texture(iChannel1, uv).rgb, r;
  if (u_mode < 0.5) r = b; else if (u_mode < 1.5) r = a + b;
  else if (u_mode < 2.5) r = 1.0 - (1.0 - a) * (1.0 - b); else r = a * b;
  o = vec4(mix(a, r, u_opacity), 1.0);
}`,
  hatch: `void mainImage(out vec4 o, in vec2 fc) { float s = step(0.5, fract((fc.x + fc.y) / 22.0)); o = vec4(mix(vec3(0.09, 0.10, 0.12), vec3(0.15, 0.16, 0.19), s), 1.0); }`,
};
const BLEND = { normal: 0, add: 1, screen: 2, multiply: 3 };

// GLSL 330 -> GLSL ES: uniform defaults move out of the code (ES has no initializers)
function adapt(code) {
  const defaults = {};
  const body = code.replace(/^\s*#version[^\n]*$/m, "").replace(/uniform\s+(float|int)\s+([^;]+);/g, (m, ty, decls) => {
    if (!decls.includes("=")) return m;
    const names = [];
    for (const d of decls.split(",")) {
      const [name, val] = d.split("=").map(s => s.trim());
      names.push(name);
      if (val !== undefined) defaults[name] = parseFloat(val);
    }
    return `uniform ${ty} ${names.join(", ")};`;
  });
  return { body, defaults };
}
function compile(type, text) {
  const s = gl.createShader(type); gl.shaderSource(s, text); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { const e = gl.getShaderInfoLog(s); gl.deleteShader(s); throw new Error(e); }
  return s;
}
let vsShader = null;
function program(code, alpha = false) {
  const { body, defaults } = adapt(code);
  const p = gl.createProgram();
  gl.attachShader(p, vsShader || (vsShader = compile(gl.VERTEX_SHADER, VS)));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, HEAD + body + (alpha ? FOOT_ALPHA : FOOT)));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return { prog: p, defaults, locs: {} };
}
const PROGS = {};
const shaders = new Map();      // catalog key -> {status, P, defaults, frameCh, error}
function shaderFor(key) {
  if (shaders.has(key)) return shaders.get(key);
  const s = { status: "pending" };
  shaders.set(key, s);
  call("shader", key).then(r => {
    if (r.error) { s.status = "bad"; s.error = r.error; return; }
    try {
      const P = program(r.code);
      Object.assign(s, { status: "ok", P, defaults: { ...r.defaults, ...P.defaults }, frameCh: r.frame_channels });
      log("sys", `compiled ${r.file} for the live preview`);
    } catch (e) {
      Object.assign(s, { status: "bad", error: e.message.trim().split("\n").slice(0, 3).join(" | ") });
      log("err", `${r.file} does not compile as GLSL ES (it plays at the render): ${s.error}`);
    }
    renderBrowser(); renderTimeline();
  });
  return s;
}
function makeTarget() {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  texParams();
  const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { tex: t, fbo };
}
function texParams() {
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}
function mediaTex() {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
  texParams(); return t;
}
function setSize(res) {
  const [rw, rh] = String(res).toLowerCase().split("x").map(Number);
  const k = 640 / Math.max(rw || 854, rh || 480);
  W = Math.round((rw || 854) * k); H = Math.round((rh || 480) * k);
  canvas.width = W; canvas.height = H;
  if (!gl) return;
  for (const T of Object.values(TEX)) { gl.deleteTexture(T.tex); gl.deleteFramebuffer(T.fbo); }
  TEX = {}; for (const k2 of ["a", "b", "c", "d", "e", "f", "g"]) TEX[k2] = makeTarget();
}
function initGL() {
  if (!gl) { log("err", "no WebGL2 here: the preview is off, the Arrange still works"); return; }
  gl.bindVertexArray(gl.createVertexArray());
  blackTex = mediaTex();
  for (const [k, code] of Object.entries(OWN)) PROGS[k] = program(code);
  for (const [k, code] of Object.entries(OWN_ALPHA)) PROGS[k] = program(code, true);
  setSize("854x480");
}
function loc(P, n) { if (!(n in P.locs)) P.locs[n] = gl.getUniformLocation(P.prog, n); return P.locs[n]; }
function run(P, uniforms, inputs, target, t, vec4s = null) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fbo : null);
  gl.viewport(0, 0, W, H);
  gl.useProgram(P.prog);
  gl.uniform3f(loc(P, "iResolution"), W, H, 1);
  gl.uniform1f(loc(P, "iTime"), t);
  gl.uniform1f(loc(P, "iTimeDelta"), 1 / 60); gl.uniform1f(loc(P, "iFrameRate"), 60);
  const fl = loc(P, "iFrame"); if (fl) gl.uniform1i(fl, frameNo);
  for (const [k, v] of Object.entries({ ...P.defaults, ...uniforms })) { const l = loc(P, k); if (l && typeof v === "number" && isFinite(v)) gl.uniform1f(l, v); }
  for (let i = 0; i < 4; i++) {
    gl.activeTexture(gl.TEXTURE0 + i);
    gl.bindTexture(gl.TEXTURE_2D, inputs[i] || blackTex);
    const l = loc(P, "iChannel" + i); if (l) gl.uniform1i(l, i);
  }
  const cr = loc(P, "iChannelResolution[0]"); if (cr) gl.uniform3fv(cr, new Float32Array([W, H, 1, W, H, 1, W, H, 1, W, H, 1]));
  for (const [k, v] of Object.entries(vec4s || {})) { const l = loc(P, k); if (l) gl.uniform4fv(l, new Float32Array(v)); }
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}
function clearTarget(T, alpha = 1) { gl.bindFramebuffer(gl.FRAMEBUFFER, T.fbo); gl.viewport(0, 0, W, H); gl.clearColor(0, 0, 0, alpha); gl.clear(gl.COLOR_BUFFER_BIT); }

// videos: one element per clip, kept in step with the song
const vids = new Map();
function vid(url) {
  let v = vids.get(url);
  if (!v) {
    const el = document.createElement("video");
    el.muted = true; el.loop = true; el.playsInline = true; el.preload = "auto"; el.src = url;
    v = { el, tex: mediaTex(), last: 0, aspect: 16 / 9 };
    el.addEventListener("loadedmetadata", () => { if (el.videoHeight) v.aspect = el.videoWidth / el.videoHeight; });
    vids.set(url, v);
    if (vids.size > 24) {           // the longest unused goes
      const old = [...vids.entries()].filter(([u, x]) => x !== v).sort((a, b) => a[1].last - b[1].last)[0];
      if (old) { old[1].el.removeAttribute("src"); old[1].el.load(); gl.deleteTexture(old[1].tex); vids.delete(old[0]); }
    }
  }
  v.last = frameNo;
  return v;
}
function syncVid(v, want, rate = 1) {
  const el = v.el;
  if (el.readyState >= 1) {
    const d = el.duration || 0;
    if (d > 0) want = ((want % d) + d) % d;
    if (playing && rate > 0.05) {
      const r = Math.max(0.0625, Math.min(16, rate));
      if (Math.abs(el.playbackRate - r) > 1e-3) el.playbackRate = r;
      if (el.paused) el.play().catch(() => { });
      if (Math.abs(el.currentTime - want) > 0.35 && !el.seeking) el.currentTime = want;
    } else {
      if (!el.paused) el.pause();
      if (Math.abs(el.currentTime - want) > 0.05 && !el.seeking) el.currentTime = want;
    }
  }
  if (el.readyState >= 2) { gl.bindTexture(gl.TEXTURE_2D, v.tex); try { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, el); } catch (e) { } }
  return v.tex;
}
// a clip the page cannot play is played from its proxy (Python makes it once)
let proxyOf = {};
const playable = (url, state) => !!proxyOf[url] || !(state === "making" || state === "failed");
function drawVideo(url, want, target, t, cover = true) {
  const v = vid(proxyOf[url] || url);
  run(PROGS.fit, { u_aspect: v.aspect, u_cover: cover ? 1 : 0 }, [syncVid(v, want)], target, t);
}
function segAt(t) {
  let lo = 0, hi = montage.length - 1, r = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (montage[m].t0 <= t) { r = m; lo = m + 1; } else hi = m - 1; }
  return r >= 0 && t < montage[r].t1 + 0.05 ? montage[r] : null;
}
const spansOf = m => m.always ? [{ first: 1, last: NB() }] : m.spans;
const activeSpan = (m, bar) => spansOf(m).find(s => bar >= s.first && bar <= s.last);
function uniformsOf(m, t) { const u = {}; for (const p of m.params) u[p.name] = paramAt(p, t); return u; }

let chips = [];
// The characters at song time t: each FG clip where its cue (REAPER's, a cue file's) or its
// block says, in its place in the frame, cut out by its alpha video (Python makes it once)
function placeValue(v, def, t) {
  if (v == null) return def;
  if (typeof v === "number") return v;
  const got = knobAt(v.knob, t);
  return got == null ? def : got;
}
function clipTimeOf(m, t, bar) {
  const p = m.place;
  if (m.cued) {
    const cues = p.cue_times || [];
    let lo = 0, hi = cues.length - 1, k = -1;
    while (lo <= hi) { const mid = (lo + hi) >> 1; if (cues[mid][0] <= t) { k = mid; lo = mid + 1; } else hi = mid - 1; }
    if (k < 0) return null;
    const [at, start, length, rate] = cues[k];
    if (t < at + length) return { ct: start + (t - at) * rate, rate, own: true };
    if (t < at + length + (p.hold || 0)) return { ct: start + length * rate - 0.02, rate: 0, own: true };
    return null;
  }
  const s = activeSpan(m, bar);
  if (!s) return null;
  return { ct: (s.from || 0) + (t - B[s.first - 1]) * (s.rate || 1), rate: s.rate ?? 1, own: !s.file, url: s.url, proxy: s.proxy };
}
function fgCharacters(t, bar) {
  const out = [];
  for (const m of members) {
    if (m.lane !== "fg" || m.kind !== "clip" || !m.enabled || !m.place) continue;
    if ((m.mutes || []).some(([a, b]) => bar >= a && bar <= b)) continue;       // hidden there: it goes on underneath
    const at = clipTimeOf(m, t, bar);
    if (!at) continue;
    const p = m.place, cut = at.own && p.cut && p.cut.state === "ready" ? p.cut : null;
    let url = cut ? cut.url : (at.url || p.file_url), why = "";
    if (!cut) {
      why = !at.own ? "another clip: whole frame" : !p.cut ? "not cut out" : p.cut.state === "making" ? "cutting it out…" : p.cut.why || "whole frame";
      if (!playable(url, at.own ? p.file_proxy : at.proxy)) { chips.push(["engine", `FG · ${m.label} · making a proxy`]); continue; }
      url = proxyOf[url] || url;
    }
    if (!url) continue;
    out.push({ m, url, ct: at.ct, rate: at.rate, cut: !!cut, why, fit: p.fit || "contain", flip: p.flip,
      x: placeValue(p.x, 0.5, t), y: placeValue(p.y, 0.5, t), scale: placeValue(p.scale, 1, t), opacity: m.opacity == null ? 1 : m.opacity });
  }
  return out;
}
function shaderLayer(m, t, input, target) {
  const s = shaderFor(m.key);
  if (s.status !== "ok") return false;
  const inputs = [];
  for (const i of (s.frameCh || [])) inputs[i] = input;
  run(s.P, { ...s.defaults, ...uniformsOf(m, t) }, inputs, target, t);
  return true;
}
function drawFrame(t) {
  chips = [];
  const bar = barAt(t);
  let cur = TEX.a, alt = TEX.b;
  const fit = (S.project.clip_fit || "contain") === "cover";
  // the montage
  const seg = segAt(t);
  if (seg && seg.url && playable(seg.url, seg.proxy)) { drawVideo(seg.url, (seg.from || 0) + (t - seg.t0) * (seg.dir || 1), cur, t, fit); chips.push(["clip", `clips · ${seg.name}`]); }
  else { clearTarget(cur); if (seg) chips.push(seg.url ? ["engine", `clips · ${seg.name} · ${seg.proxy === "failed" ? "cannot play here" : "making a proxy"}`] : ["clip", "clips · nothing"]); }
  const lane = id => members.filter(m => m.lane === id && m.enabled && activeSpan(m, bar));
  // BG: its members over the montage
  for (const m of lane("bg")) {
    const sp = activeSpan(m, bar);
    if (m.kind === "clip" && m.live && sp.url && !playable(sp.url, sp.proxy)) { chips.push(["engine", `BG · ${m.label} · making a proxy`]); continue; }
    if (m.kind === "clip" && m.live && sp.url) {
      drawVideo(sp.url, (sp.from || 0) + (t - B[sp.first - 1]) * (sp.rate || 1), TEX.e, t, true);
      chips.push(["bg", `BG · ${m.label}`]);
    } else if (!(m.live && shaderLayer(m, t, cur.tex, TEX.e))) {
      chips.push(["engine", `BG · ${m.label} · at render`]);
      if (m.kind === "clip") continue;
      run(PROGS.hatch, {}, [], TEX.e, t);
    } else chips.push(["bg", `BG · ${m.label}`]);
    run(PROGS.comp, { u_mode: BLEND[m.blend] ?? 0, u_opacity: m.opacity }, [cur.tex, TEX.e.tex], alt, t);
    [cur, alt] = [alt, cur];
  }
  const fxChain = (id, name, c, a) => {
    for (const m of lane(id)) {
      if (m.live && shaderLayer(m, t, c.tex, a)) { [c, a] = [a, c]; chips.push(["fx", `${name} · ${m.label}`]); }
      else chips.push(["engine", `${name} · ${m.label} · at render`]);
    }
    return [c, a];
  };
  [cur, alt] = fxChain("bgfx", "BG FX", cur, alt);
  // FG: drawn on its own, then over the BG
  const fgs = lane("fg");
  if (fgs.length) {
    let fc = TEX.c, fa = TEX.d, drew = false;
    clearTarget(fc);
    for (const m of fgs) {
      if (m.kind === "clip") continue;                  // the characters come after, cut out
      if (m.live && shaderLayer(m, t, fc.tex, TEX.e)) {
        run(PROGS.comp, { u_mode: BLEND[m.blend] ?? 0, u_opacity: m.opacity }, [fc.tex, TEX.e.tex], fa, t);
        [fc, fa] = [fa, fc]; drew = true; chips.push(["fg", `FG · ${m.label}`]);
      } else chips.push(["engine", `FG · ${m.label} · at render`]);
    }
    if (drew) {
      [fc, fa] = fxChain("fgfx", "FG FX", fc, fa);
      run(PROGS.comp, { u_mode: 2, u_opacity: 1 }, [cur.tex, fc.tex], alt, t);
      [cur, alt] = [alt, cur];
    }
  }
  const chars = fgCharacters(t, bar);
  if (chars.length) {
    let cc = TEX.f, ca = TEX.g;
    clearTarget(cc, 0);
    for (const ch of chars) {
      const v = vid(ch.url), tex = syncVid(v, ch.ct, ch.rate);
      const el = v.el;
      if (!el.videoWidth) { chips.push(["engine", `FG · ${ch.m.label} · loading`]); continue; }
      const k = (ch.fit === "cover" ? Math.max : Math.min)(W / el.videoWidth, H / el.videoHeight) * ch.scale;
      const w = el.videoWidth * k / W, hh = el.videoHeight * k / H;
      run(PROGS.place, { u_flip: ch.flip ? 1 : 0, u_opacity: ch.opacity }, [cc.tex, tex], ca, t, { u_rect: [ch.x - w / 2, ch.y - hh / 2, ch.x + w / 2, ch.y + hh / 2] });
      [cc, ca] = [ca, cc];
      chips.push(ch.cut ? ["fg", `FG · ${ch.m.label}`] : ["engine", `FG · ${ch.m.label} · ${ch.why}`]);
    }
    for (const m of lane("fgfx")) {             // the FG's FX on the characters, their figures kept
      if (m.live && shaderLayer(m, t, cc.tex, TEX.e)) { run(PROGS.keepalpha, {}, [TEX.e.tex, cc.tex], ca, t); [cc, ca] = [ca, cc]; }
    }
    run(PROGS.over, {}, [cur.tex, cc.tex], alt, t);
    [cur, alt] = [alt, cur];
  }
  // the master
  for (const m of lane("master")) {
    if (m.kind === "fx" && m.live && shaderLayer(m, t, cur.tex, alt)) { [cur, alt] = [alt, cur]; chips.push(["fx", `master · ${m.label}`]); }
    else chips.push(["engine", `master · ${m.label} · at render`]);
  }
  run(PROGS.copy, {}, [cur.tex], null, t);
  for (const v of vids.values()) if (v.last !== frameNo && !v.el.paused) v.el.pause();
}
let lastChips = "";
function showChips() {
  const key = JSON.stringify(chips);
  if (key === lastChips) return;
  lastChips = key;
  const live = chips.filter(([c]) => c !== "engine"), later = chips.filter(([c]) => c === "engine");
  const shown = later.length > 2 ? [...live, ["engine", `${later.length} more at the render`]] : chips;
  const el = $("ovBot");
  el.title = later.map(([, text]) => text).join("\n");
  el.replaceChildren(...shown.map(([c, text]) => h("span", { class: `ovchip ${c}`, text })));
}

// ---------------------------------------------------------------------------
// The engine's render
// ---------------------------------------------------------------------------
let engine = null;             // {url, first, last}
let view = "live";
let rendering = false;
function setView(v) {
  if (v !== "live" && holding) { holding = false; if (playing) startSource(); }
  view = v;
  $("viewLive").setAttribute("aria-pressed", v === "live");
  $("viewSketch").setAttribute("aria-pressed", v === "sketch");
  $("viewEngine").setAttribute("aria-pressed", v === "engine");
  $("engineVideo").hidden = v !== "engine";
  $("streamImg").hidden = v !== "live";
  $("ovBot").hidden = v !== "sketch";
  $("gpuNote").hidden = v !== "sketch" || !gpuBusy.length;
  if (v !== "live") $("streamNote").hidden = true;
  if (v === "live" && S) streamAsk(now());
}

// ---------------------------------------------------------------------------
// Live: the engine itself, made as it plays (core/web/stream.py). Python makes the frames a
// few seconds ahead of the playhead with the render's own stack; the page shows the one for
// now, and when the engine is behind the song waits for it: never a picture the render
// would not make.
// ---------------------------------------------------------------------------
let stream = null, streamAsking = 0, streamDirty = 0, streamShown = -1, streamMade = -1, streamBusy = false;
let streamState = {}, streamRetryAt = 0, streamUrl = null, streamShownN = 0, streamAskBusy = false;
const picOffset = () => (S && S.project.sync_offset_ms || 0) / 1000;
const streamFi = t => stream ? Math.floor((t + picOffset() - stream.t0) * stream.fps + 1e-6) : -1;
function streamHas(t) { const fi = streamFi(t); return !!stream && fi >= 0 && fi <= streamMade; }
async function streamAsk(t, force = false) {
  if (!S) return;
  const n = ++streamAsking;
  streamAskBusy = true;
  const r = await call("stream_start", t + picOffset(), force).catch(e => ({ error: e.message }));
  if (n === streamAsking) streamAskBusy = false;
  if (n !== streamAsking) return;
  if (!r || r.error) { stream = null; streamState = { error: r ? r.error : "no answer" }; showStreamNote(); return; }
  if (!stream || r.gen !== stream.gen) { streamShown = -1; streamMade = r.made; }
  stream = r; streamState = { error: r.error, what: r.what, ended: r.ended };
}
function streamEdited() { streamDirty = performance.now(); }
function holdSong() { if (holding || !playing) return; pausedAt = now(); holding = true; stopSource(); }
function releaseSong() { if (!holding) return; holding = false; if (playing) startSource(); }
async function fetchStreamFrame(gen, fi, show) {
  streamBusy = true;
  try {
    const r = await fetch(`/stream/${gen}/${fi}.jpg`, { cache: "no-store" });
    if (!stream || stream.gen !== gen) return;
    if (r.status === 410) { stream = null; return; }
    const made = +(r.headers.get("X-Made") ?? -1);
    if (made > streamMade) streamMade = made;
    const st = r.headers.get("X-State");
    if (st) streamState = JSON.parse(decodeURIComponent(st));
    if (r.status === 200 && show) {
      const url = URL.createObjectURL(await r.blob()), old = streamUrl;
      $("streamImg").src = url; streamUrl = url; streamShown = fi; streamShownN++;
      if (old) setTimeout(() => URL.revokeObjectURL(old), 1000);
    } else if (r.status !== 200) streamRetryAt = performance.now() + 60;
  } catch (e) { streamRetryAt = performance.now() + 500; }
  finally { streamBusy = false; }
}
function streamTick(t) {
  if (!S) return;
  if (streamDirty && performance.now() - streamDirty > 250) { streamDirty = 0; streamAsk(t, true); }
  if (!stream) { if (!streamAskBusy && !streamState.error) streamAsk(t); showStreamNote(); return; }
  const fi = streamFi(t);
  if (fi < 0) { if (!streamBusy) streamAsk(t); return; }
  const lead = Math.round(stream.fps * 0.75);
  if (playing && !holding && fi > streamMade && !streamState.ended) holdSong();
  if (holding && (streamMade >= fi + lead || streamState.ended || streamState.error)) releaseSong();
  if (!streamBusy && performance.now() >= streamRetryAt) {
    if (fi !== streamShown && fi <= streamMade) fetchStreamFrame(stream.gen, fi, true);
    else if (fi > streamMade || (holding && streamMade < fi + lead)) fetchStreamFrame(stream.gen, holding ? fi + lead : fi, !holding);
  }
  showStreamNote();
}
let lastStreamNote = "";
function showStreamNote() {
  const el = $("streamNote");
  const err = streamState.error, behind = holding || (!playing && stream && streamFi(now()) > streamMade);
  const text = err ? `The engine could not make it: ${err}` : behind ? `the engine is making it${streamState.what && streamState.what !== "playing" ? `: ${streamState.what}` : "…"}` : "";
  if (text === lastStreamNote) return;
  lastStreamNote = text;
  el.hidden = !text || view !== "live"; el.classList.toggle("bad", !!err);
  const montage = err && /clip montage/.test(err);            // a montage without its clips folder: settle it here
  fill(el, h("span", { text }),
    montage ? h("button", { class: "lnkbtn", onclick: () => edit("remove_montage") }, "Remove the montage") : null,
    montage ? h("button", { class: "lnkbtn", onclick: async () => { if (await chooseClipsDir()) streamEdited(); } }, "Choose the clips folder") : null,
    err ? h("button", { class: "lnkbtn", onclick: () => setView("sketch") }, "Sketch instead") : null);
}
// the exact preview's state over the picture: {text, frac, stop, bad} | null
function showExactOv(o) {
  const el = $("exactOv");
  if (!el) return;
  if (!o) { el.hidden = true; return; }
  el.hidden = false; el.classList.toggle("bad", !!o.bad);
  fill(el, h("span", { text: o.text }),
    o.frac != null ? h("div", { class: "progress wide" }, h("i", { style: `width:${Math.round(100 * o.frac)}%` })) : null,
    o.stop ? h("button", { class: "btn sm danger", onclick: () => call("stop_render") }, "Stop") : h("button", { class: "lnkbtn", "aria-label": "Close", onclick: () => showExactOv(null) }, "✕"));
}
let lastFrame = null;
function onEngine(e) {
  if (e.event === "progress" || e.event === "queued") {
    rendering = true;
    $("bStop").hidden = false; $("prog").hidden = e.event === "queued"; $("engineDot").classList.add("busy");
    const pct = e.event === "progress" ? Math.round(100 * e.frame / Math.max(1, e.of)) : 0;
    if (e.event === "progress") { $("progBar").style.width = pct + "%"; lastFrame = [e.frame, e.of]; }
    if (e.first) renderSpan = [e.first, e.last];
    const what = e.event === "queued" ? `the exact preview waits: ${e.why}` : e.frame ? `${e.what || "rendering"}: frame ${e.frame}/${e.of} · ${pct}%` : `${e.what || "starting"}…`;
    $("engineText").textContent = what;
    $("bRender").textContent = e.event === "queued" ? "Exact: waiting" : `Rendering ${pct}%`;
    showExactOv({ text: `bars ${renderSpan[0]}–${renderSpan[1]}: ${what}`, frac: e.event === "progress" ? pct / 100 : null, stop: true });
    return;
  }
  rendering = false;
  const idle = () => { $("bRender").disabled = false; $("bStop").hidden = true; $("prog").hidden = true; $("engineDot").classList.remove("busy"); updateLoopChip(); };
  if (e.event === "stopped") {
    idle();
    const at = lastFrame ? ` at frame ${lastFrame[0]}/${lastFrame[1]}` : "";
    $("engineText").textContent = `stopped${at}`; showExactOv({ text: `the exact preview stopped${at}` }); lastFrame = null;
    return;
  }
  idle(); lastFrame = null;
  if (e.event === "done") showExactOv(null);
  else showExactOv({ text: `the exact preview failed: ${(e.tail || []).slice(-1)[0] || `exit ${e.code}`} (the log has the rest)`, bad: true });
  if (e.event === "done") {
    $("engineText").textContent = "engine idle";
    const span = e.first ? [e.first, e.last] : renderSpan;
    engine = { url: e.url, first: span[0], last: span[1] };
    const v = $("engineVideo"); v.src = e.url;
    $("viewEngine").disabled = false; $("viewEngine").title = `bars ${span[0]}–${span[1]}`;
    log("in", `rendered ${e.path}`);
    loop = [span[0], span[1]]; renderTimeline(); seek(B[span[0] - 1]); setView("engine"); setPlaying(true);
  } else {
    $("engineText").textContent = "render failed"; $("engineDot").classList.add("bad");
    log("err", `the render failed (exit ${e.code}):`); for (const l of e.tail || []) log("err", l);
  }
}
let renderSpan = [1, 4];
async function startRender() {
  if (rendering || !S) return;
  renderSpan = exactSpan();
  const r = await call("render", renderSpan[0], renderSpan[1]);
  if (r.error) { toast(r.error, true); showExactOv({ text: `the exact preview: ${r.error}`, bad: true }); return; }
  rendering = true;
  $("bRender").disabled = true; $("bStop").hidden = false; $("prog").hidden = false; $("progBar").style.width = "0";
  $("engineDot").classList.remove("bad"); $("engineDot").classList.add("busy"); $("engineText").textContent = "starting: reading the song and the clips";
  showExactOv({ text: `bars ${renderSpan[0]}–${renderSpan[1]}: starting, reading the song and the clips`, frac: 0, stop: true });
}
function syncEngineVideo(t) {
  const v = $("engineVideo");
  if (view !== "engine" || !engine || v.readyState < 1) return;
  const want = t - B[engine.first - 1], d = v.duration || 0;
  if (want < 0 || (d && want > d)) { if (!v.paused) v.pause(); return; }
  if (playing) { if (v.paused) v.play().catch(() => { }); if (Math.abs(v.currentTime - want) > 0.12) v.currentTime = want; }
  else { if (!v.paused) v.pause(); if (Math.abs(v.currentTime - want) > 0.04) v.currentTime = want; }
}

// ---------------------------------------------------------------------------
// The Arrange
// ---------------------------------------------------------------------------
// Lanes top to bottom, as the Arrange window draws them. Python packs each track's members
// into sub-rows (core.arrangement.pack) and says where each goes; FG and BG get one empty
// row more, to put something new on.
const LANES = [
  { id: "audio", name: "Audio", kind: "audio", c: "var(--muted)" },
  { id: "stems", name: "Stems", kind: "stems", c: "var(--muted)" },       // one lane per stem with sound
  { id: "midi", name: "MIDI", kind: "midi", c: "var(--accent)" },
  { id: "master", name: "Master", sub: "FX over everything", kind: "track", c: "var(--lane-fx)" },
  { id: "fg", name: "FG", sub: "foreground, on its own", kind: "track", c: "var(--lane-fg)", group: "FG" },
  { id: "fgfx", name: "FG FX", sub: "on the whole FG", kind: "track", c: "var(--lane-fx)", group: "FG" },
  { id: "bg", name: "BG", sub: "over the montage", kind: "track", c: "var(--lane-bg)", group: "BG" },
  { id: "bgfx", name: "BG FX", sub: "on the whole BG", kind: "track", c: "var(--lane-fx)", group: "BG" },
  { id: "clips", name: "Montage", sub: "the BG's bottom: a clip a bar", kind: "clips", c: "var(--lane-clip)" },
];
const FX_LANES = ["bgfx", "fgfx", "master"];
const SAME_KIND = [["bg", "fg"], ["bgfx", "fgfx"]];
const RULER_H = 42, DENS_H = 10, HEAD_W = 132, EDGE = 7, ROW_MIN = 20, ROW_MAX = 120;
let rowH = 30;                 // a row's height: the vertical zoom
let layout = null;
let rowsOf = {};               // lane -> how many sub-rows its members take (Python's pack)
let groups = {}, has = {}, board = {};
let picked = null;             // what the browser picked: {type: "cat", key, kind, label} | {type: "clip", path, name, url, from}
let pickedFx = null;           // and an FX, at the same time: {type: "cat", key, kind: "fx", label}
let montageBars = [];          // the bars the montage plays ([] all of them)
let lastNote = "";             // what the last edit did (the status line)
const isFxLane = id => FX_LANES.includes(id);
const sameKind = (a, b) => a === b || SAME_KIND.some(p => p.includes(a) && p.includes(b));
const roomy = () => rowH >= 30;
const pickFor = l => l && isFxLane(l.id) ? pickedFx : picked;

function computeLayout() {
  if (labOn()) return labLayout();               // the Feature Lab in the timeline's place
  let y = 0; const L = [];
  for (const d of LANES) {
    let hh = 0, shown = 0;
    if (d.kind === "stems") {                    // compact: the song's parts, under the mix
      const sh = Math.max(18, Math.min(44, Math.round(rowH * 0.8)));
      stemWaves.forEach((sw, i) => { L.push({ id: `stem:${sw.name}`, name: sw.name, sub: "stem", kind: "stem", c: STEM_COLORS[i % STEM_COLORS.length], wave: sw, top: y, h: sh, shown: 0 }); y += sh; });
      continue;
    }
    if (d.kind === "audio") hh = Math.max(34, Math.min(90, Math.round(rowH * 1.5)));
    else if (d.kind === "midi") hh = 8 + 9 * Math.max(1, (S ? S.tracks.length : 1));
    else if (d.kind === "clips") { if (!has.clips) continue; hh = Math.max(26, rowH + 6); }   // a scene may have no montage
    else {
      if (d.group && !has[d.group]) continue;
      const n = rowsOf[d.id] || 0;
      shown = d.id === "fg" || d.id === "bg" ? n + 1 : Math.max(1, n);
      hh = shown * rowH + 4;
    }
    L.push({ ...d, top: y, h: hh, shown }); y += hh;
  }
  return { lanes: L, height: y };
}
const laneById = id => layout && layout.lanes.find(l => l.id === id);
function rowTop(l, r) { return l.top + 2 + r * rowH; }
function canvasFor(width, height) {
  const d = devicePixelRatio || 1;
  return h("canvas", { width: Math.round(width * d), height: Math.round(height * d), style: `width:${width}px; height:${height}px` });
}
function renderTimeline() {
  const heads = $("heads"), inner = $("inner"), ruler = $("ruler");
  if (!S) { heads.replaceChildren(); inner.replaceChildren(); ruler.replaceChildren(); inner.style.height = "120px"; return; }
  layout = computeLayout();
  const width = NB() * ppb;
  $("tgrid").style.gridTemplateColumns = `${HEAD_W}px ${width}px`;
  const sg = S.song;
  $("corner").replaceChildren(h("b", { text: `${sg.meter} · ${Math.round(sg.bpm)} BPM` }), h("span", { text: `bar ${fmt(sg.bar_seconds || 0)} s · ${fmtTime(sg.duration)}` }));
  $("corner").title = `${NB()} bars · ${sg.meter} · ${fmt(sg.bpm)} BPM · a bar ${fmt(sg.bar_seconds || 0)} s · the audio ${fmtTime(sg.duration)}`;
  // the ruler: bar numbers, beats, meter changes, how busy the MIDI is, REAPER's markers
  ruler.style.width = width + "px";
  const rc = canvasFor(width, RULER_H); drawRuler(rc);
  ruler.replaceChildren(rc, ...(S.song.markers || []).map(mk => h("div", { class: "marker", style: `left:${tToX(mk.t)}px`, title: `${mk.name || "marker"} · ${fmtTime(mk.t)}` }, mk.name || "◆")));
  if (loop) ruler.append(h("div", { class: "rulerloop", style: `left:${(loop[0] - 1) * ppb}px; width:${(loop[1] - loop[0] + 1) * ppb}px` }));
  ruler.append(h("div", { class: "rulerhead-mark", id: "rulerPlay" }));
  // the heads
  heads.replaceChildren(...layout.lanes.map(l => {
    const chosen = (sel && ((sel.type === "lane" && l.group === sel.group && !isFxLane(l.id) || sel.type === "lane" && l.group === sel.group)
      || (sel.type === "master" && l.id === "master") || ((sel.type === "clips" || sel.type === "montage") && l.id === "clips")));
    const head = h("div", { class: "head" + (l.kind === "midi" ? " midi" : "") + (chosen ? " chosen" : "") + (["track", "clips"].includes(l.kind) ? " pick" : ""),
      style: `height:${l.h}px; --c:${l.c}`, dataset: { lane: l.id }, title: l.kind === "track" ? `${l.name}: click for its settings and FX` : "" }, h("b", { text: l.name }));
    const sub = l.kind === "audio" ? S.project.audio : l.id === "bg" && !has.clips ? "the background" : l.sub;
    if (l.kind === "midi") for (const tr of S.tracks) head.append(h("div", { class: "trk", text: tr.name, title: `${tr.name}: ${tr.times.length} notes` }));
    else if (sub && (roomy() || l.h >= 30)) head.append(h("small", { text: sub, title: sub }));
    return head;
  }));
  // the lanes
  inner.style.width = width + "px"; inner.style.height = layout.height + "px";
  inner.replaceChildren();
  for (const l of layout.lanes) {
    const lit = sel && ((sel.type === "lane" && l.group === sel.group) || (sel.type === "master" && l.id === "master") || ((sel.type === "clips" || sel.type === "montage") && l.id === "clips"));
    const lane = h("div", { class: "lane " + l.kind + (lit ? " lit" : ""), dataset: { lane: l.id }, style: `top:${l.top}px; height:${l.h}px; background-image: repeating-linear-gradient(to right, var(--line-2) 0 1px, transparent 1px ${4 * ppb}px), repeating-linear-gradient(to right, var(--line) 0 1px, transparent 1px ${ppb}px);` });
    if (l.kind === "audio") { const c = canvasFor(width, l.h); lane.append(c); drawWave(c); }
    if (l.kind === "stem") { const c = canvasFor(width, l.h); lane.append(c); drawStem(c, l.wave, l.c); }
    if (l.kind === "midi") { const c = canvasFor(width, l.h); lane.append(c); drawMidi(c); }
    if (l.kind.startsWith("lab")) { const c = canvasFor(width, l.h); c.dataset.lab = l.kind; c.dataset.id = l.id; lane.append(c); }
    if (l.kind === "clips") drawCells(lane, l);
    if (l.kind === "track") {
      for (let r = 1; r < l.shown; r++) lane.append(h("div", { class: "rowline", style: `top:${r * rowH + 2}px` }));
      if ((l.id === "fg" || l.id === "bg") && roomy()) lane.append(h("div", { class: "newrow", style: `top:${(l.shown - 1) * rowH + 2}px; height:${rowH}px`, text: picked && accepts(l, picked) ? `drag across: ${picked.label || picked.name}, new` : "drag here: new" }));
      if (isFxLane(l.id) && roomy() && (!(rowsOf[l.id] || 0) || pickedFx)) lane.append(h("div", { class: "newrow", style: `top:${(l.shown - 1) * rowH + 2}px; height:${rowH}px`, text: pickedFx ? `drag across: ${pickedFx.label}` : "right-click: an FX here · or pick one in FX and drag across" }));
    }
    if (l.kind === "track" || l.kind === "clips") wireLaneDrop(lane, l);
    inner.append(lane);
  }
  for (const m of members) {
    const l = laneById(m.lane);
    if (!l) continue;
    for (const [a, b] of m.mutes || []) inner.append(h("div", { class: "mute", style: `left:${(a - 1) * ppb}px; width:${(b - a + 1) * ppb}px; top:${rowTop(l, m.row) + 1}px; height:${rowH - 4}px`, title: `${m.label}: hidden in bars ${a}–${b} (it goes on underneath)` }, roomy() && (b - a + 1) * ppb > 60 ? "hidden" : ""));
    if (m.cued && m.cues !== "read") {
      inner.append(h("div", { class: "cuenote", style: `left:4px; top:${rowTop(l, m.row) + 2}px; height:${rowH - 6}px`, dataset: { id: idOf(m.ref) } }, `${m.label} · ${m.cues === "error" ? "cues not read" : "reading its cues…"}`));
      continue;
    }
    spansOf(m).forEach((s, k) => inner.append(blockEl(m, s, m.always ? 0 : k, l)));
  }
  if (loop) inner.append(h("div", { class: "loopband", id: "loopband", style: `left:${(loop[0] - 1) * ppb}px; width:${(loop[1] - loop[0] + 1) * ppb}px` }));
  inner.append(h("div", { class: "ghost", id: "ghost", hidden: true }), h("div", { class: "edge", id: "edge", hidden: true }), h("div", { class: "playhead", id: "playhead" }));
  updateLoopChip();
  $("bUndo").disabled = !undoN; $("bRedo").disabled = !redoN;
  $("zoomText").textContent = `${ppb}px a bar · rows ${rowH}px`;
  if (labOn()) { if (lab.ppb !== ppb) labRefresh(); labDraw(); }
}
function drawRuler(cv) {
  const g = cv.getContext("2d"), d = devicePixelRatio || 1;
  g.scale(d, d);
  g.font = "10px 'IBM Plex Mono', Consolas, monospace"; g.textBaseline = "top";
  const every = ppb < 18 ? 8 : ppb < 26 ? 4 : ppb < 40 ? 2 : 1;
  const dens = S.song.density || [];
  for (let b = 1; b <= NB(); b++) {
    const x = (b - 1) * ppb;
    const v = Math.max(0, Math.min(1, dens[b - 1] || 0));     // how busy the MIDI is in this bar: dark to orange
    g.fillStyle = `rgb(${Math.round(27 + v * 213)},${Math.round(30 + v * 130)},${Math.round(37 + v * 38)})`;
    g.fillRect(x, RULER_H - DENS_H, Math.max(1, ppb - 1), DENS_H);
    g.fillStyle = (b - 1) % 4 === 0 ? "#5a6273" : "#3b4352"; g.fillRect(x, RULER_H - DENS_H - 12, 1, 12);
    const n = S.song.bar_beats[b - 1] || S.song.bar_beats[S.song.bar_beats.length - 1] || 4;
    g.fillStyle = "#2b313d";
    for (let k = 1; k < n; k++) g.fillRect(x + k * ppb / n, RULER_H - DENS_H - 6, 1, 6);
    if ((b - 1) % every === 0) { g.fillStyle = "#9097a7"; g.fillText(String(b), x + 3, 3); }
    const before = b > 1 ? (S.song.bar_beats[b - 2] || n) : null;
    if ((b === 1 && n !== (S.song.bar_beats[1] || n)) || (before !== null && n !== before)) { g.fillStyle = "#f0a04b"; g.fillText(`${n}/4`, x + 3, 15); }
  }
}
function drawWave(cv) {
  if (!peaks) return;
  const g = cv.getContext("2d"), d = devicePixelRatio || 1, mid = cv.height / 2;
  g.fillStyle = "#6f7891";
  for (let x = 0; x < cv.width; x += 2) {
    const t0 = xToT(x / d), t1 = xToT((x + 2) / d);
    let m = 0; for (let i = Math.floor(t0 * peaksHz); i <= Math.floor(t1 * peaksHz) && i < peaks.length; i++) if (peaks[i] > m) m = peaks[i];
    const hh = Math.min(1, m) * (mid - 2 * d);
    g.fillRect(x, mid - hh, 1.2 * d, hh * 2);
  }
}
// a stem's energy, mirrored like the mix's wave, in its own colour
const STEM_COLORS = ["#6aa7d8", "#9fb36a", "#d78bb0", "#e3c15b", "#3fb6a8", "#c27cf0"];
function drawStem(cv, sw, colour) {
  if (!sw || !sw.values.length) return;
  const g = cv.getContext("2d"), d = devicePixelRatio || 1, mid = cv.height / 2, v = sw.values, hz = sw.hz;
  g.fillStyle = colour; g.globalAlpha = 0.85;
  for (let x = 0; x < cv.width; x += 2) {
    const t0 = xToT(x / d), t1 = xToT((x + 2) / d);
    let m = 0; for (let i = Math.floor(t0 * hz); i <= Math.floor(t1 * hz) && i < v.length; i++) if (v[i] > m) m = v[i];
    const hh = m * (mid - 1 * d);
    if (hh > 0.3) g.fillRect(x, mid - hh, 1.2 * d, hh * 2);
  }
  g.globalAlpha = 1;
}
const TRACK_COLORS = ["#f0a04b", "#c27cf0", "#3fb6a8", "#6aa7d8", "#e3c15b", "#e2665a", "#9fb36a", "#d78bb0"];
function drawMidi(cv) {
  const g = cv.getContext("2d"), d = devicePixelRatio || 1;
  S.tracks.forEach((tr, i) => {
    g.fillStyle = TRACK_COLORS[i % TRACK_COLORS.length];
    const y = (5 + i * 9) * d;
    for (let j = 0; j < tr.times.length; j++) { g.globalAlpha = 0.35 + 0.65 * tr.vels[j] / 127; g.fillRect(tToX(tr.times[j]) * d, y, 2 * d, 6 * d); }
  });
  g.globalAlpha = 1;
}
function clipAtBar(b) { return segAt(B[b - 1] + 0.02); }
// the montage as runs: a clip and the bars it plays (reverses inside one run)
function montageRuns() {
  const out = [];
  for (const s of montage) {
    const a = barAt(s.t0 + 1e-3), b = barAt(Math.max(s.t0, s.t1 - 1e-3));
    const last = out[out.length - 1];
    if (last && last.name === s.name && last.last >= a - 1) last.last = Math.max(last.last, b);
    else out.push({ first: a, last: b, name: s.name, path: s.path, url: s.url });
  }
  return out;
}
function runAt(bar) { return montageRuns().find(r => r.first <= bar && bar <= r.last) || null; }
// a clip's run whose end is under x (its first bar's left side, its last bar's right side)
function runEdgeAt(x) {
  const runs = montageRuns().filter(r => r.name !== "none"), near = Math.min(EDGE, ppb / 3);
  for (const r of runs) {
    if (Math.abs(x - (r.first - 1) * ppb) <= near) return { run: r, end: "first" };
    if (Math.abs(x - r.last * ppb) <= near) return { run: r, end: "last" };
  }
  return null;
}
function showRunEdge(bar) {          // bar: the line before it (null: none)
  const ed = $("edge"), l = laneById("clips");
  if (!ed || !l) return;
  if (bar == null) { ed.hidden = true; return; }
  ed.hidden = false; ed.style.left = ((bar - 1) * ppb - 1) + "px"; ed.style.top = l.top + "px"; ed.style.height = l.h + "px";
}
const thumbs = new Map();     // clip url -> its cover's url
function drawCells(lane, l) {
  for (let b = 1; b <= NB(); b++) {
    const s = clipAtBar(b), pinned = String(b) in pins, none = (pinned && pins[String(b)] === "none") || (s && s.name === "none");
    const th = s && s.url ? thumbs.get(s.url) : null;
    const chosen = sel && sel.type === "montage" && sel.first <= b && b <= sel.last;
    const out = montageBars.length && !montageBars.some(([a, z]) => a <= b && b <= z);     // the montage's own bars: not this one
    const cls = "cell" + (none || (montage.length && !s) ? " none" : "") + (pinned ? " pin" : "") + (chosen ? " sel" : "") + (out ? " out" : "");
    lane.append(h("div", { class: cls, dataset: { bar: b }, style: `left:${(b - 1) * ppb + 1}px; width:${ppb - 2}px;` + (th && !none ? `background-image:url(${th});` : "") + (s && s.colour && !none ? `--cc:${s.colour}` : ""),
      title: `Bar ${b}: ${none ? "nothing" : s ? `${s.name}${s.from ? ` from ${fmt(s.from)} s` : ""}${s.dir < 0 ? " (reversed)" : ""}` : "…"}${pinned ? " (pinned)" : ""}${out ? " · the montage does not play here (its bars)" : ""}${s && s.path ? `\n${s.path}` : ""}`,
      text: ppb >= 44 && s && !none ? s.name.slice(0, Math.floor(ppb / 6)) : "" }));
  }
  if (montageBars.length) lane.append(h("div", { class: "mbars", title: "the montage plays in these bars" }, ...montageBars.map(([a, z]) => h("i", { style: `left:${(a - 1) * ppb}px; width:${(z - a + 1) * ppb}px` }))));
  if (!montage.length && roomy()) lane.append(h("div", { class: "newrow", style: "top:4px", text: "the montage comes when the clips are probed" }));
}
function blockEl(m, s, k, l) {
  const top = rowTop(l, m.row) + 1, height = rowH - 4;
  const left = (s.first - 1) * ppb + 1, width = (s.last - s.first + 1) * ppb - 2;
  const live = m.live && (m.kind === "clip" || (shaders.get(m.key) || {}).status !== "bad");
  const laneCls = m.kind === "fx" || isFxLane(m.lane) ? "fx" : m.kind === "clip" ? `clip ${m.lane}` : m.lane;
  const chosen = sel && sel.type === "member" && sel.id === idOf(m.ref) && (sel.k === k || m.cued || m.always || sel.k == null);
  const cls = ["block", laneCls, live ? "" : "engine", m.cued ? "cued" : "", m.always ? "always" : "", m.enabled ? "" : "off", s.missing ? "missing" : "", chosen ? "sel" : "",
    sigLight && sigLight.ids.has(idOf(m.ref)) ? "sig" : ""].filter(Boolean).join(" ");
  const linked = m.params.filter(p => p.knob != null).length;
  let label = m.label;
  if (m.kind === "clip" && s.file && !m.name) label = clipLabel(s.file);
  label += m.always ? " (the whole song)" : "";
  const pre = m.kind === "gen" && !isFxLane(m.lane) ? "GEN " : "";
  const tags = (s.from ? `@${fmt(s.from)}s`.length + 1 : 0) + (linked ? 4 : 0) + (m.own_fx ? 6 : 0);
  const room = Math.floor((width - 14) / 6.2) - tags - pre.length;      // a long name: its end shows (where stems differ)
  const shown = pre + (room > 4 && label.length > room ? "…" + label.slice(label.length - room + 1) : label);
  const colour = m.kind === "clip" && m.colour ? `; --bc:${m.colour}` : "";
  const file = m.kind === "clip" ? s.file || m.file : null, cov = file ? blockCover(file) : null;
  const pic = cov ? `; background-image:url(${cov}); background-size:${Math.min(ppb, 2 * height)}px 100%` : "";
  const el = h("div", { class: cls + (colour ? " tinted" : "") + (cov ? " thumb" : ""), dataset: { id: idOf(m.ref), k }, style: `left:${left}px; width:${Math.max(4, width)}px; top:${top}px; height:${height}px${colour}${pic}`,
    title: `${m.label} (${kindName(m)}) · bars ${s.first}–${s.last} of ${NB()}${m.cued ? " · cued (REAPER): hide bars of it from its menu" : ""}${m.always ? " · the whole song: drag an end to give it bars" : ""}${live ? "" : " · at the render"}${m.enabled ? "" : " · off"}${s.missing ? ` · MISSING: ${s.file || m.file}` : ""}${s.file || m.file ? `\n${s.file || m.file}` : ""}` },
    h("span", { class: "nm", text: shown }),
    s.from ? h("span", { class: "sg", text: `@${fmt(s.from)}s` }) : null,
    linked ? h("span", { class: "sg", text: `${linked}~` }) : null,
    m.own_fx ? h("span", { class: "sg", text: `+${m.own_fx}fx` }) : null);
  if (m.own_fx) el.append(h("i", { class: "fxstrip", title: m.fx.map(f => f.label).join(", ") }));
  return el;
}
// the covers of the clips on the lanes (any folder), asked of Python as blocks need them
const blockCovers = new Map(), coversAsked = new Set();
let coversTimer = 0;
function blockCover(file) {
  if (blockCovers.has(file)) return blockCovers.get(file);
  if (!coversAsked.has(file)) {
    coversAsked.add(file);
    clearTimeout(coversTimer);
    coversTimer = setTimeout(async () => {
      const want = [...coversAsked].filter(f => !blockCovers.has(f));
      const got = await ask("covers_for", want);
      for (const f of want) blockCovers.set(f, (got && got[f]) || null);
      if (got && Object.keys(got).length) later(renderTimeline);
    }, 120);
  }
  return null;
}
// a clip's name on its block: its file's stem; a generic one ("final.mp4") with its folder
function clipLabel(path) {
  const parts = String(path).split(/[\\/]/).filter(Boolean), st = (parts.pop() || "").replace(/\.[^.]+$/, "");
  return /^(final|output|out|render|video|clip|take|\d+)$/i.test(st) && parts.length ? `${parts.pop()}/${st}` : st;
}
const memberById = id => members.find(m => idOf(m.ref) === id);
function laneAtY(y) { return layout && layout.lanes.find(l => y >= l.top && y < l.top + l.h) || null; }
function innerXY(e) { const r = $("inner").getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
function rowAtY(l, y) { return Math.max(0, Math.min(Math.max(0, l.shown - 1), Math.floor((y - l.top - 2) / rowH))); }
function accepts(l, it) {
  if (!it || !l) return false;
  if (l.kind === "clips") return it.type === "clip";
  if (l.kind !== "track") return false;
  if (isFxLane(l.id)) return it.type === "cat" && it.kind === "fx";
  return it.type === "clip" || (it.type === "cat" && it.kind === "gen");
}

// edits: Python answers with the lanes. They go one after another, in the order made (one
// made while another is on its way waits for it).
let editChain = Promise.resolve();
function edit(name, ...args) {
  const run = async () => {
    try {
      const st = await call(name, ...args);
      if (st && st.error) { toast(st.error, true); return st; }      // Python logged it, with where
      if (st && st.lanes) apply(st);
      if (st && st.note) { toast(st.note); log("in", st.note); lastNote = st.note; showStatus(); }
      return st;
    } catch (e) { toast(e.message, true); log("err", `${name}: ${e.message}`); }
  };
  const p = editChain.then(run);
  editChain = p.catch(() => { });
  return p;
}
function apply(st) {
  if (!st || !st.lanes) return;
  members = st.lanes; pins = st.pins || {}; undoN = st.undo; redoN = st.redo;
  rowsOf = st.rows || {}; groups = st.groups || {}; has = st.has || {}; board = st.board || {};
  if (st.montage) montage = st.montage;
  else if (!has.clips) montage = [];          // the montage taken out of the scene
  if (st.montage_bars) montageBars = st.montage_bars;
  streamEdited();
  if ("dirty" in st) { dirty = st.dirty; savedAt = st.saved_at || ""; }
  if (st.select !== undefined && !subStack.length) {     // the plugin waits for the area under it: it stays
    const s2 = st.select;
    if (!s2 || s2.ref === null) sel = null;
    else if (s2.montage) sel = { type: "montage", first: s2.montage[0], last: s2.montage[1] };
    else if (s2.clips) sel = { type: "clips" };
    else if (s2.lane) sel = s2.lane === "master" ? { type: "master" } : { type: "lane", group: s2.lane };
    else if (s2.ref) sel = { type: "member", id: idOf(s2.ref), ref: s2.ref, k: s2.k };
  }
  if (sel && sel.type === "member" && !memberById(sel.id)) sel = null;
  if (sel && (sel.type === "clips" || sel.type === "montage") && !has.clips) sel = null;
  for (const m of members) if (m.live && m.kind !== "clip") shaderFor(m.key);
  renderTimeline(); renderInspector(); showSaveState(); showStatus();
}
// the status line under the lanes: what the last edit did, the undo steps
function showStatus() {
  const el = $("statusLine");
  if (!el) return;
  el.textContent = !S ? "" : [lastNote, `undo ${undoN} · redo ${redoN}`].filter(Boolean).join(" · ");
}
function showSaveState(text, bad = false) {
  const el = $("saveState");
  el.textContent = text || (!S ? "" : dirty ? "unsaved changes" : savedAt ? `saved at ${savedAt}` : "");
  el.classList.toggle("dirty", !text && dirty); el.classList.toggle("bad", bad);
  $("bSave").textContent = dirty ? "Save *" : "Save";
  el.title = S && S.project.scene ? S.project.scene : "";
}

// ---------------------------------------------------------------------------
// Pointer work on the lanes
// ---------------------------------------------------------------------------
let drag = null, rsel = null;
const inner = $("inner");
let shiftHeld = false, hoverAt = null;     // Shift held: a block's ends resize it
function edgeOf(el, clientX, shift = true) {
  if (!shift || el.classList.contains("cued")) return null;
  const r = el.getBoundingClientRect(), e = Math.min(EDGE, r.width / 3);
  return clientX - r.left < e ? "first" : r.right - clientX < e ? "last" : null;
}
function showEdge(el, end) {
  const ed = $("edge");
  if (!ed) return;
  if (!el || !end) { ed.hidden = true; return; }
  ed.hidden = false;
  ed.style.left = (end === "first" ? el.offsetLeft - 1 : el.offsetLeft + el.offsetWidth - 2) + "px";
  ed.style.top = el.offsetTop + "px"; ed.style.height = el.offsetHeight + "px";
}
function select(s, redraw = true) {
  if (subStack.length && JSON.stringify(s) !== JSON.stringify(sel)) { subBusy(); return; }
  sel = s;
  if (redraw) renderTimeline();
  else {                       // in the middle of a drag: the blocks stay the same elements
    document.querySelectorAll(".block.sel, .cell.sel").forEach(x => x.classList.remove("sel"));
    if (s && s.type === "member") inner.querySelectorAll(".block").forEach(x => { if (x.dataset.id === s.id && (+x.dataset.k === s.k || (memberById(s.id) || {}).cued)) x.classList.add("sel"); });
  }
  renderInspector();
}
function selectMember(m, k) { select({ type: "member", id: idOf(m.ref), ref: m.ref, k }, false); }
// a member's bars (a block's; a cued one's: its first cue to its last), chosen
function barsOf(m, k) {
  const sp = spansOf(m);
  if (!sp.length) return null;
  if (k != null && sp[k]) return [sp[k].first, sp[k].last];
  return [Math.min(...sp.map(s => s.first)), Math.max(...sp.map(s => s.last))];
}
function chooseBarsOf(m, k) { const b = barsOf(m, k); if (b) loop = b; renderTimeline(); renderInspector(); }
inner.addEventListener("pointerdown", e => {
  if (e.button !== 0 || !S) return;
  closeCtx();
  const [x, y] = innerXY(e), l = laneAtY(y), bar = xToBar(x);
  const bEl = e.target.closest(".block"), note = e.target.closest(".cuenote");
  if (note) { const m = memberById(note.dataset.id); if (m) { selectMember(m, null); chooseBarsOf(m, null); } return; }
  if (bEl) {
    const m = memberById(bEl.dataset.id), k = +bEl.dataset.k;
    selectMember(m, m.always ? null : k);
    if (m.kind === "clip") { const s = spansOf(m)[k]; if (s && (s.file || m.file)) chooseClip(s.file || m.file, s.from || 0); }
    if (m.cued) { chooseBarsOf(m, null); return; }
    const s = spansOf(m)[Math.max(0, k)];
    drag = { type: "block", m, k, el: bEl, mode: edgeOf(bEl, e.clientX, e.shiftKey) || "move", x0: e.clientX, y0: y, f0: s.first, l0: s.last, first: s.first, last: s.last, lane: m.lane, row: m.row, moved: false };
    try { bEl.setPointerCapture(e.pointerId); } catch (err) { /* a pointer that is not down */ }
    return;
  }
  if (e.target.closest(".cell") || (l && l.kind === "clips")) {
    const end = e.shiftKey ? runEdgeAt(x) : null;
    if (end) { drag = { type: "runend", run: end.run, end: end.end, first: end.run.first, last: end.run.last, moved: false }; try { inner.setPointerCapture(e.pointerId); } catch (err) { /* not down */ } return; }
    drag = { type: "range", bar0: bar, first: bar, last: bar, moved: false }; return;
  }
  if (l && l.kind === "track") drag = { type: "new", lane: l.id, row: rowAtY(l, y), bar0: bar, first: bar, last: bar, moved: false };
  else drag = { type: "span", bar0: bar, first: bar, last: bar, moved: false };
  try { inner.setPointerCapture(e.pointerId); } catch (err) { /* a pointer that is not down */ }
});
inner.addEventListener("pointermove", e => {
  const [x, y] = innerXY(e);
  if (!drag) {
    const cl = laneAtY(y);                    // the montage: Shift on a clip's end shows it can be dragged
    if (cl && cl.kind === "clips") {
      const end = e.shiftKey ? runEdgeAt(x) : null;
      inner.style.cursor = end ? "ew-resize" : "";
      showRunEdge(end ? (end.end === "first" ? end.run.first : end.run.last + 1) : null);
      return;
    }
    inner.style.cursor = "";
    const bEl = e.target.closest(".block");
    hoverAt = bEl ? { el: bEl, x: e.clientX } : null;
    const end = bEl ? edgeOf(bEl, e.clientX, e.shiftKey) : null;
    if (bEl) bEl.style.cursor = bEl.classList.contains("cued") ? "default" : end ? "ew-resize" : "grab";
    showEdge(bEl, end);
    return;
  }
  const bar = xToBar(x);
  if (drag.type === "block") {
    const dB = Math.round((e.clientX - drag.x0) / ppb), len = drag.l0 - drag.f0;
    if (drag.mode === "move") { drag.first = Math.max(1, Math.min(NB() - len, drag.f0 + dB)); drag.last = drag.first + len; }
    else if (drag.mode === "first") drag.first = Math.max(1, Math.min(drag.l0, drag.f0 + dB));
    else drag.last = Math.max(drag.f0, Math.min(NB(), drag.l0 + dB));
    let l = laneById(drag.m.lane), row = drag.m.row, lane = drag.m.lane;
    if (drag.mode === "move" && Math.abs(y - drag.y0) > rowH / 2) {
      const over = laneAtY(y);
      if (over && over.kind === "track" && sameKind(over.id, drag.m.lane)) { l = over; lane = over.id; row = rowAtY(over, y); }
    }
    drag.lane = lane; drag.row = row;
    drag.moved = drag.moved || dB !== 0 || lane !== drag.m.lane || row !== drag.m.row;
    Object.assign(drag.el.style, { left: (drag.first - 1) * ppb + 1 + "px", width: (drag.last - drag.first + 1) * ppb - 2 + "px", top: rowTop(l, row) + 1 + "px", cursor: drag.mode === "move" ? "grabbing" : "ew-resize" });
    if (drag.mode !== "move") showEdge(drag.el, drag.mode);
    return;
  }
  if (drag.type === "runend") {
    const r = drag.run;
    if (drag.end === "first") drag.first = Math.max(1, Math.min(r.last, Math.round(x / ppb) + 1));
    else drag.last = Math.max(r.first, Math.min(NB(), Math.round(x / ppb)));
    drag.moved = drag.first !== r.first || drag.last !== r.last;
    document.querySelectorAll(".cell").forEach(c => c.classList.toggle("sel", +c.dataset.bar >= drag.first && +c.dataset.bar <= drag.last));
    showRunEdge(drag.end === "first" ? drag.first : drag.last + 1);
    return;
  }
  drag.moved = drag.moved || bar !== drag.bar0;
  drag.first = Math.min(drag.bar0, bar); drag.last = Math.max(drag.bar0, bar);
  if (drag.type === "new") {
    const l = laneById(drag.lane), g = $("ghost");
    g.hidden = false;
    Object.assign(g.style, { left: (drag.first - 1) * ppb + "px", width: (drag.last - drag.first + 1) * ppb + "px", top: rowTop(l, drag.row) + "px", height: rowH - 2 + "px" });
    const pk = pickFor(l);
    g.textContent = pk && accepts(l, pk) ? `${pk.label || pk.name} · bars ${drag.first}–${drag.last}` : `bars ${drag.first}–${drag.last}`;
    g.classList.toggle("put", !!(pk && accepts(l, pk)));
  } else if (drag.type === "range") {
    document.querySelectorAll(".cell").forEach(c => c.classList.toggle("sel", +c.dataset.bar >= drag.first && +c.dataset.bar <= drag.last));
  } else if (drag.type === "span" && drag.moved) { loop = [drag.first, drag.last]; showLoop(); }
});
inner.addEventListener("pointerup", e => {
  if (!drag) return;
  const d = drag; drag = null;
  const [x] = innerXY(e), bar = xToBar(x);
  if (d.type === "block") {
    showEdge(null);
    if (!d.moved) { chooseBarsOf(d.m, d.m.always ? null : d.k); flushAfterDrag(); return; }
    const toRow = d.lane !== d.m.lane || d.row !== d.m.row;
    edit("move", d.m.ref, d.k, d.mode, d.first, d.last, toRow ? d.lane : null, toRow ? d.row : null);
  } else if (d.type === "new") {
    $("ghost").hidden = true;
    const l = laneById(d.lane), pk = pickFor(l);
    if (pk && accepts(l, pk)) edit("new_member", d.lane, pk.type === "clip" ? pk.path : pk.key, d.first, d.last, d.row, pk.from || 0);
    else if (d.moved) { setLoop(d.first, d.last); toast(`bars ${d.first}–${d.last} chosen · to put something there, pick it in the browser first`); }
    else { if (!subStack.length) sel = null; seek(xToT(x)); loop = [bar, bar]; renderTimeline(); renderInspector(); }     // an empty place: that bar chosen
  } else if (d.type === "runend") {
    showRunEdge(null);
    if (d.moved) edit("resize_run", d.run.first, d.run.last, d.first, d.last);
    else renderTimeline();
  } else if (d.type === "range") {
    let first = d.first, last = d.last;
    if (!d.moved) { const run = runAt(bar); if (run) { first = run.first; last = run.last; if (run.path) chooseClip(run.path, 0); } }
    loop = [first, last];
    select({ type: "montage", first, last });
  } else if (d.type === "span") {
    if (!d.moved) { seek(xToT(x)); if (!subStack.length) sel = null; renderTimeline(); renderInspector(); }
    else { seek(B[loop[0] - 1]); log("sys", `bars ${loop[0]}–${loop[1]} chosen`); renderInspector(); }
  }
  flushAfterDrag();
});
function flushAfterDrag() { const fns = pendingAfterDrag; pendingAfterDrag = []; for (const fn of fns) fn(); }
inner.addEventListener("pointercancel", () => { if (drag) renderTimeline(); drag = null; showEdge(null); flushAfterDrag(); });
inner.addEventListener("pointerleave", () => { if (!drag) showEdge(null); });
inner.addEventListener("dblclick", e => {
  const bEl = e.target.closest(".block");
  if (bEl) { const m = memberById(bEl.dataset.id); if (m && m.kind !== "clip") openPlugin(m); }
});
// the ruler: click to go there, drag to choose bars, double-click to choose none
const rulerEl = $("ruler");
rulerEl.addEventListener("pointerdown", e => {
  if (e.button !== 0 || !S) return;
  const x = e.clientX - rulerEl.getBoundingClientRect().left;
  rsel = { bar0: xToBar(x), moved: false };
  try { rulerEl.setPointerCapture(e.pointerId); } catch (err) { /* a pointer that is not down */ }
});
rulerEl.addEventListener("pointermove", e => {
  if (!rsel) return;
  const b = xToBar(e.clientX - rulerEl.getBoundingClientRect().left);
  if (b !== rsel.bar0) rsel.moved = true;
  if (rsel.moved) { loop = [Math.min(rsel.bar0, b), Math.max(rsel.bar0, b)]; showLoop(); }
});
rulerEl.addEventListener("pointerup", e => {
  if (!rsel) return;
  const r = rsel; rsel = null;
  if (!r.moved) seek(xToT(e.clientX - rulerEl.getBoundingClientRect().left));
  else { seek(B[loop[0] - 1]); renderTimeline(); renderInspector(); }
  flushAfterDrag();
});
rulerEl.addEventListener("dblclick", () => { loop = null; renderTimeline(); renderInspector(); });
// the heads: a lane's settings and FX, the master's, the montage's
$("heads").addEventListener("click", e => {
  const hd = e.target.closest(".head.pick");
  if (!hd) return;
  const l = LANES.find(x => x.id === hd.dataset.lane);
  if (l.id === "master") select({ type: "master" });
  else if (l.id === "clips") select({ type: "clips" });
  else select({ type: "lane", group: l.group });
});
$("heads").addEventListener("contextmenu", async e => {
  const hd = e.target.closest(".head");
  if (!hd || !S) return;
  e.preventDefault();
  const l = LANES.find(x => x.id === hd.dataset.lane);
  if (!l || !["track", "clips"].includes(l.kind)) return;
  if (l.id === "master") select({ type: "master" }); else if (l.id === "clips") select({ type: "clips" }); else select({ type: "lane", group: l.group });
  const first = loop ? loop[0] : 1, last = loop ? loop[1] : NB();
  const items = l.kind === "clips" ? await ask("montage_menu", first, last, picked && picked.type === "clip" ? picked.path : null)
    : await ask("menu", l.id, null, null, barAt(now()), loop);
  if (items) menuAt(e.clientX, e.clientY, pythonItems(items));
});
// the wheel: Ctrl zooms along the song around the bar under the mouse, Shift makes the rows
// taller or lower (the row under the mouse stays), Alt moves along; else it scrolls
$("scroll").addEventListener("wheel", e => {
  if (!S || !(e.ctrlKey || e.shiftKey || e.altKey)) return;
  e.preventDefault();
  const sc = $("scroll"), up = e.deltaY < 0;
  if (e.ctrlKey) {
    const r = sc.getBoundingClientRect(), mx = e.clientX - r.left - HEAD_W;
    zoomTo(Math.round(ppb * (up ? 1.25 : 0.8)), xToT(sc.scrollLeft + mx), mx);
  } else if (e.shiftKey) {
    const r = sc.getBoundingClientRect(), my = e.clientY - r.top - RULER_H, at = sc.scrollTop + my;
    const old = rowH;
    rowH = Math.max(ROW_MIN, Math.min(ROW_MAX, Math.round(rowH * (up ? 1.2 : 1 / 1.2))));
    if (rowH === old) return;
    renderTimeline();
    sc.scrollTop = Math.max(0, at * rowH / old - my);
  } else sc.scrollLeft += (up ? -1 : 1) * Math.max(1, Math.round(sc.clientWidth / ppb / 8)) * ppb;
}, { passive: false });
function zoomTo(want, t, screenX) {
  want = Math.max(+$("zoom").min, Math.min(+$("zoom").max, want));
  $("zoom").value = want;
  if (zoomAt == null) zoomAt = { t, screenX };
  const k = want / ppb, sc = $("scroll");
  for (const el of [inner, rulerEl]) { el.style.transformOrigin = "0 0"; el.style.transform = `scaleX(${k})`; }
  sc.scrollLeft = Math.max(0, tToX(zoomAt.t) * k - (zoomAt.screenX ?? sc.clientWidth / 2));
  clearTimeout(zoomTimer); zoomTimer = setTimeout(commitZoom, 180);
}
let zoomTimer = 0, zoomAt = null;
function commitZoom() {
  clearTimeout(zoomTimer);
  if (zoomAt == null) return;
  const { t, screenX } = zoomAt; zoomAt = null;
  ppb = +$("zoom").value;
  for (const el of [inner, rulerEl]) el.style.transform = "";
  renderTimeline();
  $("scroll").scrollLeft = Math.max(0, tToX(t) - (screenX ?? $("scroll").clientWidth / 2));
}
function zoomAll() { const w = $("scroll").clientWidth - HEAD_W - 8; zoomTo(Math.max(+$("zoom").min, Math.floor(w / NB())), 0, 0); commitZoom(); }
function setRows(v) { rowH = Math.max(ROW_MIN, Math.min(ROW_MAX, v)); renderTimeline(); }
function setLoop(first, last) { loop = [Math.min(first, last), Math.max(first, last)]; renderTimeline(); renderInspector(); }
function showLoop() {
  let band = $("loopband");
  if (!band) { band = h("div", { class: "loopband", id: "loopband" }); inner.insertBefore(band, $("playhead")); }
  band.style.left = (loop[0] - 1) * ppb + "px"; band.style.width = (loop[1] - loop[0] + 1) * ppb + "px";
  updateLoopChip();
}
function updateLoopChip() {
  $("loopChip").hidden = !loop;
  if (loop) $("loopText").textContent = loop[0] === loop[1] ? `bar ${loop[0]}` : `bars ${loop[0]}–${loop[1]}`;
  if (rendering) return;
  const [a, b] = exactSpan();
  $("bRender").textContent = a === b ? `Exact bar ${a}` : `Exact bars ${a}–${b}`;
}
// the exact preview's bars: the chosen ones (at most 8), else N from the playhead's bar
let exactN = 1;
try { exactN = Math.max(1, Math.min(8, +localStorage.getItem("arc.exactN") || 1)); } catch (e) { /* no storage */ }
function exactSpan() {
  if (!S) return [1, 1];
  if (loop) return [loop[0], Math.min(loop[1], loop[0] + 7)];
  const bar = barAt(now());
  return [bar, Math.min(NB(), bar + exactN - 1)];
}
function exactBars(first, last) { loop = [first, Math.min(last, first + 7)]; renderTimeline(); renderInspector(); startRender(); }
// Preview: the chosen bars played live, over and over (at most 8)
function previewBars(first, last) {
  loop = [first, last];
  renderTimeline(); renderInspector();
  seek(B[loop[0] - 1]); setView("live"); setPlaying(true);
}
function deleteSelected() {
  if (!sel) return;
  if (sel.type === "member") {
    const m = memberById(sel.id);
    if (!m) return;
    if (sel.k == null || m.always || m.cued) edit("remove", m.ref);
    else edit("delete_block", m.ref, sel.k);
  } else if (sel.type === "montage") edit("set_pins", sel.first, sel.last, "none");
}
function duplicateSpan(m, k) {
  if (m.cued) return;
  const spans = m.always ? [] : m.spans.map(s => ({ ...s }));
  const s = spansOf(m)[Math.max(0, k)], len = s.last - s.first;
  const first = Math.min(NB() - len, s.last + 1);
  spans.push({ ...s, first, last: first + len });
  edit("spans", m.ref, spans);
}

// ---------------------------------------------------------------------------
// Menus: what Python says can be done there; the catalog's entries by category
// ---------------------------------------------------------------------------
function closeCtx() { for (const d of [document, ...outDocs()]) d.querySelectorAll(".ctx").forEach(x => x.remove()); }
function wireDoc(d) {                           // a document the page's panels live in
  d.addEventListener("pointerdown", e => { menuDoc = d; if (!e.target.closest(".ctx")) closeCtx(); }, true);
  d.addEventListener("pointerover", e => {      // a select too narrow for its choice: all of it on hover
    const s = e.target.closest && e.target.closest("select"), o = s && s.selectedOptions[0];
    if (o && (!s.title || s.dataset.autoTitle)) { s.title = o.textContent; s.dataset.autoTitle = "1"; }
  }, true);
  d.addEventListener("contextmenu", () => { menuDoc = d; }, true);
}
wireDoc(document);
function catalogItems(kind, choose) {
  const byCat = new Map();
  for (const c of S.catalog) if (c.kind === kind) { if (!byCat.has(c.cat)) byCat.set(c.cat, []); byCat.get(c.cat).push(c); }
  return [...byCat].map(([cat, es]) => ({
    label: cat, sub: es.map(c => ({
      label: c.label + (c.takes_input ? " *" : "") + c.needs.map(n => ` (${n})`).join(""), title: c.about, on: () => choose(c.key) })) }));
}
function runItem(it, key) {
  if (it.page === "plugin") { const m = members.find(x => idOf(x.ref) === idOf(it.args[0])); if (m) openPlugin(m); return; }
  if (it.page === "preview") { previewBars(it.args[0], it.args[1]); return; }
  if (it.page === "exact") { exactBars(it.args[0], it.args[1]); return; }
  if (it.catalog === "gen" && key) { const c = catalogByKey.get(key); if (c) { picked = { type: "cat", key, kind: "gen", label: c.label }; showPicked(); } }
  if (it.call === "add_montage") { addMontage(...it.args); return; }
  const args = it.args.map(a => a === "$key" ? key : a === "$from" ? (picked && picked.from) || 0 : a);
  edit(it.call, ...args).then(r => {                // a block's new FX: its seeds, or its parameters below
    if (r && !r.error && it.call === "add_fx" && Array.isArray(args[0]) && args[0].length > 1) newOwnFx(args[0], key);
  });
}
// a clip montage at the BG's bottom (bars first..last, or the whole song); a project with no
// clips folder chooses one first (it is probed, then the montage is drawn)
async function addMontage(...span) {
  let r = await edit("add_montage", ...span);
  if (r && r.needs === "clips_dir" && await chooseClipsDir()) r = await edit("add_montage", ...span);
  return r;
}
async function chooseClipsDir() {
  let path = null;
  if (inWebview) { const r = await ask("pick_file", "folder", false, "", ""); path = r && r.path; }
  else { const v = await dialog("The clips folder", "The montage draws its clips from a folder (the project's clips folder): its path.", [{ label: "Use it", value: "ok", primary: true }, { label: "Cancel", value: null }], ""); path = v && v.value && v.text; }
  if (!path) return false;
  const r = await call("set_render", { clips_dir: path }).catch(e => ({ error: e.message }));
  if (!r || r.error) { toast(r ? r.error : "no answer", true); return false; }
  toast(`clips folder: ${r.clips_dir || path}`);
  return true;
}
// items: {label, on, title, danger, disabled, sub: [...]} | {head} | {sep}
function menuAt(x, y, items, parent = null) {
  if (!parent) closeCtx();
  const box = h("div", { class: "ctx" + (parent ? " sub" : ""), role: "menu" });
  for (const it of items) {
    if (it.head) { box.append(h("div", { class: "sub", text: it.head })); continue; }
    if (it.sep) { box.append(h("hr")); continue; }
    const b = h("button", { role: "menuitem", class: it.sub ? "hassub" : null, title: it.title || null, disabled: it.disabled, style: it.danger ? "color:var(--danger)" : null }, it.label, it.hint ? h("small", { text: it.hint }) : null);
    if (it.sub) {
      const open = () => {
        box.querySelectorAll(":scope > .ctx").forEach(c => c.remove());
        const r = b.getBoundingClientRect();
        menuAt(r.right - 2, r.top - 4, it.sub, box);
      };
      b.addEventListener("pointerenter", open); b.addEventListener("click", open);
    } else b.addEventListener("click", () => { closeCtx(); it.on(); });
    if (!it.sub) b.addEventListener("pointerenter", () => box.querySelectorAll(":scope > .ctx").forEach(c => c.remove()));
    box.append(b);
  }
  const doc = parent ? parent.ownerDocument : menuDoc, view = doc.defaultView || window;
  (parent || doc.body).append(box);
  const r = box.getBoundingClientRect();
  if (parent) { box.style.position = "fixed"; }
  box.style.left = Math.max(4, Math.min(x, view.innerWidth - r.width - 6)) + "px";
  box.style.top = Math.max(4, Math.min(y, view.innerHeight - r.height - 6)) + "px";
  if (!parent) box.querySelector("button")?.focus();
  return box;
}
function pythonItems(items) {
  return items.map(it => it.head || it.sep ? it : it.catalog
    ? { label: it.label, sub: catalogItems(it.catalog, key => runItem(it, key)) }
    : { label: it.label, danger: it.danger, on: () => runItem(it) });
}
inner.addEventListener("contextmenu", async e => {
  if (!S) return;
  e.preventDefault();
  const [x, y] = innerXY(e), l = laneAtY(y), bar = xToBar(x);
  if (!l) return;
  if (l.kind === "clips") {
    if (!(sel && sel.type === "montage" && sel.first <= bar && bar <= sel.last)) {
      const run = runAt(bar); select(run ? { type: "montage", first: run.first, last: run.last } : { type: "montage", first: bar, last: bar });
    }
    const items = await ask("montage_menu", sel.first, sel.last, picked && picked.type === "clip" ? picked.path : null);
    if (items) menuAt(e.clientX, e.clientY, pythonItems(items));
    return;
  }
  if (l.kind !== "track") { const items = await ask("menu", "master", null, null, bar, loop); if (items) menuAt(e.clientX, e.clientY, pythonItems(items)); return; }
  const bEl = e.target.closest(".block, .cuenote");
  let m = null, k = null;
  if (bEl) { m = memberById(bEl.dataset.id); k = bEl.dataset.k != null ? +bEl.dataset.k : null; if (m) selectMember(m, m.always || m.cued ? null : k); }
  const items = await ask("menu", l.id, m ? m.ref : null, m && !m.always && !m.cued ? k : null, bar, loop);
  if (items) menuAt(e.clientX, e.clientY, pythonItems(items));
});

// drops from the browser: what it picked, four bars from the bar under the mouse
let dragItem = null;
function dragBars(it) {
  if (!it) return 4;
  if (it.type !== "clip") return 4;
  const len = clipLens[it.url] || (clipChosen && clipChosen.path === it.path ? clipChosen.duration : null);
  if (!len || !S.song.bar_seconds) return 4;
  return Math.max(1, Math.round((len - (it.from || 0)) / S.song.bar_seconds));
}
function dragGhost(l, e) {
  const g = $("ghost");
  if (!g || !dragItem) return;
  const [x, y] = innerXY(e), bar = xToBar(x), ok = accepts(l, dragItem);
  const n = dragBars(dragItem), last = Math.min(NB(), bar + n - 1), row = l.kind === "track" ? rowAtY(l, y) : 0;
  g.hidden = false; g.classList.toggle("put", ok); g.classList.toggle("no", !ok);
  Object.assign(g.style, { left: (bar - 1) * ppb + "px", width: (last - bar + 1) * ppb + "px", top: (l.kind === "track" ? rowTop(l, row) : l.top + 2) + "px", height: rowH - 2 + "px" });
  g.textContent = ok ? `${dragItem.label || dragItem.name} · bars ${bar}–${last}${n > 16 ? " (it asks how many)" : ""}`
    : dragItem.type === "clip" ? "a clip goes on the BG, the FG or the montage" : dragItem.kind === "fx" ? "an FX goes on an FX track" : "a Generator goes on the BG or the FG";
}
function wireLaneDrop(el, l) {
  el.addEventListener("dragover", e => {
    if (!dragItem) return;
    dragGhost(l, e);
    if (accepts(l, dragItem)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; el.classList.add("drop-ok"); }
  });
  el.addEventListener("dragleave", () => { el.classList.remove("drop-ok"); const g = $("ghost"); if (g) g.hidden = true; });
  el.addEventListener("drop", e => {
    el.classList.remove("drop-ok"); const g = $("ghost"); if (g) g.hidden = true;
    if (!accepts(l, dragItem)) return;
    e.preventDefault();
    const it = dragItem; dragItem = null;
    const [x, y] = innerXY(e), bar = xToBar(x);
    putItem(it, l.id, bar, l.kind === "track" ? rowAtY(l, y) : null);
  });
}
function putItem(it, laneId, bar, row = null) {
  if (it.type === "clip") return placeClips(laneId === "clips" ? "clips" : laneId, bar, [{ path: it.path, from: it.from || 0 }], row);
  const last = Math.min(NB(), bar + 3);
  if (laneId === "clips") return edit("put_in_montage", bar, bar, it.path, it.from || 0);
  return edit("new_member", laneId, it.type === "clip" ? it.path : it.key, bar, last, row, it.from || 0);
}

// ---------------------------------------------------------------------------
// Browser: clips, Generators, FX. A click picks one (then drag across a lane to put it
// there); dragging one onto a lane puts four bars of it.
// ---------------------------------------------------------------------------
let tab = "gen";
const foldedCats = new Set();      // "gen:Category" folded in the browser
function pickClip(path, from = 0) {
  const c = (S.clips || []).find(x => x.path === path || x.path.replace(/\\/g, "/").endsWith(String(path).replace(/\\/g, "/")));
  picked = { type: "clip", path, name: String(path).split(/[\\/]/).pop(), label: String(path).split(/[\\/]/).pop().replace(/\.[^.]+$/, ""), url: c ? c.url : null, from };
  showPicked(); if (tab === "clips") renderBrowser();
}
function pick(it) {
  const same = x => x && (x.key || x.path) === (it.key || it.path);
  if (it.type === "cat" && it.kind === "fx") pickedFx = same(pickedFx) ? null : it;
  else picked = same(picked) ? null : it;
  showPicked(); renderBrowser(); renderTimeline();
}
// what is picked: a clip or a Generator (for the BG and the FG), and an FX (for the FX tracks)
function showPicked() {
  const el = $("picked");
  if (!picked && !pickedFx) { el.hidden = true; return; }
  el.hidden = false;
  const line = (p, hint, unpick) => h("div", { class: "row nowrap" }, h("span", { class: "tlabel", text: p.kind === "fx" ? "FX" : p.type === "clip" ? "clip" : "gen" }), h("b", { text: p.label || p.name }),
    p.type === "clip" && p.from ? h("span", { class: "mono", text: `from ${fmt(p.from)} s` }) : null,
    h("span", { class: "hint", text: hint }), h("button", { class: "lnkbtn", onclick: unpick, "aria-label": "Unpick" }, "✕"));
  el.replaceChildren(...[picked ? line(picked, "drag across the BG or the FG", () => { picked = null; showPicked(); renderBrowser(); renderTimeline(); }) : null,
    pickedFx ? line(pickedFx, "drag across an FX track", () => { pickedFx = null; showPicked(); renderBrowser(); renderTimeline(); }) : null].filter(Boolean));
}
function renderBrowser() {
  if (!S) return;
  const q = $("search").value.trim().toLowerCase(), list = $("blist");
  document.querySelectorAll(".browser .tab").forEach(b => b.setAttribute("aria-selected", b.dataset.tab === tab));
  list.replaceChildren();
  const drag = (el, it) => {
    el.draggable = true;
    el.addEventListener("dragstart", e => { dragItem = it; e.dataTransfer.setData("text/plain", it.key || it.path); e.dataTransfer.effectAllowed = "copy"; });
    el.addEventListener("dragend", () => { dragItem = null; document.querySelectorAll(".drop-ok").forEach(x => x.classList.remove("drop-ok")); });
  };
  const isPicked = it => [picked, pickedFx].some(p => p && (p.key || p.path) === (it.key || it.path));
  if (tab === "signals") { list.append(h("div", { class: "about", text: "…" })); renderSignals(list); return; }
  if (tab === "clips") { renderClipTab(list); return; }
  const byCat = new Map();
  for (const c of S.catalog) if (c.kind === tab && `${c.label} ${c.cat} ${c.about}`.toLowerCase().includes(q)) {
    if (!byCat.has(c.cat)) byCat.set(c.cat, []); byCat.get(c.cat).push(c);
  }
  list.append(h("div", { class: "about tababout", text: tab === "gen" ? "Generators make a picture: drag one across the BG or the FG (click: its demo and parameters)." : "FX change the picture under them: drag one across an FX track, or Add FX on a block, a lane, the master." }));
  if (!byCat.size) list.append(h("div", { class: "empty-note", text: "Nothing matches." }));
  for (const [cat, items] of byCat) {
    const shut = foldedCats.has(`${tab}:${cat}`) && !q;
    list.append(h("button", { class: "bcat" + (shut ? " shut" : ""), "aria-expanded": String(!shut), onclick: () => { const k = `${tab}:${cat}`; foldedCats.has(k) ? foldedCats.delete(k) : foldedCats.add(k); renderBrowser(); } },
      h("span", { text: cat }), h("small", { text: String(items.length) })));
    if (shut) continue;
    for (const c of items) {
      const st = shaders.get(c.key);
      const bad = st && st.status === "bad";
      const it = { type: "cat", key: c.key, kind: c.kind, label: c.label };
      const flags = (c.takes_input ? " *" : "") + c.needs.map(n => ` (${n})`).join("");
      const row = h("div", { class: "bitem" + (isPicked(it) ? " picked" : ""), title: (c.about || c.key) + (c.takes_input ? "\n* takes the picture below as input" : "") + (bad ? `\nGLSL ES: ${st.error}` : ""),
        onclick: e => { if (e.target.closest("button")) return; pick(it); entryKey = c.key; showEntryPanel(); } },
        h("span", { class: "nm", text: c.label + flags }),
        h("span", { class: `badge ${c.live && !bad ? "live" : bad ? "bad" : "engine"}`, text: c.live && !bad ? "live" : bad ? "glsl" : "engine" }),
        h("button", { class: "add", title: c.kind === "gen" ? "Put 4 bars of it in the BG at the playhead" : "Put it on the master, all the song", "aria-label": `Add ${c.label}`, onclick: () => c.kind === "gen" ? putItem(it, "bg", barAt(now())) : addFx(["master"], c.key) }, "+"));
      drag(row, it); list.append(row);
    }
  }
}
// a clip's picture: its cover, made by Python (core.arrangement.cover) and told as the covers come
let thumbRedraw = 0;
function scheduleThumbRedraw() { clearTimeout(thumbRedraw); thumbRedraw = setTimeout(() => later(() => { renderTimeline(); if (tab === "clips") renderBrowser(); }), 300); }

// ---------------------------------------------------------------------------
// The inspector: what is chosen (a block, a lane, the master, the montage's bars or settings)
// ---------------------------------------------------------------------------
const insp = () => $("insp");
function sect(text) { return h("div", { class: "sect", text }); }
function field(label, input, hint) { return h("div", { class: "fld" }, h("label", { text: label }), input, hint ? h("small", { class: "hint", text: hint }) : null); }
function textField(value, placeholder, onEnter, aria) {
  const inp = h("input", { value: value || "", placeholder, spellcheck: "false", "aria-label": aria || placeholder });
  inp.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); onEnter(inp.value); inp.blur(); } });
  inp.addEventListener("change", () => onEnter(inp.value));
  return inp;
}
function numField(value, onSet, opts = {}) {
  const inp = h("input", { type: "number", step: opts.step || "any", min: opts.min, max: opts.max, value: value ?? "", "aria-label": opts.aria || "" });
  inp.addEventListener("change", () => { if (inp.value !== "") onSet(+inp.value); });
  return inp;
}
function check(label, on, onSet) {
  return h("label", { class: "toggle" }, h("input", { type: "checkbox", checked: on, onchange: e => onSet(e.target.checked) }), label);
}
function openPlugin(m) {
  select({ type: "member", id: idOf(m.ref), ref: m.ref, k: sel && sel.id === idOf(m.ref) ? sel.k : null });
  const knobsAt = insp().querySelector("[data-sect='knobs']");
  if (knobsAt) knobsAt.scrollIntoView({ block: "start", behavior: "smooth" });
}
function renderInspector() {
  renderInspectorBody();
  const box = insp(), frozen = subStack.length > 0;
  box.classList.toggle("frozen", frozen);
  if (frozen) box.prepend(h("div", { class: "frozennote" }, h("span", { text: "Waiting for the link or FX below: Apply or Cancel it there" }), h("span", { class: "spacer" }),
    h("button", { class: "lnkbtn", onclick: closeSubAll }, "Cancel it")));
}
function renderInspectorBody() {
  const box = insp(); box.replaceChildren();
  if (!sel && loop && S) {
    $("inspTitle").textContent = "Bars";
    box.append(h("div", { class: "ptitle" }, h("h3", { text: `Bars ${loop[0]}–${loop[1]}` }), h("div", { class: "src", text: "chosen" })),
      h("div", { class: "btnrow" }, h("button", { class: "btn", title: "the live picture, over and over (Space plays it too)", onclick: () => previewBars(loop[0], loop[1]) }, `Play bars ${loop[0]}–${loop[1]}`),
        h("button", { class: "btn", title: "as the render makes them, on the GPU (Shift+Space)", onclick: () => exactBars(loop[0], loop[1]) }, `Exact bars ${loop[0]}–${Math.min(loop[1], loop[0] + 7)}`),
        h("button", { class: "btn", onclick: () => { loop = null; renderTimeline(); renderInspector(); } }, "Choose none")),
      h("div", { class: "about", text: "Pick a clip, a Generator or an FX in the browser and drag across a track to put it there; click a block to choose it." }));
    return;
  }
  if (!sel) {
    $("inspTitle").textContent = "Arrange";
    box.append(h("div", { class: "about", text: "Every edit goes to Python (core.arrangement) and comes back as the lanes." }),
      h("div", { class: "keys" },
        h("span", { class: "kbd", text: "click" }), "pick in the browser, then drag across a lane to put it",
        h("span", { class: "kbd", text: "drag" }), "move a block, onto another row or the other lane",
        h("span", { class: "kbd", text: "Shift+drag" }), "an end of a block: resize it",
        h("span", { class: "kbd", text: "Explorer" }), "drop clips on a lane, or a folder on the browser",
        h("span", { class: "kbd", text: "right click" }), "its menu: FX, hide bars, copy and paste, delete",
        h("span", { class: "kbd", text: "ruler" }), "click to go there, drag to choose bars",
        h("span", { class: "kbd", text: "a lane's name" }), "its settings and FX (Montage: its own)",
        h("span", { class: "kbd", text: "Ctrl+wheel" }), "zoom · Shift+wheel: row height · Alt+wheel: along",
        h("span", { class: "kbd", text: "Space" }), "play · Shift+Space: render the chosen bars",
        h("span", { class: "kbd", text: "Delete" }), "the chosen block (a member's last: the member)",
        h("span", { class: "kbd", text: "Ctrl+Z" }), "undo (Ctrl+Y redo)",
        h("span", { class: "kbd", text: "Ctrl+S" }), "save the scene (Ctrl+Shift+S: save as)"));
    return;
  }
  if (sel.type === "entry") return inspEntry(box, sel.key);
  if (sel.type === "knob" || sel.type === "plugin") { sel = null; return renderInspectorBody(); }
  if (sel.type === "montage") return inspMontageBars(box, sel.first, sel.last);
  if (sel.type === "clips") return inspMontage(box);
  if (sel.type === "lane") return inspLane(box, sel.group);
  if (sel.type === "master") return inspMaster(box);
  const m = memberById(sel.id);
  if (!m) { sel = null; return renderInspectorBody(); }
  inspMember(box, m);
}
function kindName(m) { return m.kind === "fx" || isFxLane(m.lane) ? "FX" : m.kind === "gen" ? "Generator" : m.lane === "fg" ? "Character" : "Clip"; }
function inspMember(box, m) {
  const c = catalogByKey.get(m.key), k = sel.k, st = shaders.get(m.key);
  const live = m.live && !(st && st.status === "bad");
  $("inspTitle").textContent = kindName(m);
  const laneName = (LANES.find(l => l.id === m.lane) || {}).name || m.lane;
  box.append(h("div", { class: "ptitle" }, h("h3", { text: m.label + (m.enabled ? "" : " (off)") }), h("div", { class: "src", text: m.file || (c ? c.file || c.key : m.key) }),
    h("div", { class: "row" }, h("span", { class: "chip", text: laneName }), h("span", { class: `badge ${live ? "live" : "engine"}`, text: live ? "live" : "engine" }),
      m.cued ? h("span", { class: "badge cue", text: "cued" }) : null, m.always ? h("span", { class: "chip", text: "the whole song" }) : null)));
  if (c && c.about) box.append(h("div", { class: "about", text: c.about }));
  if (st && st.status === "bad") box.append(h("div", { class: "about", style: "color:var(--danger)", text: `GLSL ES: ${st.error}` }));
  box.append(field("Name", textField(m.name, m.default_label, v => edit("set_layer", m.ref, "name", v.trim() || null), "Name"), "Enter; empty: its file's or its kind's"));
  if (m.cued) {
    box.append(h("div", { class: "about" }, `${m.cue_source}: `, m.cues === "read" && m.spans.length ? `${m.spans.length} cuts, bars ${m.spans[0].first} to ${m.spans[m.spans.length - 1].last}. ` : m.cues === "error" ? `not read (${m.cue_error}). ` : "its cuts are still being read. ", "Edit the cuts there (in REAPER); here, its place in the frame, its cutout and its FX."));
  } else if (m.always) box.append(h("div", { class: "about", text: "It plays the whole song: drag an end of its block, or give it bars here." }));
  if ((m.cued || m.always) && m.kind === "clip")
    box.append(field("Hidden in bars", textField(m.mutes.map(([a, b]) => b > a ? `${a}-${b}` : `${a}`).join(", "), "none", v => edit("set_mutes", m.ref, v)), "like 9-12, 30-33; or right-click a cut: Hide. It stays in sync."));
  if (m.kind !== "clip") box.append(field("Plays in bars", textField(m.bars_text, "all the song", v => edit("set_windows", m.ref, v)), "like 9-12, 30-33; Enter"));
  // the chosen block
  const blocks = m.always || m.cued ? [] : m.spans;
  if (k != null && blocks[k]) {
    const b = blocks[k], put = (key, v) => edit("set_block", m.ref, k, { [key]: v });
    const f = h("div", { class: "fields" },
      field("First bar", numField(b.first, v => put("first", Math.round(v)), { min: 1, max: NB(), step: 1 })),
      field("Last bar", numField(b.last, v => put("last", Math.round(v)), { min: 1, max: NB(), step: 1 })));
    if (m.kind === "clip") f.append(field("From (s)", numField(b.from, v => put("start", v), { min: 0, step: 0.1 })), field("Speed", numField(b.rate, v => put("rate", v), { min: 0.05, step: 0.1 })));
    box.append(sect(blocks.length > 1 ? `Block ${k + 1} of ${blocks.length}` : "Its block"), f);
    if (m.kind === "clip" && picked && picked.type === "clip")
      box.append(h("div", { class: "btnrow" }, m.lane === "bg"
        ? h("button", { class: "btn", onclick: () => edit("set_block", m.ref, k, { file: picked.path, start: picked.from || 0 }) }, `Use ${picked.label} and its 'from'`)
        : h("button", { class: "btn", onclick: () => edit("set_block", m.ref, k, { start: picked.from || 0 }) }, "Use the browser's 'from'")));
  }
  const pbs = barsOf(m, k != null && blocks[k] ? k : null), pb = pbs ? { first: pbs[0], last: pbs[1] } : null;
  box.append(h("div", { class: "btnrow" },
    pb ? h("button", { class: "btn", title: "the live picture, over and over", onclick: () => previewBars(pb.first, pb.last) }, `Play bars ${pb.first}–${pb.last}`) : null,
    pb ? h("button", { class: "btn", title: "as the render makes them, on the GPU (at most 8 bars)", onclick: () => exactBars(pb.first, pb.last) }, `Exact ${pb.first}–${Math.min(pb.last, pb.first + 7)}`) : null,
    h("button", { class: "btn", onclick: () => edit("toggle", m.ref) }, m.enabled ? "Turn off" : "Turn on"),
    isFxLane(m.lane) ? h("button", { class: "btn", onclick: () => edit("reorder", m.ref, -1) }, "Earlier") : null,
    isFxLane(m.lane) ? h("button", { class: "btn", onclick: () => edit("reorder", m.ref, 1) }, "Later") : null,
    k != null && blocks.length > 1 ? h("button", { class: "btn", onclick: () => edit("delete_block", m.ref, k) }, "Delete this block") : null,
    h("button", { class: "btn danger", onclick: () => edit("remove", m.ref) }, `Delete ${m.label.slice(0, 20)}${blocks.length > 1 ? " (all its blocks)" : ""}`)));
  if (isFxLane(m.lane)) box.append(h("div", { class: "about", text: "Its chain works in order: earlier and later move it. Drag it, or its ends, on its track." }));
  // how it is put on the picture
  if (m.kind !== "fx" && !isFxLane(m.lane)) {
    const out = h("output", { class: "mono", text: m.opacity == null ? "music" : fmt(m.opacity) });
    box.append(h("div", { class: "fields" },
      field("Blend", h("select", { onchange: e => edit("set_layer", m.ref, "blend", e.target.value === "normal" ? null : e.target.value) },
        Object.keys(BLEND).map(b => h("option", { value: b, selected: b === m.blend, text: b })))),
      m.opacity == null ? field("Opacity", h("div", { class: "about", text: `follows the music: ${m.opacity_knob}` }))
        : field("Opacity", h("input", { type: "range", min: 0, max: 1, step: 0.01, value: m.opacity, style: "accent-color:var(--accent)",
          oninput: e => { out.textContent = fmt(+e.target.value); }, onchange: e => edit("set_layer", m.ref, "opacity", +e.target.value) }), null)));
  }
  if (m.kind === "clip" && m.lane === "fg") inspPlacement(box, m);
  else if (m.kind === "clip" && m.own_fx) box.append(check("its FX may spill outside it (trails, glow, split edges)", m.spill, v => edit("set_layer", m.ref, "spill", v)));
  // its own FX
  if (!isFxLane(m.lane)) box.append(fxList(`FX on ${m.label.slice(0, 24)} only, in order`, m.fx.map((f, j) => ({ ...f, ref: [...m.ref, "fx", j] })), [...m.ref], true));
  // what it is and does: every parameter (the plugin), mask:
  pluginInto(box, m.ref, m).then(fresh => { if (fresh) layerTextInto(box, m.ref); });
  $("yaml").textContent = m.yaml;
}
const PLACES = [["x", 0.5, "0 left .. 1 right"], ["y", 0.5, "0 top .. 1 bottom"], ["scale", 1.0, "1: as tall as the frame"]];
function inspPlacement(box, m) {
  const p = m.place;
  box.append(sect("In the frame"));
  for (const [key, def, about] of PLACES) {
    const v = p[key];
    if (v && typeof v === "object") box.append(h("div", { class: "prm linked" + (editingRow(m.ref, key) ? " editing" : ""), dataset: { param: key } }, h("div", { class: "top" }, h("label", { text: key }), h("output", { text: "follows the music" })),
      v.said ? h("div", { class: "about", text: v.said }) : null, h("div", { class: "about mono small", text: v.knob }),
      h("div", { class: "btnrow" }, h("button", { class: "lnkbtn", onclick: () => linkPlace(m, key, def) }, "Edit link…"),
        h("button", { class: "lnkbtn", onclick: () => edit("set_param", m.ref, key, String(knobBias(v.knob) ?? def)) }, "Fixed"))));
    else box.append(h("div", { class: "prm" + (editingRow(m.ref, key) ? " editing" : ""), dataset: { param: key } }, h("div", { class: "top" }, h("label", { text: key }), h("small", { class: "hint", text: about })),
      h("div", { class: "slide" }, h("input", { type: "range", min: key === "scale" ? 0.05 : -0.5, max: key === "scale" ? 2 : 1.5, step: 0.01, value: v ?? def, "aria-label": key, onchange: e => edit("set_layer", m.ref, key, +e.target.value) }),
        numField(v ?? def, val => edit("set_layer", m.ref, key, val), { step: 0.05, aria: key })),
      h("button", { class: "lnkbtn", onclick: () => linkPlace(m, key, v ?? def) }, "Follow the music…")));
  }
  box.append(h("div", { class: "fields" },
    field("Fit", h("select", { onchange: e => edit("set_layer", m.ref, "fit", e.target.value) }, ["contain", "cover"].map(f => h("option", { value: f, selected: f === p.fit, text: f })))),
    field("", h("div", { class: "row" }, check("flip", p.flip, v => edit("set_layer", m.ref, "flip", v)), check("loop", p.loop, v => edit("set_layer", m.ref, "loop", v))))));
  if (p.sam != null) {
    box.append(h("div", { class: "row" }, h("span", { class: "about", style: "color:var(--live)", text: p.sam_state || "Cut out by SAM2 at the render." }),
      h("button", { class: "lnkbtn", onclick: () => openSam([...m.ref, "fx", p.sam]) }, "Seeds…")));
  } else if (p.cutout) {
    box.append(h("div", { class: "row" }, h("span", { class: "chip", text: `cut out by ${p.cutout}` }),
      h("button", { class: "lnkbtn", onclick: () => edit("set_layer", m.ref, "cutout", null) }, "Whole frame")),
      check("all but object 1 (a backdrop)", p.drop, v => { const cut = { ...(p.cutout_spec || {}) }; delete cut.drop; delete cut.objects; if (v) cut.drop = [1]; edit("set_layer", m.ref, "cutout", cut); }));
  } else {
    box.append(h("div", { class: "row" }, h("span", { class: "chip", text: "not cut out (whole frame)" }),
      p.mask_track ? h("button", { class: "lnkbtn", onclick: () => edit("set_layer", m.ref, "cutout", { track: p.mask_track, feather: 2 }) }, `Cut out (${p.mask_track.split(/[\\/]/).pop()})`) : null,
      h("button", { class: "lnkbtn", onclick: async () => { const r = await edit("add_fx", m.ref, "sam_cutout"); if (r && !r.error) { const mm = memberById(idOf(m.ref)); if (mm) openSam([...m.ref, "fx", mm.fx.length - 1], true); } } }, "Cut out by SAM2…")));
  }
  if (m.own_fx) box.append(check("its FX may spill outside the figure (trails, glow, split edges)", m.spill, v => edit("set_layer", m.ref, "spill", v)));
}
function knobBias(text) { const m = /bias:\s*(-?[\d.]+)/.exec(text || ""); return m ? +m[1] : null; }
// x / y / scale linked to the music: the knob editor, as every knob
function linkPlace(m, key, def) {
  openKnob({ ref: m.ref, keypath: [key], mode: "knob", default: typeof def === "number" ? def : 0.5, title: `${m.label} · ${key}`, param: key });
}
// an FX list: a block's own, a lane's, the master's. rows: [{ref, label, enabled, bars_text}]
function fxList(title, rows, owner, draggable) {
  const wrap = h("div", { class: "fxlist" });
  wrap.append(h("div", { class: "fxhead" }, h("span", { class: "sect", text: title }),
    h("button", { class: "lnkbtn", onclick: e => menuAt(e.clientX, e.clientY, catalogItems("fx", key => addFx(owner, key))) }, "Add FX…")));
  if (!rows.length) wrap.append(h("div", { class: "about", text: "none" }));
  rows.forEach((f, j) => {
    const row = h("div", { class: "fxrow" + (f.enabled ? "" : " off") + (editingFx(f.ref) ? " editing" : ""), draggable: draggable ? "true" : null, dataset: { j } },
      draggable ? h("span", { class: "grip", title: "drag to reorder", "aria-hidden": "true" }, "⋮⋮") : null,
      h("input", { type: "checkbox", checked: f.enabled, "aria-label": `${f.label} on`, onchange: () => edit("toggle", f.ref) }),
      h("span", { class: "nm", text: `${j + 1}. ${f.label}`, title: f.label }),
      textField(f.bars_text, "all", v => edit("set_windows", f.ref, v), `${f.label}: bars`),
      f.sam ? h("button", { class: "ib", title: "its seeds: how SAM2 finds what to cut out", "aria-label": "SAM2 seeds", onclick: () => openSam(f.ref) }, "✂")
        : h("button", { class: "ib", title: "its parameters", "aria-label": `${f.label}: its parameters`, onclick: () => { const mm = memberById(idOf(f.ref)); if (mm) openPlugin(mm); else openSubPlugin(f.ref, f.label); } }, "⚙"),
      h("button", { class: "ib", title: "earlier", "aria-label": "Earlier", onclick: () => edit("reorder", f.ref, -1) }, "↑"),
      h("button", { class: "ib", title: "later", "aria-label": "Later", onclick: () => edit("reorder", f.ref, 1) }, "↓"),
      h("button", { class: "ib", title: "remove", "aria-label": "Remove", onclick: () => edit("remove", f.ref) }, "✕"));
    if (row.classList.contains("editing")) setTimeout(() => centreInPanel(row), 0);
    if (draggable) {
      row.addEventListener("dragstart", e => { e.dataTransfer.setData("text/x-fx", String(j)); e.dataTransfer.effectAllowed = "move"; row.classList.add("dragging"); });
      row.addEventListener("dragend", () => row.classList.remove("dragging"));
      row.addEventListener("dragover", e => { if (e.dataTransfer.types.includes("text/x-fx")) { e.preventDefault(); row.classList.add("over"); } });
      row.addEventListener("dragleave", () => row.classList.remove("over"));
      row.addEventListener("drop", e => { e.preventDefault(); row.classList.remove("over"); const from = +e.dataTransfer.getData("text/x-fx"); if (from !== j) edit("move_fx", rows[from].ref, j); });
    }
    wrap.append(row);
  });
  return wrap;
}
function inspLane(box, group) {
  $("inspTitle").textContent = group;
  box.append(h("div", { class: "ptitle" }, h("h3", { text: group === "FG" ? "FG · the foreground" : "BG · the background" })),
    h("div", { class: "about", text: group === "FG" ? "Drawn on its own and put over the BG; its FX work on it alone."
      : has.clips ? "Its members over the clip montage (its bottom); its FX work on all of them." : "Its members, bar by bar; a bar with nothing in it is black. Its FX work on all of them." }));
  if (group === "BG" && !has.clips) box.append(h("div", { class: "btnrow" },
    loop ? h("button", { class: "btn sm", onclick: () => addMontage(loop[0], loop[1]) }, `Add a clip montage in bars ${loop[0]}–${loop[1]}`) : null,
    h("button", { class: "btn sm", onclick: () => addMontage() }, "Add a clip montage, all the song")),
    h("div", { class: "about", text: "A clip montage: a clip of a folder in each bar (its order, seed and triggers), at the BG's bottom. Optional: the BG takes Generators and clips without it." }));
  if (group === "FG") box.append(check("its FX may spill outside the figures (trails, glow, split edges)", !!(groups.FG || {}).spill, v => edit("set_layer", ["FG"], "spill", v)));
  const lane = group === "FG" ? "fgfx" : "bgfx";
  box.append(fxList(`FX on the whole ${group}, in order`, members.filter(m => m.lane === lane).map(m => ({ ref: m.ref, label: m.label, enabled: m.enabled, bars_text: m.bars_text })), [group], true));
}
function inspMaster(box) {
  $("inspTitle").textContent = "Master";
  box.append(h("div", { class: "ptitle" }, h("h3", { text: "Master" })), h("div", { class: "about", text: "Over everything, in order (the last on top)." }));
  box.append(fxList("Over everything, in order", members.filter(m => m.lane === "master").map(m => ({ ref: m.ref, label: (m.kind === "gen" ? "GEN " : m.kind === "clip" ? "CLIP " : "") + m.label, enabled: m.enabled, bars_text: m.bars_text })), ["master"], false));
}
async function inspMontageBars(box, first, last) {
  $("inspTitle").textContent = "Montage";
  const bars = last > first ? `bars ${first}–${last}` : `bar ${first}`;
  box.append(h("div", { class: "ptitle" }, h("h3", { text: `Montage · ${bars}` })));
  const runs = montageRuns().filter(r => r.last >= first && r.first <= last);
  if (!runs.length) box.append(h("div", { class: "about", text: "The montage is not known yet (it comes when the clips are probed)." }));
  for (const r of runs.slice(0, 12)) {
    const pinned = Object.keys(pins).some(b => +b >= r.first && +b <= r.last);
    const th = r.url ? thumbs.get(r.url) : null;
    box.append(h("div", { class: "runrow", title: `${r.path || r.name}: click to play it`, onclick: () => { if (r.path) chooseClip(r.path, 0); } }, h("span", { class: "sw", style: th ? `background-image:url(${th})` : "" }),
      h("span", { class: "nm2" }, h("span", { text: r.name }), r.path ? h("small", { class: "mono", text: r.path.split(/[\\/]/).slice(-2, -1)[0] || "" }) : null),
      h("span", { class: "mono", text: r.first === r.last ? `${r.first}` : `${r.first}–${r.last}` }), pinned ? h("span", { class: "chip", text: "pinned" }) : null));
  }
  if (runs.length > 12) box.append(h("div", { class: "about", text: `… and ${runs.length - 12} more` }));
  const btns = h("div", { class: "btnrow" });
  box.append(btns, h("div", { class: "about", text: "The montage draws a clip for each bar (clip order, the seed, the triggers' cuts and reverses). A pin keeps a clip there; the bars around stay as they were drawn. A clip from another folder, or from another point, goes in the BG over these bars." }));
  const items = await ask("montage_menu", first, last, picked && picked.type === "clip" ? picked.path : null);
  if (!items || !(sel && sel.type === "montage" && sel.first === first && sel.last === last)) return;
  for (const it of items) if (!it.head && !it.sep && !it.catalog) btns.append(h("button", { class: "btn sm", title: it.label, onclick: () => runItem(it) },
    it.label.replace(/ - Delete$/, "").replace(/ (bars? [\d–-]+|\(the engine\)|\(live\))/g, "").replace(/ \(.*\)$/, "")));
}
async function inspMontage(box) {
  $("inspTitle").textContent = "Montage";
  const plays = montageBars.length ? montageBars.map(([a, z]) => a === z ? `${a}` : `${a}–${z}`).join(", ") : "all the song";
  box.append(h("div", { class: "ptitle" }, h("h3", { text: "The montage" }), h("div", { class: "src", text: "the scene's video: settings" })),
    h("div", { class: "about", text: "The BG's bottom: a clip of the project's clips folder in each bar, cut by the bars and the triggers; the BG's members go over it. Click a clip on its row (or drag across bars) to choose it." }),
    h("div", { class: "about", text: `It plays: ${plays}. Outside its bars it is black, or the BG.` }),
    h("div", { class: "btnrow" },
      loop ? h("button", { class: "btn sm", onclick: () => edit("set_montage_bars", loop[0], loop[1]) }, `Only bars ${loop[0]}–${loop[1]}`) : null,
      montageBars.length ? h("button", { class: "btn sm", onclick: () => edit("set_montage_bars") }, "All the song") : null,
      h("span", { class: "spacer" }),
      h("button", { class: "btn sm danger", title: "out of the scene, its pins with it (undo brings it back)", onclick: () => edit("remove_montage") }, "Remove the montage")));
  const v = await ask("video_settings");
  if (!v || !(sel && sel.type === "clips")) return;
  const put = (k, val) => edit("set_video", k, val);
  box.append(h("div", { class: "fields" },
    field("New clip", check("at every bar start", v.clip_per_bar !== false, val => put("clip_per_bar", val))),
    field("Every N bars", numField(v.bars_per_clip ?? 1, val => put("bars_per_clip", Math.max(1, Math.round(val))), { min: 1, step: 1 })),
    field("Clip order", h("select", { onchange: e => put("clip_order", e.target.value) }, v.choices.clip_order.map(o => h("option", { value: o, selected: o === (v.clip_order || "sequential"), text: o })))),
    field("A clip shorter than its turn", h("select", { onchange: e => put("clip_end", e.target.value) }, v.choices.clip_end.map(o => h("option", { value: o, selected: o === (v.clip_end || "bounce"), text: o })))),
    field("Seed", numField(v.seed, val => put("seed", Math.round(val)), { step: 1 }), "the shuffle's; the project's clips.seed wins")));
  const ta = h("textarea", { class: "codeedit", spellcheck: "false", rows: 8, "aria-label": "Triggers" }, v.triggers_text);
  box.append(sect("Triggers (YAML)"), ta, h("div", { class: "btnrow" }, h("button", { class: "btn", onclick: () => put("triggers", ta.value) }, "Apply")),
    h("div", { class: "about", text: "Who cuts the montage, what each hit does, its gravity. Their own rows come with the plugin's trigger editor." }));
}
async function addFx(owner, key) {
  const r = await edit("add_fx", owner, key);
  if (r && !r.error && owner.length > 1) newOwnFx(owner, key);
  return r;
}
// an FX just put on a block (not a lane's): SAM2's seeds, or its parameters under the plugin
function newOwnFx(owner, key) {
  const mm = memberById(idOf(owner)); if (!mm || !mm.fx.length) return;
  const ref = [...owner, "fx", mm.fx.length - 1];
  if (key === "sam_cutout") openSam(ref, true);
  else openSubPlugin(ref, mm.fx[mm.fx.length - 1].label);
}
function kickTrack() {
  const t = S.tracks.find(x => /kick|bumbo/i.test(x.name)) || S.tracks[0];
  return t ? t.name : "kick";
}

// ---------------------------------------------------------------------------
// The plugin: every parameter of the chosen layer with the editor it takes (Python decides
// which: core/web/plugin.py), its sections (reactivity and triggers, gestures, the cast's
// roles), mask:
// ---------------------------------------------------------------------------
async function pluginInto(box, ref, m = null) {
  const at = h("div", { class: "plugin", dataset: { sect: "knobs" } }, h("div", { class: "about", text: "…" }));
  box.append(at);
  const n = {}; box.plugReq = n;                 // a newer drawing of this box wins
  const p = await ask("plugin", ref);
  if (!p || box.plugReq !== n) return false;
  at.replaceChildren();
  if (!p.rows.length && !p.about) { at.append(h("div", { class: "about", text: "No parameters the catalog knows of." })); }
  at.append(h("div", { class: "phead2" }, h("span", { class: "chip", text: p.kind || "layer" }), p.category ? h("span", { class: "chip", text: p.category }) : null,
    p.rows.some(r => r.follows) ? h("span", { class: "hint", text: "~ the music can move it (its link)" }) : null));
  if (p.owner && !m) {                          // an FX on a block: its on, its blend, what it does
    if (p.about) at.append(h("div", { class: "about", text: p.about }));
    at.append(h("div", { class: "row" }, check("on", p.enabled, () => edit("toggle", p.ref)),
      h("select", { "aria-label": "Blend", onchange: e => edit("set_layer", p.ref, "blend", e.target.value === "normal" ? null : e.target.value) },
        Object.keys(BLEND).map(b => h("option", { value: b, selected: b === p.blend, text: b })))));
  }
  if (p.rows.length) at.append(sect(p.owner ? `${p.label}, on ${p.owner}` : "Its parameters"));
  for (const row of p.rows) at.append(pluginRow(p, row, m));
  at.append(maskBox(p));
  return true;
}
const fnum = v => typeof v === "number" ? fmt(v) : v;
function pluginRow(p, r, m) {
  const ref = p.ref, wrap = h("div", { class: "prm" + (r.widget === "knob" && r.set ? " linked" : "") + (r.set ? " set" : "") + (editingRow(ref, r.name) ? " editing" : ""), dataset: { param: r.name } });
  if (wrap.classList.contains("editing")) setTimeout(() => centreInPanel(wrap), 0);
  const knobText = m && (m.params.find(x => x.name === r.name) || {}).knob;
  const out = h("output", { dataset: { out: r.name }, text: "" });
  wrap.append(h("div", { class: "top" }, h("label", { text: (r.follows ? "~ " : "") + r.name, title: r.about || "" }), out,
    r.button ? h("button", { class: "lnkbtn", onclick: () => openKnob({ ref, keypath: r.keypath, mode: r.button.mode, default: typeof r.default === "number" ? r.default : 0, title: `${p.label} · ${r.name}`, fromText: r.button.from_text ? (r.value || r.default) : null, param: r.name }) }, r.button.label) : null,
    r.set && !["gestures", "reactivity", "roles"].includes(r.widget) ? h("button", { class: "ib", title: "back to its default", "aria-label": `${r.name}: back to its default`, onclick: () => edit("set_value", ref, r.keypath, null) }, "↺") : null));
  const put = v => edit("set_value", ref, r.keypath, v);
  if (r.widget === "number") {
    const def = r.value, lo = Math.min(0, def), hi = Math.max(1, Math.abs(def) * 2);
    const num = h("input", { type: "number", step: r.int ? 1 : "any", value: fnum(r.value), "aria-label": r.name, onchange: e => put(r.int ? Math.round(+e.target.value) : +e.target.value) });
    out.textContent = fnum(r.value);
    wrap.append(h("div", { class: "slide" }, h("input", { type: "range", min: lo, max: hi, step: r.int ? 1 : (hi - lo) / 400, value: r.value, "aria-label": r.name,
      oninput: e => { out.textContent = fmt(+e.target.value); num.value = fmt(+e.target.value); }, onchange: e => put(r.int ? Math.round(+e.target.value) : +e.target.value) }), num));
  } else if (r.widget === "flag") {
    wrap.append(check(r.value ? "on" : "off", r.value, v => put(v)));
  } else if (r.widget === "colour") {
    const hex = "#" + r.value.map(c => Math.round(c > 1 ? c : c * 255).toString(16).padStart(2, "0")).join("");
    wrap.append(h("input", { type: "color", value: hex, "aria-label": r.name, onchange: e => { const x = e.target.value; put([1, 3, 5].map(i => parseInt(x.slice(i, i + 2), 16))); } }));
  } else if (r.widget === "numbers") {
    wrap.append(h("div", { class: "row" }, r.value.map((v, k) => h("input", { type: "number", step: "any", value: fnum(v), class: "small", "aria-label": `${r.name} ${k + 1}`,
      onchange: e => { const xs = [...r.value]; xs[k] = +e.target.value; put(xs); } }))));
  } else if (r.widget === "knob") {
    wrap.append(h("div", { class: "about mono", text: (r.sig ? `*${r.sig}  ` : "") + r.summary }));
    if (knobText) wrap.append(h("div", { class: "bar" }, h("i", { dataset: { meter: r.name } })));
    if (r.envelope != null) wrap.append(field("Envelope (s)", numField(r.envelope, v => edit("set_value", ref, [...r.keypath, "envelope"], Math.max(0, v)), { min: 0, step: 0.05 })));
  } else if (r.widget === "text") {
    const inp = textField(r.value, r.hint, v => edit("set_text", ref, r.keypath, v, r.is_text), r.name);
    wrap.append(h("div", { class: "row nowrap" }, inp, r.pick ? h("button", { class: "ib", title: `choose ${r.pick === "dir" ? "a folder" : "a file"}`, onclick: async () => {
      const got = await ask("pick_file", r.pick === "dir" ? "folder" : "any", false, "", r.value || ""); if (got && got.path) edit("set_text", ref, r.keypath, got.path, true); } }, "…") : null));
  } else if (r.widget === "gestures") wrap.append(gesturesBox(ref, r));
  else if (r.widget === "reactivity") wrap.append(reactivityBox(ref, r));
  else if (r.widget === "roles") wrap.append(rolesBox(ref, r));
  if (r.about && !["gestures", "reactivity", "roles"].includes(r.widget)) wrap.append(h("div", { class: "about", text: r.about }));
  return wrap;
}
function gesturesBox(ref, r) {
  const box = h("div", { class: "subrows" });
  r.items.forEach((g, i) => box.append(h("div", { class: "subrow" },
    h("select", { "aria-label": "gesture", onchange: e => edit("set_value", ref, [...r.keypath, i, "gesture"], e.target.value) }, r.gestures.map(x => h("option", { value: x, selected: x === g.gesture, text: x }))),
    h("button", { class: "lnkbtn", title: g.who, onclick: () => openKnob({ ref, keypath: [...r.keypath, i], mode: "trigger", title: `gesture ${i + 1} · who plays`, param: r.name }) }, g.who),
    numField(g.envelope, v => edit("set_value", ref, [...r.keypath, i, "envelope"], Math.max(0.001, v)), { min: 0, step: 0.05, aria: "envelope" }),
    h("button", { class: "ib", "aria-label": "Remove", onclick: () => edit("drop_item", ref, r.keypath, i) }, "✕"))));
  box.append(h("button", { class: "lnkbtn", onclick: async () => { const got = await edit("add_gesture", ref, r.keypath, r.gestures[0]); if (got && got.added != null) openKnob({ ref, keypath: [...r.keypath, got.added], mode: "trigger", title: "the new gesture · who plays", param: r.name }); } }, "+ gesture"));
  return box;
}
function reactivityBox(ref, r) {
  const c = r.cfg, kp = r.keypath, put = (k, v) => edit("set_value", ref, [...kp, k], v);
  const box = h("div", { class: "subrows" },
    h("div", { class: "fields" },
      field("New clip", check("every bar", c.clip_per_bar, v => put("clip_per_bar", v))),
      field("Every N bars", numField(c.bars_per_clip, v => put("bars_per_clip", Math.max(1, Math.round(v))), { min: 1, step: 1 })),
      field("Order", h("select", { onchange: e => put("clip_order", e.target.value) }, r.orders.map(o => h("option", { value: o, selected: o === c.clip_order, text: o })))),
      field("Shorter clip", h("select", { onchange: e => put("clip_end", e.target.value) }, r.ends.map(o => h("option", { value: o, selected: o === c.clip_end, text: o })))),
      field("Seed", textField(c.seed ?? "", "none", v => put("seed", /^-?\d+$/.test(v.trim()) ? +v : null)))));
  box.prepend(h("div", { class: "about", text: "When the montage takes a new clip: at every bar (or every N), in what order, and what a clip shorter than its turn does (bounce, loop, hold)." }));
  const tkp = [...kp, "triggers"], names = r.triggers.map(t => t.name);
  box.append(h("div", { class: "fxhead" }, h("span", { class: "sect", text: "Triggers" }),
    h("button", { class: "lnkbtn", onclick: async () => { const got = await edit("add_trigger", ref, tkp); if (got && got.added) openKnob({ ref, keypath: [...tkp, got.added], mode: "trigger", title: `${got.added} · who plays`, param: r.name }); } }, "+ trigger")),
    h("div", { class: "about", text: "A trigger: who plays (a MIDI track and its notes, an audio file, a melody) and what each hit does to the montage: cut to a new clip, reverse it, and so on. Gravity pulls the clips' speed around each hit." }));
  for (const t of r.triggers) {
    const others = names.filter(n => n !== t.name), tp = (k, v) => edit("set_trigger", ref, tkp, t.name, k, v);
    box.append(h("div", { class: "trig" },
      h("div", { class: "row nowrap" }, textField(t.name, "name", v => edit("rename_trigger", ref, tkp, t.name, v), "trigger name"),
        h("button", { class: "lnkbtn", onclick: () => openKnob({ ref, keypath: [...tkp, t.name], mode: "trigger", title: `${t.name} · who plays`, param: r.name }) }, `Who plays: ${t.who}`),
        h("button", { class: "ib", "aria-label": `Remove ${t.name}`, onclick: () => edit("drop_trigger", ref, tkp, t.name) }, "✕")),
      h("div", { class: "row" }, r.actions.map(a => check(a.replace("_", " "), t.actions.includes(a), v => tp("actions", v ? [...t.actions, a] : t.actions.filter(x => x !== a))))),
      h("div", { class: "fields" },
        field("Softer hits left out", numField(t.min_velocity, v => tp("min_velocity", Math.round(v)), { min: 0, max: 127, step: 1 }), "below this velocity (0: none)"),
        field("Until another comes in", h("select", { onchange: e => tp("until", e.target.value || null) }, h("option", { value: "", text: "-" }), others.map(o => h("option", { value: o, selected: o === t.until, text: o })))),
        field("Not near another's hits", h("select", { onchange: e => tp("exclude", e.target.value || null) }, h("option", { value: "", text: "-" }), others.map(o => h("option", { value: o, selected: o === t.exclude, text: o }))))),
      check("gravity (the clips' speed pulled around each hit)", !!t.gravity, v => tp("gravity", v)),
      t.gravity ? h("div", { class: "fields" }, ["peak", "floor", "radius", "curve"].map(k => field(k, numField(t.gravity[k], v => tp(`gravity.${k}`, v), { step: 0.05 })))) : null));
  }
  return box;
}
function rolesBox(ref, r) {
  const box = h("div", { class: "subrows" });
  r.items.forEach((x, i) => {
    const sr = (k, v) => edit("set_role", ref, r.keypath, i, k, v);
    box.append(h("div", { class: "trig" },
      h("div", { class: "row nowrap" }, h("b", { text: `Role ${i + 1}` }), h("span", { class: "spacer" }), h("button", { class: "ib", "aria-label": `Remove role ${i + 1}`, onclick: () => edit("drop_item", ref, r.keypath, i) }, "✕")),
      h("div", { class: "fields" },
        field("Who", h("div", { class: "row nowrap" }, h("select", { onchange: e => sr("who", e.target.value) }, [...r.who, ...(r.who.includes(x.who) ? [] : [x.who])].map(w => h("option", { value: w, selected: w === x.who, text: w }))),
          textField("", "or ids 1, 2", v => v.trim() && sr("who", v)))),
        field("Act", h("select", { onchange: e => sr("act", e.target.value) }, r.acts.map(a => h("option", { value: a, selected: a === x.act, text: a })))),
        field("Envelope (s)", numField(x.envelope ?? r.envelopes[x.act] ?? 0, v => sr("envelope", v), { min: 0, step: 0.05 })),
        field("Amount", numField(x.amount ?? 1, v => sr("amount", v), { step: 0.05 })),
        field("Colour", textField(x.color, "r, g, b · chord · pitch", v => sr("color", v)))),
      h("button", { class: "lnkbtn", onclick: () => openKnob({ ref, keypath: [...r.keypath, i, "plays"], mode: "trigger", title: `role ${i + 1} · who plays it`, param: r.name }) }, `Plays: ${x.plays}`),
      field("Its other keys (YAML)", textField(x.extra, "e.g. {shadow: 0.5, hold: 1}", v => sr("extra", v)))));
  });
  box.append(h("button", { class: "lnkbtn", onclick: () => edit("add_role", ref, r.keypath) }, "+ role"));
  return box;
}
function maskBox(p) {
  const mk = p.mask, put = ch => edit("set_mask", p.ref, ch);
  const box = h("div", { class: "maskbox" }, sect("Mask (in or out of a mask track's figures)"), check("masked", mk.on, v => put(v ? { on: true } : { on: false })));
  if (mk.on) box.append(h("div", { class: "fields" },
    field("Mask track", h("div", { class: "row nowrap" }, textField(mk.track, "the render's --masks", v => put({ track: v })),
      h("button", { class: "ib", title: "choose its folder", onclick: async () => { const got = await ask("pick_file", "folder", false, "", mk.track || ""); if (got && got.path) put({ track: got.path }); } }, "…"))),
    field("Where", h("select", { onchange: e => put({ where: e.target.value }) }, mk.wheres.map(w => h("option", { value: w, selected: w === mk.where, text: w })))),
    field("Objects", textField(mk.objects, "all", v => put({ objects: v })), "ids, like 1, 3"),
    field("Grow (px)", numField(mk.grow, v => put({ grow: v }), { step: 1 })),
    field("Feather (px)", numField(mk.feather, v => put({ feather: v }), { min: 0, step: 1 }))));
  return box;
}

// the layer's own YAML, edited as text (its comments kept while it is as the scene wrote it)
async function layerTextInto(box, ref) {
  const n = box.plugReq, got = await ask("layer_text", ref);
  if (!got || box.plugReq !== n) return;
  const ta = h("textarea", { class: "codeedit", spellcheck: "false", rows: Math.min(18, got.text.split("\n").length + 1), "aria-label": "The layer's YAML" }, got.text);
  const err = h("div", { class: "err" });
  box.append(sect("The layer, as the scene says (but its bars)"), ta, err,
    h("div", { class: "btnrow" }, h("button", { class: "btn sm", onclick: async () => {
      const r = await call("set_layer_text", ref, ta.value).catch(e => ({ error: e.message }));
      if (r && r.error) { err.textContent = r.error; return; }
      err.textContent = ""; apply(r); toast("The layer, as written");
    } }, "Apply the text"), h("span", { class: "hint", text: "anything without a control of its own; *name follows a signal" })));
  $("yaml").textContent = got.text;
}

// ---------------------------------------------------------------------------
// The knob editor: what in the music, from where, how it responds (core/knob_forms.py's
// forms, sent by Python), its YAML and its curve; Apply writes it (one undo step)
// ---------------------------------------------------------------------------
let knobCtx = null;

// ---------------------------------------------------------------------------
// The area under the plugin, beside the Arrange: what the plugin opens (a link's editor, a
// block's own FX) works there while the plugin waits, frozen, the row it came from lit.
// A stack: a link opened from that FX goes over it, Cancel goes back.
// ---------------------------------------------------------------------------
let subStack = [];             // [{type: "knob" | "plugin", ref, param, title}]
const subTop = () => subStack[subStack.length - 1] || null;
function pushSub(it) { subStack.push(it); renderSub(); renderInspector(); $("subBody").scrollTop = 0; }
function popSub() {
  const it = subStack.pop();
  if (it && it.type === "knob") { knobCtx = null; leaveLink(); }
  renderSub(); renderInspector(); renderTimeline();
}
function closeSubAll() { const had = !!knobCtx; subStack = []; knobCtx = null; if (had) leaveLink(); renderSub(); renderInspector(); renderTimeline(); }
function leaveLink() { if (lab) { const was = lab.hear; lab = null; showLabBar(); if (was !== "mix") rehear(); } clearDraft(); }
// the lit row in the middle of its panel (only the panel scrolls, not the page)
function centreInPanel(el) {
  const box = el.isConnected && el.closest(".insp"); if (!box) return;
  box.scrollTop += el.getBoundingClientRect().top - box.getBoundingClientRect().top - (box.clientHeight - el.offsetHeight) / 2;
}
function subBusy() {
  toast("The link or FX below first: Apply it or Cancel it", true);
  const p = $q(".subpanel"); if (p) { p.classList.remove("flash"); void p.offsetWidth; p.classList.add("flash"); }
  if (outWins.subpanel) outWins.subpanel.focus();
}
// the inspector's row the bottom of the stack came from (a parameter, x / y / scale), and the FX row
function editingRow(ref, param) { const b = subStack[0]; return !!b && b.type === "knob" && b.param != null && b.param === param && idOf(b.ref) === idOf(ref); }
function editingFx(ref) { const b = subStack[0]; return !!b && b.type === "plugin" && idOf(b.ref) === idOf(ref); }
function openSubPlugin(ref, label) {
  if (subStack.some(x => x.type === "plugin" && idOf(x.ref) === idOf(ref))) return;
  pushSub({ type: "plugin", ref, title: label || "FX" });
}
function renderSub() {
  const box = $("subBody"); if (!box) return;
  box.replaceChildren(); $("subActions").replaceChildren();     // its buttons in its head: always in sight
  const top = subTop();
  $("subTrail").textContent = subStack.length > 1 ? subStack.map(x => x.title).join(" › ") : "";
  if (!top) {
    $("subTitle").textContent = "Link · FX";
    box.append(h("div", { class: "about", text: "A parameter's link (how the music moves it) and the FX on a block open here, under the plugin; the plugin waits for them, its row lit." }));
    return;
  }
  if (top.type === "knob" && knobCtx) { inspKnob(box); return; }
  $("subTitle").textContent = "FX";
  box.append(h("div", { class: "ptitle" }, h("h3", { text: top.title }), h("div", { class: "src", text: "an FX of the block above" })));
  fill($("subActions"), h("button", { class: "btn sm primary", onclick: popSub }, "Done"));
  pluginInto(box, top.ref);
}

async function openKnob(o) {
  const data = await ask("knob_open", o.ref, o.keypath, o.mode || "knob", o.default || 0);
  if (!data) return;
  if (o.fromText && data.state.form === "constant") {            // a ..._from param: its feature first
    const st = await ask("knob_new", "feature", null); if (st) { st.source = o.fromText; data.state = st; }
  }
  openKnobCtx({ ...o, data, state: data.state, link: data.link || "", anchor: "" });
}
function openKnobCtx(ctx) {
  if (subStack.length && subTop().type === "knob") { subStack.pop(); leaveLink(); }    // one link at a time
  knobCtx = ctx;
  pushSub({ type: "knob", ref: ctx.ref, param: ctx.param || null, title: ctx.title });
}
function closeKnob() { if (subStack.length && subTop().type === "knob") popSub(); else { knobCtx = null; renderSub(); } }
function inspKnob(box) {
  const k = knobCtx, d = k.data, F = d.schema.forms;
  $("subTitle").textContent = k.mode === "trigger" ? "Who plays" : k.mode === "score" ? "Chords from" : "Link";
  box.append(h("div", { class: "ptitle" }, h("h3", { text: k.title || "a knob" }), h("div", { class: "src", text: d.shared ? `a signal: ${d.shared}` : "" })));
  if (d.signals.length || k.mode === "knob") {
    const sigSel = h("select", { "aria-label": "Use a signal", onchange: e => { k.link = e.target.value; renderSub(); } },
      h("option", { value: "", text: "its own" }), d.signals.map(s => h("option", { value: s.name, selected: s.name === k.link, text: `*${s.name} · ${s.yaml.slice(0, 40)}` })));
    box.append(h("div", { class: "fields" }, field("Use a signal", sigSel, "it follows that one, wherever it changes"),
      k.mode === "knob" ? field("Or name this one", d.shared && d.shared.startsWith("&")
        ? h("input", { value: d.shared.slice(1), disabled: true, "aria-label": "signal name" })
        : textField(k.anchor, "a new signal", v => { k.anchor = v.trim(); }, "signal name"),
        d.shared && d.shared.startsWith("&") ? "it is a signal (its name stays): what follows it changes too" : "others can follow it then") : null));
    box.append(h("div", { class: "hint", text: "A signal is a knob with a name (&name in the YAML): others follow it (*name) and move with it; edit it here and they all change." }));
  }
  const body = h("div", { class: "knobbody" });
  if (k.link) body.append(h("div", { class: "about", text: `It follows *${k.link}: ${(d.signals.find(s => s.name === k.link) || {}).yaml || ""}` }));
  else drawState(body, k.state, 0, k.mode !== "knob", F, d);
  const yamlOut = h("pre", { class: "code", id: "knobYaml", text: "…" }), err = h("div", { class: "err", id: "knobErr" });
  const cv = h("canvas", { class: "curve", id: "knobCurve", width: 600, height: 120 });
  box.append(body);
  const onHits = st => !!st && !!F[st.form] && (F[st.form].source === "trigger" || (st.form === "rate" && onHits(st.from)));
  if ((k.mode === "trigger" || (k.mode === "knob" && onHits(k.state))) && !k.link) box.append(sect("Its hits"), h("div", { class: "about", id: "knobHits", text: "…" }),
    h("canvas", { class: "strip", id: "knobStrip", title: "the whole song: the hits, the layer's bars shaded, the playhead" }));
  box.append(sect("It writes"), yamlOut, err);
  if (k.mode === "knob" && !k.link) box.append(h("div", { class: "fxhead" }, h("span", { class: "sect", text: "Its curve" }),
    h("select", { "aria-label": "Over", onchange: e => { k.span = e.target.value; checkKnob(); } }, [["around", "16 s around"], ["view", "the bars in view"], ["here", "8 bars on"], ["song", "the whole song"]].map(([v, t]) => h("option", { value: v, selected: v === (k.span || "around"), text: t })))), cv);
  const labBtn = k.mode === "knob" && !k.link ? h("button", { class: "btn sm" + (labOn() ? " on" : ""), "aria-pressed": labOn(),
    title: "The Feature Lab: this link drawn step by step in the timeline's place (the song, what it hears, the feature, the value), its changes on Live at once",
    onclick: () => labOn() ? closeLab() : openLab() }, "Lab") : null;
  fill($("subActions"), labBtn, h("button", { class: "btn sm primary", onclick: applyKnob }, "Apply"), h("button", { class: "btn sm", onclick: closeKnob }, "Cancel"));
  checkKnob();
}
// who plays, picked from what the project has: a MIDI track (all its notes, or some), the
// attacks of one of its audio files (a stem, the song); numbers and a transcription too
const samePath = (a, b) => String(a || "").replace(/\\/g, "/").toLowerCase() === String(b || "").replace(/\\/g, "/").toLowerCase();
function whoPlaysInto(t, src, d) {
  const midi = d.sources.midi || [], audios = d.sources.audios || [];
  const names = String(src.track || "").split(",").map(x => x.trim()).filter(Boolean);
  const tr = src.kind === "track" && names.length === 1 ? midi.find(m => m.track.toLowerCase() === names[0].toLowerCase()) : null;
  const au = src.kind === "audio" ? audios.find(a => samePath(a.path, src.audio)) : null;
  const cur = tr ? `track:${tr.track}` : au ? `audio:${au.path}` : src.kind === "notes" || src.kind === "melody" ? src.kind : "written";
  const pick = h("select", { "aria-label": "Who plays", onchange: e => {
    const v = e.target.value;
    if (v.startsWith("track:")) Object.assign(src, { kind: "track", track: v.slice(6), notes: "" });
    else if (v.startsWith("audio:")) Object.assign(src, { kind: "audio", audio: v.slice(6), notes: "" });
    else if (v === "notes" || v === "melody") src.kind = v;
    renderSub();
  } },
    midi.length ? h("optgroup", { label: "The MIDI: a track" }, midi.map(m => h("option", { value: `track:${m.track}`, selected: cur === `track:${m.track}`, text: `${m.label} · ${m.count} notes` }))) : null,
    audios.length ? h("optgroup", { label: "The audio: its attacks" }, audios.map(a => h("option", { value: `audio:${a.path}`, selected: cur === `audio:${a.path}`, text: a.label }))) : null,
    h("optgroup", { label: "Other" }, h("option", { value: "notes", selected: cur === "notes", text: "notes by number, on any track" }),
      d.sources.score || src.kind === "melody" ? h("option", { value: "melody", selected: cur === "melody", text: "the melody (a transcription)" }) : null,
      cur === "written" ? h("option", { value: "written", selected: true, text: `as written: ${src.kind === "audio" ? src.audio : src.track}` }) : null));
  t.append(field("Who plays", pick, "what in the song makes a hit"));
  if (tr) {                                     // its notes: all, or the ones chosen
    const chosen = new Set(String(src.notes || "").split(",").map(x => parseInt(x, 10)).filter(x => !isNaN(x)));
    const put = () => { src.notes = [...chosen].sort((a, b) => a - b).join(", "); renderSub(); };
    t.append(field("Which of its notes", h("div", { class: "notechips" },
      h("button", { class: "chip" + (chosen.size ? "" : " on"), onclick: () => { chosen.clear(); put(); } }, "all"),
      tr.pitches.map(p => h("button", { class: "chip" + (chosen.has(p.pitch) ? " on" : ""), title: `MIDI ${p.pitch}: played ${p.count} times`,
        onclick: () => { if (chosen.has(p.pitch)) chosen.delete(p.pitch); else chosen.add(p.pitch); put(); } }, `${p.name} ·${p.count}`))),
      chosen.size ? "only these" : "every note of the track"));
  } else if (src.kind === "track") {
    t.append(field("Tracks", textField(src.track, "kick, snare", v => { src.track = v; checkKnob(); }, "tracks"), "names (or numbers), as in the MIDI"));
  }
  if (src.kind === "audio") {
    if (!au) t.append(field("Audio file", textField(src.audio, "stems/snare.wav", v => { src.audio = v; checkKnob(); }, "audio file")));
    const ex = src.extra || (src.extra = {});
    t.append(h("div", { class: "fields" },
      field("Sensitivity", numField(ex.threshold ?? 0.3, v => { ex.threshold = Math.max(0, v); checkKnob(); }, { min: 0, step: 0.05 }), "higher: only the strong attacks"),
      field("Apart by (s)", numField(ex.min_gap ?? 0.05, v => { ex.min_gap = Math.max(0, v); checkKnob(); }, { min: 0, step: 0.01 }), "two hits at least this far apart")));
  }
  if (src.kind === "notes") t.append(field("Notes", textField(src.notes, "36, 38", v => { src.notes = v; checkKnob(); }, "notes"), "MIDI numbers, on any track (36: a kick in General MIDI)"));
  if (src.kind === "melody") {
    if (!src.score && d.sources.score) src.score = d.sources.score;
    t.append(field("Transcription", textField(src.score || d.sources.score, "its folder", v => { src.score = v; checkKnob(); })), field("Voice", textField(src.voice, "any", v => { src.voice = v; checkKnob(); })));
  }
}
// the hits of who plays over the song: how many, how many in the layer's own bars, a strip
function showHits(r) {
  const el = $("knobHits"), cv = $("knobStrip"), k = knobCtx;
  if (!el || !k) return;
  const at = k.ref.lastIndexOf("fx");             // a block's own FX: the block's bars
  const m = memberById(idOf(k.ref)) || (at >= 2 ? memberById(idOf(k.ref.slice(0, at))) : null);
  const spans = m && !m.always ? m.spans.map(s => [B[s.first - 1], B[s.last]]) : null;
  const inside = spans ? r.hit_times.filter(t => spans.some(([a, b]) => t >= a && t < b)).length : null;
  el.textContent = r.error ? "" : !r.hits ? "No hits: nothing in the song plays it." :
    `${r.hits} hits in the song, bars ${r.first_bar}–${r.last_bar}` + (inside != null ? ` · ${inside} in ${m.label}'s bars (${m.bars_text || "all"})` : "");
  if (!cv) return;
  const d = devicePixelRatio || 1, w = cv.clientWidth || 300, hh = 34, dur = S.song.duration || 1;
  cv.width = w * d; cv.height = hh * d;
  const g = cv.getContext("2d"); g.scale(d, d);
  g.fillStyle = "#0f1115"; g.fillRect(0, 0, w, hh);
  const X = t => t / dur * w;
  if (spans) { g.fillStyle = "rgba(240,160,75,.12)"; for (const [a, b] of spans) g.fillRect(X(a), 0, Math.max(1, X(b) - X(a)), hh); }
  g.fillStyle = "#2b313d"; for (let b = 0; b < B.length; b += 4) g.fillRect(X(B[b]), 0, 1, hh);
  g.fillStyle = "#f0a04b"; for (const t of r.hit_times) g.fillRect(X(t), 6, 1, hh - 12);
  const p = now(); g.fillStyle = "#e9e5db"; g.fillRect(X(p), 0, 1, hh);
}
function knobSetting(name, def, kind, help, get, set) {
  const v = get() ?? def;
  let input;
  if (Array.isArray(kind)) input = h("select", { onchange: e => { set(e.target.value); checkKnob(); } }, kind.map(c => h("option", { value: c, selected: c === v, text: c })));
  else if (kind === "bool") input = h("input", { type: "checkbox", checked: !!v, onchange: e => { set(e.target.checked); checkKnob(); } });
  else if (kind === "text") input = textField(String(v), String(def), x => { set(x); checkKnob(); }, name);
  else input = h("input", { type: "number", step: kind === "int" ? 1 : "any", value: v, onchange: e => { set(kind === "int" ? Math.round(+e.target.value) : +e.target.value); checkKnob(); } });
  return field(name, input, help);
}
// clip: the value held inside lo..hi (after gain and bias); on, it starts at the parameter's range
function clipField(st, d) {
  const rng = d.param && d.param.range;
  const on = h("input", { type: "checkbox", checked: !!st.clip, "aria-label": "hold inside", onchange: e => {
    st.clip = e.target.checked ? (rng ? [...rng] : [0, 1]) : null; renderSub(); } });
  const num = i => h("input", { type: "number", step: "any", value: st.clip ? st.clip[i] : "", disabled: !st.clip, "aria-label": i ? "clip high" : "clip low",
    onchange: e => { if (st.clip) { st.clip[i] = +e.target.value; checkKnob(); if (labOn()) showLabBar(); } } });
  return field("clip", h("span", { class: "row nowrap" }, on, num(0), h("span", { class: "hint", text: ".." }), num(1)),
    rng ? `held inside (its range: ${fmt(rng[0])}..${fmt(rng[1])})` : "held inside lo..hi");
}
function drawState(box, st, depth, onlySource, F, d) {
  if (st.form === "custom") { box.append(h("div", { class: "about", text: "A knob the editor has no form for: it stays as written (edit it in the layer's YAML)." }), h("pre", { class: "code", text: JSON.stringify(st.custom) })); return; }
  const f = F[st.form];
  if (!onlySource) {
    const groups = new Map();
    for (const [key, label] of d.schema.choices) { const g = F[key].group; if (!groups.has(g)) groups.set(g, []); groups.get(g).push([key, label.split(" · ").slice(1).join(" · ")]); }
    box.append(field(depth ? "From (what moves the rate)" : "What in the music", h("select", { onchange: async e => {
      const nst = await ask("knob_new", e.target.value, st); if (!nst) return;
      for (const key of Object.keys(st)) delete st[key]; Object.assign(st, nst); renderSub(); } },
      [...groups].map(([g, xs]) => h("optgroup", { label: g }, xs.map(([key, label]) => h("option", { value: key, selected: key === st.form, text: label })))))));
    const said = (d.schema.choices.find(([key]) => key === st.form) || [])[1];     // all of it, under it
    if (said && said.split(" · ").slice(1).join(" · ").length > 34) box.append(h("div", { class: "about", text: said.split(" · ").slice(1).join(" · ") }));
  }
  const grid = h("div", { class: "fields" });
  if (f.main && !onlySource) grid.append(knobSetting(f.main[0], f.main[1], f.main[2], f.main[3], () => st.main, v => { st.main = v; }));
  box.append(grid);
  // its source
  const src = st.source;
  if (f.source === "trigger" && src) {
    const t = h("div", { class: "fields one" });
    whoPlaysInto(t, src, d);
    const sigs = d.signals.filter(s => s.kind === "trigger");
    if (sigs.length && k_mode() === "knob") t.append(field("Or a who-plays signal", h("select", { onchange: e => { src.link = e.target.value || undefined; checkKnob(); } }, h("option", { value: "", text: "-" }), sigs.map(s => h("option", { value: s.name, selected: s.name === src.link, text: `*${s.name}` })))));
    box.append(t);
  } else if (f.source === "score" && src) {
    if (!src.score && d.sources.score) src.score = d.sources.score;      // the song's transcription, written
    box.append(h("div", { class: "fields" }, field("Transcription", textField(src.score || d.sources.score, "its folder", v => { src.score = v; checkKnob(); })),
      field("Snap", h("select", { onchange: e => { src.snap = e.target.value; checkKnob(); } }, d.schema.snaps.map(x => h("option", { value: x, selected: x === (src.snap || "auto"), text: x }))))));
  } else if (f.source === "feature") {
    const info = d.sources.feature_info || {}, groups = d.sources.feature_groups || [];
    const feats = d.sources.features.includes(st.source) ? d.sources.features : [st.source, ...d.sources.features];
    const opt = x => h("option", { value: x, selected: x === st.source, text: info[x] ? info[x].label : x, title: info[x] ? `${x}: ${info[x].about}` : x });
    const other = feats.filter(x => !info[x]);
    box.append(field("Feature", h("select", { onchange: e => { st.source = e.target.value; renderSub(); } },
      groups.map(g => { const xs = feats.filter(x => info[x] && info[x].group === g); return xs.length ? h("optgroup", { label: g }, xs.map(opt)) : null; }),
      other.length ? h("optgroup", { label: "other" }, other.map(opt)) : null),
      info[st.source] ? `${info[st.source].about} · ${st.source}, ${fmt(info[st.source].range[0])}..${fmt(info[st.source].range[1])}${info[st.source].unit ? ` ${info[st.source].unit}` : ""}` : st.source));
  }
  if (onlySource) return;
  // how it responds
  const ps = h("div", { class: "fields" });
  for (const [name, def, kind, help] of f.params) ps.append(knobSetting(name, def, kind, help, () => st.params[name], v => { st.params[name] = v; if (!(st.keep || []).includes(name)) (st.keep = st.keep || []).push(name); }));
  if (st.form !== "constant") for (const [name, def, kind, help] of d.schema.shaping) ps.append(knobSetting(name, def, kind, help, () => st[name], v => { st[name] = v; }));
  if (st.form !== "constant") ps.append(clipField(st, d));
  box.append(ps);
  if (st.form === "rate") {
    if (st.from) {
      const sub = h("div", { class: "nested" }); box.append(sub); drawState(sub, st.from, depth + 1, false, F, d);
      sub.append(h("button", { class: "lnkbtn", onclick: () => { st.from = null; renderSub(); } }, "Steady instead"));
    } else box.append(h("button", { class: "lnkbtn", onclick: async () => { st.from = await ask("knob_new", "hits", null); renderSub(); } }, "Make the rate follow the music"));
  }
}
const k_mode = () => knobCtx ? knobCtx.mode : "knob";
let knobTimer = 0;
function checkKnob() {
  clearTimeout(knobTimer);
  knobTimer = setTimeout(async () => {
    const k = knobCtx; if (!k) return;
    if (k.link) { const y = $("knobYaml"); if (y) y.textContent = `*${k.link}`; return; }
    let t0 = 0, t1 = S.song.duration;
    const span = k.span || "around";
    if (span === "around") { const p = now(); t0 = Math.max(0, p - 8); t1 = Math.min(S.song.duration, t0 + 16); t0 = Math.max(0, t1 - 16); }
    else if (span === "view") { const sc = $("scroll"); t0 = xToT(sc.scrollLeft); t1 = xToT(sc.scrollLeft + sc.clientWidth - HEAD_W); }
    else if (k.span === "here") { const b = barAt(now()); t0 = B[b - 1]; t1 = B[Math.min(NB(), b + 8)]; }
    const r = await call("knob_check", k.state, k.mode, t0, t1, 600).catch(e => ({ error: e.message }));
    if (knobCtx !== k) return;
    $("knobYaml").textContent = r.yaml || ""; $("knobErr").textContent = r.error || "";
    if (r.values) drawCurve($("knobCurve"), r, t0, t1);
    if (r.hits != null) showHits(r);
    if (!r.error) sendDraft(k, r.yaml);
    if (labOn()) { lab.hits = r.hit_times || null; labRefresh(); }
  }, 150);
}
function drawCurve(cv, r, t0, t1) {
  if (!cv) return;
  const d = devicePixelRatio || 1, w = cv.clientWidth || 300, hh = 120;
  cv.width = w * d; cv.height = hh * d;
  const g = cv.getContext("2d"); g.scale(d, d);
  g.fillStyle = "#0f1115"; g.fillRect(0, 0, w, hh);
  const X = t => (t - t0) / Math.max(1e-6, t1 - t0) * w;
  let lo = Math.min(...r.values), hi = Math.max(...r.values);                 // its own min and max
  if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
  const Y = v => hh - 16 - (v - lo) / (hi - lo) * (hh - 26);
  g.fillStyle = "#2b313d"; for (const b of r.bars) g.fillRect(X(b), 0, 1, hh - 12);
  g.fillStyle = "rgba(106,167,216,.6)"; for (const c of r.cuts || []) g.fillRect(X(c), hh - 10, 1, 10);
  g.strokeStyle = "#f0a04b"; g.lineWidth = 1.5; g.beginPath();
  r.times.forEach((t, i) => { const x = X(t), y = Y(r.values[i]); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
  const p = now(); if (p >= t0 && p <= t1) { g.fillStyle = "#e9e5db"; g.fillRect(X(p), 0, 1, hh); }
  g.fillStyle = "#9097a7"; g.font = "10px 'IBM Plex Mono', monospace"; g.fillText(`max ${fmt(hi)}`, 4, 11); g.fillText(`min ${fmt(lo)}`, 4, hh - 15);
  g.textAlign = "center";                        // the time along it
  const step = [1, 2, 4, 8, 15, 30, 60, 120].find(s => (t1 - t0) / s <= 8) || 240;
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) g.fillText(fmtTime(t).replace(/\.0$/, ""), Math.max(14, Math.min(w - 14, X(t))), hh - 2);
  g.textAlign = "left";
}
async function applyKnob() {
  const k = knobCtx; if (!k) return;
  await clearDraft(false);                        // the edit itself makes Live again
  const r = await edit("knob_apply", k.ref, k.keypath, k.state, k.mode, k.anchor, k.link);
  if (r && !r.error) { toast("Applied"); closeKnob(); }
  else { k.drafted = false; checkKnob(); }
}
// the link being edited, on Live before Apply (core/web/lab.py knob_draft): only once it
// differs from what the scene says; Cancel takes it away
async function sendDraft(k, yaml) {
  if (k.mode !== "knob" || k.link) return;
  if (k.openedYaml == null) { k.openedYaml = yaml; return; }
  if (yaml === k.openedYaml && !k.drafted) return;
  const r = await call("knob_draft", k.ref, k.keypath, k.state, k.mode).catch(() => null);
  if (!r || !r.ok || knobCtx !== k) return;
  k.drafted = true;
  if (view === "live") streamAsk(now(), true);
}
async function clearDraft(remake = true) {
  const r = await call("knob_draft_clear").catch(() => null);
  if (remake && r && r.had && view === "live") streamAsk(now(), true);
}

// ---------------------------------------------------------------------------
// The Feature Lab: the link being edited drawn step by step in the timeline's place (the
// same bars, zoom and playhead): the song, what the feature listens to, the feature, each
// step of the link, and what the parameter gets inside the range it expects. The settings
// stay in the area on the right; every change redraws the lanes and goes to Live at once.
// The curves are the render's own (core/web/lab.py knob_stages), for the bars in view.
// ---------------------------------------------------------------------------
let lab = null;          // {k: the link (knobCtx) it is for, data, heard, wave, hear: mix | heard, map, hits}
const labOn = () => !!lab && !!knobCtx && lab.k === knobCtx;
// the lanes share the timeline's height (it stays as it is): a weight and a least height each
const LAB_H = { audio: [0.55, 22], labheard: [0.7, 26], labfeat: [1.3, 42], labshape: [1.0, 34], labparam: [1.6, 52] };
const isFeatureForm = st => st && (st.form === "feature" || st.form === "relative");
const stemBufs = new Map();   // a stem's URL -> its decoded audio (the Lab plays it)
let labSeq = 0, labScrollT = 0;
function openLab() {
  const k = knobCtx; if (!k || k.mode !== "knob" || k.link) return;
  lab = { k, data: null, heard: null, wave: null, feature: null, hear: lab ? lab.hear : "mix", map: null, hits: null, ppb };
  showLabBar(); renderTimeline(); renderSub(); labRefresh();
}
function closeLab() {
  if (!lab) return;
  const was = lab.hear; lab = null;
  showLabBar(); renderTimeline();
  if (was !== "mix") rehear();
  if (knobCtx) renderSub();
}
function labLayout() {
  const L = []; let y = 0;
  const add = (o, hh) => { L.push({ shown: 0, ...o, top: 0, h: hh }); };
  const st = lab.k.state, F = lab.k.data.schema.forms[st.form] || {}, info = featInfo(st);
  add({ id: "audio", name: "Song", kind: "audio", c: "var(--dim)" }, LAB_H.audio);
  if (isFeatureForm(st)) add({ id: "lab:heard", name: "Heard", sub: heardSub(), kind: "labheard", c: "#6aa7d8" }, LAB_H.labheard);
  const trig = F.source === "trigger" || (st.form === "rate" && st.from && (lab.k.data.schema.forms[st.from.form] || {}).source === "trigger");
  if (trig && S.tracks.length) add({ id: "midi", name: "MIDI", kind: "midi", c: "var(--lane-midi, #c27cf0)" }, 8 + 9 * S.tracks.length);
  if (isFeatureForm(st)) add({ id: "lab:feature", name: "Feature", sub: info ? info.label : st.source, kind: "labfeat", c: "#9fb36a" }, LAB_H.labfeat);
  const mid = labStage("glided") || labStage("ratio");
  if (mid) add({ id: "lab:shape", name: "Shaped", sub: mid.label, kind: "labshape", c: "#d78bb0" }, LAB_H.labshape);
  const p = lab.k.data.param || {};
  add({ id: "lab:param", name: "Parameter", sub: `${p.name || "the value"}${p.range ? ` · ${fmt(p.range[0])}..${fmt(p.range[1])}` : ""}`, kind: "labparam", c: "var(--accent)" }, LAB_H.labparam);
  const fixed = L.filter(l => !Array.isArray(l.h)).reduce((a, l) => a + l.h, 0);
  const room = Math.max(0, $("scroll").clientHeight - RULER_H - 14 - fixed);
  const weight = L.filter(l => Array.isArray(l.h)).reduce((a, l) => a + l.h[0], 0);
  for (const l of L) { if (Array.isArray(l.h)) l.h = Math.max(l.h[1], Math.floor(room * l.h[0] / weight)); l.top = y; y += l.h; }
  return { lanes: L, height: y };
}
const labStage = key => lab && lab.data && lab.data.stages ? lab.data.stages.find(s => s.key === key) || null : null;
const featInfo = st => isFeatureForm(st) && lab && lab.k.data.sources.feature_info ? lab.k.data.sources.feature_info[st.source] || null : null;
function heardSub() {
  const hd = lab.heard && lab.heard.heard;
  if (!hd) return "…";
  return hd.kind === "stem" ? `the ${hd.stem} stem` : hd.kind === "band" ? `the mix, ${fmtHz(hd.lo)}–${fmtHz(hd.hi)}` : hd.why || "the mix";
}
const fmtHz = f => f >= 1000 ? `${fmt(f / 1000)} kHz` : `${fmt(f)} Hz`;
// the span the lanes are made for: the bars in view and half a view on each side
function labSpan() {
  const sc = $("scroll"), a = xToT(sc.scrollLeft), b = xToT(sc.scrollLeft + Math.max(50, sc.clientWidth - HEAD_W));
  const w = Math.max(1, b - a), dur = S.song.duration;
  return [Math.max(0, a - w / 2), Math.min(dur, b + w / 2)];
}
async function labRefresh() {
  if (!labOn()) return;
  const k = lab.k, n = ++labSeq, [t0, t1] = labSpan();
  const cols = Math.max(20, Math.min(4000, Math.round(tToX(t1) - tToX(t0))));
  if (isFeatureForm(k.state) && lab.feature !== k.state.source) {
    lab.feature = k.state.source; lab.heard = null; lab.wave = null;
    const feature = lab.feature;
    call("lab_heard", feature).then(r => { if (labOn() && lab.feature === feature) { lab.heard = r && !r.error ? r : null; renderTimeline(); if (lab.hear !== "mix") rehear(); } }).catch(() => { });
    call("lab_wave", feature).then(r => { if (labOn() && lab.feature === feature) { lab.wave = r && !r.error ? r : null; labDraw(); } }).catch(() => { });
  }
  const r = await call("knob_stages", k.state, t0, t1, cols).catch(e => ({ error: e.message }));
  if (n !== labSeq || !labOn()) return;
  const before = lab.data ? lab.data.stages.map(s => s.key).join() : null;
  lab.data = r && !r.error ? r : null; lab.error = r && r.error; lab.ppb = ppb;
  if (!lab.map) lab.map = mapOf(k.state);
  showLabBar();
  if (!lab.data || before !== lab.data.stages.map(s => s.key).join()) renderTimeline(); else labDraw();
}
// the map gain / bias say: the input's lo..hi (on the step before the final) onto the output's lo..hi
function mapOf(st) {
  const pre = labStage("glided") || labStage("ratio") || labStage("feature"), fin = labStage("final");
  const p = lab.k.data.param || {}, g = +st.gain || 1, b = +st.bias || 0;
  const out = p.range ? [...p.range] : fin ? [fin.p5, fin.p95] : [0, 1];
  if (Math.abs(g) < 1e-9) return { inLo: pre ? pre.p5 : 0, inHi: pre ? pre.p95 : 1, outLo: out[0], outHi: out[1] };
  return { inLo: (out[0] - b) / g, inHi: (out[1] - b) / g, outLo: out[0], outHi: out[1] };
}
function commitMap(m) {
  const st = lab.k.state;
  if (Math.abs(m.inHi - m.inLo) < 1e-9) return;
  st.gain = +((m.outHi - m.outLo) / (m.inHi - m.inLo)).toFixed(4);
  st.bias = +(m.outLo - st.gain * m.inLo).toFixed(4);
  lab.map = m; renderSub();
}
function labFit() {
  const pre = labStage("glided") || labStage("ratio") || labStage("feature"); if (!pre) return;
  const p = lab.k.data.param || {}, cur = lab.map || mapOf(lab.k.state);
  const out = p.range ? p.range : [cur.outLo, cur.outHi];
  if (pre.p95 - pre.p5 < 1e-6) { toast("The feature hardly moves in these bars: nothing to fit", true); return; }
  commitMap({ inLo: pre.p5, inHi: pre.p95, outLo: out[0], outHi: out[1] });
}
async function setLabRange(lo, hi) {
  const p = lab.k.data.param; if (!p) return;
  const r = await call("set_param_range", p.key, lo, hi).catch(e => ({ error: e.message }));
  if (!r || r.error) { toast(r ? r.error : "not kept", true); return; }
  Object.assign(p, await ask("param_range", lab.k.ref, lab.k.keypath) || {});
  lab.map = null; renderTimeline(); labRefresh();
}
function setHear(mode) { if (!lab) return; lab.hear = mode; showLabBar(); rehear(); }
// what plays: the song, the stem the feature listens to, or the song through its band
function hearing() {
  const hd = labOn() && lab.hear === "heard" && lab.heard ? lab.heard.heard : null;
  if (hd && hd.kind === "stem" && stemBufs.get(hd.url) instanceof AudioBuffer) return { buf: stemBufs.get(hd.url) };
  if (hd && hd.kind === "band" && actx) return { buf: audioBuf, lo: hd.lo, hi: Math.min(hd.hi, actx.sampleRate / 2 - 50) };
  return { buf: audioBuf };
}
async function rehear() {
  const hd = labOn() && lab.hear === "heard" && lab.heard ? lab.heard.heard : null;
  if (hd && hd.kind === "stem" && !stemBufs.has(hd.url)) {
    stemBufs.set(hd.url, "loading"); showLabBar();
    try {
      if (!actx) actx = new AudioContext();
      const r = await fetch(hd.url); if (!r.ok) throw new Error(`HTTP ${r.status}`);
      stemBufs.set(hd.url, await actx.decodeAudioData(await r.arrayBuffer()));
    } catch (e) { stemBufs.delete(hd.url); toast(`The stem could not be played: ${e.message}`, true); }
    showLabBar();
  }
  if (playing && !holding) { const t = now(); stopSource(); pausedAt = t; startSource(); }
}
function showLabBar() {
  const bar = $("labBar"); if (!bar) return;
  if (!labOn()) { bar.hidden = true; bar.replaceChildren(); return; }
  bar.hidden = false;
  const k = lab.k, st = k.state, p = k.data.param || {}, info = featInfo(st), feat = isFeatureForm(st);
  const hd = lab.heard && lab.heard.heard, loading = hd && hd.kind === "stem" && stemBufs.get(hd.url) === "loading";
  const seg = (v, label, title) => h("button", { class: "btn sm" + (lab.hear === v ? " on" : ""), "aria-pressed": lab.hear === v, title, disabled: v === "heard" && (!hd || hd.kind === "mix"), onclick: () => setHear(v) }, label);
  const lo = h("input", { type: "number", step: "any", value: p.range ? p.range[0] : "", placeholder: "lo", "aria-label": "the range's low end" });
  const hi = h("input", { type: "number", step: "any", value: p.range ? p.range[1] : "", placeholder: "hi", "aria-label": "the range's high end" });
  const keep = () => { if (lo.value !== "" && hi.value !== "") setLabRange(+lo.value, +hi.value); };
  lo.onchange = keep; hi.onchange = keep;
  const clip = h("input", { type: "checkbox", checked: !!st.clip, disabled: !p.range && !st.clip, onchange: e => { st.clip = e.target.checked && p.range ? [...p.range] : null; renderSub(); } });
  fill(bar, h("b", { text: "Feature Lab" }),
    h("span", { class: "hint", text: `${p.name || "the value"} ← ${info ? info.label : st.form}` }),
    feat ? h("span", { class: "tgroup" }, h("span", { class: "tlabel", text: "Hear" }), seg("mix", "the mix", "the song as it is"),
      seg("heard", loading ? "loading…" : "what it hears", hd && hd.kind === "mix" ? "it listens to the whole mix" : hd && hd.kind === "band" ? `the mix through ${fmtHz(hd.lo)}–${fmtHz(hd.hi)} only` : hd ? `the ${hd.stem} stem alone` : "…")) : null,
    feat ? h("button", { class: "btn sm", title: "The feature's usual values in these bars (p5..p95) mapped onto the parameter's range: writes gain and bias", onclick: labFit }, "Fit") : null,
    h("span", { class: "tgroup", title: p.from === "yours" ? "the range you set (kept for this parameter)" : p.from === "about" ? "the range its description says" : "its description says no range: set the one you want" },
      h("span", { class: "tlabel", text: "Range" }), lo, h("span", { class: "hint", text: ".." }), hi,
      p.from === "yours" ? h("button", { class: "lnkbtn", onclick: () => setLabRange(null, null) }, "as described") : h("span", { class: "hint", text: p.from === "about" ? "described" : "none" })),
    h("label", { class: "row nowrap", title: "the value held inside the range (clip), so a rare peak cannot throw it out" }, clip, h("span", { text: "hold inside" })),
    h("span", { class: "hint mono", text: `gain ${fmt(+st.gain || 1)} · bias ${fmt(+st.bias || 0)}${st.clip ? ` · clip ${fmt(st.clip[0])}..${fmt(st.clip[1])}` : ""}` }),
    lab.error ? h("span", { class: "err", text: lab.error }) : null,
    h("span", { class: "spacer" }),
    h("button", { class: "btn sm", onclick: closeLab }, "Back to the timeline"));
}
// the lanes' pictures, for the span the data has
function labDraw() {
  if (!labOn() || !layout) return;
  const inner = $("inner");
  for (const cv of inner.querySelectorAll("canvas[data-lab]")) {
    const l = laneById(cv.dataset.id); if (!l) continue;
    const g = cv.getContext("2d"), d = devicePixelRatio || 1;
    g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, cv.width, cv.height); g.setTransform(d, 0, 0, d, 0, 0);
    if (l.kind === "labheard") drawHeard(g, l);
    else drawLabLane(g, l);
  }
  for (const el of inner.querySelectorAll(".labhandle")) el.remove();
  if (lab.data && isFeatureForm(lab.k.state)) labHandles();
}
function drawHeard(g, l) {
  const w = lab.wave, mid = l.h / 2;
  if (!w) { g.fillStyle = "#4b5263"; g.font = "10px 'IBM Plex Mono', monospace"; g.fillText(lab.heard ? "…" : "reading it…", $("scroll").scrollLeft + 6, mid); return; }
  const src = w.kind === "mix" ? peaks : w.peaks, hz = w.kind === "mix" ? peaksHz : w.hz;
  if (!src) return;
  const sc = $("scroll"), x0 = Math.max(0, sc.scrollLeft - 200), x1 = sc.scrollLeft + sc.clientWidth + 200;
  g.fillStyle = l.c;
  for (let x = x0; x < x1; x += 1.5) {
    const t0 = xToT(x), t1 = xToT(x + 1.5);
    let m = 0; for (let i = Math.floor(t0 * hz); i <= Math.floor(t1 * hz) && i < src.length; i++) if (src[i] > m) m = src[i];
    const hh = Math.min(1, m) * (mid - 2);
    if (hh > 0.2) g.fillRect(x, mid - hh, 1.2, hh * 2);
  }
}
function drawLabLane(g, l) {
  const r = lab.data, H = l.h, pad = 8;
  const sc = $("scroll"), vx = sc.scrollLeft;
  g.font = "10px 'IBM Plex Mono', Consolas, monospace";
  if (!r) { g.fillStyle = lab.error ? "#e2665a" : "#4b5263"; g.fillText(lab.error || "…", vx + 6, H / 2); return; }
  const info = featInfo(lab.k.state), p = lab.k.data.param || {};
  const st = l.kind === "labfeat" ? labStage("feature") : l.kind === "labshape" ? (labStage("glided") || labStage("ratio")) : labStage("final");
  if (!st) return;
  // the scale: the feature's own range on the feature (and a glide of it), the parameter's on the parameter
  let lo = st.min, hi = st.max;
  const fr = info && info.range, same = l.kind === "labfeat" || (l.kind === "labshape" && st.key === "glided");
  if (same && fr) { lo = Math.min(lo, fr[0]); hi = Math.max(hi, fr[1]); }
  // a rare peak does not flatten the rest: the scale stops at what is usual, past it is marked
  if (st.key === "ratio") { lo = 0; hi = Math.max(2, Math.min(st.max, st.p95 * 2)); }
  if (l.kind === "labparam") {
    const base = p.range ? [...p.range] : [st.p5, st.p95], w = Math.max(1e-6, base[1] - base[0]);
    if (lab.map) { base[0] = Math.min(base[0], lab.map.outLo); base[1] = Math.max(base[1], lab.map.outHi); }
    lo = Math.max(st.min, base[0] - 1.5 * w); hi = Math.min(st.max, base[1] + 1.5 * w);
    lo = Math.min(lo, base[0]); hi = Math.max(hi, base[1]);
  }
  if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
  const span = hi - lo; lo -= span * 0.04; hi += span * 0.04;
  const Y = v => Math.max(1, Math.min(H - 1, pad + (1 - (v - lo) / (hi - lo)) * (H - 2 * pad)));
  l.scale = { lo, hi, Y, inv: y => lo + (1 - (y - pad) / (H - 2 * pad)) * (hi - lo) };
  const n = st.lo.length, dt = (r.t1 - r.t0) / n, X = c => tToX(r.t0 + (c + 0.5) * dt);
  const W = tToX(r.t1);
  const right = vx + sc.clientWidth - HEAD_W - 6;           // the guides' words at the right of the view, the lane's at the left
  const guide = (v, colour, text, dash) => {
    const y = Y(v); g.strokeStyle = colour; g.setLineDash(dash || []); g.lineWidth = 1; g.beginPath(); g.moveTo(tToX(r.t0), y + 0.5); g.lineTo(W, y + 0.5); g.stroke(); g.setLineDash([]);
    if (text) { g.fillStyle = colour; g.textAlign = "right"; g.fillText(text, right, Math.max(10, Math.min(H - 3, y - 2))); g.textAlign = "left"; }
  };
  if (l.kind === "labparam") {
    const m = lab.k.ref && (memberById(idOf(lab.k.ref)) || (lab.k.ref.lastIndexOf("fx") >= 2 ? memberById(idOf(lab.k.ref.slice(0, lab.k.ref.lastIndexOf("fx")))) : null));
    if (m && !m.always && m.spans) {                     // outside the layer's bars: dimmed (it does not show there)
      g.fillStyle = "rgba(0,0,0,.28)"; let from = 0;
      for (const s of [...m.spans].sort((a, b) => a.first - b.first)) { g.fillRect(from, 0, (s.first - 1) * ppb - from, H); from = s.last * ppb; }
      g.fillRect(from, 0, NB() * ppb - from, H);
    }
    if (p.range) { g.fillStyle = "rgba(240,160,75,.07)"; g.fillRect(tToX(r.t0), Y(p.range[1]), W - tToX(r.t0), Y(p.range[0]) - Y(p.range[1])); }
  }
  if (same && fr) { guide(fr[0], "#3b4352", null); guide(fr[1], "#3b4352", null); }
  if (l.kind === "labfeat" && r.song) {
    guide(r.song.p5, "rgba(159,179,106,.45)", `p5 ${fmt(r.song.p5)} (the song)`, [3, 4]);
    guide(r.song.p95, "rgba(159,179,106,.45)", `p95 ${fmt(r.song.p95)}`, [3, 4]);
  }
  if (st.key === "ratio") guide(1, "rgba(215,139,176,.5)", "1 = as usual lately", [3, 4]);
  if (l.kind === "labparam" && p.range) { guide(p.range[0], "rgba(240,160,75,.55)", `${fmt(p.range[0])}`, null); guide(p.range[1], "rgba(240,160,75,.55)", `${fmt(p.range[1])} · the range it expects`, null); }
  if (l.kind === "labparam" && lab.k.state.clip) { const c = lab.k.state.clip; guide(c[0], "rgba(233,229,219,.35)", null, [2, 3]); guide(c[1], "rgba(233,229,219,.35)", "held inside", [2, 3]); }
  const curve = (s, colour, alpha) => {
    g.fillStyle = colour; g.globalAlpha = alpha * 0.35; g.beginPath();
    for (let c = 0; c < n; c++) { const x = X(c), y = Y(s.hi[c]); c ? g.lineTo(x, y) : g.moveTo(x, y); }
    for (let c = n - 1; c >= 0; c--) g.lineTo(X(c), Y(s.lo[c]));
    g.closePath(); g.fill(); g.globalAlpha = alpha;
    g.strokeStyle = colour; g.lineWidth = 1.3; g.beginPath();
    for (let c = 0; c < n; c++) { const x = X(c), y = Y((s.hi[c] + s.lo[c]) / 2); c ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.stroke(); g.globalAlpha = 1;
  };
  if (l.kind === "labfeat" && labStage("average")) { g.setLineDash([4, 3]); curve(labStage("average"), "#e9e5db", 0.5); g.setLineDash([]); }
  if (l.kind === "labshape" && st.key === "glided") curve(labStage("feature"), "#9fb36a", 0.3);   // the raw one under it
  curve(st, l.kind === "labparam" ? "#f0a04b" : l.c, 1);
  if (l.kind === "labparam" && p.range) {                      // out of its range: red
    g.fillStyle = "rgba(226,102,90,.85)";
    for (let c = 0; c < n; c++) {
      if (st.hi[c] > p.range[1] + 1e-9) g.fillRect(X(c) - 1, Y(st.hi[c]), 2, Math.max(1, Y(p.range[1]) - Y(st.hi[c])));
      if (st.lo[c] < p.range[0] - 1e-9) g.fillRect(X(c) - 1, Y(p.range[0]), 2, Math.max(1, Y(st.lo[c]) - Y(p.range[0])));
    }
  }
  if (st.max > hi || st.min < lo) {                            // past the scale: marked, and how far
    g.fillStyle = "rgba(226,102,90,.9)";
    for (let c = 0; c < n; c++) { if (st.hi[c] > hi) g.fillRect(X(c) - 1, 0, 2, 4); if (st.lo[c] < lo) g.fillRect(X(c) - 1, H - 4, 2, 4); }
    g.textAlign = "right";
    if (st.max > hi) g.fillText(`↑ up to ${fmt(st.max)}`, right, 22);
    if (st.min < lo) g.fillText(`↓ down to ${fmt(st.min)}`, right, H - 6);
    g.textAlign = "left";
  }
  if (l.kind === "labparam" && lab.hits) { g.fillStyle = "rgba(240,160,75,.6)"; for (const t of lab.hits) if (t >= r.t0 && t <= r.t1) g.fillRect(tToX(t), 0, 1, 5); }
  const said = l.kind === "labparam" ? `${fmt(st.min)}..${fmt(st.max)} in these bars` : `${fmt(st.min)}..${fmt(st.max)} · p5 ${fmt(st.p5)} · p95 ${fmt(st.p95)} in these bars`;
  g.fillStyle = "rgba(20,23,29,.75)"; g.fillRect(vx + 2, 1, g.measureText(said).width + 6, 12);
  g.fillStyle = "#9097a7"; g.fillText(said, vx + 5, 10);
}
// the map's four handles: the input's lo / hi on the step before the final, the output's on the parameter
function labHandles() {
  const pre = laneById("lab:shape") || laneById("lab:feature"), out = laneById("lab:param");
  if (!pre || !out || !pre.scale || !out.scale) return;
  const m = lab.map || mapOf(lab.k.state);
  const put = (l, key, text) => {
    const el = h("div", { class: "labhandle" + (key.startsWith("out") ? " out" : ""), style: `top:${l.top + l.scale.Y(m[key])}px`, title: `${text}: drag up or down (writes gain and bias)` },
      h("span", { class: "tag", text: `${text} ${fmt(m[key])}` }));
    el.addEventListener("pointerdown", e => {
      e.stopPropagation(); e.preventDefault();
      try { el.setPointerCapture(e.pointerId); } catch (err) { /* not down */ }
      const top = $("inner").getBoundingClientRect().top;
      const move = ev => {
        const y = Math.max(l.top + 2, Math.min(l.top + l.h - 2, ev.clientY - top));
        m[key] = +l.scale.inv(y - l.top).toFixed(4); el.style.top = `${y}px`; el.firstChild.textContent = `${text} ${fmt(m[key])}`;
      };
      const up = ev => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); ev.stopPropagation(); commitMap({ ...m }); };
      el.addEventListener("pointermove", move); el.addEventListener("pointerup", up);
    });
    $("inner").append(el);
  };
  put(pre, "inLo", "from"); put(pre, "inHi", "from");
  put(out, "outLo", "to"); put(out, "outHi", "to");
}
// scrolled: the lanes drawn where one looks, and made again past the span they have
$("scroll").addEventListener("scroll", () => {
  if (!labOn()) return;
  clearTimeout(labScrollT);
  labScrollT = setTimeout(() => {
    if (!labOn()) return;
    const [a, b] = labSpan(), r = lab.data;
    if (!r || a < r.t0 - 0.05 || b > r.t1 + 0.05) labRefresh(); else labDraw();
  }, 120);
});

// ---------------------------------------------------------------------------
// Signals: the scene's anchors (a browser tab); choosing one lights what follows it
// ---------------------------------------------------------------------------
let sigLight = null;          // {name, refs: Set of member ids}
async function renderSignals(list) {
  const sigs = await ask("signals");
  if (!sigs || tab !== "signals") return;
  list.replaceChildren(h("div", { class: "row nowrap" }, h("span", { class: "hint", text: "in the scene's order" }), h("span", { class: "spacer" }),
    h("button", { class: "lnkbtn", onclick: () => renderSignals(list) }, "Refresh")));
  if (!sigs.length) list.append(h("div", { class: "empty-note", text: "No signals: a knob named in its editor (Or name this one) becomes one; others can follow it." }));
  for (const s of sigs) {
    const ids = new Set([s.def, ...s.uses].filter(Boolean).map(p => idOf(p.ref)));
    const on = sigLight && sigLight.name === s.name;
    list.append(h("div", { class: "sig" + (on ? " picked" : ""), onclick: e => { if (e.target.closest("button")) return; sigLight = on ? null : { name: s.name, ids }; renderTimeline(); renderSignals(list); } },
      h("div", { class: "row nowrap" }, h("b", { text: `&${s.name}` }), h("span", { class: "chip", text: s.kind }), h("span", { class: "spacer" }),
        s.kind !== "other" && s.def ? h("button", { class: "lnkbtn", onclick: async () => { const got = await ask("edit_signal", s.name); if (got) openKnobCtx({ ref: got.ref, keypath: got.keypath, mode: got.mode, title: got.title, data: got, state: got.state, link: "", anchor: "" }); } }, "Edit") : null),
      h("div", { class: "mono small", text: s.yaml }),
      h("div", { class: "hint", text: (s.def ? `defined in ${s.def.label}` : "defined in the scene's settings") + (s.uses.length ? ` · followed by ${s.uses.map(u => u.label).join(", ")}` : " · nothing follows it yet") })));
  }
}

// ---------------------------------------------------------------------------
// The project's inputs (a dock tab)
// ---------------------------------------------------------------------------
async function renderProject() {
  const box = $("dockProject"), f = await ask("project_fields");
  if (!f) return;
  const ch = {}, set = (k, v) => { ch[k] = v; $("projSave").disabled = false; };
  const pickBtn = (kind, k, inp, onGot = null) => inWebview ? h("button", { class: "ib", title: "choose", onclick: async () => { const got = await ask("pick_file", kind, false, "", inp.value || ""); if (got && got.path) { inp.value = got.path; if (onGot) onGot(got.path); else set(k, got.path); } } }, "…") : null;
  const fileRow = (label, k, kind, hint) => { const inp = h("input", { value: f[k], placeholder: hint, spellcheck: "false", onchange: e => set(k, e.target.value) }); return field(label, h("div", { class: "row nowrap" }, inp, pickBtn(kind, k, inp))); };
  const stems = f.stems.map(s => ({ ...s }));
  const stemBox = h("div", { class: "subrows" });
  const drawStems = () => stemBox.replaceChildren(...stems.map((s, i) => h("div", { class: "subrow" },
    h("input", { value: s.name, placeholder: "name", class: "small", onchange: e => { s.name = e.target.value; set("stems", stems); } }),
    (() => { const inp = h("input", { value: s.path, placeholder: "its file", onchange: e => { s.path = e.target.value; set("stems", stems); } });
      return h("div", { class: "row nowrap grow" }, inp, pickBtn("audio", "stems", inp, p => { s.path = p; if (!s.name) s.name = p.split(/[\\/]/).pop().replace(/\.[^.]+$/, ""); drawStems(); set("stems", stems); })); })(),
    h("button", { class: "ib", "aria-label": "Remove", onclick: () => { stems.splice(i, 1); drawStems(); set("stems", stems); } }, "✕"))),
    h("button", { class: "lnkbtn", onclick: () => { stems.push({ name: "", path: "" }); drawStems(); } }, "+ stem"));
  drawStems();
  const syncOut = h("output", { class: "mono", text: `${f.sync_offset_ms} ms` });
  box.replaceChildren(h("div", { class: "projform" },
    h("div", { class: "fields" },
      fileRow("Audio", "audio", "audio", "the song"), fileRow("MIDI", "midi", "midi", "none: the bars from the song's beat"),
      field("MIDI offset (s)", numField(f.midi_offset, v => set("midi_offset", v), { step: 0.001 })),
      field("Meter changes", textField(f.meter, "like 22:6/4,27:5/4", v => set("meter", v))),
      fileRow("Scene", "scene", "scene", "scenes/…yaml"),
      field("", h("button", { class: "btn sm", onclick: async () => { const v = await dialog("A new scene", "Its name (letters, digits, - and _): scenes/<name>.yaml, from the template; the project opens it.", [{ label: "Make it", value: "ok", primary: true }, { label: "Cancel", value: null }], ""); if (v && v.value && v.text) { const got = await ask("new_scene", v.text); if (got) { toast("New scene"); renderProject(); } } } }, "New scene…")),
      field("Picture against sound", h("div", { class: "row nowrap" }, h("input", { type: "range", min: -500, max: 500, step: 5, value: f.sync_offset_ms, oninput: e => { syncOut.textContent = `${e.target.value} ms`; if (S) S.project.sync_offset_ms = +e.target.value; }, onchange: e => set("sync_offset_ms", +e.target.value) }), syncOut)),
      field("Stem separation", check("skip the AI separation", f.skip_separation, v => set("skip_separation", v)))),
    sect("The project's own stems"), stemBox,
    h("div", { class: "btnrow" }, h("button", { class: "btn primary", id: "projSave", title: "writes the project file (as it is, when nothing changed)", onclick: async () => {
      const got = await ask("save_project", ch); if (got) { toast(got.reload && got.reload.length ? `Saved: loading ${got.reload.join(", ")} again` : "Saved"); renderProject(); } } }, "Save the project"),
      h("span", { class: "hint", text: f.path ? `into ${f.path}` : "into projects/<song>.yaml" }))));
}

// ---------------------------------------------------------------------------
// Render: the final video (a dialog)
// ---------------------------------------------------------------------------
let finalJob = null;
async function openRenderDialog() {
  if (!S) return;
  renderBars = loop ? [loop[0], loop[1]] : null;
  const m = $("renderDlg"); m.hidden = false;
  await drawRenderDialog();
}
let renderBars = null;          // the dialog's own first / last bars
let renderProblem = null;       // why the engine cannot render this project
async function drawRenderDialog() {
  const dlg = $("renderDlg"), box = $("renderBody"), what = dlg.dataset.what || (loop ? "bars" : "song");
  if (!renderBars) renderBars = loop ? [loop[0], loop[1]] : [1, Math.min(NB(), 8)];
  const [fb, lb] = renderBars;
  const r = await ask("render_settings", what === "bars" ? fb : null, what === "bars" ? lb : null, what);
  if (!r) return;
  finalJob = r.job; renderProblem = r.problem || null;
  const put = (k, v) => ask("set_render", { [k]: v }).then(x => x && drawRenderDialog());
  const g = r.gpu, radio = (v, label) => h("label", { class: "toggle" }, h("input", { type: "radio", name: "rwhat", checked: what === v, onchange: () => { dlg.dataset.what = v; drawRenderDialog(); } }), label);
  const setBars = (a, b) => { a = Math.max(1, Math.min(NB(), Math.round(a))); b = Math.max(a, Math.min(NB(), Math.round(b))); renderBars = [a, b]; drawRenderDialog(); };
  const startIn = textField("", "bar.beat, like 17.1", async v => { if (!v.trim()) return; const got = await ask("bar_beat", v.trim()); if (got) put("start_time", got.seconds); }, "start as bar.beat");
  const start = r.start_time || 0, startBar = S ? barAt(start + 1e-4) : 1;
  const folderRow = (k, label, kind, hint) => {
    const inp = textField(r[k], "none", v => put(k, v), label);
    return field(label, h("div", { class: "row nowrap" }, inp, inWebview ? h("button", { class: "ib", title: "choose a folder", onclick: async () => { const got = await ask("pick_file", kind, false, "", r[k] || ""); if (got && got.path) put(k, got.path); } }, "…") : null), hint);
  };
  const [w, hgt] = String(S.project.resolution || "854x480").split("x").map(Number);
  const needsMasks = members.some(m => m.lane === "fg" && m.kind === "clip" && !m.place?.sam && !m.place?.cutout) || members.some(m => (catalogByKey.get(m.key) || {}).needs?.includes("masks"));
  fill(box,
    h("div", { class: "row" }, radio("song", "the whole song"),
      radio("bars", "bars"), radio("start", "from a start")),
    what === "bars" ? h("div", { class: "fields" },
      field("First bar", numField(fb, v => setBars(v, Math.max(v, lb)), { min: 1, max: NB(), step: 1 })),
      field("Last bar", numField(lb, v => setBars(Math.min(fb, v), v), { min: 1, max: NB(), step: 1 })),
      field("", h("button", { class: "btn sm", disabled: !loop, onclick: () => loop && setBars(loop[0], loop[1]) }, loop ? `the chosen ${loop[0]}–${loop[1]}` : "choose bars on the ruler"))) : null,
    what === "start" ? h("div", { class: "fields" },
      field("Start (s)", numField(start, v => put("start_time", Math.max(0, v)), { min: 0, step: 0.5 }), `bar ${startBar} · ${fmtTime(start)}`),
      field("Or as bar.beat", h("div", { class: "row nowrap" }, startIn,
        h("button", { class: "btn sm", title: "the start of the bar the playhead is in", onclick: async () => { const got = await ask("bar_beat", "", now()); if (got) put("start_time", got.seconds); } }, "← playhead"))),
      field("Bars", numField(r.bars, v => put("bars", Math.max(0, Math.round(v))), { min: 0, step: 1 }), "0: to the end")) : null,
    h("div", { class: "about mono", text: `→ ${r.where}` }),
    h("div", { class: "about", text: `${w}×${hgt} at ${r.fps} fps · ${r.codec}${S.project.resolution ? "" : " (the default size)"} · the toolbar's Video sets the size` }),
    r.problem ? h("div", { class: "about", style: "color:var(--danger)", text: r.problem }) : null,
    h("div", { class: "about", style: g && g.used_mb / g.total_mb > 0.6 ? "color:var(--accent)" : "", text: g ? `GPU ${g.name}: ${g.util}% busy, ${Math.round(g.used_mb / 1024)} of ${Math.round(g.total_mb / 1024)} GB in use` : "no NVIDIA GPU found" }),
    r.dirty ? h("div", { class: "about", style: "color:var(--accent)", text: "The scene has unsaved changes: Render saves it first (the engine reads the file)." }) : null,
    h("div", { class: "fields" },
      field("Codec", h("select", { onchange: e => put("codec", e.target.value) }, r.choices.codec.map(c => h("option", { value: c, selected: c === r.codec, text: c })))),
      field("FPS", numField(r.fps, v => put("fps", Math.round(v)), { min: 1, max: 120, step: 1 })),
      field("Clips fit", h("select", { onchange: e => put("clip_fit", e.target.value) }, r.choices.clip_fit.map(c => h("option", { value: c, selected: c === r.clip_fit, text: c })))),
      field("Clip order", h("select", { onchange: e => put("order", e.target.value) }, r.choices.order.map(c => h("option", { value: c, selected: c === r.order, text: c })))),
      field("Seed", numField(r.seed ?? "", v => put("seed", Math.round(v)), { step: 1 })),
      field("Clip cache", numField(r.cache_size, v => put("cache_size", Math.round(v)), { min: 1, step: 1 })),
      folderRow("clips_dir", "Clips folder", "folder", "the montage's clips (a scene with a clip montage needs it)"),
      folderRow("masks", "Masks folder", "masks", needsMasks ? "a cast, a cutout by mask track or a mask: needs it" : "SAM2 mask tracks (for the cast and mask:)")),
    h("div", { class: "row" }, check("turn wide clips", r.turn_wide, v => put("turn_wide", v)), check("AI stems", r.stems, v => put("stems", v)), check("profile", r.profile, v => put("profile", v)),
      check("gravity override", r.gravity.enabled, v => put("gravity.enabled", v))),
    r.gravity.enabled ? h("div", { class: "fields" }, ["peak", "floor", "radius", "curve"].map(k => field(k, numField(r.gravity[k], v => put(`gravity.${k}`, v), { step: 0.05 })))) : null,
    h("div", { class: "progress wide", id: "finalProg" }, h("i", { id: "finalBar" })), h("div", { class: "about", id: "finalText" }),
    h("div", { class: "btnrow" },
      h("button", { class: "btn primary", id: "finalGo", disabled: !!r.problem, onclick: async () => { const got = await ask("render_final", what === "bars" ? fb : null, what === "bars" ? lb : null, what); if (got && got.saved) toast(`Saved ${got.saved} first`); } }, "Render"),
      h("button", { class: "btn danger", id: "finalStop", onclick: () => ask("stop_final") }, "Stop"),
      h("button", { class: "btn", id: "finalOpen", onclick: () => ask("open_output", "video") }, "Open the video"),
      h("button", { class: "btn", id: "finalFolder", onclick: () => ask("open_output", "folder") }, "Show it in its folder"),
      h("span", { class: "spacer" }), h("button", { class: "btn", onclick: () => { $("renderDlg").hidden = true; } }, "Close")));
  showFinal(finalJob);
}
function showFinal(j) {
  finalJob = j || finalJob;
  const jb = finalJob || {};
  const running = !!jb.running;
  const frac = jb.total ? jb.done / jb.total : 0, spent = jb.started ? Date.now() / 1000 - jb.started : 0;
  $("bRenderVideo").classList.toggle("busy", running);
  $("bRenderVideo").textContent = running ? (jb.total ? `Rendering ${Math.round(frac * 100)}%` : "Rendering: starting") : "Render video…";
  if (!$("finalText")) return;
  const left = running && frac > 0.01 ? spent / frac * (1 - frac) : null;
  $("finalBar").style.width = `${Math.round(frac * 100)}%`;
  $("finalText").textContent = running ? (jb.total ? `${Math.round(frac * 100)}% · frame ${jb.done || 0} of ${jb.total}${left != null ? ` · about ${fmtTime(left)} left` : ""}` : "starting: reading the song and the clips") + (jb.line ? ` · ${jb.line.slice(0, 80)}` : "")
    : jb.result === "stopped" && jb.total ? `stopped at frame ${jb.done || 0}/${jb.total}`
    : jb.result ? `${jb.result}${jb.out ? `: ${jb.out.split(/[\\/]/).pop()}` : ""}` : "";
  $("finalGo").disabled = running || !!renderProblem; $("finalStop").disabled = !running;
  $("finalOpen").disabled = $("finalFolder").disabled = jb.result !== "done";
}

// ---------------------------------------------------------------------------
// The clip browser: any folder (the last one remembered), covers and lengths from Python,
// the chosen clip playing from its "from" (its sound on or off), dragged onto a lane as long
// as it plays (a long one asks first); files dropped from the Explorer
// ---------------------------------------------------------------------------
let clipFolder = "", clipList = [], clipLens = {}, clipChosen = null, clipSound = false;
async function loadClipList(folder) {
  const got = await ask("list_clips", folder || clipFolder || "", $("search").value.trim());
  if (!got) return;
  clipFolder = got.folder || ""; clipList = got.clips || [];
  if (got.error) toast(got.error, true);
  if (tab === "clips") renderBrowser();
}
async function chooseClip(path, from = 0) {
  const info = await ask("clip_info", path);
  if (!info) return;
  clipChosen = { ...info, from: Math.min(from, info.duration || 0) };
  pickClip(info.path, clipChosen.from);
  showPlayer();
}
function showPlayer() {
  const box = $("player");
  if (!clipChosen) { box.hidden = true; return; }
  const c = clipChosen;
  box.hidden = false;
  const url = proxyOf[c.url] || c.url;
  const v = c.image ? h("img", { src: url, draggable: "true", alt: c.name, title: "an image: drag it onto a lane (4 bars)" })
    : h("video", { src: url, playsinline: true, loop: false, preload: "auto", draggable: "true", title: "drag it onto a lane: it goes there as long as it plays" });
  v.muted = !clipSound || playing;
  v.addEventListener("loadedmetadata", () => { v.currentTime = c.from || 0; v.play().catch(() => { }); });
  v.addEventListener("ended", () => { v.currentTime = c.from || 0; v.play().catch(() => { }); });
  v.addEventListener("dragstart", e => { dragItem = { type: "clip", path: c.path, name: c.name, label: c.name, from: c.from || 0 }; e.dataTransfer.setData("text/plain", c.path); e.dataTransfer.effectAllowed = "copy"; });
  v.addEventListener("dragend", () => { dragItem = null; });
  const out = h("output", { class: "mono", text: `${fmt(c.from || 0)} s` });
  const slider = h("input", { type: "range", min: 0, max: Math.max(0.01, c.duration || 0), step: 0.01, value: c.from || 0, "aria-label": "From (s)",
    oninput: e => { c.from = +e.target.value; out.textContent = `${fmt(c.from)} s`; v.currentTime = c.from; },
    onchange: () => { pickClip(c.path, c.from); } });
  fill(box,
    h("div", { class: "pscreen" }, v, c.proxy === "making" && !proxyOf[c.url] ? h("div", { class: "pnote", text: "making a proxy the page can play…" }) : null),
    h("div", { class: "pinfo", text: `${c.name} · ${c.image ? "an image" : `${fmt(c.duration || 0)} s`} · ${c.mask ? `mask: ${c.mask}` : "no mask track"}`, title: c.path }),
    c.image ? null : h("div", { class: "row nowrap" }, h("span", { class: "tlabel", text: "from" }), slider, out),
    h("div", { class: "row" },
      c.image ? null : check("sound", clipSound, on => { clipSound = on; v.muted = !on || playing; }),
      h("button", { class: "lnkbtn", title: "the SAM2 cutout FX on its member in the FG (its seeds)", onclick: async () => {
        const got = await ask("sam_for_clip", c.path); if (!got) return; if (got.lanes) apply(got); openSam(got.fx, got.new); } }, "Cut it out (SAM2)"),
      h("button", { class: "lnkbtn", onclick: () => { clipChosen = null; showPlayer(); } }, "✕")));
}
function renderClipTab(list) {
  const q = $("search").value.trim().toLowerCase();
  const folderIn = h("input", { value: clipFolder, placeholder: "a folder of clips", spellcheck: "false", "aria-label": "Clips folder",
    onkeydown: e => { if (e.key === "Enter") loadClipList(e.target.value); } });
  list.append(h("div", { class: "row nowrap folderrow" }, folderIn,
    h("button", { class: "ib", title: "choose a folder (any drive)", onclick: async () => { const got = await ask("pick_file", "folder"); if (got && got.path) loadClipList(got.path); } }, "…")));
  if (!clipList.length) { list.append(h("div", { class: "empty-note", text: clipFolder ? "No clips here (or drop a folder from the Explorer)." : "Choose a folder of clips (or drop one from the Explorer)." })); return; }
  list.append(h("div", { class: "hint", text: `${clipList.length} clip${clipList.length === 1 ? "" : "s"} in ${clipFolder.split(/[\\/]/).filter(Boolean).pop() || clipFolder}${q ? ` · ${clipList.filter(c => c.name.toLowerCase().includes(q)).length} match` : ""}`, title: clipFolder }));
  for (const c of clipList.filter(c => c.name.toLowerCase().includes(q)).slice(0, 400)) {
    const it = { type: "clip", path: c.path, name: c.name, label: c.name.replace(/\.[^.]+$/, ""), url: c.url, from: 0 };
    const th = thumbs.get(c.url), len = clipLens[c.url];
    const chosen = clipChosen && clipChosen.path === c.path;
    const row = h("div", { class: "bitem clip" + (chosen ? " picked" : ""), title: `${c.name}: click to play it; drag it onto the BG, the FG or the montage (it goes as long as it plays)`,
      onclick: e => { if (!e.target.closest("button")) chooseClip(c.path, 0); } },
      h("span", { class: "sw big", style: th ? `background-image:url(${th})` : "" }),
      h("span", { class: "nm2" }, h("span", { text: c.name }), len ? h("small", { text: `${fmt(len)} s` }) : null),
      h("button", { class: "add", title: "Put it in the BG at the playhead, as long as it plays", "aria-label": `Add ${c.name}`, onclick: () => putItem(it, "bg", barAt(now())) }, "+"));
    row.draggable = true;
    row.addEventListener("dragstart", e => { dragItem = it; e.dataTransfer.setData("text/plain", c.path); e.dataTransfer.effectAllowed = "copy"; });
    row.addEventListener("dragend", () => { dragItem = null; document.querySelectorAll(".drop-ok").forEach(x => x.classList.remove("drop-ok")); });
    list.append(row);
  }
}
// clips put on a lane, one after another, each as long as it plays; a long one asks first
async function placeClips(lane, first, todo, row = null) {
  let got = await edit("place_clips", lane, first, todo, null, row);
  while (got && got.asking) {
    const a = got.asking;
    const v = await dialog(`${a.name}: ${fmt(a.seconds)} s`, `It plays ${a.bars} bars${a.from ? ` from ${fmt(a.from)} s` : ""}. How many bars should it take in the ${a.lane === "clips" ? "montage" : a.lane.toUpperCase()}, from bar ${a.first}?`,
      [{ label: "Put it", value: "n", primary: true }, { label: `The whole video (${a.whole} bars${a.to_end ? ", to the song's end" : ""})`, value: "whole" }, { label: "Cancel", value: null }], String(a.suggest));
    if (!v || !v.value) { toast(`${a.name} not put${a.todo.length > 1 ? ` (nor the ${a.todo.length - 1} after it)` : ""}`); return; }
    const n = v.value === "whole" ? "whole" : Math.max(1, parseInt(v.text, 10) || a.suggest);
    got = await edit("place_clips", a.lane, a.first, a.todo, n, a.row);
  }
}
// files from the Explorer: on a lane they go there (clips, one after another from the bar
// under the mouse); on the browser a folder is listed, a clip is chosen
async function onDropped(e) {
  const kinds = await ask("paths_kind", e.paths);
  if (!kinds) return;
  const el = document.elementFromPoint(e.x, e.y);
  if (S && el && el.closest("#inner")) {
    const r = inner.getBoundingClientRect(), l = laneAtY(e.y - r.top), bar = xToBar(e.x - r.left);
    if (!kinds.clips.length) { toast("only video or image files go on the lanes", true); return; }
    if (!l || !(l.id === "bg" || l.id === "fg" || l.kind === "clips")) { toast("a clip goes on the BG, the FG or the montage", true); return; }
    placeClips(l.kind === "clips" ? "clips" : l.id, bar, kinds.clips.map(p => ({ path: p, from: 0 })), l.kind === "track" ? rowAtY(l, e.y - r.top) : null);
    return;
  }
  if (kinds.folders.length) { tab = "clips"; await loadClipList(kinds.folders[0]); renderBrowser(); }
  else if (kinds.clips.length) { tab = "clips"; renderBrowser(); chooseClip(kinds.clips[0], 0); toast(`${kinds.clips[0].split(/[\\/]/).pop()} chosen: drag it (its picture) onto the BG, the FG or the montage`); }
  else toast("drop video or image files, or a folder of them", true);
}
document.addEventListener("dragover", e => { if (e.dataTransfer && [...e.dataTransfer.types].includes("Files")) e.preventDefault(); });
document.addEventListener("drop", e => {
  if (e.dataTransfer && [...e.dataTransfer.types].includes("Files")) {
    e.preventDefault();
    if (!inWebview) toast("files from the Explorer need the app window (the browser does not tell their paths)", true);
  }
});

// ---------------------------------------------------------------------------
// The SAM2 cutout: how SAM2 finds what to cut out of a clip. By itself (auto, as always), or
// seeds of one's own: boxes and points (+ keep, - leave out) per object, on a frame
// ---------------------------------------------------------------------------
const SAM_COLOURS = ["#e2665a", "#3fb6a8", "#f0a04b", "#6aa7d8", "#c27cf0", "#9fb36a", "#e3c15b", "#d78bb0"];
let sam = null;
async function openSam(path, isNew = false) {
  const got = await ask("sam_open", path);
  if (!got) return;
  sam = { ...got, path, isNew, obj: 0, tool: "box", drag: null, img: null };
  $("samDlg").hidden = false;
  drawSamPanel();
  samFrame();
}
function samForm() {
  return { mode: sam.mode, at: sam.at, objects: sam.objects, max_objects: sam.max_objects, granularity: sam.granularity,
    reseed: sam.reseed, keep: sam.keep, feather: sam.feather, kept: sam.kept };
}
let samCheckTimer = 0;
function samChanged() {
  clearTimeout(samCheckTimer);
  samCheckTimer = setTimeout(async () => { const r = await ask("sam_check", sam.clip, samForm()); if (r && sam) $("samCache").textContent = r.cache; }, 200);
  drawSamCanvas();
}
async function samFrame() {
  const r = await ask("clip_frame", sam.clip, sam.at);
  if (!r || !sam) return;
  const img = new Image();
  img.onload = () => { if (sam) { sam.img = img; drawSamCanvas(); } };
  img.src = r.url;
}
function drawSamPanel() {
  const box = $("samBody"), seeding = sam.mode === "seeds";
  $("samTitle").textContent = `SAM2 cutout · ${sam.name}`;
  const atOut = h("output", { class: "mono", text: `${fmt(sam.at)} s` });
  const cv = h("canvas", { id: "samCanvas", width: 480, height: 480, class: "samcanvas" });
  wireSamCanvas(cv);
  box.replaceChildren(h("div", { class: "samgrid" },
    h("div", { class: "samleft" }, cv,
      h("div", { class: "row nowrap" }, h("span", { class: "tlabel", text: "frame" }),
        h("input", { type: "range", min: 0, max: Math.max(0.01, sam.duration), step: 0.04, value: sam.at, "aria-label": "Frame (s)",
          oninput: e => { sam.at = +e.target.value; atOut.textContent = `${fmt(sam.at)} s`; }, onchange: () => { samFrame(); samChanged(); } }), atOut),
      h("div", { class: "hint", text: `its block starts ${fmt(sam.start)} s into the clip` })),
    h("div", { class: "samright" },
      h("div", { class: "row" },
        h("label", { class: "toggle" }, h("input", { type: "radio", name: "sammode", checked: !seeding, onchange: () => { sam.mode = "auto"; if (sam.keep === "all" && sam.wasFigures) sam.keep = "figures"; drawSamPanel(); samChanged(); } }), "auto: SAM2 finds the objects (as always)"),
        h("label", { class: "toggle" }, h("input", { type: "radio", name: "sammode", checked: seeding, onchange: () => { sam.mode = "seeds"; if (sam.keep === "figures") { sam.keep = "all"; sam.wasFigures = true; } drawSamPanel(); samChanged(); } }), "my seeds: boxes and points on a frame")),
      seeding ? h("div", { class: "samseeds" },
        h("div", { class: "row nowrap" },
          h("select", { "aria-label": "Object", onchange: e => { sam.obj = +e.target.value; drawSamCanvas(); } }, sam.objects.map((o, k) => h("option", { value: k, selected: k === sam.obj, text: `object ${k + 1}` }))),
          h("button", { class: "btn sm", onclick: () => { sam.objects.push({ box: null, points: [] }); sam.obj = sam.objects.length - 1; drawSamPanel(); samChanged(); } }, "New object"),
          h("button", { class: "btn sm", onclick: () => { sam.objects[sam.obj] = { box: null, points: [] }; samChanged(); } }, "Clear it"),
          h("button", { class: "btn sm", onclick: () => { sam.objects = [{ box: null, points: [] }]; sam.obj = 0; drawSamPanel(); samChanged(); } }, "Clear all")),
        h("div", { class: "row" }, [["box", "box"], ["plus", "+ point (keep this)"], ["minus", "− point (leave this out)"]].map(([k, label]) =>
          h("label", { class: "toggle" }, h("input", { type: "radio", name: "samtool", checked: sam.tool === k, onchange: () => { sam.tool = k; } }), label))),
        h("div", { class: "hint", text: "on the picture: drag a box, or click a point; a right click takes the chosen object's nearest seed away" }))
        : h("div", { class: "fields" },
          field("Objects at most", numField(sam.max_objects, v => { sam.max_objects = Math.max(1, Math.round(v)); samChanged(); }, { min: 1, max: 16, step: 1 })),
          field("What", h("select", { onchange: e => { sam.granularity = e.target.value; samChanged(); } }, [["whole", "whole (characters, things)"], ["parts", "parts (eyes, hands…)"]].map(([v, t]) => h("option", { value: v, selected: v === sam.granularity, text: t })))),
          field("New objects every (s)", numField(sam.reseed, v => { sam.reseed = Math.max(0, v); samChanged(); }, { min: 0, step: 0.5 }), "0: only at the cuts")),
      h("div", { class: "fields" },
        field("Keep", h("select", { onchange: e => { sam.keep = e.target.value; samChanged(); } }, [["figures", "the figures (not a backdrop SAM2 took for an object)"], ["all", "every object"], ["drop1", "all but object 1"]].map(([v, t]) => h("option", { value: v, selected: v === sam.keep, text: t })))),
        field("Soft edge (px)", numField(sam.feather, v => { sam.feather = Math.max(0, Math.round(v)); samChanged(); }, { min: 0, step: 1 }))),
      h("div", { class: "about", id: "samCache", text: sam.cache }),
      h("div", { class: "btnrow" }, h("button", { class: "btn primary", onclick: () => closeSam(true) }, "OK"), h("button", { class: "btn", onclick: () => closeSam(false) }, "Cancel")))));
  drawSamCanvas();
}
function samRect(cv) {
  if (!sam.img) return [0, 0, cv.width, cv.height];
  const k = Math.min(cv.width / sam.img.naturalWidth, cv.height / sam.img.naturalHeight);
  const w = sam.img.naturalWidth * k, hh = sam.img.naturalHeight * k;
  return [(cv.width - w) / 2, (cv.height - hh) / 2, w, hh];
}
function drawSamCanvas() {
  const cv = $("samCanvas");
  if (!cv || !sam) return;
  const g = cv.getContext("2d"), [x0, y0, w, hh] = samRect(cv);
  g.fillStyle = "#0b0c0f"; g.fillRect(0, 0, cv.width, cv.height);
  if (sam.img) g.drawImage(sam.img, x0, y0, w, hh);
  const V = (u, v) => [x0 + u * w, y0 + v * hh];
  if (sam.mode !== "seeds") { g.fillStyle = "rgba(10,11,14,.7)"; g.fillRect(6, 6, 300, 22); g.fillStyle = "#e9e5db"; g.font = "12px 'IBM Plex Sans', sans-serif"; g.fillText("SAM2 finds the objects by itself on each shot", 12, 21); return; }
  sam.objects.forEach((o, k) => {
    const c = SAM_COLOURS[k % SAM_COLOURS.length], mine = k === sam.obj;
    if (o.box) { const [a, b] = V(o.box[0], o.box[1]), [z, y] = V(o.box[2], o.box[3]); g.strokeStyle = c; g.lineWidth = mine ? 3 : 1; g.strokeRect(a, b, z - a, y - b); g.fillStyle = c; g.font = "bold 13px monospace"; g.fillText(String(k + 1), a + 4, b + 14); }
    for (const [u, v, keep] of o.points || []) { const [px, py] = V(u, v); g.beginPath(); g.arc(px, py, mine ? 6 : 4, 0, Math.PI * 2); g.fillStyle = keep ? "#3cdc5a" : "#e63c3c"; g.fill(); g.strokeStyle = c; g.lineWidth = 2; g.stroke(); }
  });
  if (sam.drag) { const [a, b] = V(...sam.drag[0]), [z, y] = V(...sam.drag[1]); g.strokeStyle = "#fff"; g.lineWidth = 1; g.strokeRect(a, b, z - a, y - b); }
}
function wireSamCanvas(cv) {
  const uv = e => { const r = cv.getBoundingClientRect(), k = cv.width / r.width, [x0, y0, w, hh] = samRect(cv);
    return [Math.min(1, Math.max(0, ((e.clientX - r.left) * k - x0) / w)), Math.min(1, Math.max(0, ((e.clientY - r.top) * k - y0) / hh))]; };
  cv.addEventListener("contextmenu", e => {
    e.preventDefault();
    if (sam.mode !== "seeds") return;
    const [u, v] = uv(e), o = sam.objects[sam.obj];
    const near = (o.points || []).map((p, i) => [Math.abs(p[0] - u) + Math.abs(p[1] - v), i]).filter(n => n[0] <= 0.08).sort((a, b) => a[0] - b[0]);
    if (near.length) o.points.splice(near[0][1], 1);
    else if (o.box && o.box[0] <= u && u <= o.box[2] && o.box[1] <= v && v <= o.box[3]) o.box = null;
    samChanged();
  });
  cv.addEventListener("pointerdown", e => {
    if (e.button !== 0 || sam.mode !== "seeds") return;
    const [u, v] = uv(e);
    if (sam.tool === "box") { sam.drag = [[u, v], [u, v]]; try { cv.setPointerCapture(e.pointerId); } catch (err) { /* not down */ } }
    else { sam.objects[sam.obj].points.push([+u.toFixed(4), +v.toFixed(4), sam.tool === "plus" ? 1 : 0]); samChanged(); }
  });
  cv.addEventListener("pointermove", e => { if (sam && sam.drag) { sam.drag[1] = uv(e); drawSamCanvas(); } });
  cv.addEventListener("pointerup", e => {
    if (!sam || !sam.drag) return;
    const [a, b] = sam.drag[0], [u, v] = uv(e);
    sam.drag = null;
    if (Math.abs(u - a) > 0.01 && Math.abs(v - b) > 0.01)
      sam.objects[sam.obj].box = [+Math.min(a, u).toFixed(4), +Math.min(b, v).toFixed(4), +Math.max(a, u).toFixed(4), +Math.max(b, v).toFixed(4)];
    samChanged();
  });
}
async function closeSam(ok) {
  if (!sam) return;
  const s = sam;
  if (ok) {
    const r = await edit("sam_apply", s.path, samForm());
    if (!r || r.error) return;
  } else if (s.isNew) await edit("remove", s.path);
  sam = null; $("samDlg").hidden = true;
}

// ---------------------------------------------------------------------------
// A catalog entry: what it is and its demo (core/demos.py), made on asking
// ---------------------------------------------------------------------------
// the entry picked in the browser, under its list: its demo, what it is, Make demo
let entryKey = null, entryReq = 0;
async function showEntryPanel() {
  const box = $("entryPanel");
  if (!entryKey || (tab !== "gen" && tab !== "fx")) { box.hidden = true; return; }
  const n = ++entryReq, key = entryKey;
  const e = await ask("entry", key);
  if (!e || n !== entryReq) return;
  box.hidden = false;
  box.replaceChildren(h("div", { class: "row nowrap entryhead" }, h("b", { text: e.label }), h("span", { class: "chip", text: e.category }), h("span", { class: "spacer" }),
    h("button", { class: "lnkbtn", "aria-label": "Close", onclick: () => { entryKey = null; showEntryPanel(); } }, "✕")));
  entryInto(box, e, key);
}
async function inspEntry(box, key) {
  const e = await ask("entry", key);
  if (!e || !(sel && sel.type === "entry" && sel.key === key)) return;
  $("inspTitle").textContent = e.kind;
  box.append(h("div", { class: "ptitle" }, h("h3", { text: e.label }), h("div", { class: "src", text: `${e.kind} · ${e.category}` })));
  entryInto(box, e, key);
}
function entryInto(box, e, key) {
  const d = e.demo;
  if (d.url) {
    const v = h("video", { src: d.url, muted: true, loop: true, autoplay: true, playsinline: true, class: "demo", title: "click it: bigger" });
    v.muted = true;
    v.addEventListener("click", () => showDemoBig(e, d.url));
    box.append(v);
  }
  const states = { rendering: "making its demo…", stale: "the layer changed since this demo", missing: "no demo yet",
    failed: `its demo failed: ${d.error || "see the log"}`, ready: "click it: bigger" };
  box.append(h("div", { class: "row" }, h("span", { class: "about", style: d.state === "failed" ? "color:var(--danger)" : "", text: states[d.state] || d.state }),
    d.state === "missing" || d.state === "failed" || d.state === "stale" ? h("button", { class: "btn sm", title: "on the GPU: a few seconds, minutes for the 3D fractals",
      onclick: async () => { const r = await ask("make_demo", key); if (r) { toast(`${e.label}: making its demo`); showEntryPanel(); } } }, d.state === "stale" ? "Make it again" : "Make demo") : null));
  if (e.about) box.append(h("div", { class: "about", text: e.about }));
  const notes = [e.takes_input ? "* can also take the picture below as its input" : null,
    e.needs.includes("masks") ? "needs a mask track (SAM2)" : null, e.needs.includes("cuts") ? "works at the clip montage's cuts" : null].filter(Boolean);
  for (const n of notes) box.append(h("div", { class: "hint", text: n }));
  if (e.file) box.append(h("div", { class: "src mono small", text: e.file }));
  if (e.kind === "FX") box.append(h("div", { class: "btnrow" }, h("button", { class: "btn", onclick: () => addFx(["master"], key) }, "Add to the master")),
    h("div", { class: "hint", text: "or drag across an FX track (a lane's, the master's); or right-click a block: its own" }));
  else box.append(h("div", { class: "hint", text: "drag across the BG or the FG to put it there" }));
  box.append(h("details", { class: "params" }, h("summary", { text: `Parameters (${e.params.length}; ~ the music can move it)` }),
    h("div", { class: "plist2" }, e.params.map(p => h("div", { class: "mono small", title: p.about || "" }, `${p.knob ? "~ " : "  "}${p.name} = ${p.default == null ? "-" : p.default}`)))));
}
function showDemoBig(e, url) {
  const m = $("modal");
  $("modalTitle").textContent = `demo · ${e.label}`; $("modalText").hidden = true; $("modalInput").hidden = true;
  const v = h("video", { src: url, muted: true, loop: true, autoplay: true, playsinline: true, class: "demobig" });
  v.muted = true;
  $("modalBtns").replaceChildren(h("button", { class: "btn primary", onclick: () => { m.hidden = true; $("modalBig").remove(); } }, "Close"));
  m.querySelector(".modalcard").insertBefore(h("div", { id: "modalBig" }, v), $("modalBtns"));
  m.hidden = false;
}

// ---------------------------------------------------------------------------
// Repaint: a rendered video repainted frame by frame through a ComfyUI workflow (a dock tab)
// ---------------------------------------------------------------------------
let repaintJob = {};
async function renderRepaint() {
  const box = $("dockRepaint"), r = await ask("repaint_settings");
  if (!r) return;
  repaintJob = r.job || {};
  const put = (k, v) => ask("set_repaint", { [k]: v }).then(x => x && renderRepaint());
  const pickBtn = inWebview ? h("button", { class: "ib", title: "choose a video", onclick: async () => { const got = await ask("pick_file", "video", false, "", r.video || ""); if (got && got.path) put("video", got.path); } }, "…") : null;
  const check = h("div", { class: "about", id: "repaintCheck" });
  box.replaceChildren(h("div", { class: "projform" },
    h("div", { class: "about", text: "Every frame of a render repainted through a ComfyUI workflow (img2img / edit). It uses the GPU, and ComfyUI's queue is shared: check it first." }),
    h("div", { class: "fields" },
      field("Video", h("div", { class: "row nowrap" }, h("input", { value: r.video, placeholder: "a render (the last one comes here)", spellcheck: "false", onchange: e => put("video", e.target.value) }), pickBtn)),
      field("It starts in the song at (s)", numField(r.start, v => put("start", v), { min: 0, step: 0.1 })),
      field("Workflow", h("select", { onchange: e => put("workflow", e.target.value) }, h("option", { value: "", text: "(the scene's)" }), r.workflows.map(w => h("option", { value: w, selected: w === r.workflow, text: w })))),
      field("Scene", h("select", { onchange: e => put("scene", e.target.value) }, r.scenes.map(sc => h("option", { value: sc, selected: sc === r.scene, text: sc })))),
      field("Prompt", textField(r.prompt, "empty: the scene's prompts (e.g. make it claymation)", v => put("prompt", v))),
      field("FPS", numField(r.fps, v => put("fps", v), { min: 1, max: 60, step: 1 })),
      field("Seconds", numField(r.seconds, v => put("seconds", v), { min: 0, step: 1 }), "0: the whole video")),
    r.output ? h("div", { class: "about mono", text: `→ ${r.output}` }) : null,
    h("div", { class: "btnrow" },
      h("button", { class: "btn", onclick: async () => { check.textContent = "asking ComfyUI…"; const x = await ask("repaint_check"); check.textContent = x ? x.text : ""; } }, "Check ComfyUI / GPU"),
      h("button", { class: "btn primary", id: "repaintGo", onclick: () => ask("repaint_start") }, "Repaint"),
      h("button", { class: "btn danger", id: "repaintStop", onclick: async () => { const x = await ask("repaint_stop"); if (x) toast(x.text); } }, "Stop")),
    check, h("div", { class: "progress wide" }, h("i", { id: "repaintBar" })), h("div", { class: "about", id: "repaintText" })));
  showRepaint(repaintJob);
}
function showRepaint(j) {
  repaintJob = j || repaintJob;
  if (!$("repaintText")) return;
  const r = repaintJob, frac = r.total ? r.done / r.total : 0;
  const spent = r.started ? Date.now() / 1000 - r.started : 0, left = r.running && frac > 0.01 ? spent / frac * (1 - frac) : null;
  $("repaintBar").style.width = `${Math.round(frac * 100)}%`;
  $("repaintText").textContent = r.running ? `frame ${r.done || 0} of ${r.total || "?"}${left != null ? ` · about ${fmtTime(left)} left` : ""}${r.line ? ` · ${r.line.slice(0, 90)}` : ""}` : r.result || "";
  $("repaintGo").disabled = !!r.running;          // Stop at any time: it also clears ComfyUI's queue
}

// ---------------------------------------------------------------------------
// Meters: the MIDI tracks, and the chosen block's knobs
// ---------------------------------------------------------------------------
function buildMeters() {
  $("meters").replaceChildren(...S.tracks.slice(0, 6).map((tr, i) => h("div", { class: "mtr" },
    h("div", { class: "lb" }, h("span", { text: tr.name, title: tr.name })), h("div", { class: "bar" }, h("i", { dataset: { tr: i }, style: `background:${TRACK_COLORS[i]}` })))));
}
const lastIdx = (arr, t) => { let lo = 0, hi = arr.length - 1, r = -1; while (lo <= hi) { const m = (lo + hi) >> 1; if (arr[m] <= t) { r = m; lo = m + 1; } else hi = m - 1; } return r; };
function updateMeters(t) {
  S.tracks.slice(0, 6).forEach((tr, i) => {
    const j = lastIdx(tr.times, t), v = j < 0 ? 0 : Math.exp(-(t - tr.times[j]) / 0.15) * tr.vels[j] / 127;
    const el = document.querySelector(`[data-tr="${i}"]`); if (el) el.style.width = Math.min(100, v * 100) + "%";
  });
  if (sel && sel.type === "member") {
    const m = memberById(sel.id);
    if (m) for (const p of m.params) {
      if (p.knob == null) continue;
      const v = knobAt(p.knob, t);
      const o = document.querySelector(`[data-out="${CSS.escape(p.name)}"]`); if (o) o.textContent = v == null ? "…" : fmt(v);
      const b = document.querySelector(`[data-meter="${CSS.escape(p.name)}"]`);
      if (b && v != null) { const k = knobData(p.knob), vs = k.values, mx = k._max ?? (k._max = Math.max(...vs.map(Math.abs), 1e-9)); b.style.width = Math.min(100, Math.abs(v) / mx * 100) + "%"; }
    }
  }
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
let fpsT = performance.now(), frames = 0;
function tick() {
  if (S) {
    let t = now();
    if (playing && loop && t >= B[loop[1]]) { seek(B[loop[0] - 1]); t = now(); }
    if (playing && t >= S.song.duration - 0.05) { setPlaying(false); pausedAt = loop ? B[loop[0] - 1] : 0; t = pausedAt; }
    frameNo++;
    if (gl && view === "sketch" && !gpuBusy.length) { drawFrame(t + ((S.project.sync_offset_ms || 0) / 1000)); showChips(); }
    if (view === "live") streamTick(t);
    syncEngineVideo(t);
    const bar = barAt(t), beats = S.song.beats;
    let beat = 1; if (beats.length) { const i = lastIdx(beats, t + 1e-4), i0 = lastIdx(beats, B[bar - 1] + 1e-4); beat = Math.max(1, i - i0 + 1); }
    $("lcdPos").textContent = `${bar}.${beat}`; $("lcdTime").textContent = fmtTime(t);
    const ph = $("playhead");
    if (ph) {
      const x = tToX(t); ph.style.transform = `translateX(${x}px)`;
      const rp = $("rulerPlay"); if (rp) rp.style.transform = `translateX(${x}px)`;
      const sc = $("scroll"), seen = sc.clientWidth - HEAD_W;
      if (playing && !drag && !rsel && zoomAt == null && (x < sc.scrollLeft || x > sc.scrollLeft + seen - 40)) sc.scrollLeft = Math.max(0, x - 80);
    }
    updateMeters(t);
    frames++;
    const n = performance.now();
    if (n - fpsT > 500) {
      const k = 1000 / (n - fpsT);
      $("fpsText").textContent = view === "sketch" ? `sketch · ${W}×${H} · ${Math.round(frames * k)} fps` : view === "live" ? `live: the engine · ${Math.round(streamShownN * k)} fps` : "exact: a render";
      frames = 0; streamShownN = 0; fpsT = n;
    }
  }
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------------------
// Dialogs
// ---------------------------------------------------------------------------
// a small dialog: buttons [{label, value, primary, danger}]; with input, answers {value, text}
function dialog(title, text, buttons, input = null) {
  return new Promise(resolve => {
    const m = $("modal"), inp = $("modalInput");
    $("modalTitle").textContent = title; $("modalText").textContent = text || ""; $("modalText").hidden = !text;
    inp.hidden = input == null; if (input != null) inp.value = input;
    const done = v => { m.hidden = true; document.removeEventListener("keydown", onKey, true); resolve(input != null ? { value: v, text: inp.value.trim() } : v); };
    const onKey = e => {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done(null); }
      else if (e.key === "Enter" && input != null) { e.preventDefault(); e.stopPropagation(); done((buttons.find(b => b.primary) || buttons[0]).value); }
    };
    $("modalBtns").replaceChildren(...buttons.map(b => h("button", { class: "btn" + (b.primary ? " primary" : "") + (b.danger ? " danger" : ""), onclick: () => done(b.value) }, b.label)));
    m.hidden = false; document.addEventListener("keydown", onKey, true);
    (input != null ? inp : $("modalBtns").querySelector(".primary") || $("modalBtns").firstChild).focus();
  });
}
function menu(anchor, items) {      // items: [{label, hint, on, cur, sub, danger}] or "hr"
  closeCtx();
  const box = h("div", { class: "ctx", role: "menu" }, items.map(it => it === "hr" ? h("hr") : it.sub ? h("div", { class: "sub", text: it.sub })
    : h("button", { role: "menuitem", class: it.cur ? "cur" : null, style: it.danger ? "color:var(--danger)" : null, disabled: it.disabled, onclick: () => { closeCtx(); it.on(); } }, it.label, it.hint ? h("small", { text: it.hint }) : null)));
  document.body.append(box);
  const r = anchor.getBoundingClientRect(), mr = box.getBoundingClientRect();
  box.style.left = Math.max(8, Math.min(r.left, innerWidth - mr.width - 8)) + "px";
  box.style.top = Math.min(r.bottom + 4, innerHeight - mr.height - 8) + "px";
  anchor.setAttribute("aria-expanded", "true");
  const close = new MutationObserver(() => { if (!box.isConnected) { anchor.setAttribute("aria-expanded", "false"); close.disconnect(); } });
  close.observe(document.body, { childList: true });
  box.querySelector("button")?.focus();
}

// ---------------------------------------------------------------------------
// Projects: the start, opening, the loading steps
// ---------------------------------------------------------------------------
let boot0 = { recent: [], projects: [], in_window: false };
function pitem(p, first) {
  return h("button", { class: "pitem" + (first ? " first" : ""), title: p.path, onclick: () => openProject(p.path) }, h("b", { text: p.name }), h("small", { text: p.path }));
}
async function showStart() {
  const got = await ask("boot");
  if (got) boot0 = { ...boot0, recent: got.recent, projects: got.projects };
  const rec = boot0.recent.filter(p => !S || p.path !== S.project.path);
  $("recentList").replaceChildren(...(rec.length ? rec.map((p, i) => pitem(p, i === 0)) : [h("div", { class: "empty-note", text: "Nothing opened yet." })]));
  $("projList").replaceChildren(...(boot0.projects.length ? boot0.projects.map(p => pitem(p, false)) : [h("div", { class: "empty-note", text: "No projects in projects/." })]));
  $("startClose").hidden = !S;
  $("pathForm").hidden = boot0.in_window;
  $("start").hidden = false;
  ($("recentList").querySelector(".pitem") || $("projList").querySelector(".pitem") || $("bOpenFile")).focus();
}
function hideStart() { $("start").hidden = true; }
async function guardUnsaved(what) {
  if (!S || !dirty) return true;
  const v = await dialog("Unsaved changes", `${S.project.scene} has edits that are not saved. ${what}?`,
    [{ label: "Save first", value: "save", primary: true }, { label: "Discard them", value: "drop", danger: true }, { label: "Cancel", value: null }]);
  if (v === "save") return !!(await save());
  return v === "drop";
}
async function openProject(source) {
  if (!(await guardUnsaved("Open another project anyway"))) return;
  hideStart();
  await setPlaying(false);
  const name = typeof source === "string" ? source : "the command line's project";
  log("out", `open ${name}`);
  loadGen++;
  subStack = []; knobCtx = null; renderSub();
  S = null; members = []; montage = []; pins = {}; sel = null; loop = null; engine = null; peaks = null; audioBuf = null; audioUrl = null; stemWaves = [];
  blockCovers.clear(); coversAsked.clear(); entryKey = null; showEntryPanel();
  picked = null; pickedFx = null; rowsOf = {}; groups = {}; has = {}; board = {}; montageBars = []; lastNote = ""; showPicked();
  knobs.clear(); thumbs.clear(); shaders.clear(); dirty = false; savedAt = "";
  for (const v of vids.values()) { v.el.removeAttribute("src"); v.el.load(); }
  vids.clear();
  stream = null; streamState = {}; streamMade = -1; streamShown = -1; holding = false;
  setView("live"); $("viewEngine").disabled = true;
  renderTimeline(); renderInspector(); showSaveState();
  $("projName").textContent = typeof source === "string" ? source.split(/[\\/]/).pop().replace(/\.ya?ml$/, "") : "untitled";
  $("lpTitle").textContent = `Opening ${$("projName").textContent}`; $("lpSub").textContent = typeof source === "string" ? source : "";
  steps = []; showLoad(true);
  const r = await ask("open", source);
  if (!r) { showLoad(false); showStart(); return; }
  loadGen = r.gen; steps = r.load; renderSteps();
}
function onOpened(st) {
  subStack = []; knobCtx = null; renderSub();
  S = st; B = st.song.bars;
  catalogByKey = new Map(st.catalog.map(c => [c.key, c]));
  pausedAt = 0;
  setSize(st.project.resolution);
  canvas.style.aspectRatio = `${W} / ${H}`;
  $("projName").textContent = st.project.name;
  $("bProject").title = `${st.project.path || "(not saved as a project)"} · scene ${st.project.scene}`;
  document.title = `${st.project.name} · ARC Studio`;
  apply(st);
  if (st.montage) montage = st.montage;
  stemWaves = st.stems || [];
  buildMeters(); renderBrowser(); renderTimeline(); showResolution();
  loadClipList(st.clips_dir || "");
  if (st.song.audio_url) loadAudio(st.song.audio_url);
  if (st.project.converted) {
    const say = "A flat scene (no BG / FG groups): it is shown grouped into BG, FG and the master; Save writes it so.";
    log("warn", say); toast(say); lastNote = "a flat scene: Save writes it grouped";
  }
  showStatus();
  if (st.load) { steps = st.load; renderSteps(); }
}
const STEP_ICON_LABEL = { waiting: "waiting", running: "running", done: "done", failed: "failed", skipped: "skipped" };
function onLoadStep(e) {
  if (e.gen !== loadGen && loadGen) return;
  const i = steps.findIndex(s => s.step === e.step);
  const got = { ...e, _at: performance.now() };
  if (i >= 0) steps[i] = { ...steps[i], ...got }; else steps.push(got);
  renderSteps();
}
setInterval(() => { if (!$("loadPanel").hidden && steps.some(s => s.state === "running")) renderSteps(); }, 500);
function stepSeconds(s) {
  if (s.state === "waiting" || s.seconds == null) return "";
  const t = s.seconds + (s.state === "running" && s._at ? (performance.now() - s._at) / 1000 : 0);
  return `${t < 10 ? t.toFixed(1) : Math.round(t)} s`;
}
let stepsTimer = 0, loadHideTimer = 0;
function renderSteps() {
  if (stepsTimer) return;
  stepsTimer = requestAnimationFrame(() => {
    stepsTimer = 0;
    const ol = $("steps");
    ol.replaceChildren(...steps.map(s => {
      const pct = s.of ? Math.max(0, Math.min(100, 100 * s.done / s.of)) : null;
      return h("li", { class: `step ${s.state}` },
        h("span", { class: "ic", role: "img", "aria-label": STEP_ICON_LABEL[s.state] || s.state }),
        h("span", { class: "nm", text: s.title || s.step }),
        h("span", { class: "tm", text: stepSeconds(s) }),
        s.message ? h("span", { class: "msg", text: s.message }) : null,
        s.state === "running" && pct != null ? h("div", { class: "bar" }, h("i", { style: `width:${pct}%` })) : null,
        s.state === "failed" || (s.state === "skipped" && s.message === "cancelled") ? h("button", { class: "lnkbtn retry", onclick: () => ask("retry", s.step) }, "Try again") : null);
    }));
    const n = steps.length, ended = steps.filter(s => ["done", "failed", "skipped"].includes(s.state)).length;
    const bad = steps.filter(s => s.state === "failed").length, running = steps.some(s => s.state === "running");
    $("loadChip").hidden = !n; $("loadChip").classList.toggle("bad", !!bad);
    $("loadDot").className = "dot" + (bad ? " bad" : running || ended < n ? " busy" : "");
    $("loadText").textContent = ended < n ? `loading ${ended}/${n}` : bad ? `${bad} step${bad > 1 ? "s" : ""} failed` : "loaded";
    $("lpCancel").hidden = ended === n;
    $("lpTitle").textContent = (ended < n ? "Opening " : bad ? "Opened, with problems: " : "Opened ") + $("projName").textContent;
    clearTimeout(loadHideTimer);
    if (n && ended === n && !bad && !$("loadPanel").hidden) loadHideTimer = setTimeout(() => showLoad(false), 1800);
  });
}
function showLoad(on) { $("loadPanel").hidden = !on; if (on) renderSteps(); }
function projectMenu() {
  const cur = S && S.project.path;
  const items = [{ label: "Open…", hint: inWebview ? "" : "a path", on: openFileDialog }, { label: "New from a song's folder…", hint: "its mix, MIDI and stems", on: newFromAudio }, { label: "Projects…", on: showStart }];
  const rec = (boot0.recent || []).filter(p => p.path !== cur).slice(0, 8);
  if (rec.length) items.push("hr", { sub: "Recent" }, ...rec.map(p => ({ label: p.name, hint: p.path, on: () => openProject(p.path) })));
  if (S) items.push("hr", { label: "Reload the scene from disk", on: reloadScene });
  menu($("bProject"), items);
}
async function openFileDialog() {
  if (!inWebview) { await showStart(); $("pathInput").focus(); return; }
  const r = await ask("pick_file", "project");
  if (r && r.path) openProject(r.path);
}
// a new project from a song's folder (as REAPER renders it): the mix, its MIDI and its stems,
// each heard and named here; with stems of its own the AI separation is off
async function newFromAudio() {
  let path = null;
  if (inWebview) { const r = await ask("pick_file", "folder", false, "", "input/"); path = r && r.path; }
  else { const v = await dialog("New from a song's folder", "The folder with the song (its mix), its MIDI and its stems, like input/cerne. A project and a new scene are made beside the others (projects/, scenes/).", [{ label: "Look in it", value: "ok", primary: true }, { label: "Cancel", value: null }], "input/"); path = v && v.value && v.text; }
  if (!path) return;
  const got = await ask("scan_song_folder", path);
  if (got) songDialog(got);
}
let songAudio = null;
function songDialog(f) {
  const dlg = $("songDlg"), box = $("songBody");
  const rows = f.audios.map(a => ({ ...a, use: a.path !== f.mix }));
  let mix = f.mix, midi = f.midi, separation = false;
  const nameIn = h("input", { value: f.name, spellcheck: "false", "aria-label": "Project name" });
  const stop = () => { if (songAudio) { songAudio.pause(); songAudio = null; } box.querySelectorAll(".playing").forEach(b => { b.classList.remove("playing"); b.textContent = "▶"; }); };
  const play = (a, btn) => {
    const was = btn.classList.contains("playing"); stop(); if (was) return;
    songAudio = new Audio(a.url); songAudio.currentTime = Math.min(30, (a.seconds || 0) / 4); songAudio.play().catch(() => { });
    btn.classList.add("playing"); btn.textContent = "■";
  };
  const close = () => { stop(); dlg.hidden = true; };
  const draw = () => {
    const stems = rows.filter(r => r.path !== mix);
    const lens = new Set(f.audios.map(a => Math.round((a.seconds || 0) * 10)));
    fill(box,
      h("div", { class: "about mono", text: f.folder }),
      field("Project", nameIn, f.exists ? "projects/<name>.yaml exists: give another name" : "projects/<name>.yaml and scenes/<name>.yaml"),
      h("div", { class: "fields" },
        field("The song (its mix)", h("select", { onchange: e => { mix = e.target.value; draw(); } }, f.audios.map(a => h("option", { value: a.path, selected: a.path === mix, text: a.file })))),
        field("MIDI", h("select", { onchange: e => { midi = e.target.value; } }, h("option", { value: "", text: "none: the bars from the song's beat" }), f.midis.map(m => h("option", { value: m, selected: m === midi, text: m.split("/").pop() }))))),
      sect(`Its stems (${stems.filter(r => r.use).length} of ${stems.length})`),
      stems.length ? h("div", { class: "subrows" }, stems.map(r => {
        const btn = h("button", { class: "ib", title: "listen", "aria-label": `Listen to ${r.file}`, onclick: () => play(r, btn) }, "▶");
        return h("div", { class: "subrow" + (r.use ? "" : " off") },
          h("input", { type: "checkbox", checked: r.use, "aria-label": `Use ${r.file}`, onchange: e => { r.use = e.target.checked; draw(); } }),
          h("input", { value: r.name, placeholder: "its name (listen ▶)", class: "small", spellcheck: "false", "aria-label": `${r.file}: its name`, oninput: e => { r.name = e.target.value; } }),
          h("span", { class: "mono small", text: `${r.file} · ${fmt(r.seconds || 0)} s` }), btn);
      })) : h("div", { class: "about", text: "No other audio files: the AI separation makes the stems." }),
      lens.size > 1 ? h("div", { class: "about", style: "color:var(--accent)", text: "Their lengths differ: stems should be rendered from the same start as the mix." }) : null,
      check("AI stem separation too", separation || !stems.some(r => r.use), v => { separation = v; }),
      h("div", { class: "hint", text: "The song's spectrum, onsets and sub-bands are read either way (some seconds); the AI separation is what the stems spare. " + (f.rpps.length ? `REAPER: ${f.rpps.join(", ")}` : "") }),
      h("div", { class: "err", id: "songErr" }),
      h("div", { class: "btnrow" }, h("span", { class: "spacer" }), h("button", { class: "btn", onclick: close }, "Cancel"),
        h("button", { class: "btn primary", onclick: async () => {
          const r = await call("new_song_project", { name: nameIn.value, audio: mix, midi, separation,
            stems: stems.filter(x => x.use).map(x => ({ name: x.name, path: x.path })) }).catch(e => ({ error: e.message }));
          if (!r || r.error) { $("songErr").textContent = r ? r.error : "no answer"; return; }
          close(); toast(`made ${r.path}`); openProject(r.path);
        } }, "Make the project")));
  };
  draw();
  dlg.hidden = false; nameIn.focus();
}

// ---------------------------------------------------------------------------
// The scene file: save, save as, reload
// ---------------------------------------------------------------------------
async function save() {
  if (!S) return null;
  let r = null;
  try { r = await call("save"); } catch (e) { r = { error: e.message }; }
  if (!r || r.error) { const why = (r && r.error) || "see the log"; showSaveState(`NOT saved: ${why}`.slice(0, 90), true); toast(`Not saved: ${why}`, true); log("err", `not saved: ${why}`); return null; }
  apply(r);
  $("yaml").textContent = r.text;
  showSaveState("Saved!"); setTimeout(() => showSaveState(), 1500);
  toast(`Saved ${r.path}`);
  return r;
}
async function saveAs() {
  if (!S) return;
  let path = null;
  const suggested = (S.project.scene || "scene.yaml").split(/[\\/]/).pop().replace(/\.ya?ml$/, "") + "_copy.yaml";
  if (inWebview) { const r = await ask("pick_file", "scene", true, suggested); path = r && r.path; }
  else { const v = await dialog("Save the scene as", "Where to write it (relative to the repository). The window then edits that file; the project file still opens the old one until it is saved.", [{ label: "Save", value: "ok", primary: true }, { label: "Cancel", value: null }], `scenes/${suggested}`); path = v && v.value && v.text; }
  if (!path) return;
  const r = await ask("save_as", path);
  if (!r) return;
  S.project = r.project; apply(r); $("yaml").textContent = r.text;
  toast(`Saved as ${r.path}`);
}
async function reloadScene() {
  if (!S) return;
  if (dirty) {
    const v = await dialog("Reload the scene", `Read ${S.project.scene} from disk again? The edits not saved, and the undo history, go.`, [{ label: "Reload", value: "ok", danger: true }, { label: "Cancel", value: null, primary: true }]);
    if (v !== "ok") return;
  }
  const r = await ask("reload");
  if (r) { sel = null; apply(r); toast("Read the scene again"); }
}

// ---------------------------------------------------------------------------
// Wiring and start
// ---------------------------------------------------------------------------
document.querySelectorAll(".browser .tab").forEach(b => b.addEventListener("click", () => { tab = b.dataset.tab; renderBrowser(); showEntryPanel(); }));
$("search").addEventListener("input", () => { if (tab === "clips") loadClipList(clipFolder); else renderBrowser(); });
function showDock(which) {
  document.querySelectorAll(".dock .tab").forEach(b => b.setAttribute("aria-selected", b.dataset.dock === which));
  $("dockLog").hidden = which !== "log"; $("dockYaml").hidden = which !== "yaml"; $("dockProject").hidden = which !== "project";
  $("dockRepaint").hidden = which !== "repaint";
  if (which === "project") renderProject();
  if (which === "repaint") renderRepaint();
  $("logFilter").hidden = which !== "log";
  if (which === "log") { errorsUnseen = 0; showErrBadge(); }
}
document.querySelectorAll(".dock .tab").forEach(b => b.addEventListener("click", () => showDock(b.dataset.dock)));
document.querySelectorAll("#logFilter button").forEach(b => b.addEventListener("click", () => setLogFilter(b.dataset.kind)));
$("bPlay").addEventListener("click", () => setPlaying(!playing));
$("bStart").addEventListener("click", () => seek(loop ? B[loop[0] - 1] : 0));
$("loopClear").addEventListener("click", () => { loop = null; renderTimeline(); });
// zoom: while it moves, what is drawn is stretched; it is drawn again once it stops
$("zoom").addEventListener("input", e => {
  if (!S) { ppb = +e.target.value; return; }
  const sc = $("scroll"), x = tToX(now()) - sc.scrollLeft;
  zoomTo(+e.target.value, now(), x > 0 && x < sc.clientWidth - HEAD_W ? x : (sc.clientWidth - HEAD_W) / 2);
});
$("zoom").addEventListener("change", commitZoom);
$("zoomOut").addEventListener("click", () => { if (S) { zoomTo(Math.round(ppb * 0.8), xToT($("scroll").scrollLeft), 0); commitZoom(); } });
$("zoomIn").addEventListener("click", () => { if (S) { zoomTo(Math.round(ppb * 1.25), xToT($("scroll").scrollLeft), 0); commitZoom(); } });
$("zoomAll").addEventListener("click", () => { if (S) zoomAll(); });
$("rowsLess").addEventListener("click", () => setRows(Math.round(rowH / 1.2)));
$("rowsMore").addEventListener("click", () => setRows(Math.round(rowH * 1.2)));
$("resSel").addEventListener("change", async e => {
  let want = e.target.value;
  if (want === "custom") {
    const v = await dialog("The video's size", "Width x height, both even (the encoder needs it). It is written in the project's render.resolution.", [{ label: "Set", value: "ok", primary: true }, { label: "Cancel", value: null }], S.project.resolution);
    want = v && v.value ? v.text : null;
  }
  if (!want) { showResolution(); return; }
  const got = await ask("set_resolution", want);
  if (got && !got.error) { S.project = { ...S.project, ...got }; setSize(got.resolution); canvas.style.aspectRatio = `${W} / ${H}`; streamEdited(); toast(`The video: ${got.resolution}`); }
  showResolution();
});
function showResolution() {
  const sel2 = $("resSel"), res = S ? S.project.resolution : "";
  const groups2 = (S ? S.project.resolutions : []).map(g => h("optgroup", { label: g.shape }, g.sizes.map(z => h("option", { value: z, text: z, selected: z === res }))));
  const known = S && S.project.resolutions.some(g => g.sizes.includes(res));
  sel2.replaceChildren(...(known || !res ? [] : [h("option", { value: res, text: `${res} (typed)`, selected: true })]), ...groups2, h("option", { value: "custom", text: "Type a size…" }));
  sel2.disabled = !S;
}
$("bUndo").addEventListener("click", () => edit("undo"));
$("bRedo").addEventListener("click", () => edit("redo"));
$("bSave").addEventListener("click", save);
$("bSaveMenu").addEventListener("click", () => menu($("bSaveMenu"), [
  { label: "Save as…", on: saveAs, disabled: !S }, { label: "Reload from disk", hint: "the undo history goes", on: reloadScene, disabled: !S }]));
$("bProject").addEventListener("click", projectMenu);
$("bRender").addEventListener("click", startRender);
$("exactN").value = String(exactN);
$("exactN").addEventListener("change", e => { exactN = +e.target.value; try { localStorage.setItem("arc.exactN", String(exactN)); } catch (err) { /* no storage */ } updateLoopChip(); });
$("bRenderVideo").addEventListener("click", openRenderDialog);
$("bStop").addEventListener("click", () => call("stop_render"));
$("viewLive").addEventListener("click", () => setView("live"));
$("viewSketch").addEventListener("click", () => setView("sketch"));
$("viewEngine").addEventListener("click", () => { if (engine) setView("engine"); });
$("loadChip").addEventListener("click", () => showLoad($("loadPanel").hidden));
$("lpClose").addEventListener("click", () => showLoad(false));
$("lpCancel").addEventListener("click", () => ask("cancel_load"));
$("startClose").addEventListener("click", hideStart);
$("bOpenFile").addEventListener("click", openFileDialog);
$("bNewFromAudio").addEventListener("click", newFromAudio);
$("pathForm").addEventListener("submit", e => { e.preventDefault(); const p = $("pathInput").value.trim(); if (p) openProject(p); });
document.addEventListener("keydown", onKeys);
function onKeys(e) {
  if (!$("modal").hidden) return;
  const tag = (e.target.tagName || "").toLowerCase(), typing = tag === "input" || tag === "select" || tag === "textarea";
  const mod = e.ctrlKey || e.metaKey, key = e.key.toLowerCase();
  if (mod && key === "s") { e.preventDefault(); if (e.shiftKey) saveAs(); else save(); return; }
  if (mod && key === "o") { e.preventDefault(); openFileDialog(); return; }
  if (e.key === "Escape") { const open = $q(".ctx"); closeCtx(); if (!$("start").hidden && S) hideStart(); else if (!open && !typing && subStack.length) popSub(); return; }
  if (typing || !$("start").hidden) return;
  if (mod && key === "z") { e.preventDefault(); edit(e.shiftKey ? "redo" : "undo"); return; }
  if (mod && key === "y") { e.preventDefault(); edit("redo"); return; }
  if (e.key === " " && tag !== "button") { e.preventDefault(); if (e.shiftKey) startRender(); else setPlaying(!playing); }
  else if (e.key === "Home") seek(loop ? B[loop[0] - 1] : 0);
  else if (subStack.length && (e.key === "Delete" || e.key === "Backspace" || (mod && key === "d"))) subBusy();
  else if ((e.key === "Delete" || e.key === "Backspace") && sel) { e.preventDefault(); deleteSelected(); }
  else if (mod && key === "d" && sel && sel.type === "member") { e.preventDefault(); const m = memberById(sel.id); if (m) duplicateSpan(m, sel.k); }
}
window.addEventListener("beforeunload", e => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });
window.addEventListener("pagehide", () => { for (const w of Object.values(outWins)) try { w.close(); } catch (err) { /* gone */ } });

// ---------------------------------------------------------------------------
// The window's look: a theme (Classic as it was; Fine: denser) and the right column's width,
// both kept by Python for the next time (the page's own storage is not: pywebview runs private)
// ---------------------------------------------------------------------------
let ui = { theme: "", right_w: 340 };
const RIGHT_W = [280, 340];                     // the right column: its least, its first width
function setTheme(name, keep = true) {
  const old = ui.theme;
  ui.theme = name === "fine" ? "fine" : "";
  for (const d of [document, ...outDocs()]) {
    if (ui.theme) d.documentElement.dataset.theme = ui.theme; else delete d.documentElement.dataset.theme;
  }
  $("themeSel").value = ui.theme;
  const rows = { "": 30, fine: 24 };            // the rows' height follows, unless set by hand
  if (rowH === rows[old]) rowH = rows[ui.theme];
  if (keep) call("set_ui_pref", "theme", ui.theme).catch(() => { });
  if (S) later(renderTimeline);
}
function setRightW(px, keep = true) {
  const work = document.querySelector(".work").getBoundingClientRect();
  const most = Math.max(RIGHT_W[0], Math.min(900, work.width - 250 - 420));   // the Arrange keeps 420 px
  ui.right_w = Math.round(Math.max(RIGHT_W[0], Math.min(most, px)));
  document.documentElement.style.setProperty("--right-w", ui.right_w + "px");
  if (keep) call("set_ui_pref", "right_w", ui.right_w).catch(() => { });
}
$("themeSel").addEventListener("change", e => setTheme(e.target.value));
{
  const vs = $("vsplit");
  vs.addEventListener("pointerdown", e => {
    e.preventDefault(); vs.setPointerCapture(e.pointerId); vs.classList.add("on");
    const right = document.querySelector(".work").getBoundingClientRect().right;
    const move = ev => setRightW(right - ev.clientX, false);
    const up = () => { vs.classList.remove("on"); vs.removeEventListener("pointermove", move); vs.removeEventListener("pointerup", up); setRightW(ui.right_w); if (S) renderTimeline(); };
    vs.addEventListener("pointermove", move); vs.addEventListener("pointerup", up);
  });
  vs.addEventListener("dblclick", () => { setRightW(RIGHT_W[1]); if (S) renderTimeline(); });
}

// ---------------------------------------------------------------------------
// A configuration panel (the plugin, the link · FX area) in a window of its own: the panel
// itself moves there, the page's script still driving it, and comes back with ⧉ again, the
// toolbar's chip, or when that window closes. The others take its place meanwhile.
// ---------------------------------------------------------------------------
const PANELS = { inspector: "Plugin", subpanel: "Link · FX" };
function detachPanel(cls) {
  if (outWins[cls]) { attachPanel(cls); return; }
  const sec = document.querySelector(`.work > .${cls}`); if (!sec) return;
  const w = window.open("", `arc-${cls}`, cls === "inspector" ? "width=520,height=900" : "width=560,height=760");
  if (!w) { toast("A window of its own was not allowed here", true); return; }
  const d = w.document;
  d.open(); d.write("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title></title></head><body class='detached'></body></html>"); d.close();
  d.title = `${PANELS[cls]} · ARC Studio`;
  for (const n of document.querySelectorAll("link[rel=stylesheet]")) d.head.append(h("link", { rel: "stylesheet", href: n.href }));
  if (ui.theme) d.documentElement.dataset.theme = ui.theme;
  d.body.append(d.adoptNode(sec));
  outWins[cls] = w; outSecs[cls] = sec;
  wireDoc(d);
  d.addEventListener("keydown", onKeys);
  w.addEventListener("beforeunload", () => attachPanel(cls, false));
  document.querySelector(".work").classList.add(`${cls}-out`);
  showOutChips();
  if (S) later(renderTimeline);
}
function attachPanel(cls, close = true) {
  const w = outWins[cls]; if (!w) return;
  delete outWins[cls];
  const sec = outSecs[cls]; delete outSecs[cls];
  if (sec) document.querySelector(".work").append(document.adoptNode(sec));
  document.querySelector(".work").classList.remove(`${cls}-out`);
  if (menuDoc !== document) menuDoc = document;
  if (close) try { w.close(); } catch (e) { /* gone */ }
  showOutChips();
  if (S) later(renderTimeline);
}
function showOutChips() {
  fill($("outChips"), ...Object.keys(outWins).map(cls => h("button", { class: "chip", title: `${PANELS[cls]} is in a window of its own: bring it back`,
    onclick: () => attachPanel(cls) }, `${PANELS[cls]} ⧉ back`)));
  for (const b of [document, ...outDocs()].flatMap(d => [...d.querySelectorAll(".ib.detach")])) {
    const out = !!outWins[b.dataset.panel];
    b.textContent = out ? "⇲" : "⧉"; b.title = out ? "Back into the window" : "In a window of its own (and back)";
  }
}
for (const b of document.querySelectorAll(".ib.detach")) b.addEventListener("click", () => detachPanel(b.dataset.panel));
setInterval(() => { for (const [cls, w] of Object.entries(outWins)) if (w.closed) attachPanel(cls, false); }, 500);   // its window closed
for (const kind of ["keydown", "keyup"]) document.addEventListener(kind, e => {
  if (e.key !== "Shift" || drag) return;
  shiftHeld = kind === "keydown";
  if (hoverAt && hoverAt.el.isConnected) {
    const end = edgeOf(hoverAt.el, hoverAt.x, shiftHeld);
    hoverAt.el.style.cursor = hoverAt.el.classList.contains("cued") ? "default" : end ? "ew-resize" : "grab";
    showEdge(hoverAt.el, end);
  }
});

// the toolbar's height (it wraps on a narrow window): the browser stays in view under it
new ResizeObserver(() => document.documentElement.style.setProperty("--tbh", $("bProject").closest(".toolbar").offsetHeight + "px"))
  .observe(document.querySelector(".toolbar"));
let booted = false;
async function boot() {
  if (booted) return;
  booted = true;
  inWebview = !!(window.pywebview && window.pywebview.api && window.pywebview.api.boot);
  if (!inWebview) pollEvents();
  showDock("log"); renderInspector(); showSaveState();
  try {
    initGL();
    const got = await call("boot");
    if (got.error) throw new Error(got.error);
    boot0 = got;
    if (got.ui) { setTheme(got.ui.theme || "", false); if (got.ui.right_w) setRightW(got.ui.right_w, false); }
    log("sys", inWebview ? "pywebview: the page calls Python through js_api" : "browser: the page calls Python through /api");
    if (got.first) await openProject(got.first);
    else if (!loadGen) await showStart();         // (a project opened meanwhile stays)
  } catch (e) { log("err", `could not start: ${e.message}`); toast(`Could not start: ${e.message}`, true); console.error(e); }
  requestAnimationFrame(tick);
}
window.addEventListener("pywebviewready", boot);
setTimeout(() => { if (!window.pywebview) boot(); }, 700);
