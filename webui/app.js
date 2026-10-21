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
let steps = [];             // the loading steps, as Python tells them

const $ = id => document.getElementById(id);
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
const fmt = v => { if (!isFinite(v)) return "0"; const s = Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3); return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s; };
const fmtTime = t => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
const idOf = ref => JSON.stringify(ref);
const stem = name => String(name).replace(/\.[^.]+$/, "");
let toastTimer = 0;
function toast(msg, bad = false) {
  let el = document.querySelector(".toast");
  if (!el) { el = h("div", { class: "toast", role: "status" }); document.body.append(el); }
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
  montage: segs => { montage = segs || []; later(() => { renderTimeline(); if (sel && sel.type === "clip") renderInspector(); }); },
  knobs: () => { knobs.clear(); if (sel && sel.type === "member") renderInspector(); },
  covers: got => { for (const [clip, cover] of Object.entries(got || {})) thumbs.set(clip, cover); scheduleThumbRedraw(); },
  song: e => { if (S && e.url && e.url !== audioUrl) loadAudio(e.url); },
  engine: e => onEngine(e),
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
function now() {
  if (!playing) return pausedAt;
  if (audioBuf && actx) return Math.max(startPos, startPos + (actx.currentTime - startCtx));
  return startPos + (performance.now() / 1000 - startPerf);
}
function startSource() {
  startPos = pausedAt; startPerf = performance.now() / 1000;
  if (audioBuf && actx) {
    src = actx.createBufferSource(); src.buffer = audioBuf; src.connect(actx.destination);
    startCtx = actx.currentTime + 0.03;
    src.start(startCtx, Math.min(pausedAt, audioBuf.duration - 0.01));
  }
}
function stopSource() { if (src) { try { src.stop(); } catch (e) { } src.disconnect(); src = null; } }
async function setPlaying(on) {
  if (on === playing) return;
  if (on) {
    if (actx && actx.state === "suspended") await actx.resume();
    playing = true; startSource();
  } else { pausedAt = now(); playing = false; stopSource(); }
  $("playIcon").innerHTML = playing ? '<path d="M3 2h3.5v12H3zM9.5 2H13v12H9.5z"/>' : '<path d="M4 2l10 6-10 6z"/>';
  $("bPlay").classList.toggle("on", playing);
}
function seek(t) {
  t = Math.max(0, Math.min((S ? S.song.duration : 1) - 0.05, t));
  if (playing) { stopSource(); pausedAt = t; startSource(); } else pausedAt = t;
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
function program(code) {
  const { body, defaults } = adapt(code);
  const p = gl.createProgram();
  gl.attachShader(p, vsShader || (vsShader = compile(gl.VERTEX_SHADER, VS)));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, HEAD + body + FOOT));
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
  TEX = {}; for (const k2 of ["a", "b", "c", "d", "e"]) TEX[k2] = makeTarget();
}
function initGL() {
  if (!gl) { log("err", "no WebGL2 here: the preview is off, the Arrange still works"); return; }
  gl.bindVertexArray(gl.createVertexArray());
  blackTex = mediaTex();
  for (const [k, code] of Object.entries(OWN)) PROGS[k] = program(code);
  setSize("854x480");
}
function loc(P, n) { if (!(n in P.locs)) P.locs[n] = gl.getUniformLocation(P.prog, n); return P.locs[n]; }
function run(P, uniforms, inputs, target, t) {
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
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}
function clearTarget(T) { gl.bindFramebuffer(gl.FRAMEBUFFER, T.fbo); gl.viewport(0, 0, W, H); gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT); }

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
    if (vids.size > 14) {           // the longest unused goes
      const old = [...vids.entries()].filter(([u, x]) => x !== v).sort((a, b) => a[1].last - b[1].last)[0];
      if (old) { old[1].el.removeAttribute("src"); old[1].el.load(); gl.deleteTexture(old[1].tex); vids.delete(old[0]); }
    }
  }
  v.last = frameNo;
  return v;
}
function syncVid(v, want) {
  const el = v.el;
  if (el.readyState >= 1) {
    const d = el.duration || 0;
    if (d > 0) want = ((want % d) + d) % d;
    if (playing) {
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
function drawVideo(url, want, target, t, cover = true) {
  const v = vid(url);
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
  if (seg && seg.url) { drawVideo(seg.url, t - seg.t0, cur, t, fit); chips.push(["clip", `clips · ${seg.name}`]); }
  else { clearTarget(cur); if (seg) chips.push(["clip", "clips · nothing"]); }
  const lane = id => members.filter(m => m.lane === id && m.enabled && activeSpan(m, bar));
  // BG: its members over the montage
  for (const m of lane("bg")) {
    const sp = activeSpan(m, bar);
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
      if (m.live && m.kind !== "clip" && shaderLayer(m, t, fc.tex, TEX.e)) {
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
  view = v;
  $("viewLive").setAttribute("aria-pressed", v === "live");
  $("viewEngine").setAttribute("aria-pressed", v === "engine");
  $("engineVideo").hidden = v !== "engine";
}
function onEngine(e) {
  if (e.event === "progress") {
    $("progBar").style.width = (100 * e.frame / e.of) + "%";
    $("engineText").textContent = `rendering ${e.frame}/${e.of}`;
    return;
  }
  rendering = false;
  $("bRender").disabled = false; $("bStop").hidden = true; $("prog").hidden = true; $("engineDot").classList.remove("busy");
  if (e.event === "done") {
    $("engineText").textContent = "engine idle";
    const span = renderSpan;
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
  const bar = barAt(now());
  renderSpan = loop ? [loop[0], loop[1]] : [bar, Math.min(NB(), bar + 3)];
  const r = await call("render", renderSpan[0], renderSpan[1]);
  if (r.error) { toast(r.error); return; }
  rendering = true;
  $("bRender").disabled = true; $("bStop").hidden = false; $("prog").hidden = false; $("progBar").style.width = "0";
  $("engineDot").classList.remove("bad"); $("engineDot").classList.add("busy"); $("engineText").textContent = "starting the engine";
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
const LANES = [
  { id: "audio", name: "Audio", kind: "audio", h: 40, c: "var(--muted)" },
  { id: "midi", name: "MIDI", kind: "midi", c: "var(--accent)" },
  { id: "master", name: "Master", sub: "over everything", kind: "track", rowH: 22, c: "var(--lane-fx)", takes: ["fx", "gen"] },
  { id: "fg", name: "FG", sub: "characters, on their own", kind: "track", rowH: 24, c: "var(--lane-fg)", takes: ["gen", "clip"] },
  { id: "fgfx", name: "FG FX", sub: "on the whole FG", kind: "track", rowH: 22, c: "var(--lane-fx)", takes: ["fx"] },
  { id: "bg", name: "BG", sub: "over the montage", kind: "track", rowH: 24, c: "var(--lane-bg)", takes: ["gen", "clip"] },
  { id: "bgfx", name: "BG FX", sub: "on the whole BG", kind: "track", rowH: 22, c: "var(--lane-fx)", takes: ["fx"] },
  { id: "clips", name: "Clips", sub: "the montage, a cut a bar", kind: "clips", h: 30, c: "var(--lane-clip)" },
];
const RULER_H = 30;
let layout = null;
function packLane(id) {           // core.arrangement.pack: the topmost first, a row the user chose kept
  const ms = members.filter(m => m.lane === id), rows = [];
  const spans = m => spansOf(m).map(s => [s.first, s.last]);
  const free = (r, sp) => r >= rows.length || !sp.some(([a, b]) => rows[r].some(([c, d]) => a <= d && c <= b));
  const put = (m, r, sp) => { while (rows.length <= r) rows.push([]); m._row = r; rows[r].push(...sp); };
  const placed = new Set();
  for (const m of [...ms].reverse()) if (Number.isInteger(m.row) && m.row >= 0 && free(m.row, spans(m))) { put(m, m.row, spans(m)); placed.add(m); }
  for (const m of [...ms].reverse()) if (!placed.has(m)) { const sp = spans(m); let r = 0; while (!free(r, sp)) r++; put(m, r, sp); }
  const used = [...new Set(ms.map(m => m._row))].sort((a, b) => a - b);
  for (const m of ms) m._row = used.indexOf(m._row);
  return used.length;
}
function computeLayout() {
  let y = RULER_H; const L = [];
  for (const d of LANES) {
    let hh = d.h;
    if (d.kind === "midi") hh = 8 + 9 * Math.max(1, (S ? S.tracks.length : 1));
    if (d.kind === "track") hh = Math.max(1, packLane(d.id)) * d.rowH + 6;
    L.push({ ...d, top: y, h: hh }); y += hh;
  }
  return { lanes: L, height: y };
}
function canvasFor(width, height) {
  const d = devicePixelRatio || 1;
  return h("canvas", { width: Math.round(width * d), height: Math.round(height * d), style: `width:${width}px; height:${height}px` });
}
function renderTimeline() {
  if (!S) { $("heads").replaceChildren(); $("inner").replaceChildren(); $("inner").style.height = "120px"; return; }
  layout = computeLayout();
  const heads = $("heads"), inner = $("inner"), width = NB() * ppb;
  heads.replaceChildren(h("div", { class: "rulerhead", style: `height:${RULER_H}px` }, `${S.song.meter} · ${Math.round(S.song.bpm)} BPM`));
  for (const l of layout.lanes) {
    const head = h("div", { class: "head" + (l.kind === "midi" ? " midi" : ""), style: `height:${l.h}px; --c:${l.c}` }, h("b", { text: l.name }));
    const sub = l.kind === "audio" ? S.project.audio : l.sub;
    if (l.kind === "midi") for (const tr of S.tracks) head.append(h("div", { class: "trk", text: tr.name, title: `${tr.name}: ${tr.times.length} notes` }));
    else if (sub) head.append(h("small", { text: sub, title: sub }));
    heads.append(head);
  }
  inner.style.width = width + "px"; inner.style.height = layout.height + "px";
  inner.replaceChildren();
  // ruler: bar numbers, beats
  const ruler = h("div", { class: "ruler", style: `height:${RULER_H}px` });
  const rc = canvasFor(width, RULER_H); ruler.append(rc); drawRuler(rc);
  inner.append(ruler);
  for (const l of layout.lanes) {
    const lane = h("div", { class: "lane", dataset: { lane: l.id }, style: `top:${l.top}px; height:${l.h}px; background-image: repeating-linear-gradient(to right, var(--line) 0 1px, transparent 1px ${ppb}px);` });
    if (l.kind === "audio") { const c = canvasFor(width, l.h); lane.append(c); drawWave(c); }
    if (l.kind === "midi") { const c = canvasFor(width, l.h); lane.append(c); drawMidi(c); }
    if (l.kind === "clips") drawCells(lane);
    if (l.takes || l.kind === "clips") wireLaneDrop(lane, l);
    inner.append(lane);
  }
  for (const m of members) {
    const l = layout.lanes.find(x => x.id === m.lane);
    if (!l) continue;
    spansOf(m).forEach((s, k) => inner.append(blockEl(m, s, m.always ? -1 : k, l)));
  }
  if (loop) inner.append(h("div", { class: "loopband", id: "loopband", style: `left:${(loop[0] - 1) * ppb}px; width:${(loop[1] - loop[0] + 1) * ppb}px` }));
  inner.append(h("div", { class: "playhead", id: "playhead" }));
  updateLoopChip();
  $("bUndo").disabled = !undoN; $("bRedo").disabled = !redoN;
}
function drawRuler(cv) {
  const g = cv.getContext("2d"), d = devicePixelRatio || 1;
  g.scale(d, d);
  g.font = "10px 'IBM Plex Mono', Consolas, monospace"; g.textBaseline = "top";
  const every = ppb < 18 ? 8 : ppb < 26 ? 4 : ppb < 40 ? 2 : 1;
  for (let b = 1; b <= NB(); b++) {
    const x = (b - 1) * ppb;
    g.fillStyle = "#3b4352"; g.fillRect(x, RULER_H - 10, 1, 10);
    const n = S.song.bar_beats[b - 1] || S.song.bar_beats[S.song.bar_beats.length - 1] || 4;
    g.fillStyle = "#2b313d";
    for (let k = 1; k < n; k++) g.fillRect(x + k * ppb / n, RULER_H - 5, 1, 5);
    if ((b - 1) % every === 0) { g.fillStyle = "#9097a7"; g.fillText(String(b), x + 3, 4); }
    if (b > 1 && n !== (S.song.bar_beats[b - 2] || n)) { g.fillStyle = "#f0a04b"; g.fillText(`${n}/4`, x + 3, 15); }
  }
  if (S.song.bar_beats[0] && S.song.bar_beats[0] !== (S.song.bar_beats[1] || S.song.bar_beats[0])) { g.fillStyle = "#f0a04b"; g.fillText(`${S.song.bar_beats[0]}/4`, 3, 15); }
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
function clipAtBar(b) {
  const s = segAt(B[b - 1] + 0.02);
  return s ? s : null;
}
const thumbs = new Map();     // clip url -> data URL
function drawCells(lane) {
  for (let b = 1; b <= NB(); b++) {
    const s = clipAtBar(b), pinned = String(b) in pins, none = pinned && pins[String(b)] === "none";
    const th = s && s.url ? thumbs.get(s.url) : null;
    const cls = "cell" + (none || (montage.length && !s) ? " none" : "") + (pinned ? " pin" : "") + (sel && sel.type === "clip" && sel.bar === b ? " sel" : "");
    lane.append(h("div", { class: cls, dataset: { bar: b }, style: `left:${(b - 1) * ppb + 1}px; width:${ppb - 2}px;` + (th ? `background-image:url(${th})` : ""),
      title: `Bar ${b}: ${none ? "nothing" : s ? s.name : "…"}${pinned ? " (pinned)" : ""}`, text: ppb >= 44 && s && !none ? s.name.slice(0, Math.floor(ppb / 6)) : "" }));
  }
}
function blockEl(m, s, k, l) {
  const top = l.top + 3 + (m._row || 0) * l.rowH, height = l.rowH - 4;
  const left = (s.first - 1) * ppb + 1, width = (s.last - s.first + 1) * ppb - 2;
  const live = m.live && (m.kind === "clip" || (shaders.get(m.key) || {}).status !== "bad");
  const laneCls = m.kind === "fx" ? "fx" : m.kind === "clip" ? `clip ${m.lane}` : m.lane === "master" ? "fx" : m.lane;
  const cls = ["block", laneCls, live ? "" : "engine", m.cued ? "cued" : "", m.always ? "always" : "", m.enabled ? "" : "off", s.missing ? "missing" : "",
    sel && sel.type === "member" && sel.id === idOf(m.ref) && (sel.k === k || m.cued) ? "sel" : ""].filter(Boolean).join(" ");
  const linked = m.params.filter(p => p.knob != null).length;
  return h("div", { class: cls, dataset: { id: idOf(m.ref), k }, style: `left:${left}px; width:${Math.max(4, width)}px; top:${top}px; height:${height}px`,
    title: `${m.label} · bars ${s.first}–${s.last}${m.cued ? " · cued in REAPER" : ""}${m.always ? " · the whole song" : ""}${live ? "" : " · at the render"}${s.missing ? ` · MISSING: ${s.file || m.file}` : ""}` },
    h("span", { class: "nm", text: m.label }), linked ? h("span", { class: "sg", text: `${linked} knob${linked > 1 ? "s" : ""}` }) : null,
    m.own_fx ? h("span", { class: "sg", text: `+${m.own_fx} fx` }) : null);
}
const memberById = id => members.find(m => idOf(m.ref) === id);
function laneAtY(clientY) {
  const r = $("inner").getBoundingClientRect(), y = clientY - r.top;
  return layout.lanes.find(l => y >= l.top && y < l.top + l.h) || null;
}
function barAtClientX(clientX) { return xToBar(clientX - $("inner").getBoundingClientRect().left); }
function canGo(m, laneId) {
  if (m.cued) return false;
  const l = LANES.find(x => x.id === laneId);
  if (!l || !l.takes) return false;
  if (m.kind === "clip") return laneId === m.lane;      // a clip stays a background or a character
  return l.takes.includes(m.kind);
}

// edits: Python answers with the lanes. They go one after another, in the order made (one
// made while another is on its way waits for it).
let editChain = Promise.resolve();
function edit(name, ...args) {
  const run = async () => {
    try {
      const st = await call(name, ...args);
      if (st && st.error) { toast(st.error, true); return st; }      // Python logged it, with where
      apply(st); return st;
    } catch (e) { toast(e.message, true); log("err", `${name}: ${e.message}`); }
  };
  const p = editChain.then(run);
  editChain = p.catch(() => { });
  return p;
}
function apply(st) {
  if (!st || !st.lanes) return;
  members = st.lanes; pins = st.pins || {}; undoN = st.undo; redoN = st.redo;
  if (st.montage) montage = st.montage;
  if ("dirty" in st) { dirty = st.dirty; savedAt = st.saved_at || ""; }
  if (sel && sel.type === "member" && !memberById(sel.id)) sel = null;
  for (const m of members) if (m.live && m.kind !== "clip") shaderFor(m.key);
  renderTimeline(); renderInspector(); showSaveState();
}
function showSaveState(text, bad = false) {
  const el = $("saveState");
  el.textContent = text || (!S ? "" : dirty ? "unsaved changes" : savedAt ? `saved at ${savedAt}` : "");
  el.classList.toggle("dirty", !text && dirty); el.classList.toggle("bad", bad);
  $("bSave").textContent = dirty ? "Save *" : "Save";
  el.title = S && S.project.scene ? S.project.scene : "";
}

// pointer work
let drag = null, rsel = null;
const inner = $("inner");
inner.addEventListener("pointerdown", e => {
  if (e.button !== 0 || !S) return;
  closeCtx();
  const bEl = e.target.closest(".block");
  if (bEl) {
    const m = memberById(bEl.dataset.id), k = +bEl.dataset.k;
    const r = bEl.getBoundingClientRect(), ox = e.clientX - r.left, edge = Math.min(8, r.width / 3);
    select({ type: "member", id: bEl.dataset.id, k }, false);
    if (m.cued) return;
    const s = spansOf(m)[Math.max(0, k)];
    drag = { m, k, el: bEl, mode: ox < edge ? "l" : ox > r.width - edge ? "r" : "move", x0: e.clientX, first: s.first, last: s.last, f0: s.first, l0: s.last, lane: m.lane, moved: false };
    try { bEl.setPointerCapture(e.pointerId); } catch (err) { /* a pointer that is not down */ }
    return;
  }
  const cell = e.target.closest(".cell");
  if (cell) { select({ type: "clip", bar: +cell.dataset.bar }); return; }
  rsel = { x0: e.clientX, bar0: barAtClientX(e.clientX), moved: false, onRuler: !!e.target.closest(".ruler") };
  try { inner.setPointerCapture(e.pointerId); } catch (err) { /* a pointer that is not down */ }
});
inner.addEventListener("pointermove", e => {
  if (drag) {
    const dB = Math.round((e.clientX - drag.x0) / ppb), len = drag.l0 - drag.f0;
    if (drag.mode === "move") { drag.first = Math.max(1, Math.min(NB() - len, drag.f0 + dB)); drag.last = drag.first + len; }
    else if (drag.mode === "l") drag.first = Math.max(1, Math.min(drag.l0, drag.f0 + dB));
    else drag.last = Math.max(drag.f0, Math.min(NB(), drag.l0 + dB));
    let l = layout.lanes.find(x => x.id === drag.lane);
    if (drag.mode === "move") { const over = laneAtY(e.clientY); if (over && canGo(drag.m, over.id)) { drag.lane = over.id; l = over; } }
    drag.moved = drag.moved || dB !== 0 || drag.lane !== drag.m.lane;
    const top = l.top + 3 + (drag.lane === drag.m.lane ? (drag.m._row || 0) * l.rowH : 0);
    Object.assign(drag.el.style, { left: (drag.first - 1) * ppb + 1 + "px", width: (drag.last - drag.first + 1) * ppb - 2 + "px", top: top + "px", cursor: drag.mode === "move" ? "grabbing" : "ew-resize" });
    return;
  }
  if (rsel) {
    if (Math.abs(e.clientX - rsel.x0) > 4) rsel.moved = true;
    if (rsel.moved) { const b = barAtClientX(e.clientX); loop = [Math.min(rsel.bar0, b), Math.max(rsel.bar0, b)]; showLoop(); }
    return;
  }
  const bEl = e.target.closest(".block");
  if (bEl && !bEl.classList.contains("cued")) {
    const r = bEl.getBoundingClientRect(), ox = e.clientX - r.left, edge = Math.min(8, r.width / 3);
    bEl.style.cursor = ox < edge || ox > r.width - edge ? "ew-resize" : "grab";
  }
});
inner.addEventListener("pointerup", async e => {
  if (drag) {
    const d = drag; drag = null;
    if (!d.moved) { flushAfterDrag(); return; }
    const spans = d.m.always ? [] : d.m.spans.map(s => ({ ...s }));
    const ns = { ...(d.k >= 0 ? spans[d.k] : {}), first: d.first, last: d.last };
    if (d.k >= 0) spans[d.k] = ns; else spans.push(ns);
    const done = d.lane !== d.m.lane ? (sel = null, edit("relane", d.m.ref, d.lane, spans)) : edit("spans", d.m.ref, spans);
    flushAfterDrag();
    await done;
    return;
  }
  if (rsel) {
    if (!rsel.moved) {
      seek(xToT(e.clientX - inner.getBoundingClientRect().left));
      if (!rsel.onRuler) { sel = null; renderTimeline(); renderInspector(); }
    } else { seek(B[loop[0] - 1]); log("sys", `bars ${loop[0]}–${loop[1]} chosen`); }
    rsel = null;
    flushAfterDrag();
  }
});
function flushAfterDrag() { const fns = pendingAfterDrag; pendingAfterDrag = []; for (const fn of fns) fn(); }
inner.addEventListener("pointercancel", () => { if (drag) renderTimeline(); drag = null; rsel = null; flushAfterDrag(); });
inner.addEventListener("dblclick", e => { if (e.target.closest(".ruler")) { loop = null; renderTimeline(); } });
inner.addEventListener("contextmenu", e => {
  const bEl = e.target.closest(".block");
  if (!bEl) return;
  e.preventDefault();
  const m = memberById(bEl.dataset.id), k = +bEl.dataset.k;
  select({ type: "member", id: bEl.dataset.id, k });
  openCtx(e.clientX, e.clientY, m, k);
});
function select(s, redraw = true) {
  sel = s;
  if (redraw) renderTimeline();
  else {                       // in the middle of a drag: the blocks stay the same elements
    document.querySelectorAll(".block.sel, .cell.sel").forEach(x => x.classList.remove("sel"));
    if (s.type === "member") inner.querySelectorAll(".block").forEach(x => { if (x.dataset.id === s.id && (+x.dataset.k === s.k || memberById(s.id).cued)) x.classList.add("sel"); });
  }
  renderInspector();
}
function showLoop() {
  let band = $("loopband");
  if (!band) { band = h("div", { class: "loopband", id: "loopband" }); inner.insertBefore(band, $("playhead")); }
  band.style.left = (loop[0] - 1) * ppb + "px"; band.style.width = (loop[1] - loop[0] + 1) * ppb + "px";
  updateLoopChip();
}
function updateLoopChip() {
  $("loopChip").hidden = !loop;
  if (loop) $("loopText").textContent = loop[0] === loop[1] ? `bar ${loop[0]}` : `bars ${loop[0]}–${loop[1]}`;
  $("bRender").textContent = loop ? `Render bars ${loop[0]}–${loop[1]}` : "Render 4 bars here";
}
function closeCtx() { document.querySelectorAll(".ctx").forEach(x => x.remove()); }
function openCtx(x, y, m, k) {
  closeCtx();
  if (m.cued) return;
  const other = m.kind === "gen" ? (m.lane === "bg" ? "fg" : "bg") : null;
  const menu = h("div", { class: "ctx", role: "menu" },
    h("button", { role: "menuitem", onclick: () => { closeCtx(); duplicateSpan(m, k); } }, "Again right after"),
    other ? h("button", { role: "menuitem", onclick: () => { closeCtx(); edit("relane", m.ref, other, null); } }, `Move to ${other.toUpperCase()}`) : null,
    h("button", { role: "menuitem", onclick: () => { closeCtx(); edit("set_layer", m.ref, "enabled", !m.enabled); } }, m.enabled ? "Turn off" : "Turn on"),
    h("hr"),
    !m.always && m.spans.length > 1 ? h("button", { role: "menuitem", onclick: () => { closeCtx(); deleteSpan(m, k); } }, `Delete bars ${m.spans[k].first}–${m.spans[k].last}`) : null,
    h("button", { role: "menuitem", style: "color:var(--danger)", onclick: () => { closeCtx(); sel = null; edit("remove", m.ref); } }, `Delete ${m.label}`));
  document.body.append(menu);
  const r = menu.getBoundingClientRect();
  menu.style.left = Math.min(x, innerWidth - r.width - 8) + "px"; menu.style.top = Math.min(y, innerHeight - r.height - 8) + "px";
}
document.addEventListener("pointerdown", e => { if (!e.target.closest(".ctx")) closeCtx(); });
function duplicateSpan(m, k) {
  if (m.cued) return;
  const spans = m.always ? [] : m.spans.map(s => ({ ...s }));
  const s = spansOf(m)[Math.max(0, k)], len = s.last - s.first;
  const first = Math.min(NB() - len, s.last + 1);
  spans.push({ ...s, first, last: first + len });
  edit("spans", m.ref, spans);
}
function deleteSpan(m, k) {
  if (m.always || m.spans.length <= 1) { sel = null; return edit("remove", m.ref); }
  edit("spans", m.ref, m.spans.filter((_, i) => i !== k));
}

// drops from the browser
let dragItem = null;
function accepts(l, it) {
  if (!it) return false;
  if (it.type === "clip") return l.id === "bg" || l.id === "fg" || l.kind === "clips";
  return !!l.takes && l.takes.includes(it.kind);
}
function wireLaneDrop(el, l) {
  el.addEventListener("dragover", e => { if (accepts(l, dragItem)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; el.classList.add("drop-ok"); } });
  el.addEventListener("dragleave", () => el.classList.remove("drop-ok"));
  el.addEventListener("drop", e => {
    el.classList.remove("drop-ok");
    if (!accepts(l, dragItem)) return;
    e.preventDefault();
    const it = dragItem; dragItem = null;
    const bar = barAtClientX(e.clientX);
    putItem(it, l.id, bar);
  });
}
function putItem(it, laneId, bar) {
  if (it.type === "clip" && laneId === "clips") return edit("pin", bar, stem(it.name));
  const last = Math.min(NB(), bar + 3);
  return edit("add", laneId, it.type === "clip" ? it.path : it.key, bar, last).then(() => toast(`${it.label || it.name} in ${laneId.toUpperCase()}, bars ${bar}–${last}`));
}

// ---------------------------------------------------------------------------
// Browser
// ---------------------------------------------------------------------------
let tab = "gen";
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
  if (tab === "clips") {
    if (!S.clips.length) { list.append(h("div", { class: "empty-note", text: "The project has no clips folder." })); return; }
    for (const c of S.clips.filter(c => c.name.toLowerCase().includes(q))) {
      const it = { type: "clip", path: c.path, name: c.name, url: c.url };
      const th = thumbs.get(c.url);
      const row = h("div", { class: "bitem", title: `${c.name}: drag onto BG or FG for a block, onto Clips to pin a bar` },
        h("span", { class: "sw", style: th ? `background-image:url(${th})` : "" }), h("span", { class: "nm", text: c.name }),
        h("button", { class: "add", title: "Put 4 bars of it in the BG at the playhead", "aria-label": `Add ${c.name}`, onclick: () => putItem(it, "bg", barAt(now())) }, "+"));
      drag(row, it); list.append(row);
    }
    return;
  }
  const byCat = new Map();
  for (const c of S.catalog) if (c.kind === tab && `${c.label} ${c.cat} ${c.about}`.toLowerCase().includes(q)) {
    if (!byCat.has(c.cat)) byCat.set(c.cat, []); byCat.get(c.cat).push(c);
  }
  if (!byCat.size) list.append(h("div", { class: "empty-note", text: "Nothing matches." }));
  for (const [cat, items] of byCat) {
    list.append(h("div", { class: "bcat", text: cat }));
    for (const c of items) {
      const st = shaders.get(c.key);
      const bad = st && st.status === "bad";
      const it = { type: "cat", key: c.key, kind: c.kind, label: c.label };
      const row = h("div", { class: "bitem", title: (c.about || c.key) + (bad ? `\nGLSL ES: ${st.error}` : "") },
        h("span", { class: "nm", text: c.label }),
        h("span", { class: `badge ${c.live && !bad ? "live" : bad ? "bad" : "engine"}`, text: c.live && !bad ? "live" : bad ? "glsl" : "engine" }),
        h("button", { class: "add", title: "Put 4 bars of it at the playhead", "aria-label": `Add ${c.label}`, onclick: () => putItem(it, c.kind === "gen" ? "bg" : "bgfx", barAt(now())) }, "+"));
      drag(row, it); list.append(row);
    }
  }
}
// a clip's picture: its cover, made by Python (core.arrangement.cover) and told as the covers come
let thumbRedraw = 0;
function scheduleThumbRedraw() { clearTimeout(thumbRedraw); thumbRedraw = setTimeout(() => later(() => { renderTimeline(); if (tab === "clips") renderBrowser(); }), 300); }

// ---------------------------------------------------------------------------
// The plugin (what is chosen)
// ---------------------------------------------------------------------------
function renderInspector() {
  const box = $("insp"); box.replaceChildren();
  if (!sel) {
    $("inspTitle").textContent = "Plugin";
    box.append(h("div", { class: "about", text: "Pick a block in the Arrange. Every edit goes to Python (core.arrangement) and comes back as the lanes." }),
      h("div", { class: "keys" },
        h("span", { class: "kbd", text: "Space" }), "play or pause",
        h("span", { class: "kbd", text: "drag" }), "move a block; its ends resize it; up or down to another lane",
        h("span", { class: "kbd", text: "ruler" }), "click to go there, drag to choose bars",
        h("span", { class: "kbd", text: "Delete" }), "delete the chosen bars",
        h("span", { class: "kbd", text: "Ctrl+D" }), "the chosen bars again, right after",
        h("span", { class: "kbd", text: "Ctrl+Z" }), "undo (Ctrl+Y redo)",
        h("span", { class: "kbd", text: "Ctrl+S" }), "save the scene (Ctrl+Shift+S: save as)",
        h("span", { class: "kbd", text: "Ctrl+O" }), "open a project"));
    return;
  }
  if (sel.type === "clip") return renderClip(box);
  const m = memberById(sel.id);
  if (!m) { sel = null; return renderInspector(); }
  const c = catalogByKey.get(m.key);
  $("inspTitle").textContent = m.kind === "clip" ? (m.lane === "fg" ? "Character" : "Clip") : m.kind === "gen" ? "Generator" : "FX";
  const st = shaders.get(m.key);
  const live = m.live && !(st && st.status === "bad");
  box.append(h("div", { class: "ptitle" }, h("h3", { text: m.label }), h("div", { class: "src", text: m.file || (c ? c.file || c.key : m.key) }),
    h("div", { class: "row" }, h("span", { class: `badge ${live ? "live" : "engine"}`, text: live ? "live" : "engine" }),
      m.cued ? h("span", { class: "badge cue", text: "cued in REAPER" }) : null, m.always ? h("span", { class: "chip", text: "the whole song" }) : null,
      m.own_fx ? h("span", { class: "chip", text: `${m.own_fx} FX of its own` }) : null)));
  if (c && c.about) box.append(h("div", { class: "about", text: c.about }));
  if (st && st.status === "bad") box.append(h("div", { class: "about", style: "color:var(--danger)", text: `GLSL ES: ${st.error}` }));
  // where and when
  const lanes = LANES.filter(l => canGo(m, l.id));
  const fields = h("div", { class: "fields" },
    h("div", { class: "fld" }, h("label", { text: "Lane" }), h("select", { disabled: lanes.length < 2, onchange: e => edit("relane", m.ref, e.target.value, null) },
      (lanes.length ? lanes : LANES.filter(l => l.id === m.lane)).map(l => h("option", { value: l.id, selected: l.id === m.lane, text: l.name })))),
    h("div", { class: "fld" }, h("label", { text: "Plays" }), h("label", { class: "toggle" }, h("input", { type: "checkbox", checked: m.enabled, onchange: e => edit("set_layer", m.ref, "enabled", e.target.checked) }), m.enabled ? "on" : "off")));
  if (m.kind !== "fx") {
    fields.append(h("div", { class: "fld" }, h("label", { text: "Blend" }), h("select", { onchange: e => edit("set_layer", m.ref, "blend", e.target.value) },
      Object.keys(BLEND).map(k => h("option", { value: k, selected: k === m.blend, text: k })))));
    const out = h("output", { class: "mono", text: fmt(m.opacity) });
    fields.append(h("div", { class: "fld" }, h("label", {}, "Opacity ", out), h("input", { type: "range", min: 0, max: 1, step: 0.01, value: m.opacity, style: "accent-color:var(--accent)",
      oninput: e => { m.opacity = +e.target.value; out.textContent = fmt(m.opacity); }, onchange: e => edit("set_layer", m.ref, "opacity", +e.target.value) })));
  }
  box.append(fields);
  // its bars
  box.append(h("div", { class: "sect", text: m.cued ? "Bars (from its REAPER cues)" : "Bars" }));
  if (m.cued && m.cues === "reading") box.append(h("div", { class: "about", text: "Its cues are being read (a REAPER edit takes a few seconds): the bars show when they are known." }));
  if (m.cued && m.cues === "error") box.append(h("div", { class: "about", style: "color:var(--danger)", text: `Its cues could not be read: ${m.cue_error}` }));
  const spans = h("div", { class: "spans" });
  if (m.always) spans.append(h("div", { class: "about", text: "No bars: it plays the whole song. Drag an end of its block to give it some." }));
  (m.always ? [] : m.spans).forEach((s, k) => {
    const set = (key, v) => { const ns = m.spans.map(x => ({ ...x })); ns[k][key] = Math.max(1, Math.min(NB(), Math.round(v))); if (ns[k].last < ns[k].first) ns[k].last = ns[k].first; edit("spans", m.ref, ns); };
    spans.append(h("div", { class: "spanrow" },
      h("input", { type: "number", min: 1, max: NB(), value: s.first, disabled: m.cued, "aria-label": "first bar", onchange: e => set("first", +e.target.value) }),
      h("input", { type: "number", min: 1, max: NB(), value: s.last, disabled: m.cued, "aria-label": "last bar", onchange: e => set("last", +e.target.value) }),
      h("button", { disabled: m.cued, title: "Delete these bars", "aria-label": "Delete these bars", onclick: () => deleteSpan(m, k) }, "✕")));
  });
  box.append(spans);
  if (!m.cued) box.append(h("div", { class: "btnrow" }, h("button", { class: "btn", onclick: () => duplicateSpan(m, Math.max(0, sel.k)) }, "Again right after"),
    h("button", { class: "btn danger", onclick: () => { sel = null; edit("remove", m.ref); } }, "Delete")));
  // its knobs
  if (m.params.length) {
    box.append(h("div", { class: "sect", text: "Knobs" }));
    for (const p of m.params) box.append(paramRow(m, p));
  } else if (m.kind !== "clip") box.append(h("div", { class: "about", text: "No knobs the catalog knows of." }));
  box.append(h("div", { class: "sect", text: "The layer, as the scene says (but its bars)" }), h("pre", { class: "code", text: m.yaml }));
  $("yaml").textContent = m.yaml;
}
function guessRange(p) {
  const v = p.v ?? p.default, d = p.default;
  const lo = Math.min(0, v, d), hi = Math.max(1, Math.abs(d) * 2, Math.abs(v) * 1.5);
  return [lo, hi];
}
function kickTrack() {
  const t = S.tracks.find(x => /kick|bumbo/i.test(x.name)) || S.tracks[0];
  return t ? t.name : "kick";
}
function paramRow(m, p) {
  const row = h("div", { class: "prm" + (p.knob != null ? " linked" : ""), dataset: { param: p.name } });
  const out = h("output", { dataset: { out: p.name }, text: p.knob != null ? "" : fmt(p.v) });
  row.append(h("div", { class: "top" }, h("label", { text: p.name }), out));
  if (p.knob != null) {
    const ta = h("textarea", { spellcheck: "false", "aria-label": `${p.name}: what in the music moves it` }, p.knob);
    const k = knobData(p.knob), err = k && k !== "pending" && k.error;
    row.append(ta, h("div", { class: "bar" }, h("i", { dataset: { meter: p.name } })));
    if (err) row.append(h("div", { class: "err", text: err }));
    row.append(h("div", { class: "btnrow" },
      h("button", { class: "lnkbtn", onclick: () => edit("set_param", m.ref, p.name, ta.value) }, "Apply"),
      h("button", { class: "lnkbtn", onclick: () => edit("set_param", m.ref, p.name, +fmt(paramAt(p, now()))) }, "Steady at its value now")));
  } else {
    const [lo, hi] = guessRange(p);
    const num = h("input", { type: "number", step: "any", value: p.v, "aria-label": p.name, onchange: e => edit("set_param", m.ref, p.name, +e.target.value) });
    row.append(h("div", { class: "slide" },
      h("input", { type: "range", min: lo, max: hi, step: (hi - lo) / 400, value: p.v, "aria-label": p.name,
        oninput: e => { p.v = +e.target.value; out.textContent = fmt(p.v); num.value = fmt(p.v); }, onchange: e => edit("set_param", m.ref, p.name, +e.target.value) }), num));
    row.append(h("button", { class: "lnkbtn", onclick: () => edit("set_param", m.ref, p.name, `{hits: {track: '${kickTrack()}'}, envelope: 0.25, gain: ${fmt((hi - lo) / 2)}, bias: ${fmt(p.v)}}`) }, "Follow the music"));
  }
  if (p.about) row.append(h("div", { class: "about", text: p.about }));
  return row;
}
function renderClip(box) {
  const bar = sel.bar, s = clipAtBar(bar), pinned = String(bar) in pins;
  $("inspTitle").textContent = "Clips";
  const th = s && s.url ? thumbs.get(s.url) : null;
  box.append(h("div", { class: "ptitle" }, h("h3", { text: `Bar ${bar}` }), h("div", { class: "src", text: s ? s.name : "nothing" }),
    h("div", { class: "row" }, h("span", { class: "chip", text: pinned ? "pinned" : "the montage's choice" }))));
  if (th) box.append(h("div", { style: `aspect-ratio:16/9; max-width:100%; border-radius:6px; border:1px solid var(--line); background:center/cover url(${th})` }));
  box.append(h("div", { class: "fld" }, h("label", { text: "Plays in this bar" }),
    h("select", { onchange: e => edit("pin", bar, e.target.value === "auto" ? null : e.target.value) },
      h("option", { value: "auto", selected: !pinned, text: "The montage's choice" }),
      S.clips.map(c => h("option", { value: stem(c.name), selected: pins[String(bar)] === stem(c.name), text: c.name })),
      h("option", { value: "none", selected: pins[String(bar)] === "none", text: "Nothing" }))));
  box.append(h("div", { class: "about", text: "Python resolves the montage again (the composer's dry run) and the row follows. Drop a clip from the browser onto a bar to pin it." }));
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
    if (gl && view === "live") { drawFrame(t); showChips(); }
    syncEngineVideo(t);
    const bar = barAt(t), beats = S.song.beats;
    let beat = 1; if (beats.length) { const i = lastIdx(beats, t + 1e-4), i0 = lastIdx(beats, B[bar - 1] + 1e-4); beat = Math.max(1, i - i0 + 1); }
    $("lcdPos").textContent = `${bar}.${beat}`; $("lcdTime").textContent = fmtTime(t);
    const ph = $("playhead");
    if (ph) {
      const x = tToX(t); ph.style.transform = `translateX(${x}px)`;
      const sc = $("scroll");
      if (playing && !drag && !rsel && (x < sc.scrollLeft || x > sc.scrollLeft + sc.clientWidth - 40)) sc.scrollLeft = Math.max(0, x - 80);
    }
    updateMeters(t);
    frames++;
    const n = performance.now();
    if (n - fpsT > 500) { $("fpsText").textContent = view === "live" ? `live · ${W}×${H} · ${Math.round(frames * 1000 / (n - fpsT))} fps` : "engine render"; frames = 0; fpsT = n; }
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
  S = null; members = []; montage = []; pins = {}; sel = null; loop = null; engine = null; peaks = null; audioBuf = null; audioUrl = null;
  knobs.clear(); thumbs.clear(); shaders.clear(); dirty = false; savedAt = "";
  for (const v of vids.values()) { v.el.removeAttribute("src"); v.el.load(); }
  vids.clear();
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
  buildMeters(); renderBrowser(); renderTimeline();
  if (st.song.audio_url) loadAudio(st.song.audio_url);
  if (st.project.converted) log("warn", "a flat scene: grouped into BG / FG / master; Save writes it so");
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
  const items = [{ label: "Open…", hint: inWebview ? "" : "a path", on: openFileDialog }, { label: "New from an audio file…", on: newFromAudio }, { label: "Projects…", on: showStart }];
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
async function newFromAudio() {
  let path = null;
  if (inWebview) { const r = await ask("pick_file", "audio"); path = r && r.path; }
  else { const v = await dialog("New from an audio file", "The song's path. A project and a new scene are made beside the others (projects/, scenes/); a MIDI file of the same name comes with it.", [{ label: "Make it", value: "ok", primary: true }, { label: "Cancel", value: null }], "input/"); path = v && v.value && v.text; }
  if (!path) return;
  const r = await ask("new_project", path);
  if (r) { toast(`made ${r.path}`); openProject(r.path); }
}

// ---------------------------------------------------------------------------
// The scene file: save, save as, reload
// ---------------------------------------------------------------------------
async function save() {
  if (!S) return null;
  const r = await ask("save");
  if (!r) { showSaveState("NOT saved: see the log", true); return null; }
  apply(r);
  $("yaml").textContent = r.text;
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
document.querySelectorAll(".browser .tab").forEach(b => b.addEventListener("click", () => { tab = b.dataset.tab; renderBrowser(); }));
$("search").addEventListener("input", renderBrowser);
function showDock(which) {
  document.querySelectorAll(".dock .tab").forEach(b => b.setAttribute("aria-selected", b.dataset.dock === which));
  $("dockLog").hidden = which !== "log"; $("dockYaml").hidden = which !== "yaml";
  $("logFilter").hidden = which !== "log";
  if (which === "log") { errorsUnseen = 0; showErrBadge(); }
}
document.querySelectorAll(".dock .tab").forEach(b => b.addEventListener("click", () => showDock(b.dataset.dock)));
document.querySelectorAll("#logFilter button").forEach(b => b.addEventListener("click", () => setLogFilter(b.dataset.kind)));
$("bPlay").addEventListener("click", () => setPlaying(!playing));
$("bStart").addEventListener("click", () => seek(loop ? B[loop[0] - 1] : 0));
$("loopClear").addEventListener("click", () => { loop = null; renderTimeline(); });
// zoom: while the slider moves, what is drawn is stretched; it is drawn again once it stops
let zoomTimer = 0, zoomAt = null;
$("zoom").addEventListener("input", e => {
  if (!S) { ppb = +e.target.value; return; }
  if (zoomAt == null) zoomAt = now();
  const k = +e.target.value / ppb, sc = $("scroll");
  inner.style.transformOrigin = "0 0"; inner.style.transform = `scaleX(${k})`;
  sc.scrollLeft = Math.max(0, tToX(zoomAt) * k - sc.clientWidth / 2);
  clearTimeout(zoomTimer); zoomTimer = setTimeout(commitZoom, 180);
});
$("zoom").addEventListener("change", commitZoom);
function commitZoom() {
  clearTimeout(zoomTimer);
  if (zoomAt == null) return;
  const t = zoomAt; zoomAt = null;
  ppb = +$("zoom").value; inner.style.transform = "";
  renderTimeline();
  $("scroll").scrollLeft = Math.max(0, tToX(t) - $("scroll").clientWidth / 2);
}
$("bUndo").addEventListener("click", () => edit("undo"));
$("bRedo").addEventListener("click", () => edit("redo"));
$("bSave").addEventListener("click", save);
$("bSaveMenu").addEventListener("click", () => menu($("bSaveMenu"), [
  { label: "Save as…", on: saveAs, disabled: !S }, { label: "Reload from disk", hint: "the undo history goes", on: reloadScene, disabled: !S }]));
$("bProject").addEventListener("click", projectMenu);
$("bRender").addEventListener("click", startRender);
$("bStop").addEventListener("click", () => call("stop_render"));
$("viewLive").addEventListener("click", () => setView("live"));
$("viewEngine").addEventListener("click", () => { if (engine) setView("engine"); });
$("loadChip").addEventListener("click", () => showLoad($("loadPanel").hidden));
$("lpClose").addEventListener("click", () => showLoad(false));
$("lpCancel").addEventListener("click", () => ask("cancel_load"));
$("startClose").addEventListener("click", hideStart);
$("bOpenFile").addEventListener("click", openFileDialog);
$("bNewFromAudio").addEventListener("click", newFromAudio);
$("pathForm").addEventListener("submit", e => { e.preventDefault(); const p = $("pathInput").value.trim(); if (p) openProject(p); });
document.addEventListener("keydown", e => {
  if (!$("modal").hidden) return;
  const tag = (e.target.tagName || "").toLowerCase(), typing = tag === "input" || tag === "select" || tag === "textarea";
  const mod = e.ctrlKey || e.metaKey, key = e.key.toLowerCase();
  if (mod && key === "s") { e.preventDefault(); if (e.shiftKey) saveAs(); else save(); return; }
  if (mod && key === "o") { e.preventDefault(); openFileDialog(); return; }
  if (e.key === "Escape") { closeCtx(); if (!$("start").hidden && S) hideStart(); return; }
  if (typing || !$("start").hidden) return;
  if (mod && key === "z") { e.preventDefault(); edit(e.shiftKey ? "redo" : "undo"); return; }
  if (mod && key === "y") { e.preventDefault(); edit("redo"); return; }
  if (e.key === " " && tag !== "button") { e.preventDefault(); setPlaying(!playing); }
  else if (e.key === "Home") seek(loop ? B[loop[0] - 1] : 0);
  else if ((e.key === "Delete" || e.key === "Backspace") && sel) {
    e.preventDefault();
    if (sel.type === "member") { const m = memberById(sel.id); if (m && !m.cued) deleteSpan(m, Math.max(0, sel.k)); }
    else if (String(sel.bar) in pins) edit("pin", sel.bar, null);
  }
  else if (mod && key === "d" && sel && sel.type === "member") { e.preventDefault(); const m = memberById(sel.id); if (m) duplicateSpan(m, sel.k); }
});
window.addEventListener("beforeunload", e => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });

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
    log("sys", inWebview ? "pywebview: the page calls Python through js_api" : "browser: the page calls Python through /api");
    if (got.first) await openProject(got.first);
    else await showStart();
  } catch (e) { log("err", `could not start: ${e.message}`); toast(`Could not start: ${e.message}`, true); console.error(e); }
  requestAnimationFrame(tick);
}
window.addEventListener("pywebviewready", boot);
setTimeout(() => { if (!window.pywebview) boot(); }, 700);
