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
  montage: segs => { montage = segs || []; later(() => { renderTimeline(); if (sel && sel.type === "montage") renderInspector(); }); },
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
  if (seg && seg.url) { drawVideo(seg.url, (seg.from || 0) + (t - seg.t0) * (seg.dir || 1), cur, t, fit); chips.push(["clip", `clips · ${seg.name}`]); }
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
// Lanes top to bottom, as the Arrange window draws them. Python packs each track's members
// into sub-rows (core.arrangement.pack) and says where each goes; FG and BG get one empty
// row more, to put something new on.
const LANES = [
  { id: "audio", name: "Audio", kind: "audio", c: "var(--muted)" },
  { id: "midi", name: "MIDI", kind: "midi", c: "var(--accent)" },
  { id: "master", name: "Master", sub: "FX over everything", kind: "track", c: "var(--lane-fx)" },
  { id: "fg", name: "FG", sub: "foreground, on its own", kind: "track", c: "var(--lane-fg)", group: "FG" },
  { id: "fgfx", name: "FG FX", sub: "on the whole FG", kind: "track", c: "var(--lane-fx)", group: "FG" },
  { id: "bg", name: "BG", sub: "over the montage", kind: "track", c: "var(--lane-bg)", group: "BG" },
  { id: "bgfx", name: "BG FX", sub: "on the whole BG", kind: "track", c: "var(--lane-fx)", group: "BG" },
  { id: "clips", name: "Clips", sub: "the montage, under the BG", kind: "clips", c: "var(--lane-clip)" },
];
const FX_LANES = ["bgfx", "fgfx", "master"];
const SAME_KIND = [["bg", "fg"], ["bgfx", "fgfx"]];
const RULER_H = 34, HEAD_W = 132, EDGE = 7, ROW_MIN = 20, ROW_MAX = 120;
let rowH = 26;                 // a row's height: the vertical zoom
let layout = null;
let rowsOf = {};               // lane -> how many sub-rows its members take (Python's pack)
let groups = {}, has = {}, board = {};
let picked = null;             // what the browser picked: {type: "cat", key, kind, label} | {type: "clip", path, name, url, from}
const isFxLane = id => FX_LANES.includes(id);
const sameKind = (a, b) => a === b || SAME_KIND.some(p => p.includes(a) && p.includes(b));
const roomy = () => rowH >= 30;

function computeLayout() {
  let y = 0; const L = [];
  for (const d of LANES) {
    let hh = 0, shown = 0;
    if (d.kind === "audio") hh = Math.max(34, Math.min(90, Math.round(rowH * 1.5)));
    else if (d.kind === "midi") hh = 8 + 9 * Math.max(1, (S ? S.tracks.length : 1));
    else if (d.kind === "clips") { if (!has.clips && !montage.length) continue; hh = Math.max(26, rowH + 6); }
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
  $("corner").textContent = `${S.song.meter} · ${Math.round(S.song.bpm)} BPM`;
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
    const sub = l.kind === "audio" ? S.project.audio : l.sub;
    if (l.kind === "midi") for (const tr of S.tracks) head.append(h("div", { class: "trk", text: tr.name, title: `${tr.name}: ${tr.times.length} notes` }));
    else if (sub && (roomy() || l.h >= 30)) head.append(h("small", { text: sub, title: sub }));
    return head;
  }));
  // the lanes
  inner.style.width = width + "px"; inner.style.height = layout.height + "px";
  inner.replaceChildren();
  for (const l of layout.lanes) {
    const lane = h("div", { class: "lane " + l.kind, dataset: { lane: l.id }, style: `top:${l.top}px; height:${l.h}px; background-image: repeating-linear-gradient(to right, var(--line) 0 1px, transparent 1px ${ppb}px);` });
    if (l.kind === "audio") { const c = canvasFor(width, l.h); lane.append(c); drawWave(c); }
    if (l.kind === "midi") { const c = canvasFor(width, l.h); lane.append(c); drawMidi(c); }
    if (l.kind === "clips") drawCells(lane, l);
    if (l.kind === "track") {
      for (let r = 1; r < l.shown; r++) lane.append(h("div", { class: "rowline", style: `top:${r * rowH + 2}px` }));
      if ((l.id === "fg" || l.id === "bg") && roomy()) lane.append(h("div", { class: "newrow", style: `top:${(l.shown - 1) * rowH + 2}px; height:${rowH}px`, text: picked && accepts(l, picked) ? `drag across: ${picked.label || picked.name}, new` : "drag here: new" }));
      if (isFxLane(l.id) && !(rowsOf[l.id] || 0) && roomy()) lane.append(h("div", { class: "newrow", style: `top:2px; height:${rowH}px`, text: "right-click: an FX here · or pick one in FX and drag across" }));
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
}
function drawRuler(cv) {
  const g = cv.getContext("2d"), d = devicePixelRatio || 1;
  g.scale(d, d);
  g.font = "10px 'IBM Plex Mono', Consolas, monospace"; g.textBaseline = "top";
  const every = ppb < 18 ? 8 : ppb < 26 ? 4 : ppb < 40 ? 2 : 1;
  const dens = S.song.density || [];
  for (let b = 1; b <= NB(); b++) {
    const x = (b - 1) * ppb;
    const v = dens[b - 1] || 0;                 // how busy the MIDI is in this bar
    if (v > 0) { g.fillStyle = `rgba(240,160,75,${0.12 + 0.6 * v})`; g.fillRect(x, RULER_H - 4, ppb, 4); }
    g.fillStyle = "#3b4352"; g.fillRect(x, RULER_H - 14, 1, 10);
    const n = S.song.bar_beats[b - 1] || S.song.bar_beats[S.song.bar_beats.length - 1] || 4;
    g.fillStyle = "#2b313d";
    for (let k = 1; k < n; k++) g.fillRect(x + k * ppb / n, RULER_H - 9, 1, 5);
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
const thumbs = new Map();     // clip url -> its cover's url
function drawCells(lane, l) {
  for (let b = 1; b <= NB(); b++) {
    const s = clipAtBar(b), pinned = String(b) in pins, none = (pinned && pins[String(b)] === "none") || (s && s.name === "none");
    const th = s && s.url ? thumbs.get(s.url) : null;
    const chosen = sel && sel.type === "montage" && sel.first <= b && b <= sel.last;
    const cls = "cell" + (none || (montage.length && !s) ? " none" : "") + (pinned ? " pin" : "") + (chosen ? " sel" : "");
    lane.append(h("div", { class: cls, dataset: { bar: b }, style: `left:${(b - 1) * ppb + 1}px; width:${ppb - 2}px;` + (th && !none ? `background-image:url(${th})` : ""),
      title: `Bar ${b}: ${none ? "nothing" : s ? s.name : "…"}${pinned ? " (pinned)" : ""}`, text: ppb >= 44 && s && !none ? s.name.slice(0, Math.floor(ppb / 6)) : "" }));
  }
  if (!montage.length && roomy()) lane.append(h("div", { class: "newrow", style: "top:4px", text: "the montage comes when the clips are probed" }));
}
function blockEl(m, s, k, l) {
  const top = rowTop(l, m.row) + 1, height = rowH - 4;
  const left = (s.first - 1) * ppb + 1, width = (s.last - s.first + 1) * ppb - 2;
  const live = m.live && (m.kind === "clip" || (shaders.get(m.key) || {}).status !== "bad");
  const laneCls = m.kind === "fx" || isFxLane(m.lane) ? "fx" : m.kind === "clip" ? `clip ${m.lane}` : m.lane;
  const chosen = sel && sel.type === "member" && sel.id === idOf(m.ref) && (sel.k === k || m.cued || m.always || sel.k == null);
  const cls = ["block", laneCls, live ? "" : "engine", m.cued ? "cued" : "", m.always ? "always" : "", m.enabled ? "" : "off", s.missing ? "missing" : "", chosen ? "sel" : ""].filter(Boolean).join(" ");
  const linked = m.params.filter(p => p.knob != null).length;
  let label = m.label;
  if (m.kind === "clip" && s.file && !m.name) label = s.file.split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
  const el = h("div", { class: cls, dataset: { id: idOf(m.ref), k }, style: `left:${left}px; width:${Math.max(4, width)}px; top:${top}px; height:${height}px`,
    title: `${m.label} · bars ${s.first}–${s.last}${m.cued ? " · cued (REAPER): hide bars of it from its menu" : ""}${m.always ? " · the whole song: drag an end to give it bars" : ""}${live ? "" : " · at the render"}${s.missing ? ` · MISSING: ${s.file || m.file}` : ""}` },
    h("span", { class: "nm", text: (m.kind === "gen" && !isFxLane(m.lane) ? "" : "") + label + (m.always ? " (the whole song)" : "") }),
    s.from ? h("span", { class: "sg", text: `@${fmt(s.from)}s` }) : null,
    linked ? h("span", { class: "sg", text: `${linked}~` }) : null,
    m.own_fx ? h("span", { class: "sg", text: `+${m.own_fx}fx` }) : null);
  if (m.own_fx) el.append(h("i", { class: "fxstrip", title: m.fx.map(f => f.label).join(", ") }));
  return el;
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
      if (st && st.note) { toast(st.note); log("in", st.note); }
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
  if ("dirty" in st) { dirty = st.dirty; savedAt = st.saved_at || ""; }
  if (st.select !== undefined) {
    const s2 = st.select;
    if (!s2 || s2.ref === null) sel = null;
    else if (s2.montage) sel = { type: "montage", first: s2.montage[0], last: s2.montage[1] };
    else if (s2.ref) sel = { type: "member", id: idOf(s2.ref), ref: s2.ref, k: s2.k };
  }
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

// ---------------------------------------------------------------------------
// Pointer work on the lanes
// ---------------------------------------------------------------------------
let drag = null, rsel = null;
const inner = $("inner");
function edgeOf(el, clientX) {
  if (el.classList.contains("cued")) return null;
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
  sel = s;
  if (redraw) renderTimeline();
  else {                       // in the middle of a drag: the blocks stay the same elements
    document.querySelectorAll(".block.sel, .cell.sel").forEach(x => x.classList.remove("sel"));
    if (s && s.type === "member") inner.querySelectorAll(".block").forEach(x => { if (x.dataset.id === s.id && (+x.dataset.k === s.k || (memberById(s.id) || {}).cued)) x.classList.add("sel"); });
  }
  renderInspector();
}
function selectMember(m, k) { select({ type: "member", id: idOf(m.ref), ref: m.ref, k }, false); }
inner.addEventListener("pointerdown", e => {
  if (e.button !== 0 || !S) return;
  closeCtx();
  const [x, y] = innerXY(e), l = laneAtY(y), bar = xToBar(x);
  const bEl = e.target.closest(".block"), note = e.target.closest(".cuenote");
  if (note) { const m = memberById(note.dataset.id); if (m) selectMember(m, null); return; }
  if (bEl) {
    const m = memberById(bEl.dataset.id), k = +bEl.dataset.k;
    selectMember(m, m.always ? null : k);
    if (m.kind === "clip") { const s = spansOf(m)[k]; if (s && (s.file || m.file)) pickClip(s.file || m.file, s.from || 0); }
    if (m.cued) return;
    const s = spansOf(m)[Math.max(0, k)];
    drag = { type: "block", m, k, el: bEl, mode: edgeOf(bEl, e.clientX) || "move", x0: e.clientX, y0: y, f0: s.first, l0: s.last, first: s.first, last: s.last, lane: m.lane, row: m.row, moved: false };
    try { bEl.setPointerCapture(e.pointerId); } catch (err) { /* a pointer that is not down */ }
    return;
  }
  if (e.target.closest(".cell") || (l && l.kind === "clips")) { drag = { type: "range", bar0: bar, first: bar, last: bar, moved: false }; return; }
  if (l && l.kind === "track") drag = { type: "new", lane: l.id, row: rowAtY(l, y), bar0: bar, first: bar, last: bar, moved: false };
  else drag = { type: "span", bar0: bar, first: bar, last: bar, moved: false };
  try { inner.setPointerCapture(e.pointerId); } catch (err) { /* a pointer that is not down */ }
});
inner.addEventListener("pointermove", e => {
  const [x, y] = innerXY(e);
  if (!drag) {
    const bEl = e.target.closest(".block");
    const end = bEl ? edgeOf(bEl, e.clientX) : null;
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
  drag.moved = drag.moved || bar !== drag.bar0;
  drag.first = Math.min(drag.bar0, bar); drag.last = Math.max(drag.bar0, bar);
  if (drag.type === "new") {
    const l = laneById(drag.lane), g = $("ghost");
    g.hidden = false;
    Object.assign(g.style, { left: (drag.first - 1) * ppb + "px", width: (drag.last - drag.first + 1) * ppb + "px", top: rowTop(l, drag.row) + "px", height: rowH - 2 + "px" });
    g.textContent = picked && accepts(l, picked) ? `${picked.label || picked.name} · bars ${drag.first}–${drag.last}` : `bars ${drag.first}–${drag.last}`;
    g.classList.toggle("put", !!(picked && accepts(l, picked)));
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
    if (!d.moved) { renderTimeline(); flushAfterDrag(); return; }
    const toRow = d.lane !== d.m.lane || d.row !== d.m.row;
    edit("move", d.m.ref, d.k, d.mode, d.first, d.last, toRow ? d.lane : null, toRow ? d.row : null);
  } else if (d.type === "new") {
    $("ghost").hidden = true;
    const l = laneById(d.lane);
    if (picked && accepts(l, picked)) edit("new_member", d.lane, picked.type === "clip" ? picked.path : picked.key, d.first, d.last, d.row, picked.from || 0);
    else if (d.moved) { setLoop(d.first, d.last); toast(`bars ${d.first}–${d.last} chosen · to put something there, pick it in the browser first`); }
    else { sel = null; seek(xToT(x)); renderTimeline(); renderInspector(); }
  } else if (d.type === "range") {
    let first = d.first, last = d.last;
    if (!d.moved) { const run = runAt(bar); if (run) { first = run.first; last = run.last; if (run.path) pickClip(run.path, 0); } }
    select({ type: "montage", first, last });
  } else if (d.type === "span") {
    if (!d.moved) { seek(xToT(x)); sel = null; renderTimeline(); renderInspector(); }
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
  $("bRender").textContent = loop ? `Render bars ${loop[0]}–${loop[1]}` : "Render 4 bars here";
}
// Preview: the chosen bars played live, over and over (at most 8)
function previewBars(first, last) {
  loop = [first, Math.min(last, first + 7)];
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
function closeCtx() { document.querySelectorAll(".ctx").forEach(x => x.remove()); }
document.addEventListener("pointerdown", e => { if (!e.target.closest(".ctx")) closeCtx(); });
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
  const args = it.args.map(a => a === "$key" ? key : a === "$from" ? (picked && picked.from) || 0 : a);
  edit(it.call, ...args);
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
  (parent || document.body).append(box);
  const r = box.getBoundingClientRect();
  if (parent) { box.style.position = "fixed"; }
  box.style.left = Math.max(4, Math.min(x, innerWidth - r.width - 6)) + "px";
  box.style.top = Math.max(4, Math.min(y, innerHeight - r.height - 6)) + "px";
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
function wireLaneDrop(el, l) {
  el.addEventListener("dragover", e => { if (accepts(l, dragItem)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; el.classList.add("drop-ok"); } });
  el.addEventListener("dragleave", () => el.classList.remove("drop-ok"));
  el.addEventListener("drop", e => {
    el.classList.remove("drop-ok");
    if (!accepts(l, dragItem)) return;
    e.preventDefault();
    const it = dragItem; dragItem = null;
    const [x, y] = innerXY(e), bar = xToBar(x);
    putItem(it, l.id, bar, l.kind === "track" ? rowAtY(l, y) : null);
  });
}
function putItem(it, laneId, bar, row = null) {
  const last = Math.min(NB(), bar + 3);
  if (laneId === "clips") return edit("put_in_montage", bar, bar, it.path, it.from || 0);
  return edit("new_member", laneId, it.type === "clip" ? it.path : it.key, bar, last, row, it.from || 0);
}

// ---------------------------------------------------------------------------
// Browser: clips, Generators, FX. A click picks one (then drag across a lane to put it
// there); dragging one onto a lane puts four bars of it.
// ---------------------------------------------------------------------------
let tab = "gen";
function pickClip(path, from = 0) {
  const c = (S.clips || []).find(x => x.path === path || x.path.replace(/\\/g, "/").endsWith(String(path).replace(/\\/g, "/")));
  picked = { type: "clip", path, name: String(path).split(/[\\/]/).pop(), label: String(path).split(/[\\/]/).pop().replace(/\.[^.]+$/, ""), url: c ? c.url : null, from };
  showPicked(); if (tab === "clips") renderBrowser();
}
function pick(it) { picked = picked && (picked.key || picked.path) === (it.key || it.path) ? null : it; showPicked(); renderBrowser(); }
function showPicked() {
  const el = $("picked");
  if (!picked) { el.hidden = true; return; }
  el.hidden = false;
  el.replaceChildren(...[h("span", { class: "tlabel", text: "picked" }), h("b", { text: picked.label || picked.name }),
    picked.type === "clip" && picked.from ? h("span", { class: "mono", text: `from ${fmt(picked.from)} s` }) : null,
    h("span", { class: "hint", text: picked.type === "cat" && picked.kind === "fx" ? "drag across an FX track, or Add FX on a block" : "drag across a lane to put it there" }),
    h("button", { class: "lnkbtn", onclick: () => { picked = null; showPicked(); renderBrowser(); }, "aria-label": "Unpick" }, "✕")].filter(Boolean));
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
  const isPicked = it => picked && (picked.key || picked.path) === (it.key || it.path);
  if (tab === "clips") {
    if (!S.clips.length) { list.append(h("div", { class: "empty-note", text: "The project has no clips folder." })); return; }
    for (const c of S.clips.filter(c => c.name.toLowerCase().includes(q))) {
      const it = { type: "clip", path: c.path, name: c.name, label: c.name.replace(/\.[^.]+$/, ""), url: c.url, from: 0 };
      const th = thumbs.get(c.url);
      const row = h("div", { class: "bitem" + (isPicked(it) ? " picked" : ""), title: `${c.name}: click to pick, then drag across BG or FG; or drag it onto a lane (4 bars), onto Clips to pin a bar`, onclick: e => { if (!e.target.closest("button")) pick(it); } },
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
      const flags = (c.takes_input ? " *" : "") + c.needs.map(n => ` (${n})`).join("");
      const row = h("div", { class: "bitem" + (isPicked(it) ? " picked" : ""), title: (c.about || c.key) + (c.takes_input ? "\n* takes the picture below as input" : "") + (bad ? `\nGLSL ES: ${st.error}` : ""), onclick: e => { if (!e.target.closest("button")) pick(it); } },
        h("span", { class: "nm", text: c.label + flags }),
        h("span", { class: `badge ${c.live && !bad ? "live" : bad ? "bad" : "engine"}`, text: c.live && !bad ? "live" : bad ? "glsl" : "engine" }),
        h("button", { class: "add", title: c.kind === "gen" ? "Put 4 bars of it in the BG at the playhead" : "Put it on the master, all the song", "aria-label": `Add ${c.label}`, onclick: () => c.kind === "gen" ? putItem(it, "bg", barAt(now())) : edit("add_fx", ["master"], c.key) }, "+"));
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
  const box = insp(); box.replaceChildren();
  if (!sel && loop && S) {
    $("inspTitle").textContent = "Bars";
    box.append(h("div", { class: "ptitle" }, h("h3", { text: `Bars ${loop[0]}–${loop[1]}` }), h("div", { class: "src", text: "chosen" })),
      h("div", { class: "btnrow" }, h("button", { class: "btn", onclick: () => previewBars(loop[0], loop[1]) }, `Preview bars ${loop[0]}–${Math.min(loop[1], loop[0] + 7)} (Space)`),
        h("button", { class: "btn", onclick: () => { loop = null; renderTimeline(); renderInspector(); } }, "Choose none")),
      h("div", { class: "about", text: "Pick a clip, a Generator or an FX in the browser and drag across a track to put it there; click a block to choose it." }));
    return;
  }
  if (!sel) {
    $("inspTitle").textContent = "Arrange";
    box.append(h("div", { class: "about", text: "Every edit goes to Python (core.arrangement) and comes back as the lanes." }),
      h("div", { class: "keys" },
        h("span", { class: "kbd", text: "click" }), "pick in the browser, then drag across a lane to put it",
        h("span", { class: "kbd", text: "drag" }), "move a block, onto another row or the other lane; its ends resize it",
        h("span", { class: "kbd", text: "right click" }), "its menu: FX, hide bars, copy and paste, delete",
        h("span", { class: "kbd", text: "ruler" }), "click to go there, drag to choose bars",
        h("span", { class: "kbd", text: "a lane's name" }), "its settings and FX (Clips: the montage's)",
        h("span", { class: "kbd", text: "Ctrl+wheel" }), "zoom · Shift+wheel: row height · Alt+wheel: along",
        h("span", { class: "kbd", text: "Space" }), "play · Shift+Space: render the chosen bars",
        h("span", { class: "kbd", text: "Delete" }), "the chosen block (a member's last: the member)",
        h("span", { class: "kbd", text: "Ctrl+Z" }), "undo (Ctrl+Y redo)",
        h("span", { class: "kbd", text: "Ctrl+S" }), "save the scene (Ctrl+Shift+S: save as)"));
    return;
  }
  if (sel.type === "montage") return inspMontageBars(box, sel.first, sel.last);
  if (sel.type === "clips") return inspMontage(box);
  if (sel.type === "lane") return inspLane(box, sel.group);
  if (sel.type === "master") return inspMaster(box);
  const m = memberById(sel.id);
  if (!m) { sel = null; return renderInspector(); }
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
  const pb = k != null && blocks[k] ? blocks[k] : m.spans[0];
  box.append(h("div", { class: "btnrow" },
    pb ? h("button", { class: "btn", onclick: () => previewBars(pb.first, pb.last) }, `Preview bars ${pb.first}–${Math.min(pb.last, pb.first + 7)}`) : null,
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
  // its knobs
  if (m.params.length) {
    box.append(h("div", { class: "sect", dataset: { sect: "knobs" }, text: "Knobs" }));
    for (const p of m.params) box.append(paramRow(m, p));
  } else if (m.kind !== "clip") box.append(h("div", { class: "about", dataset: { sect: "knobs" }, text: "No knobs the catalog knows of." }));
  box.append(sect("The layer, as the scene says (but its bars)"), h("pre", { class: "code", text: m.yaml }));
  $("yaml").textContent = m.yaml;
}
const PLACES = [["x", 0.5, "0 left .. 1 right"], ["y", 0.5, "0 top .. 1 bottom"], ["scale", 1.0, "1: as tall as the frame"]];
function inspPlacement(box, m) {
  const p = m.place;
  box.append(sect("In the frame"));
  for (const [key, def, about] of PLACES) {
    const v = p[key];
    if (v && typeof v === "object") box.append(h("div", { class: "prm linked" }, h("div", { class: "top" }, h("label", { text: key }), h("output", { text: "follows the music" })),
      h("div", { class: "about mono", text: v.knob }),
      h("div", { class: "btnrow" }, h("button", { class: "lnkbtn", onclick: () => linkPlace(m, key, v.knob) }, "Edit link…"),
        h("button", { class: "lnkbtn", onclick: () => edit("set_param", m.ref, key, String(knobBias(v.knob) ?? def)) }, "Fixed"))));
    else box.append(h("div", { class: "prm" }, h("div", { class: "top" }, h("label", { text: key }), h("small", { class: "hint", text: about })),
      h("div", { class: "slide" }, h("input", { type: "range", min: key === "scale" ? 0.05 : -0.5, max: key === "scale" ? 2 : 1.5, step: 0.01, value: v ?? def, "aria-label": key, onchange: e => edit("set_layer", m.ref, key, +e.target.value) }),
        numField(v ?? def, val => edit("set_layer", m.ref, key, val), { step: 0.05, aria: key })),
      h("button", { class: "lnkbtn", onclick: () => linkPlace(m, key, `{hits: {track: '${kickTrack()}'}, envelope: 0.25, gain: 0.1, bias: ${fmt(v ?? def)}}`) }, "Follow the music…")));
  }
  box.append(h("div", { class: "fields" },
    field("Fit", h("select", { onchange: e => edit("set_layer", m.ref, "fit", e.target.value) }, ["contain", "cover"].map(f => h("option", { value: f, selected: f === p.fit, text: f })))),
    field("", h("div", { class: "row" }, check("flip", p.flip, v => edit("set_layer", m.ref, "flip", v)), check("loop", p.loop, v => edit("set_layer", m.ref, "loop", v))))));
  if (p.sam != null) {
    box.append(h("div", { class: "about", style: "color:var(--live)", text: "Cut out by SAM2 at the render (its seeds: the sam_cutout FX below)." }));
  } else if (p.cutout) {
    box.append(h("div", { class: "row" }, h("span", { class: "chip", text: `cut out by ${p.cutout}` }),
      h("button", { class: "lnkbtn", onclick: () => edit("set_layer", m.ref, "cutout", null) }, "Whole frame")),
      check("all but object 1 (a backdrop)", p.drop, v => { const cut = { ...(p.cutout_spec || {}) }; delete cut.drop; delete cut.objects; if (v) cut.drop = [1]; edit("set_layer", m.ref, "cutout", cut); }));
  } else {
    box.append(h("div", { class: "row" }, h("span", { class: "chip", text: "not cut out (whole frame)" }),
      p.mask_track ? h("button", { class: "lnkbtn", onclick: () => edit("set_layer", m.ref, "cutout", { track: p.mask_track, feather: 2 }) }, `Cut out (${p.mask_track.split(/[\\/]/).pop()})`) : null,
      h("button", { class: "lnkbtn", onclick: () => edit("add_fx", m.ref, "sam_cutout") }, "Cut out by SAM2")));
  }
  if (m.own_fx) box.append(check("its FX may spill outside the figure (trails, glow, split edges)", m.spill, v => edit("set_layer", m.ref, "spill", v)));
}
function knobBias(text) { const m = /bias:\s*(-?[\d.]+)/.exec(text || ""); return m ? +m[1] : null; }
async function linkPlace(m, key, text) {
  const v = await dialog(`${m.label} · ${key}`, "What in the music moves it, as YAML (the knob editor comes with the plugin's items). It is checked before it is written.", [{ label: "Apply", value: "ok", primary: true }, { label: "Cancel", value: null }], text);
  if (v && v.value && v.text) edit("set_param", m.ref, key, v.text);
}
// an FX list: a block's own, a lane's, the master's. rows: [{ref, label, enabled, bars_text}]
function fxList(title, rows, owner, draggable) {
  const wrap = h("div", { class: "fxlist" });
  wrap.append(h("div", { class: "fxhead" }, h("span", { class: "sect", text: title }),
    h("button", { class: "lnkbtn", onclick: e => menuAt(e.clientX, e.clientY, catalogItems("fx", key => edit("add_fx", owner, key))) }, "Add FX…")));
  if (!rows.length) wrap.append(h("div", { class: "about", text: "none" }));
  rows.forEach((f, j) => {
    const row = h("div", { class: "fxrow" + (f.enabled ? "" : " off"), draggable: draggable ? "true" : null, dataset: { j } },
      draggable ? h("span", { class: "grip", title: "drag to reorder", "aria-hidden": "true" }, "⋮⋮") : null,
      h("input", { type: "checkbox", checked: f.enabled, "aria-label": `${f.label} on`, onchange: () => edit("toggle", f.ref) }),
      h("span", { class: "nm", text: `${j + 1}. ${f.label}`, title: f.label }),
      textField(f.bars_text, "all", v => edit("set_windows", f.ref, v), `${f.label}: bars`),
      h("button", { class: "ib", title: "earlier", "aria-label": "Earlier", onclick: () => edit("reorder", f.ref, -1) }, "↑"),
      h("button", { class: "ib", title: "later", "aria-label": "Later", onclick: () => edit("reorder", f.ref, 1) }, "↓"),
      h("button", { class: "ib", title: "remove", "aria-label": "Remove", onclick: () => edit("remove", f.ref) }, "✕"));
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
    h("div", { class: "about", text: group === "FG" ? "Drawn on its own and put over the BG; its FX work on it alone." : "Over the clip montage; its FX work on it and the montage." }));
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
  $("inspTitle").textContent = "Clips";
  const bars = last > first ? `bars ${first}–${last}` : `bar ${first}`;
  box.append(h("div", { class: "ptitle" }, h("h3", { text: `Clips · ${bars}` })));
  const runs = montageRuns().filter(r => r.last >= first && r.first <= last);
  if (!runs.length) box.append(h("div", { class: "about", text: "The montage is not known yet (it comes when the clips are probed)." }));
  for (const r of runs.slice(0, 12)) {
    const pinned = Object.keys(pins).some(b => +b >= r.first && +b <= r.last);
    const th = r.url ? thumbs.get(r.url) : null;
    box.append(h("div", { class: "runrow" }, h("span", { class: "sw", style: th ? `background-image:url(${th})` : "" }),
      h("span", { class: "nm", text: r.name, title: r.path || "" }), h("span", { class: "mono", text: r.first === r.last ? `${r.first}` : `${r.first}–${r.last}` }), pinned ? h("span", { class: "chip", text: "pinned" }) : null));
  }
  if (runs.length > 12) box.append(h("div", { class: "about", text: `… and ${runs.length - 12} more` }));
  const btns = h("div", { class: "btnrow" });
  box.append(btns, h("div", { class: "about", text: "The montage draws a clip for each bar (clip order, the seed, the triggers' cuts and reverses). A pin keeps a clip there; the bars around stay as they were drawn. A clip from another folder, or from another point, goes in the BG over these bars." }));
  const items = await ask("montage_menu", first, last, picked && picked.type === "clip" ? picked.path : null);
  if (!items || !(sel && sel.type === "montage" && sel.first === first && sel.last === last)) return;
  for (const it of items) if (!it.head && !it.sep) btns.append(h("button", { class: "btn", onclick: () => runItem(it) }, it.label.replace(/ - Delete$/, "")));
}
async function inspMontage(box) {
  $("inspTitle").textContent = "Clips";
  box.append(h("div", { class: "ptitle" }, h("h3", { text: "The montage" }), h("div", { class: "src", text: "the scene's video: settings" })),
    h("div", { class: "about", text: "The project's clips folder, cut by the bars and the triggers; it shows wherever the BG has nothing. Click a clip on its row (or drag across bars) to choose it." }));
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
      const rp = $("rulerPlay"); if (rp) rp.style.transform = `translateX(${x}px)`;
      const sc = $("scroll"), seen = sc.clientWidth - HEAD_W;
      if (playing && !drag && !rsel && zoomAt == null && (x < sc.scrollLeft || x > sc.scrollLeft + seen - 40)) sc.scrollLeft = Math.max(0, x - 80);
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
  picked = null; rowsOf = {}; groups = {}; has = {}; board = {}; showPicked();
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
  buildMeters(); renderBrowser(); renderTimeline(); showResolution();
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
  if (got) { S.project = { ...S.project, ...got }; setSize(got.resolution); canvas.style.aspectRatio = `${W} / ${H}`; toast(`The video: ${got.resolution}`); }
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
  if (e.key === " " && tag !== "button") { e.preventDefault(); if (e.shiftKey) startRender(); else setPlaying(!playing); }
  else if (e.key === "Home") seek(loop ? B[loop[0] - 1] : 0);
  else if ((e.key === "Delete" || e.key === "Backspace") && sel) { e.preventDefault(); deleteSelected(); }
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
