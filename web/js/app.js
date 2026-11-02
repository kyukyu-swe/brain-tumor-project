import { Client, handle_file } from "./vendor/gradio-client.js";
import { marked } from "./vendor/marked.esm.js";
import { BACKEND, FALLBACK_LANGUAGES, SUGGESTED_LANGUAGES, MAX_TRIAGE_FILES } from "./config.js";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pct = (x) => `${Math.round(x * 100)}%`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const md = (text) => marked.parse(esc(text));   // escape first: the model text can never inject HTML

const NAMES = { glioma: "Glioma", meningioma: "Meningioma", notumor: "No tumor", pituitary: "Pituitary tumor" };
const PRIORITY_ICON = { High: "🔴", Medium: "🟠", Low: "🟢" };

/* ------------------------------------------------------------------ backend */
let client = null;
let LANGS = FALLBACK_LANGUAGES;          // replaced by the full list from the backend (/info)
let SUGGESTED = SUGGESTED_LANGUAGES;
let connecting = null;

function setStatus(state, text) {
  const el = $("#status");
  el.classList.remove("ready", "error");
  if (state) el.classList.add(state);
  $("#statusText").textContent = text;
}

async function connect() {
  if (client) return client;
  if (connecting) return connecting;
  setStatus("", "Waking up the AI… (can take ~1 min)");
  connecting = Client.connect(BACKEND)
    .then(async (c) => {
      client = c;
      setStatus("ready", "AI ready");
      try {
        const info = (await c.predict("/info", [])).data[0];
        if (Array.isArray(info.languages) && info.languages.length) {
          // older backends send plain names; newer ones send objects
          LANGS = info.languages.map((l) => (typeof l === "string"
            ? (FALLBACK_LANGUAGES.find((f) => f.name === l) || { name: l, native: l, code: "", official: true, rtl: false })
            : l));
        }
        if (Array.isArray(info.suggested)) SUGGESTED = info.suggested;
        $("#aboutModel").textContent =
          `Our model (${info.model}) looks at one MRI slice and gives the chance of glioma, meningioma, ` +
          `pituitary tumor, or no tumor. Test accuracy: ${(info.test_accuracy * 100).toFixed(2)}%.`;
      } catch { /* info is optional */ }
      return c;
    })
    .catch((e) => {
      setStatus("error", "AI offline — click to retry");
      connecting = null;
      throw e;
    });
  return connecting;
}

async function analyze(blob) {
  const c = await connect();
  const res = await c.predict("/analyze", [handle_file(blob)]);
  return res.data[0];
}

async function explain(result, language, mode) {
  const c = await connect();
  const { image, heatmap, ...light } = result;      // don't send the images back
  const res = await c.predict("/explain", [light, language, mode]);
  return res.data[0];
}

/* ------------------------------------------------------------------ UI helpers */
function toast(msg, isError = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.toggle("error", isError);
  t.classList.add("show");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), 4200);
}

function setLoading(btn, on, label) {
  btn.classList.toggle("loading", on);
  btn.disabled = on;
  if (label) btn.lastChild.textContent = label;
}

// ripple on every button
document.addEventListener("pointerdown", (e) => {
  const btn = e.target.closest(".btn");
  if (!btn || btn.disabled) return;
  const r = btn.getBoundingClientRect();
  const s = Math.max(r.width, r.height);
  const span = document.createElement("span");
  span.className = "ripple";
  span.style.cssText = `width:${s}px;height:${s}px;left:${e.clientX - r.left - s / 2}px;top:${e.clientY - r.top - s / 2}px`;
  btn.appendChild(span);
  setTimeout(() => span.remove(), 700);
});

function countUp(el, to) {
  const from = Number(el.textContent) || 0;
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / 600);
    el.textContent = Math.round(from + (to - from) * (1 - Math.pow(1 - k, 3)));
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function makeDropzone(zone, input, onFiles) {
  ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("drag"); }));
  zone.addEventListener("drop", (e) => onFiles([...e.dataTransfer.files].filter((f) => f.type.startsWith("image/"))));
  input.addEventListener("change", () => { onFiles([...input.files]); input.value = ""; });
}

/* ------------------------------------------------------------------ result rendering */
function resultHTML(r, { showProbs = true } = {}) {
  const C = 2 * Math.PI * 38;
  const reasons = r.refer
    ? `<div class="alert"><span>⚠️</span><div><b>The AI is not sure. A specialist should check this scan.</b>
         <ul>${r.reasons.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div></div>`
    : `<div class="alert ok"><span>✅</span><div><b>The AI is confident</b> — but it can still be wrong. Only a doctor can diagnose.</div></div>`;
  const probs = Object.entries(r.probs).sort((a, b) => b[1] - a[1]);
  return `
    <div class="result-head">
      <div class="ring">
        <svg width="92" height="92"><circle class="bg" cx="46" cy="46" r="38"/>
          <circle class="fg" cx="46" cy="46" r="38" stroke-dasharray="${C}" stroke-dashoffset="${C}" data-target="${C * (1 - r.confidence)}"/></svg>
        <div class="val">${pct(r.confidence)}</div>
      </div>
      <div>
        <div class="result-sub">AI result</div>
        <h3 class="result-title">${esc(r.display_name)}</h3>
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px">
          <span class="badge ${r.priority}">${PRIORITY_ICON[r.priority]} ${r.priority} priority</span>
          ${r.refer ? `<span class="badge review">⚠️ Needs review</span>` : ""}
        </div>
      </div>
    </div>
    ${reasons}
    <p class="section-title">Where the AI looked</p>
    <div class="compare">
      <img src="${r.image}" alt="MRI scan" />
      <img class="top-layer" src="${r.heatmap}" alt="AI heatmap" />
      <div class="handle"></div>
      <span class="tag l">Original</span><span class="tag r">Heatmap</span>
      <input type="range" min="0" max="100" value="50" aria-label="Compare original and heatmap" />
    </div>
    <div class="compare-hint">Drag to compare · red = areas that most influenced the AI</div>
    ${showProbs ? `
    <p class="section-title">Probabilities</p>
    <div class="bars">
      ${probs.map(([k, v], i) => `
        <div class="bar-row ${i === 0 ? "top" : ""}"><span>${NAMES[k] || k}</span>
          <div class="bar-track"><div class="bar-fill" data-w="${Math.max(v * 100, 0.5)}"></div></div>
          <span class="pct">${pct(v)}</span></div>`).join("")}
    </div>` : ""}`;
}

function activateResult(root) {
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $$(".ring .fg", root).forEach((c) => (c.style.strokeDashoffset = c.dataset.target));
    $$(".bar-fill", root).forEach((b) => (b.style.width = `${b.dataset.w}%`));
  }));
  $$(".compare", root).forEach((cmp) => {
    const range = $("input", cmp), top = $(".top-layer", cmp), handle = $(".handle", cmp);
    const update = () => { top.style.clipPath = `inset(0 0 0 ${range.value}%)`; handle.style.left = `${range.value}%`; };
    range.addEventListener("input", update);
    update();
  });
}

const skeleton = `<div class="skeleton"><div></div><div></div><div></div></div>`;

/* ------------------------------------------------------------------ language picker */
const fold = (x) => String(x).toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
const langByName = (n) => LANGS.find((l) => l.name === n);
function savedLang() {
  try { return localStorage.getItem("lang") || "English"; } catch { return "English"; }
}
function saveLang(n) {
  try { localStorage.setItem("lang", n); } catch { /* storage may be blocked */ }
}

const openPickers = new Set();
document.addEventListener("click", (e) => openPickers.forEach((p) => !p.root.contains(e.target) && p.close()));

function langPicker(root) {
  let selected = savedLang();
  let items = [];       // currently shown options (for keyboard navigation)
  let active = -1;
  root.innerHTML = `
    <button type="button" class="lp-button" aria-haspopup="listbox" aria-expanded="false">
      <span class="lp-globe">🌐</span><span class="lp-label"></span><span class="lp-caret">▾</span>
    </button>
    <div class="lp-panel" role="dialog">
      <div class="lp-search">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
        <input type="text" placeholder="Search: Thai, ไทย, th…" aria-label="Search language" autocomplete="off" spellcheck="false" />
      </div>
      <div class="lp-list" role="listbox"></div>
    </div>
    <div class="lp-beta hidden">⚠️ <b>Beta language:</b> Gemini does not officially support it. Please ask a native speaker to check the text.</div>`;
  const btn = $(".lp-button", root), panel = $(".lp-panel", root), input = $("input", root), list = $(".lp-list", root);
  const api = { root, close, get: () => selected, info: () => langByName(selected) || { name: selected, rtl: false, official: true } };

  function label() {
    const l = api.info();
    $(".lp-label", root).innerHTML = `<b><bdi>${esc(l.native || l.name)}</bdi></b>${l.native && l.native !== l.name ? `<small>${esc(l.name)}</small>` : ""}`;
    $(".lp-beta", root).classList.toggle("hidden", l.official !== false);
  }
  function option(l) {
    return `<div class="lp-option ${l.name === selected ? "selected" : ""}" role="option" data-name="${esc(l.name)}"
              aria-selected="${l.name === selected}">
      <bdi class="lp-native">${esc(l.native)}</bdi>
      <span class="lp-en">${l.native !== l.name ? esc(l.name) : ""}</span>
      ${l.official === false ? '<span class="lp-tag">beta</span>' : ""}
      ${l.name === selected ? '<span class="lp-check">✓</span>' : ""}
    </div>`;
  }
  function render() {
    const q = fold(input.value.trim());
    let html = "";
    if (!q) {
      const sug = SUGGESTED.map(langByName).filter(Boolean);
      const official = LANGS.filter((l) => l.official !== false).sort((a, b) => a.name.localeCompare(b.name));
      const beta = LANGS.filter((l) => l.official === false);
      html = `<div class="lp-group">Suggested</div>${sug.map(option).join("")}` +
             `<div class="lp-group">All supported languages (${official.length})</div>${official.map(option).join("")}` +
             (beta.length ? `<div class="lp-group">Beta — please check with a native speaker</div>${beta.map(option).join("")}` : "");
    } else {
      // rank: exact code > name/native starts with the text > contains the text
      const score = (l) => {
        const n = fold(l.name), nat = fold(l.native), c = fold(l.code);
        if (c === q) return 0;
        if (n.startsWith(q) || nat.startsWith(q)) return 1;
        if (n.split(/[\s()]+/).some((w) => w.startsWith(q))) return 2;
        if (n.includes(q) || nat.includes(q) || c.startsWith(q)) return 3;
        return 9;
      };
      const hits = LANGS.map((l) => [score(l), l]).filter(([sc]) => sc < 9)
        .sort((a, b) => a[0] - b[0] || a[1].name.localeCompare(b[1].name)).map(([, l]) => l);
      html = hits.length ? hits.map(option).join("") : `<div class="lp-empty">No language found for “${esc(input.value)}”</div>`;
    }
    list.innerHTML = html;
    items = $$(".lp-option", list);
    active = Math.max(0, items.findIndex((x) => x.dataset.name === selected));
    if (q) active = 0;
    highlight(false);
  }
  function highlight(scroll = true) {
    items.forEach((x, i) => x.classList.toggle("active", i === active));
    if (scroll && items[active]) items[active].scrollIntoView({ block: "nearest" });
  }
  function open() {
    openPickers.forEach((p) => p !== api && p.close());
    input.value = "";
    render();
    // open upwards if there is not enough space below
    const r = btn.getBoundingClientRect();
    const below = window.innerHeight - r.bottom, above = r.top;
    root.classList.toggle("up", below < 360 && above > below);
    root.classList.add("open");
    btn.setAttribute("aria-expanded", "true");
    openPickers.add(api);
    setTimeout(() => { input.focus(); items[active]?.scrollIntoView({ block: "center" }); }, 60);
  }
  function close() {
    root.classList.remove("open");
    btn.setAttribute("aria-expanded", "false");
    openPickers.delete(api);
  }
  function choose(name) {
    selected = name;
    saveLang(name);
    label();
    close();
    btn.classList.add("bump");
    setTimeout(() => btn.classList.remove("bump"), 400);
  }

  btn.addEventListener("click", () => (root.classList.contains("open") ? close() : open()));
  input.addEventListener("input", render);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); active = Math.min(items.length - 1, active + 1); highlight(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); active = Math.max(0, active - 1); highlight(); }
    else if (e.key === "Enter") { e.preventDefault(); if (items[active]) choose(items[active].dataset.name); }
    else if (e.key === "Escape") { close(); btn.focus(); }
  });
  list.addEventListener("click", (e) => { const o = e.target.closest(".lp-option"); if (o) choose(o.dataset.name); });
  label();
  api.refresh = label;          // call again after the full language list arrives
  return api;
}

function explainHTML(text, lang) {
  const beta = lang.official === false
    ? `<p class="beta-note">⚠️ ${esc(lang.name)} is a beta language — please ask a native speaker to check this text.</p>` : "";
  return `<div class="explain" dir="${lang.rtl ? "rtl" : "ltr"}" lang="${esc(lang.code || "")}">${md(text)}${beta}</div>`;
}

/* ------------------------------------------------------------------ tabs */
function showView(name) {
  const tabs = $$(".tab");
  const idx = tabs.findIndex((t) => t.dataset.view === name);
  tabs.forEach((t, i) => t.classList.toggle("active", i === idx));
  $("#tabIndicator").style.transform = `translateX(${idx * 100}%)`;
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
}
$$(".tab").forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));

/* ------------------------------------------------------------------ samples */
let SAMPLES = [];
async function loadSamples() {
  try {
    SAMPLES = await (await fetch("samples/manifest.json")).json();
  } catch { SAMPLES = []; }
}
async function sampleBlob(item) {
  const blob = await (await fetch(`samples/${item.file}`)).blob();
  return new File([blob], item.file, { type: blob.type || "image/jpeg" });
}

/* ------------------------------------------------------------------ PATIENT view */
const patient = { file: null, lang: null };

function setPatientFile(file) {
  patient.file = file;
  $("#patientImg").src = URL.createObjectURL(file);
  $("#patientDrop").classList.add("hidden");
  $("#patientPreview").classList.remove("hidden");
  $("#patientRun").disabled = false;
}

function initPatient() {
  patient.lang = langPicker($("#patientLangs"));
  makeDropzone($("#patientDrop"), $("#patientInput"), (files) => files[0] && setPatientFile(files[0]));
  $("#patientPreview").addEventListener("click", () => {
    if (!$("#patientRun").classList.contains("loading")) $("#patientInput").click();
  });

  // one sample per class
  const picks = ["glioma", "meningioma", "notumor", "pituitary"].map((c) => SAMPLES.find((s) => s.label === c)).filter(Boolean);
  $("#patientSamples").innerHTML = picks.map((s, i) =>
    `<button class="sample" data-i="${i}" title="Sample ${i + 1}"><img src="samples/${esc(s.file)}" alt="Sample MRI ${i + 1}" loading="lazy"/></button>`).join("");
  $("#patientSamples").addEventListener("click", async (e) => {
    const b = e.target.closest(".sample");
    if (!b) return;
    $$(".sample", $("#patientSamples")).forEach((x) => x.classList.toggle("active", x === b));
    setPatientFile(await sampleBlob(picks[Number(b.dataset.i)]));
  });

  $("#patientRun").addEventListener("click", runPatient);
}

async function runPatient() {
  if (!patient.file) return;
  const btn = $("#patientRun"), steps = $("#patientSteps"), out = $("#patientResult");
  const mark = (i, state) => { const li = $(`li[data-step="${i}"]`, steps); li.classList.remove("doing", "done"); if (state) li.classList.add(state); };
  setLoading(btn, true, "Analyzing…");
  steps.classList.remove("hidden");
  [0, 1, 2].forEach((i) => mark(i, null));
  $("#patientPreview").classList.add("scanning");
  try {
    mark(0, "doing");
    const r = await analyze(patient.file);
    mark(0, "done"); mark(1, "doing");
    await sleep(450);
    $("#patientEmpty").classList.add("hidden");
    out.classList.remove("hidden");
    out.innerHTML = resultHTML(r) + `<p class="section-title">In simple words</p><div id="patientExplain">${skeleton}</div>`;
    activateResult(out);
    mark(1, "done"); mark(2, "doing");
    if (window.innerWidth < 880) out.scrollIntoView({ behavior: "smooth", block: "start" });
    const lang = patient.lang.info();
    const text = await explain(r, lang.name, "patient");
    $("#patientExplain").innerHTML = explainHTML(text, lang);
    mark(2, "done");
  } catch (e) {
    console.error(e);
    toast(`Something went wrong: ${e.message || e}`, true);
    [0, 1, 2].forEach((i) => mark(i, null));
  } finally {
    $("#patientPreview").classList.remove("scanning");
    setLoading(btn, false, "Explain my scan");
  }
}

/* ------------------------------------------------------------------ TRIAGE view */
const triage = { files: [], results: [] };

function setTriageFiles(files) {
  let list = files.filter((f) => f.type.startsWith("image/") || /\.(jpe?g|png)$/i.test(f.name));
  if (list.length > MAX_TRIAGE_FILES) {
    toast(`Only the first ${MAX_TRIAGE_FILES} scans will be used (demo limit).`);
    list = list.slice(0, MAX_TRIAGE_FILES);
  }
  triage.files = list;
  $("#triageCount").textContent = list.length ? `📁 ${list.length} scan${list.length > 1 ? "s" : ""} ready` : "";
  $("#triageRun").disabled = !list.length;
}

function sortResults(results) {
  return [...results].sort((a, b) =>
    (b.r.p_tumor >= 0.5) - (a.r.p_tumor >= 0.5) || b.r.refer - a.r.refer || b.r.p_tumor - a.r.p_tumor);
}

function renderQueue() {
  const sorted = sortResults(triage.results);
  $("#queue").innerHTML = sorted.map((x, i) => `
    <button class="queue-item ${x.r.priority}" data-id="${x.id}" style="animation-delay:${i * 40}ms">
      <span class="rank">#${i + 1}</span>
      <img src="${x.thumb}" alt="" />
      <div>
        <div class="name">${esc(x.name)}</div>
        <div class="meta">AI: ${esc(x.r.display_name)} · ${pct(x.r.confidence)} confident</div>
        <div class="mini-bar" title="P(tumor) ${pct(x.r.p_tumor)}"><span style="width:${Math.max(x.r.p_tumor * 100, 2)}%"></span></div>
      </div>
      <div class="right">
        <span class="badge ${x.r.priority}">${PRIORITY_ICON[x.r.priority]} ${x.r.priority}</span>
        ${x.r.refer ? `<span class="badge review">⚠️ Review</span>` : ""}
      </div>
    </button>`).join("");
  countUp($("#statTotal"), triage.results.length);
  countUp($("#statHigh"), triage.results.filter((x) => x.r.priority === "High").length);
  countUp($("#statReview"), triage.results.filter((x) => x.r.refer).length);
}

async function runTriage() {
  const btn = $("#triageRun"), files = triage.files;
  if (!files.length) return;
  setLoading(btn, true, "Analyzing…");
  $("#triageSamples").disabled = true;
  triage.results = [];
  $("#queue").innerHTML = "";
  const prog = $("#triageProgress");
  prog.classList.remove("hidden");
  let done = 0;
  const update = () => {
    $("#triageFill").style.width = `${(done / files.length) * 100}%`;
    $("#triageLabel").textContent = done < files.length ? `Analyzing scan ${Math.min(done + 1, files.length)} of ${files.length}…` : `✅ All ${files.length} scans sorted`;
  };
  update();
  let next = 0, failed = 0;
  const worker = async () => {
    while (next < files.length) {
      const i = next++;
      try {
        const r = await analyze(files[i]);
        triage.results.push({ id: i, name: files[i].name, thumb: URL.createObjectURL(files[i]), r });
        renderQueue();
      } catch (e) {
        console.error(e); failed++;
      }
      done++; update();
    }
  };
  try {
    await connect();
    await Promise.all([worker(), worker()]);   // 2 scans at a time
    if (failed) toast(`${failed} scan(s) could not be analyzed.`, true);
  } catch (e) {
    toast(`Could not reach the AI: ${e.message || e}`, true);
  } finally {
    setLoading(btn, false, "Sort by priority");
    $("#triageSamples").disabled = false;
    setTimeout(() => prog.classList.add("hidden"), 2500);
  }
}

function openDrawer(x) {
  const sorted = sortResults(triage.results);
  const rank = sorted.findIndex((y) => y.id === x.id) + 1;
  $("#drawerRank").innerHTML = `<b>${rank}</b> ${esc(x.name)}`;
  const body = $("#drawerBody");
  body.innerHTML = resultHTML(x.r) + `
    <p class="section-title">📝 Draft note for the doctor</p>
    <div class="note-box" id="drawerNote">${skeleton}</div>
    <p class="section-title">💙 Explain to the patient</p>
    <div class="lang-picker" id="drawerLangs"></div>
    <div class="actions"><button class="btn btn-primary" id="drawerExplain"><span class="btn-icon">💬</span><span class="spinner"></span>Make patient explanation</button></div>
    <div id="drawerPatient"></div>`;
  activateResult(body);
  const picker = langPicker($("#drawerLangs"));
  $("#drawer").classList.add("open");
  $("#backdrop").classList.add("open");
  $("#drawer").setAttribute("aria-hidden", "false");

  explain(x.r, "English", "doctor_note")
    .then((t) => { if ($("#drawerNote")) $("#drawerNote").innerHTML = md(t); })
    .catch((e) => { if ($("#drawerNote")) $("#drawerNote").textContent = `Could not load note: ${e.message || e}`; });

  $("#drawerExplain").addEventListener("click", async () => {
    const b = $("#drawerExplain");
    setLoading(b, true, "Writing…");
    $("#drawerPatient").innerHTML = skeleton;
    try {
      const lang = picker.info();
      const t = await explain(x.r, lang.name, "patient");
      $("#drawerPatient").innerHTML = explainHTML(t, lang);
    } catch (e) {
      $("#drawerPatient").innerHTML = "";
      toast(`Could not make the explanation: ${e.message || e}`, true);
    } finally {
      setLoading(b, false, "Make patient explanation");
    }
  });
}

function closeDrawer() {
  $("#drawer").classList.remove("open");
  $("#backdrop").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
}

function initTriage() {
  makeDropzone($("#triageDrop"), $("#triageInput"), setTriageFiles);
  $("#triageRun").addEventListener("click", runTriage);
  $("#triageSamples").addEventListener("click", async () => {
    const b = $("#triageSamples");
    setLoading(b, true, "Loading…");
    const picks = [...SAMPLES].sort(() => Math.random() - 0.5).slice(0, 12);
    setTriageFiles(await Promise.all(picks.map(sampleBlob)));
    setLoading(b, false, "Try 12 sample scans");
    toast("12 sample scans loaded — now click “Sort by priority”");
  });
  $("#queue").addEventListener("click", (e) => {
    const item = e.target.closest(".queue-item");
    if (item) openDrawer(triage.results.find((x) => x.id === Number(item.dataset.id)));
  });
  $("#drawerClose").addEventListener("click", closeDrawer);
  $("#backdrop").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => e.key === "Escape" && closeDrawer());
}

/* ------------------------------------------------------------------ start */
$("#status").addEventListener("click", () => { if (!client) connect().catch(() => {}); });
await loadSamples();
initTriage();
initPatient();
connect().then(() => patient.lang?.refresh()).catch((e) => console.error("connect failed", e));
