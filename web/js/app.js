import { Client, handle_file } from "./vendor/gradio-client.js";
import { marked } from "./vendor/marked.esm.js";
import { BACKEND, LANGUAGES, MAX_TRIAGE_FILES } from "./config.js";

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

function langChips(container, initial = "English") {
  container.innerHTML = LANGUAGES.map((l) =>
    `<button class="chip ${l.value === initial ? "active" : ""}" data-lang="${esc(l.value)}">${esc(l.label)}</button>`).join("");
  container.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    $$(".chip", container).forEach((c) => c.classList.toggle("active", c === chip));
  });
  return () => $(".chip.active", container)?.dataset.lang || "English";
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
const patient = { file: null, getLang: () => "English" };

function setPatientFile(file) {
  patient.file = file;
  $("#patientImg").src = URL.createObjectURL(file);
  $("#patientDrop").classList.add("hidden");
  $("#patientPreview").classList.remove("hidden");
  $("#patientRun").disabled = false;
}

function initPatient() {
  patient.getLang = langChips($("#patientLangs"));
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
    const text = await explain(r, patient.getLang(), "patient");
    $("#patientExplain").innerHTML = `<div class="explain">${md(text)}</div>`;
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
    <div class="chips" id="drawerLangs"></div>
    <div class="actions"><button class="btn btn-primary" id="drawerExplain"><span class="btn-icon">💬</span><span class="spinner"></span>Make patient explanation</button></div>
    <div id="drawerPatient"></div>`;
  activateResult(body);
  const getLang = langChips($("#drawerLangs"));
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
      const t = await explain(x.r, getLang(), "patient");
      $("#drawerPatient").innerHTML = `<div class="explain">${md(t)}</div>`;
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
connect().catch((e) => console.error("connect failed", e));
