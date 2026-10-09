const state = { file: null, dataUrl: null, lastCase: null, metrics: null, models: [], chartMode: "scores" };
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const safe = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
const percent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
const niceDate = (value) => {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
};
const modelName = (id) => state.models.find((model) => model.id === id)?.name || id;
const classPill = (name) => `<span class="class-pill ${safe(name)}"><i class="class-dot"></i>${safe(name)}</span>`;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const type = response.headers.get("content-type") || "";
  if (!response.ok) {
    const body = type.includes("json") ? await response.json() : { error: `Request failed (${response.status})` };
    throw new Error(body.error || "Request failed");
  }
  return type.includes("json") ? response.json() : response;
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.remove("show"), 2600);
}

function navigate(page) {
  if (!$("#page-" + page)) return;
  $$(".page").forEach((node) => node.classList.toggle("active", node.id === `page-${page}`));
  $$(".nav-item").forEach((node) => node.classList.toggle("active", node.dataset.page === page));
  const title = { overview: "Overview", analysis: "Image analysis", history: "Case history", performance: "Model performance" }[page];
  $("#crumb-current").textContent = title;
  history.replaceState(null, "", `#${page}`);
  if (page === "history") loadHistory();
  if (page === "performance") loadMetrics();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderModels(models) {
  state.models = models;
  const available = models.filter((model) => model.available);
  $("#stat-models").textContent = `${available.length}/${models.length}`;
  $("#model-list").innerHTML = models.map((model) => `
    <div class="model-list-row"><span class="model-indicator">◈</span><div class="model-details"><strong>${safe(model.name)}</strong><small>${safe(model.checkpoint)}</small></div><span class="${model.available ? "available-chip" : "unavailable-chip"}">${model.available ? "READY" : "MISSING"}</span></div>`).join("");
}

function caseRow(item, history = false) {
  const title = safe(item.original_name);
  const id = safe(item.id);
  const date = niceDate(item.created_at);
  const model = safe(modelName(item.primary_model));
  const confidence = percent(item.confidence);
  if (history) return `<tr data-case="${id}"><td><span class="case-id">${id.slice(0, 8).toUpperCase()}</span></td><td><div class="image-cell"><img class="case-thumb" src="/media/${id}/image" alt=""><div><strong>${title}</strong><small>Saved locally</small></div></div></td><td>${classPill(item.predicted_class)}</td><td>${model}</td><td><div class="confidence-cell"><div class="mini-progress"><i style="width:${confidence}"></i></div><strong>${confidence}</strong></div></td><td>${safe(date)}</td><td><button class="row-action" data-open="${id}">Open →</button></td></tr>`;
  return `<tr data-case="${id}"><td><span class="case-id">${id.slice(0, 8).toUpperCase()}</span></td><td><div class="image-cell"><img class="case-thumb" src="/media/${id}/image" alt=""><div><strong>${title}</strong><small>${model}</small></div></div></td><td>${classPill(item.predicted_class)}</td><td><div class="confidence-cell"><div class="mini-progress"><i style="width:${confidence}"></i></div><strong>${confidence}</strong></div></td><td>${safe(date)}</td><td><button class="row-action" data-open="${id}">Open →</button></td></tr>`;
}

async function loadHistory() {
  const query = encodeURIComponent($("#history-search")?.value || "");
  const data = await api(`/api/history?q=${query}`);
  const cases = data.cases || [];
  $("#history-count").textContent = cases.length;
  $("#history-total").textContent = `${cases.length} saved case${cases.length === 1 ? "" : "s"}`;
  $("#history-cases").innerHTML = cases.length ? cases.map((item) => caseRow(item, true)).join("") : '<tr><td colspan="7" class="empty-cell">No matching cases yet. Analyze an image to create your first case.</td></tr>';
  $("#stat-total").textContent = cases.length;
  const recent = cases.slice(0, 5);
  $("#recent-cases").innerHTML = recent.length ? recent.map((item) => caseRow(item)).join("") : '<tr><td colspan="6" class="empty-cell">No analyses yet. Upload a dermoscopic image to get started.</td></tr>';
}

function renderRecordedExperiments(data) {
  const values = Object.entries(data || {});
  $("#experiment-cards").innerHTML = values.slice(0, 3).map(([name, metrics]) => `
    <article class="experiment-card"><span class="experiment-kappa">κ ${Number(metrics.Kappa || 0).toFixed(3)}</span><div class="experiment-label">${safe(name.replaceAll("_", " · "))}</div><strong>${percent(metrics.accuracy)}</strong><small>Recorded validation accuracy</small></article>`).join("") || '<article class="experiment-card">No saved experiment metrics found.</article>';
  $("#recorded-experiments").innerHTML = values.map(([name, metrics]) => `
    <article class="recorded-row"><strong>${safe(name)}</strong><small>Previously recorded result · project notebook</small><div class="recorded-values"><span>Accuracy <b>${percent(metrics.accuracy)}</b></span>${metrics.Kappa === undefined ? "" : `<span>Kappa <b>${Number(metrics.Kappa).toFixed(3)}</b></span>`}</div></article>`).join("");
}

function renderResearchRuns(runs) {
  const order = ["plain_ce", "attention_ce", "attention_focal"];
  const found = new Map((runs || []).map((run) => [run.experiment, run]));
  $("#research-run-count").textContent = `${found.size}/3 runs`;
  if (found.size === 3) {
    $("#research-status").textContent = "3 CONTROLLED RUNS";
    $("#research-status").classList.add("complete");
  }
  $("#research-runs").innerHTML = order.map((name) => {
    const run = found.get(name);
    const label = { plain_ce: "EfficientNet · Cross-entropy", attention_ce: "Feature attention · Cross-entropy", attention_focal: "Feature attention · Focal loss" }[name];
    if (!run) return `<article class="research-run pending"><span class="run-status-dot"></span><div><strong>${label}</strong><small>Waiting for a trained checkpoint</small></div><span class="run-state">PENDING</span></article>`;
    const m = run.metrics;
    return `<article class="research-run"><span class="run-status-dot done"></span><div><strong>${label}</strong><small>Selected epoch ${safe(run.best_epoch)} · held-out tuning split</small></div><span class="run-metric">SCC recall <b>${percent(m.scc_recall)}</b></span><span class="run-metric">Macro F1 <b>${percent(m.macro_f1)}</b></span></article>`;
  }).join("");
}

function setSelectedFile(file) {
  const error = $("#upload-error");
  error.classList.add("hidden");
  if (!file) return;
  if (!/^image\/(png|jpeg|webp)$/.test(file.type)) return showUploadError("Please choose a PNG, JPG or WebP image.");
  if (file.size > 12 * 1024 * 1024) return showUploadError("This image is larger than 12 MB.");
  const reader = new FileReader();
  reader.onload = () => {
    state.file = file;
    state.dataUrl = reader.result;
    $("#preview-image").src = state.dataUrl;
    $("#file-name").textContent = file.name;
    $("#file-size").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
    $("#preview-wrap").classList.remove("hidden");
    $("#dropzone").classList.add("hidden");
    $("#run-analysis").disabled = false;
  };
  reader.onerror = () => showUploadError("The selected image could not be read.");
  reader.readAsDataURL(file);
}

function showUploadError(message) {
  const node = $("#upload-error");
  node.textContent = message;
  node.classList.remove("hidden");
}

function clearFile() {
  state.file = null;
  state.dataUrl = null;
  $("#file-input").value = "";
  $("#preview-wrap").classList.add("hidden");
  $("#dropzone").classList.remove("hidden");
  $("#run-analysis").disabled = true;
  $("#upload-error").classList.add("hidden");
}

function renderAnalysis(result) {
  state.lastCase = result;
  const selected = result.results[result.primary_model] || Object.values(result.results)[0];
  $("#result-placeholder").classList.add("hidden");
  $("#result-content").classList.remove("hidden");
  $("#prediction-class").textContent = selected.predicted_class;
  $("#prediction-confidence").textContent = `${percent(selected.confidence)} confidence`;
  $("#prediction-dot").style.color = selected.predicted_class === "MEL" ? "#bc706b" : selected.predicted_class === "SCC" ? "#bd985b" : "#7799b4";
  $("#prediction-dot").style.background = "currentColor";
  $(".prediction-label").textContent = `${safe(selected.name).toUpperCase()} PREDICTION`;
  $("#probability-bars").innerHTML = Object.entries(selected.probabilities).map(([name, value]) => `
    <div class="probability-row ${name === selected.predicted_class ? "primary" : ""}"><span>${safe(name)}</span><div class="probability-track"><i style="width:${percent(value)}"></i></div><strong>${percent(value)}</strong></div>`).join("");
  $("#comparison-list").innerHTML = Object.entries(result.results).map(([id, item]) => `
    <div class="comparison-row"><strong>${safe(item.name)}</strong>${classPill(item.predicted_class)}<small>top score</small><span class="comparison-confidence">${percent(item.confidence)}</span></div>`).join("");
  $("#heatmap-placeholder").classList.add("hidden");
  $("#heatmap-view").classList.remove("hidden");
  $("#original-result-image").src = result.image_url;
  $("#heatmap-image").src = result.heatmap_url;
  $("#history-count").textContent = Number($("#history-count").textContent || 0) + 1;
  $("#stat-total").textContent = $("#history-count").textContent;
  $("#download-report").onclick = () => { window.location.href = result.report_url; };
}

async function runAnalysis() {
  if (!state.file || !state.dataUrl) return;
  const button = $("#run-analysis");
  const previous = button.innerHTML;
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span> Running model inference…';
  $("#upload-error").classList.add("hidden");
  try {
    const result = await api("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: state.file.name, image: state.dataUrl }),
    });
    renderAnalysis(result);
    toast("Analysis complete · case saved locally");
    loadHistory();
  } catch (error) {
    showUploadError(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = previous;
  }
}

function renderMetrics(data) {
  state.metrics = data;
  const models = Object.entries(data.models || {});
  $("#metric-status").classList.add("ready");
  $("#metric-status").classList.remove("error");
  $("#metric-status").innerHTML = `<i></i> ${Number(data.validation_images).toLocaleString()} validation images`;
  $("#metric-source p").textContent = `${data.source}. Metrics include accuracy, macro precision/recall/F1, Cohen’s kappa and confusion matrices.`;
  $("#metric-card-grid").innerHTML = models.map(([id, metrics]) => `
    <article class="metric-card"><div class="metric-card-head">${safe(metrics.name)}<span>Validation split</span></div><div class="metric-primary">${percent(metrics.accuracy)}</div><div class="metric-secondary">Accuracy <b>·</b> Macro F1 ${percent(metrics.f1_macro)} <b>·</b> κ ${Number(metrics.kappa).toFixed(3)}</div></article>`).join("");
  const select = $("#confusion-select");
  select.innerHTML = models.map(([id, m]) => `<option value="${safe(id)}">${safe(m.name)}</option>`).join("");
  select.onchange = () => renderConfusion(select.value);
  renderCurve(state.chartMode);
  renderConfusion(models[0]?.[0]);
}

function renderCurve(mode = state.chartMode) {
  state.chartMode = mode;
  const tabs = $$("[data-chart]");
  tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.chart === mode));
  const chart = $("#score-chart");
  const legend = $("#score-legend");
  const entries = Object.entries(state.metrics?.models || {});
  if (mode === "scores") {
    $("#curve-subtitle").textContent = "Macro precision, recall and F1-score · per model";
    chart.classList.remove("roc-mode");
    legend.innerHTML = '<span><i class="legend-blue"></i> Precision</span><span><i class="legend-green"></i> Recall</span><span><i class="legend-amber"></i> F1-score</span>';
    chart.innerHTML = entries.map(([id, m]) => `<div class="chart-group" title="${safe(m.name)}">
      <i class="chart-bar bar-blue" data-value="${percent(m.precision_macro)}" style="height:${Math.max(2, m.precision_macro * 100)}%"></i>
      <i class="chart-bar bar-green" data-value="${percent(m.recall_macro)}" style="height:${Math.max(2, m.recall_macro * 100)}%"></i>
      <i class="chart-bar bar-amber" data-value="${percent(m.f1_macro)}" style="height:${Math.max(2, m.f1_macro * 100)}%"></i>
      <span class="chart-label">${safe(m.name.replace("EfficientNet-B0 + Feature Attention", "Attention").replace("EfficientNet-B0", "EffNet"))}</span></div>`).join("");
    return;
  }
  $("#curve-subtitle").textContent = "One-vs-rest ROC · choose a model in the confusion matrix selector";
  chart.classList.add("roc-mode");
  const modelId = $("#confusion-select").value || entries[0]?.[0];
  const metrics = state.metrics?.models?.[modelId];
  if (!metrics?.roc) {
    chart.innerHTML = '<div class="chart-loading">ROC data is not available.</div>';
    return;
  }
  const colors = ["#718fbc", "#72ac92", "#d1a257"];
  const paths = metrics.classes.map((name, index) => {
    const curve = metrics.roc[name];
    const points = curve.fpr.map((fpr, i) => `${24 + fpr * 300},${145 - curve.tpr[i] * 125}`).join(" ");
    return `<polyline points="${points}" fill="none" stroke="${colors[index]}" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>`;
  }).join("");
  chart.innerHTML = `<svg viewBox="0 0 350 175" role="img" aria-label="One-versus-rest ROC curves"><path d="M24 20V145H324" fill="none" stroke="#dfe8e3" stroke-width="1"/><path d="M24 145 324 20" stroke="#d3ddd8" stroke-width="1" stroke-dasharray="4 4"/><text x="24" y="162" fill="#9aa6a1" font-size="8">0</text><text x="315" y="162" fill="#9aa6a1" font-size="8">1 − Specificity</text><text x="4" y="25" fill="#9aa6a1" font-size="8">Sensitivity</text>${paths}</svg>`;
  legend.innerHTML = metrics.classes.map((name, index) => `<span><i style="background:${colors[index]}"></i>${safe(name)} · AUC ${Number(metrics.roc[name].auc).toFixed(3)}</span>`).join("");
}

function renderConfusion(id) {
  const metrics = state.metrics?.models?.[id];
  if (!metrics) return;
  $("#confusion-model-name").textContent = metrics.name;
  const max = Math.max(1, ...metrics.confusion_matrix.flat());
  const heads = metrics.classes.map((name) => `<span class="matrix-head">${safe(name)}</span>`).join("");
  const rows = metrics.confusion_matrix.map((row, r) => `<span class="matrix-row-head">${safe(metrics.classes[r])}</span>${row.map((value, c) => {
    const intensity = .08 + .35 * (value / max);
    const color = r === c ? `background:rgba(91,164,128,${intensity + .22})` : `background:rgba(170,190,183,${intensity})`;
    return `<span class="matrix-cell ${r === c ? "diagonal" : ""}" style="${color}" title="Actual ${safe(metrics.classes[r])} predicted ${safe(metrics.classes[c])}">${Number(value).toLocaleString()}</span>`;
  }).join("")}`).join("");
  $("#confusion-matrix").innerHTML = `<span class="matrix-corner">ACTUAL ↓<br>PRED. →</span>${heads}${rows}`;
  if (state.chartMode === "roc") renderCurve("roc");
}

async function loadMetrics() {
  if (state.metrics) return;
  $("#metric-status").classList.remove("ready", "error");
  $("#metric-status").innerHTML = '<i></i> Recomputing on validation data…';
  try {
    renderMetrics(await api("/api/metrics"));
  } catch (error) {
    $("#metric-status").classList.add("error");
    $("#metric-status").textContent = `Metrics unavailable: ${error.message}`;
    $("#metric-card-grid").innerHTML = `<div class="clinical-notice">${safe(error.message)}</div>`;
    $("#score-chart").innerHTML = '<div class="chart-loading">Validation metrics could not be calculated.</div>';
  }
}

async function openCase(id) {
  try {
    const item = await api(`/api/cases/${encodeURIComponent(id)}`);
    const probabilities = item.probabilities;
    const selected = item.results[item.primary_model];
    $("#modal-content").innerHTML = `
      <div class="eyebrow">SAVED CASE · ${safe(item.id.slice(0, 8).toUpperCase())}</div><h2 class="modal-title">${safe(item.original_name)}</h2><p class="modal-meta">Analyzed ${safe(niceDate(item.created_at))} · ${safe(selected.name)}</p>
      <div class="modal-grid"><figure><img src="${safe(item.image_url)}" alt="Uploaded image"><figcaption class="modal-meta">Original image</figcaption></figure><figure><img src="${safe(item.heatmap_url)}" alt="Grad-CAM explanation"><figcaption class="modal-meta">Grad-CAM overlay</figcaption></figure></div>
      <div class="prediction-banner"><div><span class="prediction-label">MODEL PREDICTION</span><div class="prediction-class"><strong>${safe(item.predicted_class)}</strong><span class="confidence-chip">${percent(item.confidence)} confidence</span></div><small>Not a diagnosis</small></div></div>
      <div class="modal-bars">${Object.entries(probabilities).map(([name, value]) => `<div class="probability-row"><span>${safe(name)}</span><div class="probability-track"><i style="width:${percent(value)}"></i></div><strong>${percent(value)}</strong></div>`).join("")}</div>
      <div class="comparison-block"><div class="subheading"><strong>Model comparison</strong></div>${Object.entries(item.results).map(([, result]) => `<div class="comparison-row"><strong>${safe(result.name)}</strong>${classPill(result.predicted_class)}<small>top score</small><span class="comparison-confidence">${percent(result.confidence)}</span></div>`).join("")}</div>
      <div class="modal-actions"><a class="button button-secondary" href="${safe(item.report_url)}">↓ Download PDF report</a></div>`;
    $("#case-modal").classList.remove("hidden");
  } catch (error) { toast(error.message); }
}

function setup() {
  $$("[data-page]").forEach((node) => node.addEventListener("click", () => navigate(node.dataset.page)));
  $$('[data-go]').forEach((node) => node.addEventListener("click", () => navigate(node.dataset.go)));
  $("#dropzone").addEventListener("click", () => $("#file-input").click());
  $("#dropzone").addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") $("#file-input").click(); });
  $("#file-input").addEventListener("change", (event) => setSelectedFile(event.target.files?.[0]));
  $("#dropzone").addEventListener("dragover", (event) => { event.preventDefault(); $("#dropzone").classList.add("dragging"); });
  $("#dropzone").addEventListener("dragleave", () => $("#dropzone").classList.remove("dragging"));
  $("#dropzone").addEventListener("drop", (event) => { event.preventDefault(); $("#dropzone").classList.remove("dragging"); setSelectedFile(event.dataTransfer.files?.[0]); });
  $("#remove-image").addEventListener("click", clearFile);
  $("#run-analysis").addEventListener("click", runAnalysis);
  $("#history-search").addEventListener("input", () => { clearTimeout(setup.searchTimer); setup.searchTimer = setTimeout(loadHistory, 240); });
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-open]");
    const row = event.target.closest("tr[data-case]");
    if (button) openCase(button.dataset.open);
    else if (row && !event.target.closest("button")) openCase(row.dataset.case);
  });
  $("#modal-close").addEventListener("click", () => $("#case-modal").classList.add("hidden"));
  $$("[data-chart]").forEach((tab) => tab.addEventListener("click", () => renderCurve(tab.dataset.chart)));
  $("#case-modal").addEventListener("click", (event) => { if (event.target.id === "case-modal") $("#case-modal").classList.add("hidden"); });
  $("#help-button").addEventListener("click", () => toast("Research demo · predictions are not medical advice."));
  window.addEventListener("hashchange", () => navigate(location.hash.slice(1) || "overview"));
  const startPage = location.hash.slice(1);
  if (startPage && $("#page-" + startPage)) navigate(startPage);
  Promise.all([api("/api/health"), api("/api/experiments"), loadHistory()]).then(([health, experiments]) => {
    renderModels(health.models || []);
    const device = String(health.device || "cpu");
    $("#device-label").textContent = device.includes("cuda") ? "CUDA enabled · GPU" : "CPU inference";
    $("#stat-device").textContent = device.includes("cuda") ? "CUDA GPU" : "CPU";
    renderRecordedExperiments(experiments.recorded);
    renderResearchRuns(experiments.research_runs);
  }).catch((error) => {
    $("#device-label").textContent = "Backend unavailable";
    toast(`Dashboard backend error: ${error.message}`);
  });
}

document.addEventListener("DOMContentLoaded", setup);
