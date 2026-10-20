/**
 * NeuroScan AI - Frontend Application Core Logic
 * Handles interactive MRI pipeline, Chart.js visualizations, In-app Python Sandbox, and Clinical Report Generator.
 */

// Global Application State
const AppState = {
  currentFile: null,
  currentSampleName: "sample_glioma_case1.png",
  selectedModel: "svm",
  testSplit: 0.20,
  latestPreprocessData: null,
  latestFeaturesData: null,
  latestTrainingData: null,
  latestDiagnosisData: null,
  templates: {},
  charts: {
    radar: null,
    roc: null,
    benchmark: null
  }
};

// DOM Elements
const elements = {
  navTabs: document.querySelectorAll(".nav-tab-btn"),
  tabContents: document.querySelectorAll(".tab-content"),
  stageBtns: document.querySelectorAll(".stage-btn"),
  stageViews: document.querySelectorAll(".pipeline-view-section"),
  uploadDropzone: document.getElementById("uploadDropzone"),
  mriFileInput: document.getElementById("mriFileInput"),
  samplePillsContainer: document.getElementById("samplePillsContainer"),
  selectedImagePreview: document.getElementById("selectedImagePreview"),
  selectedImageName: document.getElementById("selectedImageName"),
  runPipelineBtn: document.getElementById("runPipelineBtn"),
  
  // Preprocess Elements
  mriVisualGrid: document.getElementById("mriVisualGrid"),
  prepMetaDimensions: document.getElementById("prepMetaDimensions"),
  prepMetaPixels: document.getElementById("prepMetaPixels"),
  prepMetaIntensity: document.getElementById("prepMetaIntensity"),
  prepTumorCandidateStatus: document.getElementById("prepTumorCandidateStatus"),

  // Features Elements
  glcmTableBody: document.getElementById("glcmTableBody"),
  statsTableBody: document.getElementById("statsTableBody"),
  morphTableBody: document.getElementById("morphTableBody"),

  // Training Elements
  modelSelector: document.getElementById("modelSelector"),
  testSplitSlider: document.getElementById("testSplitSlider"),
  testSplitLabel: document.getElementById("testSplitLabel"),
  trainBtn: document.getElementById("trainBtn"),
  metricAccuracy: document.getElementById("metricAccuracy"),
  metricF1: document.getElementById("metricF1"),
  metricPrecision: document.getElementById("metricPrecision"),
  metricRecall: document.getElementById("metricRecall"),
  metricLatency: document.getElementById("metricLatency"),
  benchmarksTableBody: document.getElementById("benchmarksTableBody"),

  // Confusion Matrix
  cmTP: document.getElementById("cmTP"),
  cmTN: document.getElementById("cmTN"),
  cmFP: document.getElementById("cmFP"),
  cmFN: document.getElementById("cmFN"),

  // Diagnosis Banner
  diagnosisBanner: document.getElementById("diagnosisBanner"),
  diagnosisStatusText: document.getElementById("diagnosisStatusText"),
  diagnosisConfidenceText: document.getElementById("diagnosisConfidenceText"),
  diagnosisSubtypeText: document.getElementById("diagnosisSubtypeText"),
  diagnosisOverlayImg: document.getElementById("diagnosisOverlayImg"),

  // Code Sandbox
  templateSelector: document.getElementById("templateSelector"),
  codeEditor: document.getElementById("codeEditor"),
  runCodeBtn: document.getElementById("runCodeBtn"),
  resetCodeBtn: document.getElementById("resetCodeBtn"),
  terminalBody: document.getElementById("terminalBody"),
  terminalLatency: document.getElementById("terminalLatency"),
  terminalPlotsArea: document.getElementById("terminalPlotsArea"),

  // Medical Report Modal
  reportModal: document.getElementById("reportModal"),
  openReportBtn: document.getElementById("openReportBtn"),
  closeReportBtn: document.getElementById("closeReportBtn"),
  printReportBtn: document.getElementById("printReportBtn")
};

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initWorkflowStages();
  initFileUpload();
  initTrainingControls();
  initCodeSandbox();
  initCopyButtons();
  initReportModal();
  
  // Load sample gallery and templates
  loadSamplesGallery();
  loadCodeTemplates();

  // Run initial diagnostic cycle on default sample
  setTimeout(() => {
    executeFullDiagnosticPipeline();
  }, 300);
});

// Navigation Handling
function initNavigation() {
  elements.navTabs.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      elements.navTabs.forEach(b => b.classList.remove("active"));
      elements.tabContents.forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(targetId);
      if (target) target.classList.add("active");
    });
  });
}

// Workflow Stage Switching
function initWorkflowStages() {
  elements.stageBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const stageId = btn.getAttribute("data-stage");
      elements.stageBtns.forEach(b => b.classList.remove("active"));
      elements.stageViews.forEach(v => v.classList.remove("active"));
      btn.classList.add("active");
      const view = document.getElementById(stageId);
      if (view) view.classList.add("active");
    });
  });
}

// File Upload & Drag-and-Drop
function initFileUpload() {
  const dropzone = elements.uploadDropzone;
  const input = elements.mriFileInput;

  dropzone.addEventListener("click", () => input.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  input.addEventListener("change", () => {
    if (input.files.length > 0) {
      handleFileSelected(input.files[0]);
    }
  });

  if (elements.runPipelineBtn) {
    elements.runPipelineBtn.addEventListener("click", () => {
      executeFullDiagnosticPipeline();
    });
  }
}

function handleFileSelected(file) {
  AppState.currentFile = file;
  AppState.currentSampleName = null;
  
  // Uncheck sample pills
  document.querySelectorAll(".sample-pill").forEach(p => p.classList.remove("active"));
  
  // Preview
  const reader = new FileReader();
  reader.onload = (e) => {
    elements.selectedImagePreview.src = e.target.result;
    elements.selectedImageName.innerText = file.name;
    executeFullDiagnosticPipeline();
  };
  reader.readAsDataURL(file);
}

// Fetch Sample MRI Gallery
async function loadSamplesGallery() {
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    if (data.samples && elements.samplePillsContainer) {
      elements.samplePillsContainer.innerHTML = "";
      data.samples.forEach((s, idx) => {
        const pill = document.createElement("div");
        pill.className = `sample-pill ${idx === 0 ? "active" : ""}`;
        pill.innerHTML = `
          <img src="${s.thumbnail}" class="sample-thumb" alt="${s.title}">
          <div class="sample-info">
            <span class="sample-name">${s.title}</span>
            <span class="sample-type">${s.filename}</span>
          </div>
        `;
        pill.addEventListener("click", () => {
          document.querySelectorAll(".sample-pill").forEach(p => p.classList.remove("active"));
          pill.classList.add("active");
          AppState.currentFile = null;
          AppState.currentSampleName = s.filename;
          elements.selectedImagePreview.src = s.thumbnail;
          elements.selectedImageName.innerText = s.title;
          executeFullDiagnosticPipeline();
        });
        elements.samplePillsContainer.appendChild(pill);
      });
    }
  } catch (err) {
    console.error("Failed to load samples gallery:", err);
  }
}

// Master Diagnostic Pipeline Execution
async function executeFullDiagnosticPipeline() {
  showLoadingNotification("Running End-to-End Diagnostic Pipeline...");
  
  const formData = new FormData();
  if (AppState.currentFile) {
    formData.append("file", AppState.currentFile);
  } else {
    formData.append("sample_name", AppState.currentSampleName || "sample_glioma_case1.png");
  }
  formData.append("model", AppState.selectedModel);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData
    });
    const result = await res.json();
    if (!result.success) {
      alert("Pipeline Error: " + (result.error || "Unknown"));
      return;
    }

    AppState.latestDiagnosisData = result.diagnosis;
    AppState.latestPreprocessData = result.preprocessing;
    AppState.latestFeaturesData = result.features;

    // Render all modules
    renderPreprocessingStages(result.preprocessing);
    renderExtractedFeatures(result.features.detailed);
    renderDiagnosisBanner(result.diagnosis, result.preprocessing.stages.tumor_overlay);
    
    // Also trigger training & evaluation refresh
    triggerTraining();

  } catch (err) {
    console.error("Pipeline Execution failed:", err);
  }
}

// Render Preprocessing 7-Stage Visual Gallery
function renderPreprocessingStages(prepData) {
  const stages = prepData.stages;
  const meta = prepData.metadata;
  const candidate = prepData.tumor_candidate;

  if (elements.prepMetaDimensions) elements.prepMetaDimensions.innerText = meta.original_dimensions;
  if (elements.prepMetaPixels) elements.prepMetaPixels.innerText = `${meta.skull_stripped_pixels.toLocaleString()} px`;
  if (elements.prepMetaIntensity) elements.prepMetaIntensity.innerText = meta.mean_intensity;
  if (elements.prepTumorCandidateStatus) {
    elements.prepTumorCandidateStatus.innerHTML = candidate.has_candidate 
      ? `<span style="color: var(--accent-red); font-weight: 700;">Lesion Detected (${candidate.area_pixels} px, ${candidate.roi_brain_ratio_percent}%)</span>`
      : `<span style="color: var(--accent-green); font-weight: 700;">No Significant Hyper-intensity</span>`;
  }

  const stageDisplayNames = [
    { key: "original", title: "1. Raw MRI Scan", desc: "Native DICOM/JPEG input" },
    { key: "cropped", title: "2. Brain Contour Crop", desc: "Removed background padding" },
    { key: "grayscale_denoised", title: "3. Bilateral Denoise", desc: "Preserved anatomical edges" },
    { key: "skull_mask", title: "4. Skull Strip Mask", desc: "Otsu threshold & morphology" },
    { key: "skull_stripped", title: "5. Parenchyma Isolated", desc: "Extracranial tissue removed" },
    { key: "clahe_enhanced", title: "6. CLAHE Contrast", desc: "Adaptive histogram equalization" },
    { key: "edges_contours", title: "7. Canny Edge Map", desc: "Tissue boundary gradients" },
    { key: "tumor_overlay", title: "8. Segmented Tumor ROI", desc: "Candidate hyper-intensity mask" }
  ];

  if (elements.mriVisualGrid) {
    elements.mriVisualGrid.innerHTML = "";
    stageDisplayNames.forEach(item => {
      if (stages[item.key]) {
        const card = document.createElement("div");
        card.className = "mri-card";
        card.innerHTML = `
          <img src="${stages[item.key]}" class="mri-card-img" alt="${item.title}">
          <div class="mri-card-title">${item.title}</div>
          <div class="mri-card-desc">${item.desc}</div>
        `;
        elements.mriVisualGrid.appendChild(card);
      }
    });
  }
}

// Render GLCM & Statistical Features
function renderExtractedFeatures(detailed) {
  // 1. GLCM Texture Table
  if (elements.glcmTableBody && detailed.glcm_texture) {
    elements.glcmTableBody.innerHTML = detailed.glcm_texture.features.map(f => `
      <tr>
        <td><strong>${f.name}</strong><br><span class="ref-range">${f.description}</span></td>
        <td class="feature-val">${f.value}</td>
        <td><span class="ref-range">${f.reference_range}</span></td>
      </tr>
    `).join("");
  }

  // 2. Statistical Moments Table
  if (elements.statsTableBody && detailed.statistical_intensity) {
    elements.statsTableBody.innerHTML = detailed.statistical_intensity.features.map(f => `
      <tr>
        <td><strong>${f.name}</strong></td>
        <td class="feature-val">${f.value} ${f.unit}</td>
        <td><span class="ref-range">${f.reference_range}</span></td>
      </tr>
    `).join("");
  }

  // 3. Morphological Features Table
  if (elements.morphTableBody && detailed.morphological_shape) {
    elements.morphTableBody.innerHTML = detailed.morphological_shape.features.map(f => `
      <tr>
        <td><strong>${f.name}</strong></td>
        <td class="feature-val">${f.value}</td>
        <td><span class="ref-range">${f.reference_range}</span></td>
      </tr>
    `).join("");
  }

  // Render Radar Feature Chart
  renderRadarChart(detailed);
}

function renderRadarChart(detailed) {
  const ctx = document.getElementById("glcmRadarChart");
  if (!ctx || typeof Chart === "undefined") return;

  const glcmItems = detailed.glcm_texture ? detailed.glcm_texture.features : [];
  const labels = glcmItems.map(f => f.name);
  const currentValues = glcmItems.map(f => Math.min(100, (f.value / 1.5)));
  const normalBaseline = [25, 20, 85, 30, 90, 45];

  if (AppState.charts.radar) {
    AppState.charts.radar.destroy();
  }

  AppState.charts.radar = new Chart(ctx, {
    type: "radar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Current Scan Texture Profile",
          data: currentValues,
          backgroundColor: "rgba(0, 242, 254, 0.25)",
          borderColor: "#00f2fe",
          pointBackgroundColor: "#00f2fe",
          borderWidth: 2
        },
        {
          label: "Healthy Brain Baseline Reference",
          data: normalBaseline,
          backgroundColor: "rgba(0, 245, 160, 0.15)",
          borderColor: "#00f5a0",
          pointBackgroundColor: "#00f5a0",
          borderWidth: 1.5,
          borderDash: [4, 4]
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          angleLines: { color: "rgba(255, 255, 255, 0.1)" },
          grid: { color: "rgba(255, 255, 255, 0.08)" },
          pointLabels: { color: "#94a3b8", font: { size: 11, family: "Plus Jakarta Sans" } },
          ticks: { display: false }
        }
      },
      plugins: {
        legend: {
          labels: { color: "#f8fafc", font: { family: "Plus Jakarta Sans", size: 11 } }
        }
      }
    }
  });
}

// Diagnosis Banner & Bounding ROI Overlay
function renderDiagnosisBanner(diagnosis, overlayUrl) {
  if (!elements.diagnosisBanner) return;

  const isTumor = diagnosis.is_tumor_detected;
  elements.diagnosisBanner.className = `glass-panel ${isTumor ? "tumor-positive" : "tumor-negative"}`;
  
  if (elements.diagnosisStatusText) {
    elements.diagnosisStatusText.innerHTML = isTumor
      ? `<span style="color: var(--accent-red); font-size: 1.35rem; font-weight: 800;">🚨 BRAIN TUMOR DETECTED (POSITIVE)</span>`
      : `<span style="color: var(--accent-green); font-size: 1.35rem; font-weight: 800;">✅ HEALTHY BRAIN SCAN (NEGATIVE)</span>`;
  }

  if (elements.diagnosisConfidenceText) {
    elements.diagnosisConfidenceText.innerText = `${diagnosis.confidence_percentage}% Confidence (${diagnosis.model_used})`;
  }

  if (elements.diagnosisSubtypeText) {
    elements.diagnosisSubtypeText.innerText = `Lesion Subtype: ${diagnosis.tumor_subtype}`;
  }

  if (elements.diagnosisOverlayImg && overlayUrl) {
    elements.diagnosisOverlayImg.src = overlayUrl;
  }
}

// Training & Benchmarking Controls
function initTrainingControls() {
  if (elements.modelSelector) {
    elements.modelSelector.addEventListener("change", (e) => {
      AppState.selectedModel = e.target.value;
    });
  }

  if (elements.testSplitSlider && elements.testSplitLabel) {
    elements.testSplitSlider.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      AppState.testSplit = val;
      elements.testSplitLabel.innerText = `${Math.round(val * 100)}% Test Split`;
    });
  }

  if (elements.trainBtn) {
    elements.trainBtn.addEventListener("click", () => {
      triggerTraining();
    });
  }
}

async function triggerTraining() {
  if (elements.trainBtn) {
    elements.trainBtn.disabled = true;
    elements.trainBtn.innerHTML = `<i class="pulse-dot"></i> Training Models...`;
  }

  try {
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: AppState.selectedModel,
        test_split: AppState.testSplit
      })
    });
    const json = await res.json();
    if (json.success) {
      AppState.latestTrainingData = json.data;
      renderTrainingEvaluation(json.data);
    }
  } catch (err) {
    console.error("Training error:", err);
  } finally {
    if (elements.trainBtn) {
      elements.trainBtn.disabled = false;
      elements.trainBtn.innerHTML = `⚡ Train Model Live`;
    }
  }
}

function renderTrainingEvaluation(data) {
  const m = data.metrics;
  if (elements.metricAccuracy) elements.metricAccuracy.innerText = `${m.accuracy}%`;
  if (elements.metricF1) elements.metricF1.innerText = `${m.f1_score}%`;
  if (elements.metricPrecision) elements.metricPrecision.innerText = `${m.precision}%`;
  if (elements.metricRecall) elements.metricRecall.innerText = `${m.recall}%`;
  if (elements.metricLatency) elements.metricLatency.innerText = `${m.training_latency_ms} ms`;

  // Confusion Matrix Values
  const cm = data.confusion_matrix;
  if (elements.cmTP) elements.cmTP.innerText = cm.tp;
  if (elements.cmTN) elements.cmTN.innerText = cm.tn;
  if (elements.cmFP) elements.cmFP.innerText = cm.fp;
  if (elements.cmFN) elements.cmFN.innerText = cm.fn;

  // Benchmark Table
  if (elements.benchmarksTableBody && data.benchmark_rankings) {
    elements.benchmarksTableBody.innerHTML = data.benchmark_rankings.map(r => `
      <tr>
        <td><strong>#${r.rank}</strong></td>
        <td><strong>${r.name}</strong></td>
        <td><span class="feature-val">${r.accuracy}%</span></td>
        <td>${r.precision}%</td>
        <td>${r.recall}%</td>
        <td><strong style="color: var(--accent-green);">${r.f1_score}%</strong></td>
        <td>${r.latency_ms} ms</td>
        <td><span class="brand-tag">${r.badge}</span></td>
      </tr>
    `).join("");
  }

  // Render ROC Curve & Benchmark Charts
  renderRocCurveChart(data.roc_curve);
  renderBenchmarkChart(data.benchmark_rankings);
}

function renderRocCurveChart(roc) {
  const ctx = document.getElementById("rocCurveChart");
  if (!ctx || !roc || typeof Chart === "undefined") return;

  if (AppState.charts.roc) {
    AppState.charts.roc.destroy();
  }

  AppState.charts.roc = new Chart(ctx, {
    type: "line",
    data: {
      labels: roc.fpr,
      datasets: [
        {
          label: `ROC Curve (AUC = ${roc.auc})`,
          data: roc.tpr,
          borderColor: "#ff007f",
          backgroundColor: "rgba(255, 0, 127, 0.15)",
          fill: true,
          tension: 0.2,
          borderWidth: 2.5
        },
        {
          label: "Random Chance Line",
          data: roc.fpr,
          borderColor: "rgba(255, 255, 255, 0.3)",
          borderDash: [5, 5],
          fill: false,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: "False Positive Rate (1 - Specificity)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" }
        },
        y: {
          title: { display: true, text: "True Positive Rate (Sensitivity)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8" }
        }
      },
      plugins: {
        legend: { labels: { color: "#f8fafc" } }
      }
    }
  });
}

function renderBenchmarkChart(rankings) {
  const ctx = document.getElementById("benchmarkBarChart");
  if (!ctx || !rankings || typeof Chart === "undefined") return;

  if (AppState.charts.benchmark) {
    AppState.charts.benchmark.destroy();
  }

  const labels = rankings.map(r => r.name.replace(" Classifier", "").replace(" (SVM)", ""));
  const accuracies = rankings.map(r => r.accuracy);
  const f1Scores = rankings.map(r => r.f1_score);

  AppState.charts.benchmark = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Accuracy (%)",
          data: accuracies,
          backgroundColor: "rgba(0, 242, 254, 0.8)",
          borderRadius: 4
        },
        {
          label: "F1-Score (%)",
          data: f1Scores,
          backgroundColor: "rgba(0, 245, 160, 0.8)",
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { display: false } },
        y: { min: 80, max: 100, ticks: { color: "#94a3b8" }, grid: { color: "rgba(255, 255, 255, 0.05)" } }
      },
      plugins: {
        legend: { labels: { color: "#f8fafc" } }
      }
    }
  });
}

// In-App Python Code Sandbox
async function loadCodeTemplates() {
  try {
    const res = await fetch("/api/code-templates");
    const data = await res.json();
    AppState.templates = data.templates;

    if (elements.templateSelector) {
      elements.templateSelector.innerHTML = "";
      Object.keys(data.templates).forEach((key, idx) => {
        const item = data.templates[key];
        const opt = document.createElement("option");
        opt.value = key;
        opt.innerText = item.title;
        elements.templateSelector.appendChild(opt);
      });

      // Load first template by default
      const firstKey = Object.keys(data.templates)[0];
      if (firstKey) {
        elements.codeEditor.value = data.templates[firstKey].code;
      }
    }
  } catch (err) {
    console.error("Failed to load code templates:", err);
  }
}

function initCodeSandbox() {
  if (elements.templateSelector) {
    elements.templateSelector.addEventListener("change", (e) => {
      const selected = AppState.templates[e.target.value];
      if (selected) {
        elements.codeEditor.value = selected.code;
      }
    });
  }

  if (elements.resetCodeBtn) {
    elements.resetCodeBtn.addEventListener("click", () => {
      const selectedKey = elements.templateSelector.value;
      if (AppState.templates[selectedKey]) {
        elements.codeEditor.value = AppState.templates[selectedKey].code;
      }
    });
  }

  if (elements.runCodeBtn) {
    elements.runCodeBtn.addEventListener("click", () => {
      executeSandboxCode();
    });
  }

  // Allow Tab key in textarea
  if (elements.codeEditor) {
    elements.codeEditor.addEventListener("keydown", (e) => {
      if (e.key === "Tab") {
        e.preventDefault();
        const start = elements.codeEditor.selectionStart;
        const end = elements.codeEditor.selectionEnd;
        elements.codeEditor.value = elements.codeEditor.value.substring(0, start) + "    " + elements.codeEditor.value.substring(end);
        elements.codeEditor.selectionStart = elements.codeEditor.selectionEnd = start + 4;
      } else if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        executeSandboxCode();
      }
    });
  }
}

async function executeSandboxCode() {
  const code = elements.codeEditor.value;
  elements.runCodeBtn.disabled = true;
  elements.runCodeBtn.innerHTML = `<i class="pulse-dot"></i> Executing...`;
  elements.terminalBody.innerHTML = `Running script in isolated Python sandbox...\n`;
  elements.terminalPlotsArea.innerHTML = "";

  try {
    const res = await fetch("/api/run-code", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: code })
    });
    const result = await res.json();

    if (elements.terminalLatency) {
      elements.terminalLatency.innerText = `${result.execution_time_ms} ms`;
    }

    if (result.success) {
      elements.terminalBody.innerHTML = `<span style="color:#00f5a0;">[Program Exited Successfully with code 0]</span>\n\n` + escapeHtml(result.stdout);
      
      // Render any captured Matplotlib plots
      if (result.generated_plots && result.generated_plots.length > 0) {
        elements.terminalPlotsArea.innerHTML = `<div style="font-size:0.75rem; color:#94a3b8; margin-bottom:4px;">📊 Generated Matplotlib Visualizations (${result.generated_plots.length}):</div>`;
        result.generated_plots.forEach((pUrl, idx) => {
          const img = document.createElement("img");
          img.src = pUrl;
          img.className = "terminal-plot-img";
          elements.terminalPlotsArea.appendChild(img);
        });
      }
    } else {
      elements.terminalBody.innerHTML = `<span class="terminal-error">[Runtime Error Encountered]:</span>\n\n` + escapeHtml(result.stderr || result.error_message);
    }
  } catch (err) {
    elements.terminalBody.innerHTML = `<span class="terminal-error">Connection or Sandbox Execution Failed: ${err.message}</span>`;
  } finally {
    elements.runCodeBtn.disabled = false;
    elements.runCodeBtn.innerHTML = `▶ Compile & Run Script`;
  }
}

// Copy Buttons for Developer Guide
function initCopyButtons() {
  document.querySelectorAll(".code-copy-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const codeBlock = btn.closest(".code-block-wrapper").querySelector(".code-block-content");
      if (codeBlock) {
        navigator.clipboard.writeText(codeBlock.innerText.trim());
        const orig = btn.innerHTML;
        btn.innerHTML = `✓ Copied!`;
        setTimeout(() => { btn.innerHTML = orig; }, 2000);
      }
    });
  });
}

// Clinical Medical Report Modal
function initReportModal() {
  if (elements.openReportBtn && elements.reportModal) {
    elements.openReportBtn.addEventListener("click", () => {
      populateMedicalReport();
      elements.reportModal.classList.add("active");
    });
  }

  if (elements.closeReportBtn) {
    elements.closeReportBtn.addEventListener("click", () => {
      elements.reportModal.classList.remove("active");
    });
  }

  if (elements.printReportBtn) {
    elements.printReportBtn.addEventListener("click", () => {
      window.print();
    });
  }

  // Click outside to dismiss
  if (elements.reportModal) {
    elements.reportModal.addEventListener("click", (e) => {
      if (e.target === elements.reportModal) {
        elements.reportModal.classList.remove("active");
      }
    });
  }
}

function populateMedicalReport() {
  const diag = AppState.latestDiagnosisData || {
    is_tumor_detected: true,
    diagnosis_title: "Brain Tumor Detected",
    tumor_subtype: "Glioma / Glioblastoma Multiforme (High Grade)",
    confidence_percentage: 94.5,
    model_used: "Support Vector Machine (SVM)",
    severity_assessment: "Urgent Neurosurgical Consultation Advised"
  };

  const prep = AppState.latestPreprocessData || { stages: {} };

  // Generate unique clinical study id
  const studyId = "RAD-" + Math.floor(100000 + Math.random() * 900000);
  const now = new Date().toLocaleString();

  const container = document.getElementById("reportContentContainer");
  if (!container) return;

  const isPos = diag.is_tumor_detected;

  container.innerHTML = `
    <div class="report-header">
      <div>
        <div class="report-hospital-title">NEUROLOGY & RADIOLOGICAL AI CLINIC</div>
        <div class="report-sub">Automated Clinical AI Diagnostic Summary Report</div>
      </div>
      <div style="text-align: right;">
        <div style="font-weight: 700; color: #0f172a;">REPORT ID: ${studyId}</div>
        <div class="report-sub">Scan Date: ${now}</div>
      </div>
    </div>

    <div class="report-meta-grid">
      <div>
        <div class="report-meta-label">Patient Status</div>
        <div class="report-meta-val">Outpatient MRI Study</div>
      </div>
      <div>
        <div class="report-meta-label">Modality</div>
        <div class="report-meta-val">Brain MRI (T1-Gd / T2 FLAIR)</div>
      </div>
      <div>
        <div class="report-meta-label">Diagnostic Algorithm</div>
        <div class="report-meta-val">${diag.model_used}</div>
      </div>
      <div>
        <div class="report-meta-label">Clinical Confidence</div>
        <div class="report-meta-val">${diag.confidence_percentage}%</div>
      </div>
    </div>

    <div class="report-finding-box ${isPos ? 'positive' : 'negative'}">
      <h3 style="margin-bottom: 0.35rem;">
        ${isPos ? '🚨 POSITIVE: ABNORMAL BRAIN LESION DETECTED' : '✅ NEGATIVE: NO TUMOR DETECTED'}
      </h3>
      <div style="font-size: 0.9rem; line-height: 1.5;">
        <strong>Histopathological / Texture Indication:</strong> ${diag.tumor_subtype}<br>
        <strong>Recommended Clinical Action:</strong> ${diag.severity_assessment}
      </div>
    </div>

    <h4 style="margin: 1.25rem 0 0.5rem 0; color: #0f172a;">Radiological Imaging Progression</h4>
    <div class="report-images-row">
      <div class="report-img-card">
        <img src="${prep.stages.original || ''}" alt="Raw MRI">
        <div style="font-size: 0.75rem; font-weight: 700; margin-top: 4px;">1. Raw MRI Scan</div>
      </div>
      <div class="report-img-card">
        <img src="${prep.stages.skull_stripped || ''}" alt="Skull Stripped">
        <div style="font-size: 0.75rem; font-weight: 700; margin-top: 4px;">2. Skull-Stripped Brain</div>
      </div>
      <div class="report-img-card">
        <img src="${prep.stages.tumor_overlay || ''}" alt="Segmented ROI">
        <div style="font-size: 0.75rem; font-weight: 700; margin-top: 4px;">3. ROI Segmentation Overlay</div>
      </div>
    </div>

    <div style="margin-top: 1.5rem; font-size: 0.75rem; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 0.75rem;">
      <strong>Disclaimer:</strong> This automated report was synthesized by the NeuroScan AI Machine Learning Diagnostic Suite using GLCM feature extraction and trained scikit-learn models. For clinical decisions, correlation with formal radiological review and biopsy is required.
    </div>
  `;
}

function showLoadingNotification(msg) {
  // Optional toast indicator
  console.log("[NeuroScan AI]", msg);
}

function escapeHtml(text) {
  if (!text) return "";
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
