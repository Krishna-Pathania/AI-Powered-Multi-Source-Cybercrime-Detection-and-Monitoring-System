const riskLevelElement = document.getElementById("riskLevel");
const riskScoreElement = document.getElementById("riskScore");
const mlProbabilityElement = document.getElementById("mlProbability");
const currentUrlElement = document.getElementById("currentUrl");
const reasonsListElement = document.getElementById("reasonsList");
const backendStatusElement = document.getElementById("backendStatus");
const historyListElement = document.getElementById("historyList");
const statusCardElement = document.getElementById("statusCard");
const lastUpdatedElement = document.getElementById("lastUpdated");
const refreshButton = document.getElementById("refreshButton");
const falsePositiveButton = document.getElementById("falsePositiveButton");
const feedbackStatusElement = document.getElementById("feedbackStatus");

function riskClass(level) {
  if (level === "Unavailable") return "unavailable";
  if (level === "High Risk") return "high-risk";
  if (level === "Suspicious") return "suspicious";
  return "safe";
}

function formatTimestamp(timestamp) {
  if (!timestamp) return "-";
  try {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit"
    });
  } catch (error) {
    return "-";
  }
}

function renderReasons(reasons = []) {
  reasonsListElement.innerHTML = "";
  if (!reasons.length) {
    const item = document.createElement("li");
    item.textContent = "No explanation available.";
    reasonsListElement.appendChild(item);
    return;
  }

  reasons.forEach((reason) => {
    const item = document.createElement("li");
    item.textContent = reason;
    reasonsListElement.appendChild(item);
  });
}

function renderHistory(history = []) {
  historyListElement.innerHTML = "";
  if (!history.length) {
    const item = document.createElement("li");
    item.textContent = "No recent scans.";
    historyListElement.appendChild(item);
    return;
  }

  history.slice(0, 10).forEach((entry) => {
    const item = document.createElement("li");
    item.innerHTML = `
      <div class="history-url">${entry.url}</div>
      <div class="history-meta">${entry.risk_level} • Score ${entry.risk_score}</div>
    `;
    historyListElement.appendChild(item);
  });
}

function renderAnalysis(analysis, history = []) {
  if (!analysis) {
    riskLevelElement.textContent = "Unavailable";
    riskScoreElement.textContent = "-";
    mlProbabilityElement.textContent = "-";
    currentUrlElement.textContent = "No active tab data.";
    backendStatusElement.textContent = "⚠️ Backend not reachable";
    renderReasons(["No result available."]);
    renderHistory(history);
    return;
  }

  statusCardElement.className = `status-card ${riskClass(analysis.risk_level)}`;
  riskLevelElement.textContent = analysis.risk_level || "Unknown";
  riskScoreElement.textContent = analysis.risk_score ?? "-";
  mlProbabilityElement.textContent = typeof analysis.ml_probability === "number"
    ? `${(analysis.ml_probability * 100).toFixed(1)}%`
    : "-";
  currentUrlElement.textContent = analysis.url || "Unknown URL";
  backendStatusElement.textContent = analysis.status === "backend_error"
    ? "⚠️ Backend not reachable"
    : analysis.status === "invalid"
      ? "⚠️ Invalid or unsupported URL"
      : "Backend connected";
  renderReasons(analysis.reason || []);
  renderHistory(history);
}

async function loadAnalysis(messageType = "GET_ACTIVE_ANALYSIS") {
  const response = await chrome.runtime.sendMessage({ type: messageType });
  if (response?.error) {
    renderAnalysis({
      status: "backend_error",
      risk_level: "Unavailable",
      risk_score: 0,
      ml_probability: 0,
      reason: [response.error]
    });
    return;
  }

  renderAnalysis(response?.analysis, response?.scanHistory || []);
}

refreshButton.addEventListener("click", async () => {
  refreshButton.disabled = true;
  await loadAnalysis("REANALYZE_ACTIVE_TAB");
  refreshButton.disabled = false;
});

document.addEventListener("DOMContentLoaded", async () => {
  await loadAnalysis();
});

function renderHistory(history = []) {
  historyListElement.innerHTML = "";
  if (!history.length) {
    const item = document.createElement("li");
    item.textContent = "No recent scans.";
    historyListElement.appendChild(item);
    return;
  }

  history.slice(0, 10).forEach((entry) => {
    const item = document.createElement("li");
    item.innerHTML = `
      <div class="history-url">${entry.url}</div>
      <div class="history-meta">${entry.risk_level} - Score ${entry.risk_score}</div>
    `;
    historyListElement.appendChild(item);
  });
}

function renderAnalysis(analysis, history = []) {
  if (!analysis) {
    statusCardElement.className = "status-card unavailable";
    riskLevelElement.textContent = "Unavailable";
    riskScoreElement.textContent = "-";
    mlProbabilityElement.textContent = "-";
    currentUrlElement.textContent = "No active tab data.";
    backendStatusElement.textContent = "Backend not reachable";
    lastUpdatedElement.textContent = "-";
    renderReasons(["No result available."]);
    renderHistory(history);
    return;
  }

  statusCardElement.className = `status-card ${riskClass(analysis.risk_level)}`;
  riskLevelElement.textContent = analysis.risk_level || "Unknown";
  riskScoreElement.textContent = analysis.risk_score ?? "-";
  mlProbabilityElement.textContent = typeof analysis.ml_probability === "number"
    ? `${(analysis.ml_probability * 100).toFixed(1)}%`
    : "-";
  currentUrlElement.textContent = analysis.url || "Unknown URL";
  lastUpdatedElement.textContent = formatTimestamp(analysis.updatedAt);
  backendStatusElement.textContent = analysis.status === "backend_error"
    ? "Backend not reachable"
    : analysis.status === "invalid"
      ? "Invalid or unsupported URL"
      : "Backend connected";
  falsePositiveButton.disabled = analysis.status !== "ok" || analysis.risk_level === "Safe";
  feedbackStatusElement.textContent = analysis.risk_level === "Safe"
    ? "This domain is already considered safe."
    : "Trust the current domain if this is an official site.";
  renderReasons(analysis.reason || []);
  renderHistory(history);
}

falsePositiveButton.addEventListener("click", async () => {
  falsePositiveButton.disabled = true;
  feedbackStatusElement.textContent = "Sending false-positive report...";
  const response = await chrome.runtime.sendMessage({ type: "REPORT_FALSE_POSITIVE" });
  if (response?.error) {
    feedbackStatusElement.textContent = response.error;
  } else {
    feedbackStatusElement.textContent = response?.feedback?.message || "Trusted domain added.";
    renderAnalysis(response?.analysis);
  }
});
