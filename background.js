const API_ENDPOINT = "http://localhost:5000/predict";
const FEEDBACK_ENDPOINT = "http://localhost:5000/feedback/false-positive";
const ANALYSIS_TIMEOUT_MS = 6000;
const HISTORY_LIMIT = 10;
const SUPPORTED_PROTOCOLS = ["http:", "https:"];
const BADGE_STYLE = {
  Safe: { text: "OK", color: "#15803d" },
  Suspicious: { text: "WARN", color: "#ca8a04" },
  "High Risk": { text: "RISK", color: "#b91c1c" },
  Unavailable: { text: "ERR", color: "#475569" }
};

async function withTimeout(url, options = {}, timeoutMs = ANALYSIS_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}

function isSupportedUrl(rawUrl) {
  try {
    const parsed = new URL(rawUrl);
    return SUPPORTED_PROTOCOLS.includes(parsed.protocol);
  } catch (error) {
    return false;
  }
}

function normalizeReasons(reasons) {
  if (Array.isArray(reasons) && reasons.length) {
    return reasons.filter(Boolean);
  }
  if (typeof reasons === "string" && reasons.trim()) {
    return [reasons.trim()];
  }
  return ["No explanation returned."];
}

function buildInvalidUrlResult(url) {
  return {
    url,
    status: "invalid",
    risk_level: "Unavailable",
    risk_score: 0,
    ml_probability: 0,
    reason: ["Invalid or unsupported URL."],
    updatedAt: Date.now()
  };
}

function buildBackendErrorResult(url, message) {
  return {
    url,
    status: "backend_error",
    risk_level: "Unavailable",
    risk_score: 0,
    ml_probability: 0,
    reason: [message],
    updatedAt: Date.now()
  };
}

function buildApiResult(url, payload) {
  return {
    url,
    status: "ok",
    risk_level: payload.risk_level || "Safe",
    risk_score: Number(payload.risk_score ?? 0),
    ml_probability: Number(payload.ml_probability ?? 0),
    reason: normalizeReasons(payload.reason),
    prediction: payload.prediction || "Safe",
    debug: payload.debug || {},
    updatedAt: Date.now()
  };
}

async function storeHistory(entry) {
  const storage = await chrome.storage.local.get({ scanHistory: [] });
  const existing = storage.scanHistory.filter((item) => item.url !== entry.url);
  existing.unshift({
    url: entry.url,
    risk_level: entry.risk_level,
    risk_score: entry.risk_score,
    updatedAt: entry.updatedAt
  });
  await chrome.storage.local.set({ scanHistory: existing.slice(0, HISTORY_LIMIT) });
}

async function updateBadge(tabId, result) {
  if (typeof tabId !== "number") {
    return;
  }

  const style = BADGE_STYLE[result.risk_level] || BADGE_STYLE.Unavailable;
  await chrome.action.setBadgeBackgroundColor({ tabId, color: style.color });
  await chrome.action.setBadgeText({ tabId, text: style.text });
}

async function sendBannerState(tabId, result) {
  if (typeof tabId !== "number") {
    return;
  }

  try {
    await chrome.tabs.sendMessage(tabId, {
      type: "ANALYSIS_UPDATED",
      payload: result
    });
  } catch (error) {
    // Ignore pages where content scripts are unavailable.
  }
}

async function analyzeUrl(url) {
  if (!url || !isSupportedUrl(url)) {
    return buildInvalidUrlResult(url);
  }

  try {
    const response = await withTimeout(API_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ url })
    });

    if (!response.ok) {
      const message = response.status === 429
        ? "Backend rate limit reached."
        : `Backend returned ${response.status}.`;
      return buildBackendErrorResult(url, message);
    }

    const payload = await response.json();
    return buildApiResult(url, payload);
  } catch (error) {
    const message = error.name === "AbortError"
      ? "Backend timeout. Please try again."
      : "Backend not reachable.";
    return buildBackendErrorResult(url, message);
  }
}

async function reportFalsePositive(url) {
  if (!url || !isSupportedUrl(url)) {
    return { ok: false, message: "Invalid or unsupported URL." };
  }

  try {
    const response = await withTimeout(FEEDBACK_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ url })
    });

    if (!response.ok) {
      return { ok: false, message: `Backend returned ${response.status}.` };
    }

    const payload = await response.json();
    return {
      ok: true,
      trusted_domain: payload.trusted_domain,
      message: payload.message || "Trusted domain added."
    };
  } catch (error) {
    const message = error.name === "AbortError"
      ? "Backend timeout. Please try again."
      : "Backend not reachable.";
    return { ok: false, message };
  }
}

async function updateTabAnalysis(tabId, url) {
  const result = await analyzeUrl(url);
  const storageKey = `analysis_${tabId}`;
  await chrome.storage.local.set({ [storageKey]: result });
  await storeHistory(result);
  await updateBadge(tabId, result);
  await sendBannerState(tabId, result);
}

async function analyzeActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || typeof tab.id !== "number") {
    return;
  }
  await updateTabAnalysis(tab.id, tab.url || "");
}

chrome.runtime.onInstalled.addListener(async () => {
  await chrome.storage.local.set({ scanHistory: [] });
  await analyzeActiveTab();
});

chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  try {
    const tab = await chrome.tabs.get(tabId);
    await updateTabAnalysis(tabId, tab.url || "");
  } catch (error) {
    // Ignore transient activation errors.
  }
});

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" || changeInfo.url) {
    await updateTabAnalysis(tabId, tab.url || changeInfo.url || "");
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "GET_ACTIVE_ANALYSIS") {
    chrome.tabs.query({ active: true, currentWindow: true }).then(async ([tab]) => {
      if (!tab || typeof tab.id !== "number") {
        sendResponse({ error: "No active tab available." });
        return;
      }

      const key = `analysis_${tab.id}`;
      const storage = await chrome.storage.local.get({ [key]: null, scanHistory: [] });
      let analysis = storage[key];
      if (!analysis || analysis.url !== tab.url) {
        analysis = await analyzeUrl(tab.url || "");
        await chrome.storage.local.set({ [key]: analysis });
        await storeHistory(analysis);
        await updateBadge(tab.id, analysis);
        await sendBannerState(tab.id, analysis);
      }

      sendResponse({
        analysis,
        scanHistory: storage.scanHistory
      });
    });
    return true;
  }

  if (message?.type === "REANALYZE_ACTIVE_TAB") {
    chrome.tabs.query({ active: true, currentWindow: true }).then(async ([tab]) => {
      if (!tab || typeof tab.id !== "number") {
        sendResponse({ error: "No active tab available." });
        return;
      }

      const analysis = await analyzeUrl(tab.url || "");
      await chrome.storage.local.set({ [`analysis_${tab.id}`]: analysis });
      await storeHistory(analysis);
      await updateBadge(tab.id, analysis);
      await sendBannerState(tab.id, analysis);
      sendResponse({ analysis });
    });
    return true;
  }

  if (message?.type === "REPORT_FALSE_POSITIVE") {
    chrome.tabs.query({ active: true, currentWindow: true }).then(async ([tab]) => {
      if (!tab || typeof tab.id !== "number") {
        sendResponse({ error: "No active tab available." });
        return;
      }

      const feedback = await reportFalsePositive(tab.url || "");
      if (!feedback.ok) {
        sendResponse({ error: feedback.message });
        return;
      }

      const analysis = await analyzeUrl(tab.url || "");
      await chrome.storage.local.set({ [`analysis_${tab.id}`]: analysis });
      await storeHistory(analysis);
      await updateBadge(tab.id, analysis);
      await sendBannerState(tab.id, analysis);

      sendResponse({ analysis, feedback });
    });
    return true;
  }

  return false;
});
