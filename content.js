const BANNER_ID = "cybercrime-detection-banner";

function removeBanner() {
  const banner = document.getElementById(BANNER_ID);
  if (banner) {
    banner.remove();
  }
}

function buildBanner(result) {
  const banner = document.createElement("div");
  banner.id = BANNER_ID;
  banner.style.position = "fixed";
  banner.style.top = "0";
  banner.style.left = "0";
  banner.style.right = "0";
  banner.style.zIndex = "2147483647";
  banner.style.padding = "14px 18px";
  banner.style.display = "flex";
  banner.style.alignItems = "center";
  banner.style.justifyContent = "space-between";
  banner.style.gap = "12px";
  banner.style.fontFamily = "'Segoe UI', Tahoma, sans-serif";
  banner.style.boxShadow = "0 10px 30px rgba(15, 23, 42, 0.24)";
  banner.style.color = "#ffffff";
  banner.style.background = result.risk_level === "High Risk"
    ? "linear-gradient(135deg, #991b1b, #dc2626)"
    : "linear-gradient(135deg, #a16207, #f59e0b)";

  const reasons = Array.isArray(result.reason) && result.reason.length
    ? result.reason.join(" | ")
    : "Suspicious behavior detected.";

  banner.innerHTML = `
    <div style="display:flex; flex-direction:column; gap:4px;">
      <strong style="font-size:14px;">${result.risk_level} page detected</strong>
      <span style="font-size:12px; opacity:0.95;">${reasons}</span>
    </div>
    <button id="${BANNER_ID}-close" style="border:none; border-radius:999px; padding:8px 12px; cursor:pointer; font-weight:600; color:#0f172a;">Dismiss</button>
  `;

  banner.querySelector(`#${BANNER_ID}-close`)?.addEventListener("click", () => {
    banner.remove();
  });

  return banner;
}

function updateBanner(result) {
  removeBanner();
  if (!result || !["Suspicious", "High Risk"].includes(result.risk_level)) {
    return;
  }

  const banner = buildBanner(result);
  document.documentElement.appendChild(banner);
}

chrome.runtime.onMessage.addListener((message) => {
  if (message?.type === "ANALYSIS_UPDATED") {
    updateBanner(message.payload);
  }
});
