from __future__ import annotations

import pickle
import subprocess
import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover - optional at runtime
    def st_autorefresh(*args, **kwargs):
        return None

from feature_extraction import append_trusted_domain
from model import CybercrimeDetectionSystem


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "detector.pkl"
LOGS_FILE = BASE_DIR / "logs.json"


@st.cache_resource(show_spinner=False)
def load_or_train_system() -> CybercrimeDetectionSystem:
    if MODEL_FILE.exists():
        with MODEL_FILE.open("rb") as model_file:
            system = pickle.load(model_file)
        if getattr(system, "is_trained", False):
            return system

    subprocess.run([sys.executable, "train.py"], cwd=BASE_DIR, check=True)
    with MODEL_FILE.open("rb") as model_file:
        return pickle.load(model_file)


def render_prediction_card(result) -> None:
    public_level = {
        "High": "High Risk",
        "Medium": "Suspicious",
        "Low": "Safe",
        "High Risk": "High Risk",
        "Suspicious": "Suspicious",
        "Safe": "Safe",
    }.get(result.risk_level, result.risk_level)
    color = {
        "High": "#ef4444",
        "Medium": "#f59e0b",
        "Low": "#22c55e",
        "High Risk": "#ef4444",
        "Suspicious": "#f59e0b",
        "Safe": "#22c55e",
    }.get(result.risk_level, "#3b82f6")
    st.markdown(
        f"""
        <div style="padding:1rem;border-radius:16px;background:#0f172a;color:white;">
            <div style="font-size:1.25rem;font-weight:700;">{result.prediction}</div>
            <div style="margin-top:0.4rem;">Risk Level:
                <span style="background:{color};padding:0.25rem 0.65rem;border-radius:999px;font-weight:700;">
                    {public_level}
                </span>
            </div>
            <div style="margin-top:0.35rem;">Risk Score: {result.risk_score}/10</div>
            <div style="margin-top:0.35rem;">ML Probability: {result.probability:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explainability(result) -> None:
    st.subheader("Explainable Output")
    debug = result.debug or {}

    st.write(f"ML probability: `{result.probability:.2%}`")

    rule_triggers = debug.get("rule_triggers", [])
    if rule_triggers:
        st.write("Rule triggers:")
        for trigger in rule_triggers[:5]:
            st.write(f"- {trigger.get('reason', trigger.get('rule_name', 'Rule triggered'))}")
    else:
        st.write("Rule triggers: `None`")

    api_result = debug.get("api_result") or debug.get("reputation", {}).get("api_result")
    reputation = debug.get("reputation", {})
    if api_result:
        st.write(
            "API result: `{}` | confidence: `{:.2f}` | fallback used: `{}`".format(
                api_result.get("verdict", "unknown"),
                float(api_result.get("confidence_score", 0.0)),
                api_result.get("fallback_used", False),
            )
        )
    elif reputation.get("status") in {"safe_override", "malicious_override"}:
        st.write(f"API result: `{reputation.get('status')}`")
    else:
        st.write("API result: `No API verdict available`")

    nlp_detector = debug.get("nlp_detector", {})
    if nlp_detector:
        st.write(
            "NLP patterns detected: `{}` | scam probability: `{:.2%}`".format(
                ", ".join(nlp_detector.get("detected_patterns", [])) or "None",
                float(nlp_detector.get("scam_probability", 0.0)),
            )
        )
    else:
        st.write("NLP patterns detected: `None`")

    explanation_lines = []
    if rule_triggers:
        explanation_lines.append(rule_triggers[0].get("reason", "Rule engine triggered"))
    if result.probability >= 0.75:
        explanation_lines.append("High ML probability")
    elif result.probability >= 0.55:
        explanation_lines.append("Elevated ML probability")
    if api_result and api_result.get("verdict") == "malicious":
        explanation_lines.append("API flagged malicious")
    if nlp_detector.get("detected_patterns"):
        explanation_lines.append(f"NLP patterns: {', '.join(nlp_detector.get('detected_patterns', [])[:3])}")

    if explanation_lines:
        st.info(f"{result.risk_level} because: " + "; ".join(explanation_lines))


def metrics_frame(system: CybercrimeDetectionSystem) -> pd.DataFrame:
    rows = []
    for source, info in system.metrics.items():
        report = info.get("report", {})
        rows.append(
            {
                "Source": source.upper(),
                "Accuracy": round(info.get("accuracy", 0.0), 4),
                "Precision": round(report.get("threat", {}).get("precision", 0.0), 4),
                "Recall": round(report.get("threat", {}).get("recall", 0.0), 4),
                "F1": round(report.get("threat", {}).get("f1-score", 0.0), 4),
                "CV F1": round(info.get("cross_validation", {}).get("mean", 0.0), 4),
            }
        )
    return pd.DataFrame(rows)


def pie_data(predictions: pd.Series) -> pd.DataFrame:
    counts = predictions.value_counts().rename_axis("label").reset_index(name="count")
    return counts


def load_logs() -> list[dict[str, object]]:
    if not LOGS_FILE.exists():
        return []

    try:
        with LOGS_FILE.open("r", encoding="utf-8") as logs_file:
            payload = json.load(logs_file)
    except (OSError, json.JSONDecodeError):
        return []

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []

    normalized_logs: list[dict[str, object]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        normalized_logs.append(
            {
                "timestamp": str(entry.get("timestamp", "")),
                "input": str(entry.get("input", "")),
                "risk": str(entry.get("risk", "Unknown")),
                "score": float(entry.get("score", 0.0) or 0.0),
                "ml_prob": float(entry.get("ml_prob", 0.0) or 0.0),
                "reasons": entry.get("reasons", []) if isinstance(entry.get("reasons", []), list) else [str(entry.get("reasons", ""))],
            }
        )
    return normalized_logs


def logs_frame(logs: list[dict[str, object]]) -> pd.DataFrame:
    if not logs:
        return pd.DataFrame(columns=["timestamp", "input", "risk", "score", "ml_prob", "reasons"])

    frame = pd.DataFrame(logs)
    frame["reasons"] = frame["reasons"].apply(lambda value: ", ".join(value) if isinstance(value, list) else str(value))
    return frame


def render_live_alerts(logs: list[dict[str, object]]) -> None:
    st.subheader("Live Alerts")
    if not logs:
        st.info("No live logs available yet. Threat events from `logs.json` will appear here automatically.")
        return

    for entry in reversed(logs[-5:]):
        message = (
            f"{entry.get('timestamp', 'Unknown time')} | Score: {entry.get('score', 0):.1f} | "
            f"ML: {float(entry.get('ml_prob', 0.0)):.2%}\n\n{entry.get('input', '')}\n\n"
            f"Reasons: {', '.join(entry.get('reasons', [])) or 'No reasons recorded.'}"
        )
        risk = str(entry.get("risk", "Unknown"))
        if risk == "High Risk":
            st.error(message)
        elif risk == "Suspicious":
            st.warning(message)
        else:
            st.success(message)


def render_system_stats(logs: list[dict[str, object]]) -> None:
    st.subheader("System Stats")
    total_scans = len(logs)
    high_risk_count = sum(1 for entry in logs if entry.get("risk") == "High Risk")
    suspicious_count = sum(1 for entry in logs if entry.get("risk") == "Suspicious")

    total_col, high_col, suspicious_col = st.columns(3)
    total_col.metric("Total Scans", total_scans)
    high_col.metric("High Risk Count", high_risk_count)
    suspicious_col.metric("Suspicious Count", suspicious_count)


def render_scan_history(logs_df: pd.DataFrame) -> None:
    st.subheader("Scan History")
    if logs_df.empty:
        st.info("No scan history available yet.")
        return
    st.dataframe(logs_df, use_container_width=True)


def render_detailed_view(logs: list[dict[str, object]]) -> None:
    st.subheader("Detailed Analysis View")
    if not logs:
        st.info("No log entries available for detailed inspection.")
        return

    options = [
        f"{entry.get('timestamp', 'Unknown time')} | {entry.get('risk', 'Unknown')} | {str(entry.get('input', ''))[:70]}"
        for entry in reversed(logs)
    ]
    selected_option = st.selectbox("Select a logged scan", options)
    selected_index = options.index(selected_option)
    selected_entry = list(reversed(logs))[selected_index]

    st.write(f"Input: `{selected_entry.get('input', '')}`")
    st.write(f"Risk Level: `{selected_entry.get('risk', 'Unknown')}`")
    st.write(f"Score: `{float(selected_entry.get('score', 0.0)):.1f}`")
    st.write(f"ML Probability: `{float(selected_entry.get('ml_prob', 0.0)):.2%}`")
    st.write("Reasons:")
    for reason in selected_entry.get("reasons", []):
        st.write(f"- {reason}")


def main() -> None:
    st.set_page_config(page_title="Cybercrime Detection Dashboard", layout="wide")
    st_autorefresh(interval=3000, key="dashboard-live-refresh")
    st.title("Cybercrime Detection Dashboard")
    st.caption("Hybrid detection system using ML, rule engine, and reputation checks.")

    system = load_or_train_system()
    logs = load_logs()
    logs_df = logs_frame(logs)

    left, middle, right = st.columns(3)
    left.metric("Email Accuracy", f"{system.metrics.get('email', {}).get('accuracy', 0):.2%}")
    middle.metric("SMS Accuracy", f"{system.metrics.get('sms', {}).get('accuracy', 0):.2%}")
    right.metric("URL Accuracy", f"{system.metrics.get('url', {}).get('accuracy', 0):.2%}")
    st.metric("Text Accuracy", f"{system.metrics.get('text', {}).get('accuracy', 0):.2%}")

    render_live_alerts(logs)
    render_system_stats(logs)
    render_scan_history(logs_df)
    render_detailed_view(logs)

    tab1, tab2, tab3, tab4 = st.tabs(["Single Scan", "Batch Analysis", "Model Analytics", "Feature Insights"])

    with tab1:
        source = st.selectbox("Source", ["url", "email", "sms", "text"], format_func=str.upper)
        content = st.text_area("Input", height=220, placeholder="Paste URL, email body, or SMS text here.")
        if "single_scan_result" not in st.session_state:
            st.session_state.single_scan_result = None
        if st.button("Analyze Item", type="primary"):
            try:
                st.session_state.single_scan_result = system.analyze(source, content)
            except ValueError as exc:
                st.error(str(exc))
                st.session_state.single_scan_result = None

        result = st.session_state.single_scan_result
        if result is not None:
            render_prediction_card(result)
            render_explainability(result)
            st.subheader("Top Reasons")
            for reason in result.reasons[:3]:
                st.write(f"- {reason}")
            if result.debug.get("rule_triggers"):
                st.subheader("Rule Triggers")
                st.dataframe(pd.DataFrame(result.debug["rule_triggers"]), use_container_width=True)
            if source == "url" and content.strip():
                if st.button("Report False Positive / Trust This Domain"):
                    try:
                        trusted_domain = append_trusted_domain(content)
                        if system.url_predictor is not None:
                            system.url_predictor.reputation_checker.whitelist.add(trusted_domain)
                            if system.url_bundle is not None:
                                system.url_bundle.feature_extractor.trusted_domains = sorted(
                                    set(system.url_bundle.feature_extractor.trusted_domains) | {trusted_domain}
                                )
                        st.success(
                            f"Added `{trusted_domain}` to `trusted_domains.csv`. Future URL checks will treat it as trusted."
                        )
                        st.session_state.single_scan_result = system.analyze(source, content)
                    except ValueError as exc:
                        st.error(str(exc))
            if result.debug:
                with st.expander("Explainability / Debug"):
                    st.json(result.debug, expanded=False)

    with tab2:
        st.write("Upload a CSV for batch scanning. Recommended columns are `source, content`, but common alternatives like `url`, `text`, `message`, `body`, or `link` are also accepted.")
        batch_source = st.selectbox(
            "Default source for single-column files",
            ["url", "email", "sms", "text"],
            format_func=str.upper,
            key="batch-source",
        )
        sample_batch = pd.DataFrame(
            [
                {"source": "url", "content": "https://web.whatsapp.com/"},
                {"source": "email", "content": "Urgent, verify your payroll account at http://secure-payroll-check.com"},
                {"source": "sms", "content": "You won a reward. Claim now and share OTP immediately."},
                {"source": "text", "content": "Kindly send your bank OTP immediately to unlock your reward."},
            ]
        )
        st.download_button(
            "Download Sample Batch Template",
            sample_batch.to_csv(index=False).encode("utf-8"),
            file_name="sample_batch_template.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch-upload")
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            if batch_df.empty:
                st.warning("Uploaded CSV is empty. Please upload a file with at least one row.")
            st.dataframe(batch_df.head(10), use_container_width=True)
            if st.button("Run Batch Scan"):
                try:
                    results_df = system.analyze_batch(batch_df, default_source=batch_source)
                    st.dataframe(results_df, use_container_width=True)
                    threat_rate = (results_df["prediction"].eq("Threat").mean() * 100) if not results_df.empty else 0
                    st.metric("Threat Rate", f"{threat_rate:.2f}%")
                    pie_df = pie_data(results_df["prediction"])
                    st.subheader("Threat vs Safe Pie Chart")
                    st.vega_lite_chart(
                        pie_df,
                        {
                            "mark": {"type": "arc", "innerRadius": 40},
                            "encoding": {
                                "theta": {"field": "count", "type": "quantitative"},
                                "color": {"field": "label", "type": "nominal"},
                                "tooltip": [{"field": "label"}, {"field": "count"}],
                            },
                        },
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download Results",
                        results_df.to_csv(index=False).encode("utf-8"),
                        file_name="cybercrime_batch_results.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.error(f"Batch scan failed: {exc}")
                    st.info("Use columns like `source, content` or upload a single-column CSV and choose the default source above.")

    with tab3:
        comparison_df = metrics_frame(system)
        st.subheader("Model Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        st.bar_chart(comparison_df.set_index("Source")[["Accuracy", "Precision", "Recall", "F1"]])

        st.subheader("Threat vs Safe Mix")
        mix_df = pd.DataFrame(
            [
                {"source": source.upper(), "safe": metrics["class_distribution"].get("safe", 0), "threat": metrics["class_distribution"].get("threat", 0)}
                for source, metrics in system.metrics.items()
            ]
        )
        st.bar_chart(mix_df.set_index("source"))

    with tab4:
        url_metrics = system.metrics.get("url", {})
        feature_importance = pd.DataFrame(url_metrics.get("feature_importance", []))
        if not feature_importance.empty:
            st.subheader("Top URL Feature Importances")
            st.bar_chart(feature_importance.set_index("feature"))
        comparison = pd.DataFrame(url_metrics.get("model_comparison", []))
        if not comparison.empty:
            st.subheader("URL Model Comparison")
            st.dataframe(comparison, use_container_width=True)
            st.bar_chart(comparison.set_index("model")[["accuracy", "precision", "recall", "f1"]])


if __name__ == "__main__":
    main()
