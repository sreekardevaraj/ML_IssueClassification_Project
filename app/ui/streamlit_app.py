"""Streamlit operations console for the support classification API."""

import os

import requests
import streamlit as st


API_URL = os.getenv("SUPPORT_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Support Signal Desk",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#172126; --muted:#627178; --mint:#d9f5e9; --coral:#ff765f; --line:#dce6e1; }
    .stApp { background:linear-gradient(135deg,#f8fbf8 0%,#eef5f1 52%,#fff7f1 100%); color:var(--ink); }
    [data-testid="stSidebar"] { background:#172126; }
    [data-testid="stSidebar"] * { color:#edf8f2; }
    .hero { padding:1.3rem 0 .8rem; }
    .eyebrow { color:#e65d49; font-size:.74rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    .hero h1 { color:var(--ink); font-family:Georgia,serif; font-size:3rem; line-height:1; margin:.35rem 0; }
    .hero p { color:var(--muted); font-size:1rem; max-width:650px; }
    .metric { background:rgba(255,255,255,.72); border:1px solid var(--line); border-radius:8px; padding:1rem 1.1rem; }
    .metric-label { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; }
    .metric-value { color:var(--ink); font-size:1.45rem; font-weight:700; margin-top:.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str) -> dict:
    response = requests.get(f"{API_URL}{path}", timeout=5)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict) -> dict:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.markdown("## SIGNAL DESK")
    st.caption("IT support classification console")
    api_url = st.text_input("API endpoint", API_URL)
    if api_url.rstrip("/") != API_URL:
        API_URL = api_url.rstrip("/")
    st.divider()
    st.markdown("**Workflow**")
    st.caption("Enter one support issue and detect its category through the production API.")

st.markdown(
    '<div class="hero"><div class="eyebrow">Support Case Classification Engine</div>'
    '<h1>Signal Desk</h1><p>Turn messy support narratives into a clear first queue, with confidence and escalation state visible at a glance.</p></div>',
    unsafe_allow_html=True,
)

try:
    ready = api_get("/health/ready")
    info = api_get("/v1/model-info")
    status = "ONLINE" if ready.get("status") == "ready" else "NOT READY"
except requests.RequestException as exc:
    ready, info, status = {}, {}, "API OFFLINE"
    st.error(f"Cannot reach the classification API at {API_URL}: {exc}")

metric_columns = st.columns(4)
for column, label, value in zip(
    metric_columns,
    ["API status", "Model version", "Taxonomy", "Artifacts"],
    [status, info.get("model_version", "--"), info.get("taxonomy_version", "--"), "Loaded" if ready.get("artifacts_loaded") else "Missing"],
):
    column.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

st.divider()
left, right = st.columns([1.1, .9], gap="large")
with left:
    st.markdown("### Detect an issue")
    case_id = st.text_input("Case ID (optional)", placeholder="e.g. INC-10482")
    text = st.text_area("Describe the support issue", height=220, placeholder="Example: My VPN disconnects whenever I try to access internal applications...")
    submitted = st.button("Detect issue", type="primary", use_container_width=True)
    if submitted:
        if len(text.strip()) < 3:
            st.warning("Enter at least three characters describing the issue.")
        else:
            try:
                st.session_state["prediction"] = api_post("/v1/predict", {"case_id": case_id or None, "text": text})
            except requests.RequestException as exc:
                st.error(f"Detection failed: {exc}")
with right:
    prediction = st.session_state.get("prediction")
    st.markdown("### Detected issue")
    if not prediction:
        st.info("Enter a support issue and select Detect issue.")
    else:
        if prediction["stage"] == "abstained":
            st.warning(f"Human review recommended: {prediction['routing_reason'].replace('_', ' ')}")
        else:
            st.success(f"Detected through {prediction['stage'].replace('_', ' ').title()}")
        st.metric("Issue category", prediction["category"].replace("_", " ").title(), f"{prediction['confidence']:.1%}")
        st.markdown("**Alternative categories**")
        for candidate in prediction["candidates"][1:]:
            st.write(f"{candidate['category'].replace('_', ' ').title()} · {candidate['score']:.1%}")
        st.caption(prediction["confidence_semantics"])
        st.caption(f"Request ID: {prediction['request_id']}")
