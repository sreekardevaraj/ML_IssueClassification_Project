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
    :root { --ink:#f5f7f2; --muted:#9daaa5; --panel:#202b2d; --panel-soft:#263436; --mint:#bcefd6; --coral:#ff765f; --line:#3a4849; }
    .stApp { background:radial-gradient(circle at 78% 5%,#344948 0%,#182224 34%,#11191b 100%); color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#101719; border-right:1px solid #2b393a; }
    [data-testid="stSidebar"] * { color:#e9f1ec; }
    [data-testid="stSidebar"] .stCaption { color:#8d9b96; }
    .block-container { max-width:1180px; padding-top:3rem; padding-bottom:4rem; }
    .hero { padding:1.2rem 0 2.2rem; border-bottom:1px solid rgba(188,239,214,.16); }
    .eyebrow { color:var(--mint); font-size:.72rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
    .hero h1 { color:var(--ink); font-family:Georgia,serif; font-size:4.8rem; letter-spacing:0; line-height:.92; margin:.6rem 0 1rem; }
    .hero p { color:#b2bfba; font-size:1.04rem; line-height:1.6; max-width:650px; }
    .metric { background:rgba(32,43,45,.78); border:1px solid var(--line); border-radius:8px; padding:1rem 1.1rem; min-height:82px; }
    .metric-label { color:#8d9b96; font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; }
    .metric-value { color:var(--ink); font-size:1.25rem; font-weight:750; margin-top:.45rem; overflow-wrap:anywhere; }
    h2, h3 { color:var(--ink) !important; letter-spacing:0; }
    label, .stMarkdown p { color:#c1cbc6; }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea { background:#202b2d; color:#f5f7f2; border:1px solid #435254; border-radius:7px; }
    [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus { border-color:var(--mint); box-shadow:0 0 0 1px var(--mint); }
    .stButton > button { background:var(--coral); color:#191b1a; border:0; border-radius:7px; font-weight:800; padding:.72rem 1rem; transition:transform .16s ease, background .16s ease; }
    .stButton > button:hover { background:#ff927f; color:#191b1a; transform:translateY(-1px); }
    [data-testid="stMetric"] { background:transparent; }
    [data-testid="stMetricLabel"] { color:#91a09a; }
    [data-testid="stMetricValue"] { color:var(--mint); }
    [data-testid="stAlert"] { border-radius:7px; }
    hr { border-color:rgba(188,239,214,.14); }
    .result-kicker { color:var(--coral); font-size:.7rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
    .result-title { color:var(--ink); font-family:Georgia,serif; font-size:2rem; margin:.25rem 0 1rem; }
    .category-chip { display:inline-block; background:var(--mint); color:#162421; border-radius:5px; padding:.3rem .55rem; font-size:.78rem; font-weight:800; letter-spacing:.04em; text-transform:uppercase; }
    .alt-row { display:flex; justify-content:space-between; border-bottom:1px solid #344244; padding:.65rem 0; color:#c5d0cb; }
    .alt-score { color:var(--mint); font-variant-numeric:tabular-nums; }
    @media (max-width: 700px) { .block-container { padding:1.5rem 1rem 3rem; } .hero h1 { font-size:3.5rem; } }
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
    st.markdown('<div class="result-kicker">Input channel</div><h2>Detect an issue</h2>', unsafe_allow_html=True)
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
    st.markdown('<div class="result-kicker">Classification result</div><div class="result-title">Detected issue</div>', unsafe_allow_html=True)
    if not prediction:
        st.info("Enter a support issue and select Detect issue.")
    else:
        if prediction["stage"] == "abstained":
            st.warning(f"Human review recommended: {prediction['routing_reason'].replace('_', ' ')}")
        else:
            st.success(f"Detected through {prediction['stage'].replace('_', ' ').title()}")
        st.markdown(f'<span class="category-chip">{prediction["category"].replace("_", " ")}</span>', unsafe_allow_html=True)
        st.metric("Confidence", f"{prediction['confidence']:.1%}")
        st.markdown("**Alternative categories**")
        for candidate in prediction["candidates"][1:]:
            st.markdown(f'<div class="alt-row"><span>{candidate["category"].replace("_", " ").title()}</span><span class="alt-score">{candidate["score"]:.1%}</span></div>', unsafe_allow_html=True)
        st.caption(prediction["confidence_semantics"])
        st.caption(f"Request ID: {prediction['request_id']}")
