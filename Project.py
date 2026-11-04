import html
from datetime import datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# Configure page style
st.set_page_config(
    page_title="Credit Risk & Loan Default Predictor",
    page_icon="🏦",
    layout="wide"
)

INK, NAVY, ACCENT = "#0f172a", "#0b2545", "#13518f"
GOOD, BAD, WARN = "#2e7d5b", "#b83a3a", "#c98a1b"
MUTED, LINE = "#64748b", "#e2e6ec"

# Custom CSS to improve UI style
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp {{ font-family: 'Inter', 'Segoe UI', Arial, sans-serif; }}
    .stApp {{ background-color: #f4f6f9; }}
    .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1240px; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    label, .stMarkdown p, [data-testid="stWidgetLabel"] p {{ color: #334155; font-size: .88rem; }}
    [data-testid="stWidgetLabel"] p {{ font-weight: 500; }}

    /* ---------- top bar ---------- */
    .topbar {{ background:{NAVY}; border-radius:10px; padding:26px 34px; margin-bottom:26px;
               display:flex; justify-content:space-between; align-items:center; gap:24px; flex-wrap:wrap;
               border-bottom:3px solid {ACCENT}; }}
    .topbar .eyebrow {{ color:#8fb0d6; font-size:.72rem; font-weight:600; letter-spacing:1.6px; text-transform:uppercase; }}
    .topbar h1 {{ color:#fff; font-size:1.65rem; font-weight:700; margin:4px 0 4px 0; letter-spacing:-.2px; }}
    .stMarkdown .topbar p {{ color:#b7c9e0 !important; font-size:.92rem; margin:0; }}
    .tags span {{ display:inline-block; color:#d6e3f3; border:1px solid rgba(255,255,255,.28); border-radius:4px;
                  padding:5px 12px; font-size:.74rem; font-weight:500; margin-left:8px; letter-spacing:.3px; }}

    /* ---------- section headings ---------- */
    .sec {{ display:flex; align-items:baseline; gap:12px; margin:6px 0 4px 0; }}
    .sec .num {{ color:{ACCENT}; font-weight:700; font-size:.85rem; letter-spacing:1px; }}
    .sec .ttl {{ color:{INK}; font-weight:700; font-size:1.2rem; letter-spacing:-.1px; }}
    .sec-sub {{ color:{MUTED}; font-size:.86rem; margin:0 0 14px 0; }}
    .card-title {{ color:{NAVY}; font-weight:700; font-size:.82rem; text-transform:uppercase; letter-spacing:1.1px;
                   padding-bottom:10px; margin-bottom:6px; border-bottom:1px solid {LINE}; }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{ background:#fff; border-radius:10px; border:1px solid {LINE};
        box-shadow:0 1px 2px rgba(15,23,42,.04); }}

    /* ---------- button ---------- */
    .stButton>button {{ background:{NAVY}; color:#fff; font-size:15px; font-weight:600; letter-spacing:.4px;
        padding:12px 28px; border-radius:6px; border:1px solid {NAVY}; width:100%; transition:all .15s ease; }}
    .stButton>button:hover {{ background:{ACCENT}; border-color:{ACCENT}; color:#fff; }}
    .stButton>button:focus {{ color:#fff; box-shadow:0 0 0 3px rgba(19,81,143,.25); }}

    /* ---------- result ---------- */
    .panel {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:22px 26px;
              box-shadow:0 1px 2px rgba(15,23,42,.04); }}
    .lbl {{ color:{MUTED}; font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; }}
    .big {{ color:{INK}; font-size:2.6rem; font-weight:700; line-height:1.1; margin:6px 0 4px 0;
            font-variant-numeric:tabular-nums; letter-spacing:-1px; }}
    .note {{ color:{MUTED}; font-size:.8rem; }}
    .badge-status {{ display:inline-flex; align-items:center; gap:8px; padding:7px 14px; border-radius:4px;
                     font-weight:600; font-size:.82rem; letter-spacing:.5px; text-transform:uppercase; }}
    .badge-status i {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
    .badge-status.ok  {{ background:#e8f3ee; color:#1f6247; border:1px solid #c3e0d2; }}
    .badge-status.ok i {{ background:{GOOD}; }}
    .badge-status.bad {{ background:#f8e9e9; color:#8f2b2b; border:1px solid #ebc4c4; }}
    .badge-status.bad i {{ background:{BAD}; }}

    .gauge {{ position:relative; height:10px; border-radius:999px; margin:38px 0 62px 0;
              background:linear-gradient(90deg,#4f9d7d 0%,#d9b24c 50%,#c4584f 100%); }}
    .gauge .dot {{ position:absolute; top:-7px; width:24px; height:24px; border-radius:50%; background:#fff;
                   border:4px solid {NAVY}; transform:translateX(-50%); box-shadow:0 2px 6px rgba(15,23,42,.3); }}
    .gauge .tag {{ position:absolute; top:-32px; transform:translateX(-50%); font-size:.74rem; font-weight:600;
                   color:{NAVY}; white-space:nowrap; }}
    .gauge .base {{ position:absolute; top:-4px; width:2px; height:18px; background:{NAVY}; opacity:.5; }}
    .gauge .btag {{ position:absolute; top:22px; transform:translateX(-50%); font-size:.72rem; color:{MUTED}; white-space:nowrap; }}
    .scale {{ display:flex; justify-content:space-between; color:#94a3b8; font-size:.7rem; margin-top:-8px;
              text-transform:uppercase; letter-spacing:.6px; }}

    .callout {{ border-radius:6px; padding:13px 18px; font-size:.88rem; margin-top:12px; }}
    .callout.info {{ background:#eef3f9; color:#1c3f66; border-left:4px solid {ACCENT}; }}
    .callout.warn {{ background:#fbf3e3; color:#76500a; border-left:4px solid {WARN}; }}

    /* ---------- explanation ---------- */
    .strip {{ display:grid; grid-template-columns:repeat(4,1fr); background:#fff; border:1px solid {LINE};
              border-radius:10px; box-shadow:0 1px 2px rgba(15,23,42,.04); overflow:hidden; }}
    .strip > div {{ padding:20px 24px; border-right:1px solid {LINE}; }}
    .strip > div:last-child {{ border-right:none; background:#f8fafc; }}
    .strip .big {{ font-size:2rem; }}

    .eq {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:14px 22px; margin:12px 0 18px 0;
           color:#334155; font-size:.9rem; font-variant-numeric:tabular-nums; }}
    .eq b {{ color:{INK}; }}

    .fcard {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:16px 22px; height:100%; }}
    .fcard.good {{ border-top:3px solid {GOOD}; }} .fcard.risk {{ border-top:3px solid {BAD}; }}
    .fcard h4 {{ margin:0 0 6px 0; font-size:.78rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; }}
    .fcard.good h4 {{ color:{GOOD}; }} .fcard.risk h4 {{ color:{BAD}; }}
    .frow {{ display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid #eef1f5; font-size:.88rem; color:#334155; }}
    .frow:last-child {{ border-bottom:none; }}
    .frow b {{ color:{INK}; font-weight:600; }}
    .frow .v {{ font-weight:700; font-variant-numeric:tabular-nums; }}
    .fcard.good .v {{ color:{GOOD}; }} .fcard.risk .v {{ color:{BAD}; }}

    .reason {{ display:flex; gap:14px; align-items:flex-start; padding:11px 0; border-bottom:1px solid #eef1f5;
               color:#334155; font-size:.88rem; }}
    .reason:last-child {{ border-bottom:none; }}
    .badge {{ min-width:78px; text-align:center; padding:3px 8px; border-radius:4px; font-weight:700; font-size:.78rem;
              font-variant-numeric:tabular-nums; }}
    .badge.up   {{ background:#f8e9e9; color:#8f2b2b; }}
    .badge.down {{ background:#e8f3ee; color:#1f6247; }}
    .badge.flat {{ background:#eef1f5; color:{MUTED}; }}

    .footer {{ margin-top:28px; padding-top:14px; border-top:1px solid {LINE}; color:#94a3b8; font-size:.76rem; line-height:1.5; }}
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="topbar">
        <div>
            <div class="eyebrow">Credit Risk Analytics</div>
            <h1>Loan Default Prediction Dashboard</h1>
            <p>Enter the customer details below to evaluate credit risk and default probability.</p>
        </div>
        <div class="tags"><span>Random Forest</span><span>Explainable AI · SHAP</span></div>
    </div>
""", unsafe_allow_html=True)

# Load the large trained model from Hugging Face.
# Load the smaller preprocessing files from the same folder as this script.
APP_DIR = Path(__file__).resolve().parent

HF_REPO_ID = "bhattom2811-afk/Home-Loan-Default"
HF_MODEL_FILENAME = "best_model.pkl"


@st.cache_resource
def load_ml_assets():
    """Download the trained model from Hugging Face and load local preprocessing assets."""
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
        repo_type="model",
    )

    model = joblib.load(model_path)
    scaler = joblib.load(APP_DIR / "scaler.pkl")
    encoders = joblib.load(APP_DIR / "encoders.pkl")
    features = joblib.load(APP_DIR / "feature_names.pkl")

    return model, scaler, encoders, features


try:
    model, scaler, encoders, features = load_ml_assets()
except Exception as e:
    st.error(
        "Unable to load the model or preprocessing files. "
        "Check that best_model.pkl exists in the Hugging Face repository "
        "and scaler.pkl, encoders.pkl, and feature_names.pkl are in GitHub. "
        f"Technical details: {e}"
    )
    st.stop()


# =====================================================================
# EXPLAINABLE AI (XAI) HELPERS
# =====================================================================
FEATURE_INFO = {
    'Age':            ("Age",                    lambda v: f"{v:.0f} years"),
    'Income':         ("Annual income",          lambda v: f"${v:,.0f}"),
    'LoanAmount':     ("Loan amount",            lambda v: f"${v:,.0f}"),
    'CreditScore':    ("Credit score",           lambda v: f"{v:.0f}"),
    'MonthsEmployed': ("Months employed",        lambda v: f"{v:.0f} months"),
    'NumCreditLines': ("Number of credit lines", lambda v: f"{v:.1f}"),
    'InterestRate':   ("Interest rate",          lambda v: f"{v:.1f}%"),
    'LoanTerm':       ("Loan term",              lambda v: f"{v:.0f} months"),
    'DTIRatio':       ("Debt-to-income ratio",   lambda v: f"{v:.2f}"),
    'Education':      ("Education level",        str),
    'EmploymentType': ("Employment type",        str),
    'MaritalStatus':  ("Marital status",         str),
    'HasMortgage':    ("Has mortgage",           str),
    'HasDependents':  ("Has dependents",         str),
    'LoanPurpose':    ("Loan purpose",           str),
    'HasCoSigner':    ("Has co-signer",          str),
}


@st.cache_resource
def get_tree_explainer(_model):
    """Exact TreeSHAP explainer for the Random Forest (None if shap is unavailable)."""
    try:
        import shap
        return shap.TreeExplainer(_model)
    except Exception:
        return None


def explain_prediction(model, scaled_input, default_prob):
    """
    Returns (impacts, baseline, method).
    impacts  : per-feature effect on the default probability (0-1 scale, + = raises risk)
    baseline : the model's average predicted default probability (None for the fallback)
    With SHAP: baseline + sum(impacts) == default_prob (exact additive explanation)
    """
    explainer = get_tree_explainer(model)
    if explainer is not None:
        try:
            sv = explainer.shap_values(scaled_input)
            if isinstance(sv, list):
                vals = np.asarray(sv[1])[0]
            else:
                sv = np.asarray(sv)
                vals = sv[0, :, 1] if sv.ndim == 3 else sv[0]
            ev = np.asarray(explainer.expected_value).ravel()
            base = float(ev[1] if ev.size > 1 else ev[0])
            if abs(base + vals.sum() - default_prob) < 1e-3:
                return vals, base, "SHAP"
        except Exception:
            pass
    # Fallback: set each feature to the training average (0 in scaled space)
    vals = np.zeros(scaled_input.shape[1])
    for i in range(scaled_input.shape[1]):
        tmp = scaled_input.copy()
        tmp[0, i] = 0.0
        vals[i] = default_prob - model.predict_proba(tmp)[0, 1]
    return vals, None, "Occlusion"


def describe_feature(feat, raw, impact_pts, avg):
    """One plain-English sentence explaining why a feature moved the score."""
    label, fmt = FEATURE_INFO[feat]
    if abs(impact_pts) < 0.05:
        effect = "has almost no influence on the score"
    elif impact_pts > 0:
        effect = "pushes the default probability <b>up</b>"
    else:
        effect = "pulls the default probability <b>down</b>"
    if avg is not None:
        rel = "above" if raw > avg else "below"
        return f"{label} of {fmt(raw)} is {rel} the average borrower ({fmt(avg)}) and {effect}."
    return f"{label} <i>'{html.escape(str(raw))}'</i> {effect} compared with the average borrower."


def build_explanation_table(model, scaler, encoders, features, user_input, scaled_input, default_prob):
    impacts, baseline, method = explain_prediction(model, scaled_input, default_prob)
    impacts_pts = impacts * 100
    means = dict(zip(scaler.feature_names_in_, scaler.mean_)) if hasattr(scaler, "feature_names_in_") else {}
    rows = []
    for f, imp in zip(features, impacts_pts):
        raw = user_input[f]
        if f in encoders:
            shown, avg = encoders[f].inverse_transform([int(raw)])[0], None
        else:
            shown, avg = float(raw), means.get(f)
        rows.append({
            "Factor": FEATURE_INFO[f][0],
            "Value": FEATURE_INFO[f][1](shown),
            "Impact": round(float(imp), 3),
            "Abs": abs(float(imp)),
            "Why": describe_feature(f, shown, float(imp), avg),
        })
    df = pd.DataFrame(rows).sort_values("Abs", ascending=False).reset_index(drop=True)
    return df, baseline, method


def impact_chart(df):
    order = list(df["Factor"])   # df is already sorted by absolute impact
    bars = alt.Chart(df).mark_bar(cornerRadius=2, height=15).encode(
        x=alt.X("Impact:Q", title="Effect on default probability (percentage points)"),
        y=alt.Y("Factor:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelFontSize=12)),
        color=alt.condition(alt.datum.Impact > 0, alt.value(BAD), alt.value(GOOD)),
        tooltip=["Factor", "Value", alt.Tooltip("Impact:Q", format="+.2f", title="Effect (pts)")],
    )

    def _labels(align, dx, cond):
        return alt.Chart(df).mark_text(align=align, dx=dx, fontSize=11, color="#475569").encode(
            x="Impact:Q", y=alt.Y("Factor:N", sort=order), text=alt.Text("Impact:Q", format="+.2f"),
        ).transform_filter(cond)
    labels = _labels("left", 5, alt.datum.Impact >= 0) + _labels("right", -5, alt.datum.Impact < 0)
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color="#94a3b8").encode(x="z:Q")
    return (bars + labels + zero).properties(height=440).configure_view(strokeWidth=0).configure_axis(
        gridColor="#eef1f5", domainColor="#cbd5e1", labelColor="#475569", titleColor="#475569",
        titleFontWeight=500, titleFontSize=11)


def section(num, title, sub=""):
    st.markdown(f"<div class='sec'><span class='num'>{num}</span><span class='ttl'>{title}</span></div>"
                f"<div class='sec-sub'>{sub}</div>", unsafe_allow_html=True)


def render_result(prediction, default_prob, baseline):
    """Risk assessment result card with a visual risk gauge."""
    section("02", "Risk Assessment Result",
            f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}")
    pos = min(max(default_prob * 100, 2), 98)
    base_html = ""
    if baseline is not None:
        bpos = min(max(baseline * 100, 2), 98)
        base_html = (f"<div class='base' style='left:{bpos}%'></div>"
                     f"<div class='btag' style='left:{bpos}%'>Average customer {baseline:.1%}</div>")
    badge = ("<span class='badge-status bad'><i></i>Risk status: High default risk</span>" if prediction == 1
             else "<span class='badge-status ok'><i></i>Risk status: Low default risk / Approved</span>")
    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f"<div class='panel' style='min-height:170px'><div class='lbl'>Default Probability</div>"
            f"<div class='big'>{default_prob:.2%}</div><div class='note'>Estimated likelihood that this loan defaults</div></div>",
            unsafe_allow_html=True)
    with right:
        st.markdown(
            f"<div class='panel' style='min-height:170px'>{badge}"
            f"<div class='gauge'><div class='tag' style='left:{pos}%'>This customer {default_prob:.1%}</div>"
            f"{base_html}<div class='dot' style='left:{pos}%'></div></div>"
            f"<div class='scale'><span>0% · Low risk</span><span>50%</span><span>100% · Default</span></div></div>",
            unsafe_allow_html=True)
    if prediction == 1:
        st.markdown("<div class='callout warn'><b>WARNING:</b> This customer shows high risk signals similar to "
                    "past loan defaults. It is highly recommended to inspect credit profiles or ask for an "
                    "alternative co-signer.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='callout info'><b>INFO:</b> This customer shows stable repayment indicators "
                    "according to the Random Forest classifier.</div>", unsafe_allow_html=True)


def render_explanation(model, scaled_input, default_prob, exp_df, baseline, method):
    """Explainable AI: why THIS customer received THIS score."""
    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    section("03", "Why this customer got this score",
            "Each factor adds to or subtracts from the average customer's risk. Together they produce the final probability.")

    raisers = exp_df[exp_df["Impact"] > 0.05].head(4)
    reducers = exp_df[exp_df["Impact"] < -0.05].head(4)
    added = exp_df.loc[exp_df["Impact"] > 0, "Impact"].sum()
    removed = exp_df.loc[exp_df["Impact"] < 0, "Impact"].sum()

    if baseline is not None:
        diff = (default_prob - baseline) * 100
        st.markdown(
            f"<div class='strip'>"
            f"<div><div class='lbl'>Average customer</div><div class='big'>{baseline:.2%}</div><div class='note'>Starting point (baseline)</div></div>"
            f"<div><div class='lbl'>Risk added</div><div class='big' style='color:{BAD}'>{added:+.2f}</div><div class='note'>percentage points</div></div>"
            f"<div><div class='lbl'>Risk removed</div><div class='big' style='color:{GOOD}'>{removed:+.2f}</div><div class='note'>percentage points</div></div>"
            f"<div><div class='lbl'>This customer</div><div class='big'>{default_prob:.2%}</div>"
            f"<div class='note'>{abs(diff):.2f} pts {'above' if diff > 0 else 'below'} average</div></div>"
            f"</div>"
            f"<div class='eq'><b>How the score is built:</b> &nbsp;{baseline:.2%} (average customer) "
            f"&nbsp;<b>{added:+.2f}</b> pts &nbsp;<b>{removed:+.2f}</b> pts &nbsp;=&nbsp; <b>{default_prob:.2%}</b></div>",
            unsafe_allow_html=True)
    else:
        st.caption("SHAP is not installed, so an approximate one-factor-at-a-time explanation is shown. "
                   "Run `pip install shap` for the exact additive breakdown.")

    def factor_list(df):
        return "".join(f"<div class='frow'><span><b>{html.escape(r.Factor)}</b> &nbsp;·&nbsp; {html.escape(r.Value)}</span>"
                       f"<span class='v'>{r.Impact:+.2f} pts</span></div>" for r in df.itertuples()) or \
               "<div class='frow'>None of significance</div>"

    c1, c2 = st.columns(2)
    c1.markdown(f"<div class='fcard good'><h4>Main factors lowering the risk</h4>{factor_list(reducers)}</div>",
                unsafe_allow_html=True)
    c2.markdown(f"<div class='fcard risk'><h4>Main factors raising the risk</h4>{factor_list(raisers)}</div>",
                unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<div class='card-title'>Contribution of every factor</div>"
                    "<div class='sec-sub' style='margin-top:6px'>Green bars lower the default probability · red bars raise it · "
                    "bar length = percentage points moved.</div>", unsafe_allow_html=True)
        st.altair_chart(impact_chart(exp_df), use_container_width=True)

    with st.expander("Plain-English reasoning for every factor", expanded=True):
        rows = ""
        for r in exp_df.itertuples():
            cls = "up" if r.Impact > 0.05 else ("down" if r.Impact < -0.05 else "flat")
            rows += (f"<div class='reason'><span class='badge {cls}'>{r.Impact:+.2f} pts</span>"
                     f"<span><b>{html.escape(r.Factor)}</b> — {r.Why}</span></div>")
        st.markdown(rows, unsafe_allow_html=True)

    if hasattr(model, "estimators_"):
        tree_probs = np.array([t.predict_proba(scaled_input)[0, 1] for t in model.estimators_])
        below = (tree_probs < 0.5).mean()
        agree = "strong" if below > 0.8 or below < 0.2 else "mixed"
        st.markdown(
            f"<div class='callout info'><b>Model confidence:</b> the Random Forest has "
            f"<b>{len(tree_probs)}</b> decision trees and the final probability is the average of their estimates. "
            f"<b>{below:.0%}</b> of the trees individually rate this customer below 50% default risk "
            f"(<b>{agree}</b> agreement).</div>", unsafe_allow_html=True)

    st.markdown("<div class='footer'>Explanations describe what the model learned from historical loans and are "
                "computed with TreeSHAP. They are decision support, not a guarantee of repayment.</div>",
                unsafe_allow_html=True)


# Organize inputs into columns
section("01", "Customer Details", "Complete both sections, then run the evaluation.")
col1, col2 = st.columns(2)

user_input = {}

with col1, st.container(border=True):
    st.markdown("<div class='card-title'>Personal &amp; Employment Profile</div>", unsafe_allow_html=True)
    if 'Age' in features:
        user_input['Age'] = st.slider("Age", 18, 100, 35)
    if 'Income' in features:
        user_input['Income'] = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    if 'EmploymentType' in features:
        emp_options = list(encoders['EmploymentType'].classes_)
        selected_emp = st.selectbox("Employment Type", emp_options)
        user_input['EmploymentType'] = encoders['EmploymentType'].transform([selected_emp])[0]
    if 'MonthsEmployed' in features:
        user_input['MonthsEmployed'] = st.number_input("Months Employed", min_value=0, max_value=600, value=24)
    if 'MaritalStatus' in features:
        marital_options = list(encoders['MaritalStatus'].classes_)
        selected_marital = st.selectbox("Marital Status", marital_options)
        user_input['MaritalStatus'] = encoders['MaritalStatus'].transform([selected_marital])[0]
    if 'Education' in features:
        edu_options = list(encoders['Education'].classes_)
        selected_edu = st.selectbox("Education Level", edu_options)
        user_input['Education'] = encoders['Education'].transform([selected_edu])[0]
    if 'HasDependents' in features:
        dep_options = list(encoders['HasDependents'].classes_)
        selected_dep = st.selectbox("Has Dependents?", dep_options)
        user_input['HasDependents'] = encoders['HasDependents'].transform([selected_dep])[0]

with col2, st.container(border=True):
    st.markdown("<div class='card-title'>Financial &amp; Loan Profile</div>", unsafe_allow_html=True)
    if 'LoanAmount' in features:
        user_input['LoanAmount'] = st.number_input("Requested Loan Amount ($)", min_value=0, value=15000, step=500)
    if 'CreditScore' in features:
        user_input['CreditScore'] = st.slider("Credit Score (FICO)", 300, 850, 650)
    if 'NumCreditLines' in features:
        user_input['NumCreditLines'] = st.number_input("Number of Credit Lines", min_value=0, max_value=20, value=2)
    if 'InterestRate' in features:
        user_input['InterestRate'] = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.5, step=0.1)
    if 'LoanTerm' in features:
        user_input['LoanTerm'] = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], index=2)
    if 'DTIRatio' in features:
        user_input['DTIRatio'] = st.slider("Debt-to-Income (DTI) Ratio", 0.0, 1.0, 0.35, step=0.01)
    if 'LoanPurpose' in features:
        purpose_options = list(encoders['LoanPurpose'].classes_)
        selected_purpose = st.selectbox("Loan Purpose", purpose_options)
        user_input['LoanPurpose'] = encoders['LoanPurpose'].transform([selected_purpose])[0]
    if 'HasMortgage' in features:
        mort_options = list(encoders['HasMortgage'].classes_)
        selected_mort = st.selectbox("Has Mortgage?", mort_options)
        user_input['HasMortgage'] = encoders['HasMortgage'].transform([selected_mort])[0]
    if 'HasCoSigner' in features:
        cosign_options = list(encoders['HasCoSigner'].classes_)
        selected_cosign = st.selectbox("Has Co-Signer?", cosign_options)
        user_input['HasCoSigner'] = encoders['HasCoSigner'].transform([selected_cosign])[0]

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 1.2, 1])

# Execution Button
if btn_col.button("Evaluate Default Risk"):
    # Transform raw dictionary structure back into safe Pandas DataFrame matching exact model attributes
    input_df = pd.DataFrame([user_input])[features]

    # Preprocess & Scale features
    scaled_input = scaler.transform(input_df)

    # Generate Prediction and Probabilities
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    default_prob = probabilities[1]

    # Explainable AI: factor contributions behind this probability
    exp_df, baseline, method = build_explanation_table(
        model, scaler, encoders, features, user_input, scaled_input, default_prob)

    st.markdown("<hr style='border:none;border-top:1px solid #dbe1ea;margin:26px 0 22px 0'>", unsafe_allow_html=True)
    render_result(prediction, default_prob, baseline)
    render_explanation(model, scaled_input, default_prob, exp_df, baseline, method)
    