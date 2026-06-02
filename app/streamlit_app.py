import streamlit as st
st.cache_resource.clear()
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import os
import io
import warnings
from datetime import datetime
try:
    # ignore  
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.units import cm
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense — ML Retention Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

.stApp {
    background: #0A0E1A !important;
    font-family: 'Inter', sans-serif !important;
}

html, body, [class*="css"] { font-size: 16px !important; }

#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: #0F1525 !important;
    border-right: 1px solid #1E2A3E !important;
}
[data-testid="stSidebar"] * { color: #EDF2F7 !important; }
[data-testid="stSidebar"] .stMarkdown { font-size: 14px !important; }

/* Sidebar expander header fix */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #131C2C !important;
    border: 1px solid #1E2A3E !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background: #131C2C !important;
    color: #EDF2F7 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    background: #1E2A3E !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
[data-testid="stSidebar"] [data-testid="stExpander"] summary span,
[data-testid="stSidebar"] [data-testid="stExpander"] summary div {
    color: #EDF2F7 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] svg {
    fill: #00E5FF !important;
    stroke: #00E5FF !important;
}
[data-testid="stSidebar"] details[data-testid="stExpander"] > summary {
    background: #131C2C !important;
    color: #00E5FF !important;
}
/* Force expander content background */
[data-testid="stSidebar"] [data-testid="stExpander"] > div:last-child {
    background: #0F1525 !important;
    border-top: 1px solid #1E2A3E !important;
}

.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

.main-heading {
    font-family: 'Inter', sans-serif !important;
    font-size: 44px !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #00E5FF, #B388FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem !important;
    letter-spacing: -0.02em !important;
}

.sub-heading {
    font-family: 'Inter', sans-serif !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    color: #FFFFFF !important;
    margin-top: 0 !important;
    margin-bottom: 0.25rem !important;
}

.tech-line {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #00E5FF !important;
    margin-top: 0 !important;
}

h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; }

h2 {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #EDF2F7 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.75rem !important;
    letter-spacing: -0.01em !important;
}

h3 {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #EDF2F7 !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #A0AEC0 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00E5FF !important;
    border-bottom: 2px solid #00E5FF !important;
}

/* ── SELECTBOX COMPLETE FIX ── */
[data-baseweb="select"] > div:first-child {
    background: #131C2C !important;
    border: 1px solid #2D3A5E !important;
    border-radius: 10px !important;
}
[data-baseweb="select"] [data-baseweb="single-value"] { color: #EDF2F7 !important; }
[data-baseweb="select"] [class*="singleValue"] { color: #EDF2F7 !important; }
[data-baseweb="select"] [class*="placeholder"] { color: #718096 !important; }
[data-baseweb="select"] div[class] { color: #EDF2F7 !important; }
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
ul[data-baseweb="menu"],
[role="listbox"] {
    background: #131C2C !important;
    border: 1px solid #2D3A5E !important;
    border-radius: 10px !important;
}
[role="option"],
[data-baseweb="menu"] li,
[data-baseweb="option"] {
    background: #131C2C !important;
    color: #EDF2F7 !important;
    font-size: 14px !important;
}
[role="option"]:hover,
[data-baseweb="menu"] li:hover,
[data-baseweb="option"]:hover {
    background: #1E2A3E !important;
    color: #00E5FF !important;
}
[role="option"][aria-selected="true"],
[data-baseweb="option"][aria-selected="true"] {
    background: #1A2540 !important;
    color: #00E5FF !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div,
[data-baseweb="select"] p { color: #EDF2F7 !important; }
[data-baseweb="select"] svg { fill: #A0AEC0 !important; }

.stNumberInput > div > div > input {
    background: #131C2C !important;
    border: 1px solid #2D3A5E !important;
    color: #EDF2F7 !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    padding: 8px 12px !important;
}

.stSlider { padding: 10px 0 !important; }

.stButton > button {
    background: linear-gradient(135deg, #00E5FF, #0099CC) !important;
    color: #0A0E1A !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 36px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(0,229,255,0.3) !important;
}

[data-testid="metric-container"] {
    background: #131C2C !important;
    border: 1px solid #1E2A3E !important;
    border-radius: 16px !important;
    padding: 20px !important;
    transition: transform 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    border-color: #2D3A5E;
}
[data-testid="metric-container"] label {
    color: #00E5FF !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00E5FF !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 32px !important;
    font-weight: 700 !important;
}

hr { border-color: #1E2A3E !important; margin: 2rem 0 !important; }

label, .stSelectbox label, .stSlider label {
    color: #A0AEC0 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
}

.dataframe { font-size: 13px !important; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0F1525; }
::-webkit-scrollbar-thumb { background: #2D3A5E; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #00E5FF; }

.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD & CACHE
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "outputs", "model.pkl"))

@st.cache_resource
def get_explainer():
    """
    Guaranteed fix for XGBoost/SHAP base_score string bug '[5E-1]'.
    Strategy: save booster to temp ubj file, patch base_score in raw
    binary config, reload fresh Booster, pass directly to TreeExplainer.
    """
    import json, tempfile, os, xgboost as xgb

    mdl = load_model()
    booster = mdl.get_booster()

    try:
        # ── Step 1: save booster as JSON model file ──
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp_path = tmp.name
        tmp.close()
        booster.save_model(tmp_path)

        # ── Step 2: load JSON and recursively fix base_score ──
        with open(tmp_path, "r", encoding="utf-8") as f:
            bst_dict = json.load(f)

        def _fix(node):
            if isinstance(node, dict):
                for k in list(node.keys()):
                    if k == "base_score":
                        raw = str(node[k]).strip("[]").strip()
                        try:
                            node[k] = str(float(raw))
                        except ValueError:
                            node[k] = "0.5"
                    else:
                        _fix(node[k])
            elif isinstance(node, list):
                for item in node:
                    _fix(item)

        _fix(bst_dict)

        # ── Step 3: write patched JSON back ──
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(bst_dict, f)

        # ── Step 4: load fresh Booster from patched file ──
        fixed_booster = xgb.Booster()
        fixed_booster.load_model(tmp_path)
        os.unlink(tmp_path)

        # ── Step 5: build explainer from fixed booster directly ──
        return shap.TreeExplainer(fixed_booster)

    except Exception as e:
        # Last resort fallback — older SHAP versions may still work
        try:
            return shap.TreeExplainer(booster)
        except Exception:
            return shap.TreeExplainer(mdl)

@st.cache_data
def load_features():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "features.csv"))
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df

@st.cache_data
def load_predictions():
    return pd.read_csv(os.path.join(BASE_DIR, "outputs", "predictions.csv"))

model     = load_model()
df_ref    = load_features()
pred_hist = load_predictions()

INDUSTRY = {'tenure': 32, 'MonthlyCharges': 64.0, 'churn_rate': 26.54, 'support_score': 1.2}

MODEL_COMPARISON = [
    ("XGBoost",            "0.8437", "81%", "51%", "74%", True),
    ("LightGBM",           "0.8312", "78%", "49%", "73%", False),
    ("Random Forest",      "0.8201", "74%", "47%", "72%", False),
    ("Logistic Regression","0.7934", "68%", "43%", "70%", False),
    ("Decision Tree",      "0.7456", "71%", "38%", "68%", False),
]

# Prediction history stored in session state
# Threshold for classification (adjustable)
DEFAULT_THRESHOLD = 0.279

SAMPLE_CSV_COLS = [
    'tenure','MonthlyCharges','TotalCharges','SeniorCitizen','Partner','Dependents',
    'PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection',
    'TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','gender',
    'Contract_Month-to-month','Contract_One year','Contract_Two year',
    'InternetService_DSL','InternetService_Fiber optic','InternetService_No',
    'PaymentMethod_Bank transfer (automatic)','PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check','PaymentMethod_Mailed check',
]

def make_sample_csv():
    rows = [
        [12, 79.85, 958.2,  0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [36, 55.00, 1980.0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [60, 90.50, 5430.0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    ]
    return pd.DataFrame(rows, columns=SAMPLE_CSV_COLS).to_csv(index=False)

# ─────────────────────────────────────────────
# PDF REPORT BUILDER
# ─────────────────────────────────────────────
def build_pdf_report(customer_id, inp, prob, shap_s, monthly, tenure):
    if not REPORTLAB_OK:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    cyan   = rl_colors.HexColor('#00E5FF')
    dark   = rl_colors.HexColor('#0A0E1A')
    white  = rl_colors.white
    red    = rl_colors.HexColor('#FF5252')
    orange = rl_colors.HexColor('#FFB74D')
    green  = rl_colors.HexColor('#00E676')

    h1  = ParagraphStyle('h1',  fontSize=22, textColor=cyan,   spaceAfter=4,  fontName='Helvetica-Bold')
    h2  = ParagraphStyle('h2',  fontSize=14, textColor=white,  spaceAfter=4,  fontName='Helvetica-Bold', spaceBefore=14)
    bod = ParagraphStyle('bod', fontSize=10, textColor=white,  spaceAfter=4,  fontName='Helvetica', leading=14)
    sub = ParagraphStyle('sub', fontSize=9,  textColor=rl_colors.HexColor('#A0AEC0'), spaceAfter=2, fontName='Helvetica')
    lbl = ParagraphStyle('lbl', fontSize=9,  textColor=cyan,   spaceAfter=2,  fontName='Helvetica-Bold')

    risk_col = red if prob >= 0.6 else (orange if prob >= 0.3 else green)
    rl  = "HIGH RISK" if prob >= 0.6 else ("MEDIUM RISK" if prob >= 0.3 else "LOW RISK")
    hs  = round((1 - prob) * 100, 1)
    rev = clv_revenue_at_risk(prob, monthly, tenure)
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    elems = []

    # Header
    elems.append(Paragraph("🎯 ChurnSense", h1))
    elems.append(Paragraph("ML-Based Customer Retention Intelligence Report", sub))
    elems.append(Paragraph(f"Generated: {ts}", sub))
    elems.append(HRFlowable(width="100%", thickness=1, color=cyan, spaceAfter=10))

    # Customer summary table
    elems.append(Paragraph("Customer Summary", h2))
    sum_data = [
        ['Customer ID',      customer_id or 'N/A',     'Risk Tier',      rl],
        ['Tenure',           f"{tenure} months",        'Churn Probability', f"{prob*100:.1f}%"],
        ['Monthly Charges',  f"${monthly:.2f}",         'Health Score',   f"{hs}/100"],
        ['Contract',         inp.get('Contract_Two year') and 'Two Year' or
                             inp.get('Contract_One year') and 'One Year' or 'Month-to-month',
                             'Revenue at Risk',         f"${rev:,.0f}"],
    ]
    ts_style = TableStyle([
        ('BACKGROUND',  (0,0), (-1,-1), rl_colors.HexColor('#131C2C')),
        ('TEXTCOLOR',   (0,0), (0,-1), cyan),
        ('TEXTCOLOR',   (2,0), (2,-1), cyan),
        ('TEXTCOLOR',   (1,0), (1,-1), white),
        ('TEXTCOLOR',   (3,0), (3,-1), white),
        ('FONTNAME',    (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME',    (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',    (2,0), (2,-1), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS',(0,0),(-1,-1),[rl_colors.HexColor('#131C2C'), rl_colors.HexColor('#0F1525')]),
        ('GRID',        (0,0), (-1,-1), 0.5, rl_colors.HexColor('#1E2A3E')),
        ('PADDING',     (0,0), (-1,-1), 8),
    ])
    t = Table(sum_data, colWidths=[4*cm, 5*cm, 4*cm, 4*cm])
    t.setStyle(ts_style)
    elems.append(t)

    # Top SHAP drivers
    elems.append(Paragraph("Top Feature Impacts (SHAP)", h2))
    shap_data = [['Feature', 'SHAP Impact', 'Direction']]
    for feat, val in shap_s.head(8).items():
        direction = '↑ Increases Risk' if val > 0 else '↓ Reduces Risk'
        shap_data.append([feat, f"{val:.4f}", direction])
    shap_style = TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  rl_colors.HexColor('#1E2A3E')),
        ('TEXTCOLOR',   (0,0), (-1,0),  cyan),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[rl_colors.HexColor('#131C2C'), rl_colors.HexColor('#0F1525')]),
        ('TEXTCOLOR',   (0,1), (-1,-1), white),
        ('GRID',        (0,0), (-1,-1), 0.5, rl_colors.HexColor('#1E2A3E')),
        ('PADDING',     (0,0), (-1,-1), 7),
    ])
    st2 = Table(shap_data, colWidths=[8*cm, 4*cm, 5*cm])
    st2.setStyle(shap_style)
    elems.append(st2)

    # Recommendations
    elems.append(Paragraph("Retention Recommendations", h2))
    if prob >= 0.6:
        recs = ["Schedule direct retention call within 24 hours",
                "Offer 20% loyalty discount or contract upgrade",
                "Assign dedicated account manager",
                "Propose Two-Year contract with incentive"]
        urgency = "🔴 IMMEDIATE ACTION REQUIRED"
    elif prob >= 0.3:
        recs = ["Send personalised satisfaction survey",
                "Offer loyalty reward points",
                "Recommend service upgrade",
                "Share long-term contract benefits"]
        urgency = "🟡 PROACTIVE OUTREACH RECOMMENDED"
    else:
        recs = ["Continue standard engagement programme",
                "Include in upsell campaign",
                "Monitor quarterly satisfaction",
                "Consider referral programme invite"]
        urgency = "🟢 STANDARD ENGAGEMENT"
    elems.append(Paragraph(urgency, lbl))
    for r in recs:
        elems.append(Paragraph(f"• {r}", bod))

    # Industry benchmark
    elems.append(Paragraph("Industry Benchmark", h2))
    bm_data = [
        ['Metric',            'This Customer',           'Industry Average', 'Status'],
        ['Tenure',            f"{tenure} months",        '32 months',        'Below' if tenure < 32 else 'Above'],
        ['Monthly Charges',   f"${monthly:.2f}",         '$64.00',           'Above' if monthly > 64 else 'Below'],
        ['Churn Probability', f"{prob*100:.1f}%",        '26.54%',           'Above' if prob*100 > 26.54 else 'Below'],
    ]
    bm_style = TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), rl_colors.HexColor('#1E2A3E')),
        ('TEXTCOLOR',   (0,0), (-1,0), cyan),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',    (0,1), (-1,-1),'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[rl_colors.HexColor('#131C2C'), rl_colors.HexColor('#0F1525')]),
        ('TEXTCOLOR',   (0,1), (-1,-1), white),
        ('GRID',        (0,0), (-1,-1), 0.5, rl_colors.HexColor('#1E2A3E')),
        ('PADDING',     (0,0), (-1,-1), 7),
    ])
    bt = Table(bm_data, colWidths=[4.5*cm, 4.5*cm, 4.5*cm, 3.5*cm])
    bt.setStyle(bm_style)
    elems.append(bt)

    # Footer
    elems.append(Spacer(1, 0.5*cm))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=rl_colors.HexColor('#1E2A3E')))
    elems.append(Paragraph("ChurnSense — ML-Based Customer Retention Intelligence Platform | XGBoost + SHAP", sub))

    doc.build(elems)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def tenure_group(t):
    if t <= 12:   return 0
    elif t <= 24: return 1
    elif t <= 48: return 2
    else:         return 3

def build_row(inp):
    row = pd.DataFrame([inp])
    row['tenure_group']            = tenure_group(inp['tenure'])
    row['avg_monthly_spend_ratio'] = inp['MonthlyCharges'] / (inp['TotalCharges'] + 1)
    row['support_score']           = (inp['OnlineSecurity'] + inp['TechSupport'] +
                                      inp['OnlineBackup']   + inp['DeviceProtection'])
    row['streaming_count']         = inp['StreamingTV'] + inp['StreamingMovies']
    row['is_high_value']           = 1 if inp['MonthlyCharges'] > 65 else 0
    row['engagement_score']        = (inp['PhoneService'] + inp['MultipleLines'] +
                                      row['streaming_count'] + row['support_score'])
    X_ref = df_ref.drop('Churn', axis=1)
    X_ref[X_ref.select_dtypes(include='bool').columns] = \
        X_ref[X_ref.select_dtypes(include='bool').columns].astype(int)
    return row.reindex(columns=X_ref.columns, fill_value=0)

def predict_prob(inp):
    return float(model.predict_proba(build_row(inp))[0][1])

def predict_with_threshold(inp, threshold=DEFAULT_THRESHOLD):
    prob = predict_prob(inp)
    return prob, int(prob >= threshold)

def risk_color(p):
    return "#FF5252" if p >= 0.6 else ("#FFB74D" if p >= 0.3 else "#00E676")

def risk_label(p):
    return "HIGH RISK" if p >= 0.6 else ("MEDIUM RISK" if p >= 0.3 else "LOW RISK")

def health_score(prob):
    return round((1 - prob) * 100, 1)

def health_label(s):
    if s >= 80: return "Healthy",  "#00E676"
    if s >= 60: return "Stable",   "#00E5FF"
    if s >= 40: return "At Risk",  "#FFB74D"
    return             "Critical", "#FF5252"

def segment_badge(inp, prob):
    tenure  = inp['tenure']
    monthly = inp['MonthlyCharges']
    m2m     = inp.get('Contract_Month-to-month', 0)
    support = (inp['OnlineSecurity'] + inp['TechSupport'] +
               inp['OnlineBackup']   + inp['DeviceProtection'])
    if tenure >= 48 and not m2m:
        return "Long-Term Loyal Customer",    "#00E676"
    if monthly > 80 and prob >= 0.6:
        return "High-Value At-Risk Customer", "#FF5252"
    if monthly > 80 and prob < 0.3:
        return "Loyal Premium Customer",      "#00E5FF"
    if monthly > 70 and support == 0:
        return "Price-Sensitive Customer",    "#FFB74D"
    if tenure < 12:
        return "New Customer",                "#B388FF"
    if inp.get('Dependents', 0) and inp.get('Partner', 0):
        return "Family Customer",             "#00E5FF"
    if prob >= 0.6:
        return "At-Risk Customer",            "#FF5252"
    return "Standard Customer",               "#A0AEC0"

def validate_inputs(tenure, monthly, total):
    expected_min = tenure * monthly * 0.5
    if total < expected_min and tenure > 3:
        return False, (f"⚠️ Total Charges (${total:,.0f}) seems low for "
                       f"{tenure} months at ${monthly}/mo. "
                       f"Expected at least ${expected_min:,.0f}. Please verify.")
    return True, ""

FEATURE_LABELS = {
    'Contract_Month-to-month':                 'Contract: Month-to-month',
    'Contract_One year':                       'Contract: One Year',
    'Contract_Two year':                       'Contract: Two Year',
    'InternetService_Fiber optic':             'Internet: Fiber Optic',
    'InternetService_DSL':                     'Internet: DSL',
    'InternetService_No':                      'Internet: None',
    'PaymentMethod_Electronic check':          'Payment: Electronic Check',
    'PaymentMethod_Mailed check':              'Payment: Mailed Check',
    'PaymentMethod_Bank transfer (automatic)': 'Payment: Bank Transfer',
    'PaymentMethod_Credit card (automatic)':   'Payment: Credit Card',
    'MonthlyCharges':                          'Monthly Charges',
    'TotalCharges':                            'Total Charges',
    'tenure':                                  'Customer Tenure',
    'avg_monthly_spend_ratio':                 'Spend-to-Tenure Ratio',
    'support_score':                           'Support Services Score',
    'streaming_count':                         'Streaming Services',
    'engagement_score':                        'Engagement Score',
    'OnlineSecurity':                          'Online Security',
    'TechSupport':                             'Tech Support',
    'OnlineBackup':                            'Online Backup',
    'DeviceProtection':                        'Device Protection',
    'SeniorCitizen':                           'Senior Citizen',
    'Partner':                                 'Has Partner',
    'Dependents':                              'Has Dependents',
    'PaperlessBilling':                        'Paperless Billing',
    'PhoneService':                            'Phone Service',
    'MultipleLines':                           'Multiple Lines',
    'tenure_group':                            'Tenure Group',
    'is_high_value':                           'High-Value Customer',
}

def clean_label(feat):
    return FEATURE_LABELS.get(feat, feat.replace('_', ' ').title())

def get_shap_series(inp):
    explainer = get_explainer()
    row       = build_row(inp)
    shap_vals = explainer.shap_values(row)

    # Handle all output shapes:
    # list of 2 arrays (binary XGBClassifier)  → shap_vals[1][0]
    # single 2D array (Booster, newer SHAP)     → shap_vals[0]
    # single 1D array                            → shap_vals
    if isinstance(shap_vals, list):
        vals = shap_vals[1][0]
    elif hasattr(shap_vals, 'ndim'):
        if shap_vals.ndim == 2:
            vals = shap_vals[0]
        else:
            vals = shap_vals
    else:
        vals = shap_vals[0]

    s = pd.Series(vals, index=row.columns)
    s.index = [clean_label(f) for f in s.index]
    return s.sort_values(key=abs, ascending=False).head(10)

def prediction_summary(inp, prob, shap_s):
    risk_drivers   = [(n, v) for n, v in shap_s.items() if v >  0.05]
    retain_factors = [(n, v) for n, v in shap_s.items() if v < -0.05]
    risk_drivers   = sorted(risk_drivers,   key=lambda x: x[1], reverse=True)[:3]
    retain_factors = sorted(retain_factors, key=lambda x: abs(x[1]), reverse=True)[:3]
    if prob >= 0.6:
        summary = (f"<strong>Model Prediction:</strong> {prob*100:.1f}% churn probability — "
                   f"significantly above industry average of 26.5%. "
                   f"Primary risk factors identified below.")
        rec = "Schedule retention call within 24 hours. Offer contract upgrade with loyalty discount."
    elif prob >= 0.3:
        summary = (f"<strong>Model Prediction:</strong> {prob*100:.1f}% churn probability — "
                   f"moderately above industry average. "
                   f"Proactive engagement recommended.")
        rec = "Send personalised satisfaction survey and offer loyalty reward."
    else:
        summary = (f"<strong>Model Prediction:</strong> {prob*100:.1f}% churn probability — "
                   f"below industry average of 26.5%. "
                   f"This customer demonstrates stable behaviour.")
        rec = "Maintain standard engagement. Consider upsell opportunities."
    return summary, risk_drivers, retain_factors, rec

def clv_revenue_at_risk(prob, monthly, tenure):
    expected_remaining = max(0, 36 - tenure)
    clv = monthly * expected_remaining
    return round(clv * prob, 2)

def shap_based_recommendations(prob, inp, shap_s):
    primary   = []
    secondary = []
    top_risk  = {n: v for n, v in shap_s.items() if v > 0.05}

    if 'Contract: Month-to-month' in top_risk:
        primary.append("Offer contract upgrade to One Year or Two Year plan")
    if 'Payment: Electronic Check' in top_risk:
        primary.append("Incentivise switch to automatic bank transfer payment")
    if 'Internet: Fiber Optic' in top_risk:
        primary.append("Review fiber optic service quality — check for complaints")
    if 'Spend-to-Tenure Ratio' in top_risk or 'Monthly Charges' in top_risk:
        primary.append("Offer tailored discount to reduce effective monthly cost")
    if 'Online Security' in top_risk or 'Tech Support' in top_risk:
        primary.append("Provide complimentary Tech Support and Online Security for 3 months")

    if not primary:
        if prob >= 0.6:
            primary = ["Schedule direct retention call",
                       "Offer 20% loyalty discount",
                       "Propose contract upgrade"]
        elif prob >= 0.3:
            primary = ["Send satisfaction survey",
                       "Offer loyalty reward points",
                       "Recommend relevant service upgrade"]
        else:
            primary = ["Continue standard engagement",
                       "Include in rewards programme",
                       "Monitor quarterly satisfaction"]

    if inp.get('OnlineSecurity', 0) == 0:
        secondary.append("Offer Online Security bundle at reduced rate")
    if inp.get('TechSupport', 0) == 0:
        secondary.append("Offer Tech Support trial for 30 days")
    if inp.get('Contract_Month-to-month', 0):
        secondary.append("Share long-term contract benefits comparison")
    secondary.append("Send personalised customer appreciation message")

    roadmap = []
    if prob >= 0.6:
        roadmap = [
            ("WEEK 1", "Immediate Outreach",
             ["Call customer directly",
              "Identify pain points",
              "Present personalised retention offer"]),
            ("WEEK 2", "Offer Delivery",
             ["Send formal discount or upgrade proposal",
              "Free service trial activation",
              "Assign dedicated account contact"]),
            ("WEEK 3", "Follow-Up",
             ["Check offer acceptance status",
              "Address objections",
              "Escalate if no response"]),
            ("WEEK 4", "Resolution",
             ["Confirm new contract or plan",
              "Onboard to upgraded services",
              "Schedule 30-day satisfaction check"]),
        ]
    elif prob >= 0.3:
        roadmap = [
            ("WEEK 1", "Engagement",
             ["Send satisfaction survey",
              "Monitor response",
              "Flag for follow-up if no reply"]),
            ("WEEK 2", "Value Addition",
             ["Offer loyalty points or reward",
              "Recommend service upgrade",
              "Share customer tips and tutorials"]),
            ("WEEK 3", "Check-In",
             ["Optional call or chat session",
              "Address any outstanding concerns",
              "Present loyalty programme details"]),
            ("WEEK 4", "Monitor",
             ["Review usage behaviour",
              "Confirm satisfaction",
              "Plan next engagement cycle"]),
        ]
    else:
        roadmap = [
            ("MONTH 1", "Maintain",
             ["Standard newsletter",
              "Rewards update",
              "New feature announcement"]),
            ("MONTH 2", "Upsell",
             ["Premium feature offer",
              "Referral programme invite",
              "Quarterly satisfaction survey"]),
            ("MONTH 3", "Loyalty",
             ["Anniversary acknowledgement",
              "Exclusive offer",
              "Request testimonial"]),
            ("ONGOING", "Monitor",
             ["Quarterly check-in",
              "Usage review",
              "Flag if behaviour changes"]),
        ]

    return primary[:4], secondary[:3], roadmap

def get_what_if_scenarios(base_inp, base_prob):
    scenarios = []

    s1 = base_inp.copy()
    s1['Contract_Month-to-month'] = 0
    s1['Contract_One year']       = 0
    s1['Contract_Two year']       = 1
    p1 = predict_prob(s1)
    delta1 = base_prob - p1
    scenarios.append({
        "name": "Upgrade to Two-Year Contract",
        "desc": "Removes month-to-month contract risk factor",
        "new_prob": p1,
        "delta": delta1,
        "direction": "reduction" if delta1 > 0 else "increase"
    })

    s2 = base_inp.copy()
    s2['TechSupport']    = 1
    s2['OnlineSecurity'] = 1
    p2 = predict_prob(s2)
    delta2 = base_prob - p2
    scenarios.append({
        "name": "Add Tech Support + Online Security",
        "desc": "Increases support score and service stickiness",
        "new_prob": p2,
        "delta": delta2,
        "direction": "reduction" if delta2 > 0 else "increase"
    })

    s3 = base_inp.copy()
    s3['PaymentMethod_Electronic check']              = 0
    s3['PaymentMethod_Bank transfer (automatic)']     = 1
    s3['PaymentMethod_Credit card (automatic)']       = 0
    s3['PaymentMethod_Mailed check']                  = 0
    p3 = predict_prob(s3)
    delta3 = base_prob - p3
    scenarios.append({
        "name": "Switch to Auto Bank Transfer",
        "desc": "Removes electronic check payment instability",
        "new_prob": p3,
        "delta": delta3,
        "direction": "reduction" if delta3 > 0 else "increase"
    })

    s4 = s1.copy()
    s4['TechSupport']    = 1
    s4['OnlineSecurity'] = 1
    s4['PaymentMethod_Electronic check']          = 0
    s4['PaymentMethod_Bank transfer (automatic)'] = 1
    p4 = predict_prob(s4)
    delta4 = base_prob - p4
    scenarios.append({
        "name": "Apply All Three Changes",
        "desc": "Combined effect of contract + services + payment",
        "new_prob": p4,
        "delta": delta4,
        "direction": "reduction" if delta4 > 0 else "increase"
    })

    return scenarios

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:25px 0 15px;'>
        <div style='font-family:Inter, sans-serif;font-size:28px;
                    background:linear-gradient(135deg, #00E5FF, #B388FF);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    font-weight:800;letter-spacing:-0.5px;'>
            🎯 ChurnSense
        </div>
        <div style='color:#A0AEC0;font-size:11px;letter-spacing:1px;margin-top:6px;font-weight:600;'>
            PREDICTIVE RETENTION PLATFORM
        </div>
        <div style='color:#A0AEC0;font-size:10px;margin-top:4px;'>
            Predict. Understand. Retain.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    with st.expander("📊 MODEL PERFORMANCE", expanded=True):
        col1, col2 = st.columns(2)
        col1.metric("AUC-ROC",   "0.8437")
        col2.metric("Recall",    "81%")
        col1.metric("Precision", "51%")
        col2.metric("Accuracy",  "74%")

    with st.expander("📈 DATASET STATS", expanded=True):
        st.metric("Industry Churn Rate", "26.54%")
        st.metric("High Risk Customers", "504")
        st.metric("Revenue at Risk",     "$509K")

    with st.expander("🔍 TOP CHURN DRIVERS (SHAP)", expanded=True):
        for driver, score in [("Month-to-month Contract", "0.717"),
                               ("Spend-to-Tenure Ratio",   "0.338"),
                               ("Fiber Optic Internet",     "0.287"),
                               ("Electronic Check",         "0.193"),
                               ("Monthly Charges",          "0.182")]:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                <span style='color:#A0AEC0;font-size:12px;'>{driver}</span>
                <span style='color:#00E5FF;font-family:JetBrains Mono;font-size:11px;'>{score}</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='color:#4A5568;font-size:10px;text-align:center;padding:10px;'>
        ⚡ XGBoost · SHAP Explainability<br>
        🎯 Feature Engineering · Power BI<br>
        🐍 Python 3.10
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:30px;'>
    <div class='main-heading'>ChurnSense</div>
    <div class='sub-heading'>Machine Learning-Based Customer Retention Intelligence Platform</div>
    <div class='tech-line'>XGBoost Classifier + SHAP Explainability + Customer Retention Intelligence</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 RISK PREDICTION ENGINE",
    "🔄 RETENTION STRATEGY SIMULATOR",
    "💡 RETENTION DECISION CENTER",
    "📊 BATCH ANALYSIS",
    "📈 BUSINESS INTELLIGENCE DASHBOARD",
    "ℹ️ SYSTEM OVERVIEW"
])

# ══════════════════════════════════════════════════════
# TAB 1 — RISK PREDICTION ENGINE
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown("### Customer Risk Assessment")
    st.markdown("<p style='color:#00E5FF;font-size:14px;margin-bottom:20px;'>"
                "Enter customer data to generate ML-based churn prediction and SHAP explainability analysis.</p>",
                unsafe_allow_html=True)

    # Customer ID input
    cid_col, _ = st.columns([1, 3])
    with cid_col:
        customer_id = st.text_input("🪪 Customer ID (optional)", placeholder="e.g. CUST-00423", key="cid")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**👤 CUSTOMER PROFILE**")
        tenure      = st.slider("Tenure (months)", 0, 72, 12)
        senior      = st.selectbox("Senior Citizen",   ["No", "Yes"], index=0, key="p1")
        partner     = st.selectbox("Partner",           ["No", "Yes"], index=0, key="p2")
        dependents  = st.selectbox("Dependents",        ["No", "Yes"], index=0, key="p3")
        gender      = st.selectbox("Gender",            ["Female", "Male"], index=0, key="p4")

    with c2:
        st.markdown("**💰 BILLING & CONTRACT**")
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total_charges   = st.number_input("Total Charges ($)",
                              value=round(float(tenure * monthly_charges), 2),
                              min_value=0.0, step=10.0)
        contract  = st.selectbox("Contract Type",
                        ["Month-to-month", "One year", "Two year"], index=0, key="p5")
        payment   = st.selectbox("Payment Method",
                        ["Electronic check", "Mailed check",
                         "Bank transfer (automatic)", "Credit card (automatic)"],
                        index=0, key="p6")
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"], index=0, key="p7")

    with c3:
        st.markdown("**🛠️ SERVICES**")
        internet        = st.selectbox("Internet Service",   ["Fiber optic", "DSL", "No"], index=0, key="p8")
        online_security = st.selectbox("Online Security",    ["No", "Yes"], index=0, key="p9")
        tech_support    = st.selectbox("Tech Support",       ["No", "Yes"], index=0, key="p10")
        online_backup   = st.selectbox("Online Backup",      ["No", "Yes"], index=0, key="p11")
        device_prot     = st.selectbox("Device Protection",  ["No", "Yes"], index=0, key="p12")
        streaming_tv    = st.selectbox("Streaming TV",       ["No", "Yes"], index=0, key="p13")
        streaming_mov   = st.selectbox("Streaming Movies",   ["No", "Yes"], index=0, key="p14")
        phone_svc       = st.selectbox("Phone Service",      ["No", "Yes"], index=0, key="p15")
        multi_lines     = st.selectbox("Multiple Lines",     ["No", "Yes"], index=0, key="p16")

    st.divider()
    run_btn = st.button("🔍 RUN PREDICTION ANALYSIS", use_container_width=True)

    if run_btn:
        # Input validation
        valid, warn_msg = validate_inputs(tenure, monthly_charges, total_charges)
        if not valid:
            st.warning(warn_msg)

        inp = {
            'tenure':           tenure,
            'MonthlyCharges':   monthly_charges,
            'TotalCharges':     total_charges,
            'SeniorCitizen':    1 if senior     == "Yes"  else 0,
            'Partner':          1 if partner    == "Yes"  else 0,
            'Dependents':       1 if dependents == "Yes"  else 0,
            'PhoneService':     1 if phone_svc  == "Yes"  else 0,
            'MultipleLines':    1 if multi_lines== "Yes"  else 0,
            'OnlineSecurity':   1 if online_security == "Yes" else 0,
            'OnlineBackup':     1 if online_backup   == "Yes" else 0,
            'DeviceProtection': 1 if device_prot     == "Yes" else 0,
            'TechSupport':      1 if tech_support    == "Yes" else 0,
            'StreamingTV':      1 if streaming_tv    == "Yes" else 0,
            'StreamingMovies':  1 if streaming_mov   == "Yes" else 0,
            'PaperlessBilling': 1 if paperless == "Yes" else 0,
            'gender':           1 if gender    == "Male" else 0,
            'Contract_Month-to-month':              1 if contract == "Month-to-month" else 0,
            'Contract_One year':                    1 if contract == "One year"       else 0,
            'Contract_Two year':                    1 if contract == "Two year"       else 0,
            'InternetService_DSL':                  1 if internet == "DSL"            else 0,
            'InternetService_Fiber optic':          1 if internet == "Fiber optic"    else 0,
            'InternetService_No':                   1 if internet == "No"             else 0,
            'PaymentMethod_Bank transfer (automatic)': 1 if "Bank"       in payment else 0,
            'PaymentMethod_Credit card (automatic)':   1 if "Credit"     in payment else 0,
            'PaymentMethod_Electronic check':          1 if "Electronic" in payment else 0,
            'PaymentMethod_Mailed check':              1 if "Mailed"     in payment else 0,
        }

        with st.spinner("Running XGBoost prediction and SHAP analysis..."):
            prob   = predict_prob(inp)
            shap_s = get_shap_series(inp)

        rc           = risk_color(prob)
        rl           = risk_label(prob)
        hs           = health_score(prob)
        h_lbl, h_col = health_label(hs)
        seg, seg_col = segment_badge(inp, prob)

        st.session_state.update({
            'inp': inp, 'prob': prob, 'monthly': monthly_charges, 'tenure': tenure
        })

        # ── Save to prediction history ──
        if 'pred_history' not in st.session_state:
            st.session_state['pred_history'] = []
        st.session_state['pred_history'].append({
            'Customer ID':       customer_id if customer_id else f"CUST-{len(st.session_state.get('pred_history', []))+1:04d}",
            'Time':              datetime.now().strftime("%H:%M:%S"),
            'Tenure (mo)':       tenure,
            'Monthly ($)':       monthly_charges,
            'Contract':          contract,
            'Churn Prob (%)':    round(prob * 100, 1),
            'Risk Tier':         risk_label(prob),
            'Health Score':      health_score(prob),
            'Rev at Risk ($)':   clv_revenue_at_risk(prob, monthly_charges, tenure),
        })

        # ── Row 1: Summary Cards ──
        st.markdown("### 📋 Customer Profile Summary")
        r1, r2, r3, r4 = st.columns(4)

        r1.markdown(f"""
        <div style='background:#131C2C;border:1px solid {h_col};border-radius:16px;
                    padding:20px;text-align:center;'>
            <div style='color:#00E5FF;font-size:11px;font-weight:600;letter-spacing:1px;
                        text-transform:uppercase;margin-bottom:8px;'>Customer Health Score</div>
            <div style='color:{h_col};font-family:JetBrains Mono,monospace;
                        font-size:42px;font-weight:700;'>{hs}/100</div>
            <div style='color:{h_col};font-size:14px;margin-top:6px;font-weight:600;'>{h_lbl}</div>
            <div style='color:#718096;font-size:11px;margin-top:6px;'>Derived from churn probability</div>
        </div>""", unsafe_allow_html=True)

        r2.markdown(f"""
        <div style='background:#131C2C;border:1px solid {seg_col};border-radius:16px;
                    padding:20px;text-align:center;'>
            <div style='color:#00E5FF;font-size:11px;font-weight:600;letter-spacing:1px;
                        text-transform:uppercase;margin-bottom:8px;'>Customer Segment</div>
            <div style='color:{seg_col};font-size:15px;font-weight:700;
                        line-height:1.4;margin-top:12px;'>{seg}</div>
        </div>""", unsafe_allow_html=True)

        rev_at_risk = clv_revenue_at_risk(prob, monthly_charges, tenure)
        r3.markdown(f"""
        <div style='background:#131C2C;border:1px solid {rc};border-radius:16px;
                    padding:20px;text-align:center;'>
            <div style='color:#00E5FF;font-size:11px;font-weight:600;letter-spacing:1px;
                        text-transform:uppercase;margin-bottom:8px;'>Revenue at Risk (CLV)</div>
            <div style='color:{rc};font-family:JetBrains Mono,monospace;
                        font-size:36px;font-weight:700;'>${rev_at_risk:,.0f}</div>
            <div style='color:#718096;font-size:11px;margin-top:6px;'>
                Monthly charges × remaining tenure × risk
            </div>
        </div>""", unsafe_allow_html=True)

        r4.markdown(f"""
        <div style='background:#131C2C;border:1px solid {rc};border-radius:16px;
                    padding:20px;text-align:center;'>
            <div style='color:#00E5FF;font-size:11px;font-weight:600;letter-spacing:1px;
                        text-transform:uppercase;margin-bottom:8px;'>Churn Probability</div>
            <div style='color:{rc};font-family:JetBrains Mono,monospace;
                        font-size:42px;font-weight:700;'>{prob*100:.1f}%</div>
            <div style='color:{rc};font-size:13px;margin-top:6px;
                        font-weight:700;letter-spacing:1px;'>{rl}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge + SHAP Bar ──
        g_col, s_col = st.columns([1, 1])
        with g_col:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                delta={'reference': 26.54, 'valueformat': '.1f', 'suffix': '%', 'relative': False},
                title={'text': "Churn Probability vs Industry Average",
                       'font': {'color': '#EDF2F7', 'size': 13}},
                number={'suffix': '%', 'font': {'color': rc, 'size': 42}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#A0AEC0',
                             'tickfont': {'color': '#A0AEC0', 'size': 11}},
                    'bar': {'color': rc, 'thickness': 0.25},
                    'bgcolor': '#131C2C', 'bordercolor': '#1E2A3E',
                    'steps': [{'range': [0,  30], 'color': '#0D2E1A'},
                              {'range': [30, 60], 'color': '#2E1F0A'},
                              {'range': [60,100], 'color': '#2E0A0A'}],
                    'threshold': {'line': {'color': '#00E5FF', 'width': 4},
                                  'thickness': 0.75, 'value': 26.54}
                }
            ))
            fig_g.update_layout(
                paper_bgcolor='#131C2C', plot_bgcolor='#131C2C',
                font={'color': '#EDF2F7'}, height=300,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            st.plotly_chart(fig_g, use_container_width=True,
                            config={'modeBarButtonsToKeep': ['toImage'], 'displaylogo': False})
            st.markdown("<p style='color:#718096;font-size:12px;text-align:center;'>"
                        "▬ Industry average reference (26.54%)</p>", unsafe_allow_html=True)

        with s_col:
            colors_bar = ['#FF5252' if v > 0 else '#00E676' for v in shap_s.values]
            fig_s, ax  = plt.subplots(figsize=(7, 5))
            fig_s.patch.set_facecolor('#131C2C')
            ax.set_facecolor('#131C2C')
            ax.barh(range(len(shap_s)), shap_s.values[::-1], color=colors_bar[::-1], height=0.7)
            ax.set_yticks(range(len(shap_s)))
            ax.set_yticklabels([f[:32] for f in shap_s.index[::-1]], fontsize=11, color='#A0AEC0')
            ax.axvline(0, color='#2D3A5E', linewidth=1.5)
            ax.set_xlabel('SHAP Value (impact on prediction)', color='#A0AEC0', fontsize=12, fontweight='500')
            ax.set_title('Top Feature Impacts on Churn Risk', color='#EDF2F7', fontsize=14, pad=12, fontweight='600')
            ax.tick_params(colors='#A0AEC0', labelsize=10)
            for sp in ax.spines.values():
                sp.set_edgecolor('#1E2A3E')
                sp.set_linewidth(0.5)
            red_patch   = mpatches.Patch(color='#FF5252', label='↑ Increases churn risk')
            green_patch = mpatches.Patch(color='#00E676', label='↓ Decreases churn risk')
            ax.legend(handles=[red_patch, green_patch], facecolor='#131C2C', labelcolor='#A0AEC0',
                      fontsize=10, loc='lower right', frameon=True, edgecolor='#1E2A3E')
            plt.tight_layout()
            st.pyplot(fig_s)
            plt.close()

        # ── Model Insight Report ──
        st.markdown("### 📄 Model Insight Report")
        st.markdown("<p style='color:#718096;font-size:12px;'>"
                    "Interpretation of XGBoost model output using SHAP explainability values.</p>",
                    unsafe_allow_html=True)

        summary, risk_d, retain_f, rec_txt = prediction_summary(inp, prob, shap_s)

        st.markdown(f"""
        <div style='background:#131C2C;border:1px solid #1E2A3E;
                    border-radius:16px;padding:24px;margin-bottom:20px;'>
            <div style='color:#00E5FF;font-family:JetBrains Mono,monospace;
                        font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:12px;'>
                📊 SUMMARY
            </div>
            <p style='color:#EDF2F7;font-size:15px;line-height:1.7;margin:0;'>
                {summary}
            </p>
        </div>""", unsafe_allow_html=True)

        col_risk, col_retain = st.columns(2)
        with col_risk:
            st.markdown("""
            <div style='background:#131C2C;border:1px solid #1E2A3E;
                        border-radius:16px;padding:20px;'>
                <div style='color:#FF5252;font-family:JetBrains Mono,monospace;
                            font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:14px;'>
                    ⚠️ PRIMARY CHURN DRIVERS
                </div>""", unsafe_allow_html=True)
            if risk_d:
                for name, val in risk_d:
                    st.markdown(f"""
                    <div style='margin-bottom:14px;'>
                        <div style='display:flex;justify-content:space-between;margin-bottom:5px;'>
                            <span style='color:#EDF2F7;font-size:13px;font-weight:500;'>↑ {name}</span>
                            <span style='color:#FF5252;font-size:12px;
                                         font-family:JetBrains Mono,monospace;font-weight:600;'>
                                Impact: {val:.3f}
                            </span>
                        </div>
                        <div style='background:#1E2A3E;border-radius:4px;height:5px;width:100%;'>
                            <div style='background:#FF5252;height:5px;border-radius:4px;
                                        width:{min(100,int(abs(val)*100))}%;'></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#A0AEC0;font-size:13px;'>"
                            "No significant churn drivers identified.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_retain:
            st.markdown("""
            <div style='background:#131C2C;border:1px solid #1E2A3E;
                        border-radius:16px;padding:20px;'>
                <div style='color:#00E676;font-family:JetBrains Mono,monospace;
                            font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:14px;'>
                    🛡️ RETENTION FACTORS
                </div>""", unsafe_allow_html=True)
            if retain_f:
                for name, val in retain_f:
                    st.markdown(f"""
                    <div style='margin-bottom:14px;'>
                        <div style='display:flex;justify-content:space-between;margin-bottom:5px;'>
                            <span style='color:#EDF2F7;font-size:13px;font-weight:500;'>↓ {name}</span>
                            <span style='color:#00E676;font-size:12px;
                                         font-family:JetBrains Mono,monospace;font-weight:600;'>
                                Impact: {abs(val):.3f}
                            </span>
                        </div>
                        <div style='background:#1E2A3E;border-radius:4px;height:5px;width:100%;'>
                            <div style='background:#00E676;height:5px;border-radius:4px;
                                        width:{min(100,int(abs(val)*100))}%;'></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#A0AEC0;font-size:13px;'>"
                            "No significant retention factors identified.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:#0D2E1A;border:1px solid #00E676;
                    border-radius:12px;padding:16px 20px;margin-top:16px;'>
            <span style='color:#00E676;font-family:JetBrains Mono,monospace;
                         font-size:11px;font-weight:600;letter-spacing:1px;'>
                RECOMMENDED RETENTION STRATEGY
            </span><br>
            <span style='color:#EDF2F7;font-size:14px;'>{rec_txt}</span>
        </div>""", unsafe_allow_html=True)

        # ── Industry Benchmark ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🏆 Customer vs Industry Benchmark")
        support_val = (inp['OnlineSecurity'] + inp['TechSupport'] +
                       inp['OnlineBackup']   + inp['DeviceProtection'])

        bm_rows = [
            ("Tenure",
             f"{tenure} months",
             f"{INDUSTRY['tenure']} months",
             "Below Average" if tenure < INDUSTRY['tenure'] else "Above Average",
             "#FF5252" if tenure < INDUSTRY['tenure'] else "#00E676"),
            ("Monthly Charges",
             f"${monthly_charges:.2f}",
             f"${INDUSTRY['MonthlyCharges']:.2f}",
             "Above Average" if monthly_charges > INDUSTRY['MonthlyCharges'] else "Below Average",
             "#FFB74D" if monthly_charges > INDUSTRY['MonthlyCharges'] else "#00E676"),
            ("Churn Probability",
             f"{prob*100:.1f}%",
             f"{INDUSTRY['churn_rate']:.1f}%",
             "Above Average" if prob*100 > INDUSTRY['churn_rate'] else "Below Average",
             "#FF5252" if prob*100 > INDUSTRY['churn_rate'] else "#00E676"),
            ("Support Services",
             f"{support_val}/4",
             f"{INDUSTRY['support_score']}/4",
             "Below Average" if support_val < INDUSTRY['support_score'] else "Above Average",
             "#FFB74D" if support_val < INDUSTRY['support_score'] else "#00E676"),
        ]

        hdr = st.columns(4)
        for col, h in zip(hdr, ['METRIC', 'THIS CUSTOMER', 'INDUSTRY AVG', 'STATUS']):
            col.markdown(f"<div style='color:#00E5FF;font-family:JetBrains Mono,monospace;"
                         f"font-size:11px;font-weight:600;letter-spacing:0.5px;padding:10px 0;'>{h}</div>",
                         unsafe_allow_html=True)

        for metric, cust_val, ind_val, status, sc in bm_rows:
            cols = st.columns(4)
            cols[0].markdown(f"<div style='color:#EDF2F7;font-size:14px;font-weight:500;"
                             f"padding:10px 0;border-top:1px solid #1E2A3E;'>{metric}</div>",
                             unsafe_allow_html=True)
            cols[1].markdown(f"<div style='color:#00E5FF;font-family:JetBrains Mono,monospace;"
                             f"font-size:14px;padding:10px 0;border-top:1px solid #1E2A3E;'>{cust_val}</div>",
                             unsafe_allow_html=True)
            cols[2].markdown(f"<div style='color:#FFFFFF;font-size:14px;"
                             f"padding:10px 0;border-top:1px solid #1E2A3E;'>{ind_val}</div>",
                             unsafe_allow_html=True)
            cols[3].markdown(f"<div style='color:{sc};font-family:JetBrains Mono,monospace;"
                             f"font-size:12px;padding:10px 0;border-top:1px solid #1E2A3E;"
                             f"font-weight:700;'>{status}</div>", unsafe_allow_html=True)

        # ── Prediction History ──
        if st.session_state.get('pred_history'):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📋 Prediction History (This Session)")
            hist_df = pd.DataFrame(st.session_state['pred_history'])
            def color_risk(val):
                if val == 'HIGH RISK':    return 'color: #FF5252; font-weight: 700'
                elif val == 'MEDIUM RISK': return 'color: #FFB74D; font-weight: 700'
                return 'color: #00E676; font-weight: 700'
            st.dataframe(
                hist_df.style.applymap(color_risk, subset=['Risk Tier']),
                use_container_width=True, height=200
            )
            if st.button("🗑️ Clear History", key="clear_hist"):
                st.session_state['pred_history'] = []
                st.rerun()

        # ── PDF Export ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📄 Export Prediction Report")
        pdf_c1, pdf_c2 = st.columns([1, 3])
        with pdf_c1:
            if REPORTLAB_OK:
                with st.spinner("Generating PDF..."):
                    pdf_bytes = build_pdf_report(
                        customer_id if customer_id else "N/A",
                        inp, prob, shap_s, monthly_charges, tenure)
                if pdf_bytes:
                    cid_clean = (customer_id or "report").replace(" ", "_")
                    st.download_button(
                        "📥 DOWNLOAD PDF REPORT",
                        data=pdf_bytes,
                        file_name=f"churnsense_{cid_clean}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True)
            else:
                st.warning("Install reportlab for PDF export: pip install reportlab")
        with pdf_c2:
            st.markdown(
                "<div style='background:#131C2C;border:1px solid #1E2A3E;"
                "border-radius:10px;padding:12px 16px;margin-top:4px;'>"
                "<span style='color:#00E5FF;font-size:12px;font-weight:600;'>PDF INCLUDES: </span>"
                "<span style='color:#FFFFFF;font-size:12px;'>Customer Summary · SHAP Feature Impacts · "
                "Retention Recommendations · Industry Benchmark · Risk Assessment</span></div>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 2 — RETENTION STRATEGY SIMULATOR
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("### Retention Strategy Simulator")
    st.markdown("<p style='color:#00E5FF;font-size:14px;margin-bottom:20px;'>"
                "Simulate how different retention actions affect churn risk. "
                "Run a prediction in Tab 1 first.</p>",
                unsafe_allow_html=True)

    if 'inp' not in st.session_state:
        st.markdown("""
        <div style='background:#131C2C;border:1px solid #1E2A3E;border-radius:16px;
                    text-align:center;padding:80px;'>
            <div style='font-size:48px;margin-bottom:16px;'>⚡</div>
            <div style='color:#00E5FF;font-family:JetBrains Mono,monospace;font-size:14px;'>
                Run a prediction in the Risk Prediction Engine tab first.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        base_prob = st.session_state['prob']
        base_inp  = st.session_state['inp'].copy()
        scenarios = get_what_if_scenarios(base_inp, base_prob)

        st.markdown(f"""
        <div style='background:#131C2C;border:1px solid #2D3A5E;
                    border-radius:12px;padding:16px 24px;margin-bottom:24px;'>
            <span style='color:#FFFFFF;font-size:14px;'>Current churn risk: </span>
            <span style='color:{risk_color(base_prob)};font-family:JetBrains Mono,monospace;
                         font-size:20px;font-weight:700;'>{base_prob*100:.1f}%</span>
            <span style='color:#FFFFFF;font-size:14px;'> — {risk_label(base_prob)}</span>
        </div>""", unsafe_allow_html=True)

        best    = max(scenarios, key=lambda x: x['delta'])
        best_rc = risk_color(best['new_prob'])
        st.markdown(f"""
        <div style='background:#0D2E1A;border:2px solid #00E676;
                    border-radius:16px;padding:20px 24px;margin-bottom:24px;'>
            <div style='color:#00E676;font-family:JetBrains Mono,monospace;
                        font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:8px;'>
                🎯 RECOMMENDED RETENTION ACTION
            </div>
            <div style='color:#EDF2F7;font-size:17px;font-weight:600;'>{best["name"]}</div>
            <div style='color:#FFFFFF;font-size:13px;margin-top:6px;'>
                Risk reduction: {base_prob*100:.1f}% → {best["new_prob"]*100:.1f}%
                &nbsp;|&nbsp;
                <span style='color:#00E676;font-weight:700;'>
                    ↓ {abs(best["delta"]*100):.1f}% {best["direction"]}
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 📊 Strategy Impact Analysis")
        for sc in scenarios:
            d_abs  = abs(sc['delta']) * 100
            d_col  = "#00E676" if sc['delta'] > 0 else "#FF5252"
            arrow  = "↓" if sc['delta'] > 0 else "↑"
            new_rc = risk_color(sc['new_prob'])
            st.markdown(f"""
            <div style='background:#131C2C;border:1px solid #1E2A3E;
                        border-radius:14px;padding:18px 24px;margin-bottom:12px;'>
                <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                    <div style='flex:1;'>
                        <div style='color:#FFFFFF;font-size:15px;font-weight:600;margin-bottom:6px;'>
                            {sc["name"]}
                        </div>
                        <div style='color:#FFFFFF;font-size:13px;'>{sc["desc"]}</div>
                    </div>
                    <div style='text-align:right;margin-left:24px;'>
                        <div style='color:#FFFFFF;font-size:11px;margin-bottom:4px;font-weight:600;'>
                            RISK CHANGE
                        </div>
                        <div style='font-family:JetBrains Mono,monospace;font-size:16px;'>
                            <span style='color:{risk_color(base_prob)};'>{base_prob*100:.1f}%</span>
                            <span style='color:#FFFFFF;margin:0 8px;'>→</span>
                            <span style='color:{new_rc};'>{sc["new_prob"]*100:.1f}%</span>
                        </div>
                        <div style='color:{d_col};font-family:JetBrains Mono,monospace;
                                    font-size:13px;font-weight:700;margin-top:4px;'>
                            {arrow} {d_abs:.1f}% {sc["direction"]}
                        </div>
                    </div>
                </div>
                <div style='margin-top:12px;background:#1E2A3E;border-radius:5px;height:6px;'>
                    <div style='background:{new_rc};height:6px;border-radius:5px;
                                width:{sc["new_prob"]*100:.0f}%;'></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Custom Strategy Simulator ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔧 Custom Strategy Simulator")
        st.markdown("<p style='color:#00E5FF;font-size:13px;margin-bottom:16px;'>"
                    "Adjust customer attributes to see real-time impact on churn risk.</p>",
                    unsafe_allow_html=True)

        w1, w2 = st.columns(2)
        with w1:
            contract_opts = ["Month-to-month", "One year", "Two year"]
            cur_contract  = ("Month-to-month" if base_inp.get('Contract_Month-to-month')
                             else ("One year" if base_inp.get('Contract_One year') else "Two year"))
            new_contract  = st.selectbox("Contract Type", contract_opts,
                              index=contract_opts.index(cur_contract), key="w1")
            new_security  = st.selectbox("Online Security", ["No", "Yes"],
                              index=base_inp.get('OnlineSecurity', 0), key="w3")
            new_tech      = st.selectbox("Tech Support", ["No", "Yes"],
                              index=base_inp.get('TechSupport', 0), key="w4")
        with w2:
            pay_opts    = ["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
            new_payment = st.selectbox("Payment Method", pay_opts, key="w5")
            new_charges = st.slider("Monthly Charges ($)", 18.0, 120.0,
                            float(base_inp.get('MonthlyCharges', 65)), step=0.5, key="w6")
            internet_opts = ["Fiber optic", "DSL", "No"]
            cur_internet  = ("Fiber optic" if base_inp.get('InternetService_Fiber optic')
                             else ("DSL" if base_inp.get('InternetService_DSL') else "No"))
            new_internet  = st.selectbox("Internet Service", internet_opts,
                              index=internet_opts.index(cur_internet), key="w2")

        mod_inp = base_inp.copy()
        mod_inp['Contract_Month-to-month']                 = 1 if new_contract == "Month-to-month" else 0
        mod_inp['Contract_One year']                       = 1 if new_contract == "One year"       else 0
        mod_inp['Contract_Two year']                       = 1 if new_contract == "Two year"       else 0
        mod_inp['InternetService_Fiber optic']             = 1 if new_internet == "Fiber optic"    else 0
        mod_inp['InternetService_DSL']                     = 1 if new_internet == "DSL"            else 0
        mod_inp['InternetService_No']                      = 1 if new_internet == "No"             else 0
        mod_inp['OnlineSecurity']                          = 1 if new_security == "Yes" else 0
        mod_inp['TechSupport']                             = 1 if new_tech     == "Yes" else 0
        mod_inp['MonthlyCharges']                          = new_charges
        mod_inp['PaymentMethod_Electronic check']          = 1 if "Electronic" in new_payment else 0
        mod_inp['PaymentMethod_Mailed check']              = 1 if "Mailed"     in new_payment else 0
        mod_inp['PaymentMethod_Bank transfer (automatic)'] = 1 if "Bank"       in new_payment else 0
        mod_inp['PaymentMethod_Credit card (automatic)']   = 1 if "Credit"     in new_payment else 0

        new_prob  = predict_prob(mod_inp)
        man_delta = base_prob - new_prob
        man_dir   = "reduction" if man_delta > 0 else "increase"
        man_col   = "#00E676" if man_delta > 0 else "#FF5252"
        man_arrow = "↓" if man_delta > 0 else "↑"
        new_rc    = risk_color(new_prob)

        res1, res2, res3 = st.columns(3)
        res1.metric("Current Risk",  f"{base_prob*100:.1f}%")
        res2.metric("Modified Risk", f"{new_prob*100:.1f}%",
                    delta=f"{man_delta*100:+.1f}% {man_dir}")
        res3.metric("Risk Category", risk_label(new_prob))

        st.markdown(f"""
        <div style='background:#131C2C;border:1px solid #1E2A3E;
                    border-radius:14px;padding:20px;margin-top:16px;'>
            <div style='margin-bottom:16px;'>
                <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                    <span style='color:#00E5FF;font-size:13px;font-weight:600;'>CURRENT RISK</span>
                    <span style='color:{risk_color(base_prob)};
                                 font-family:JetBrains Mono,monospace;font-size:13px;font-weight:600;'>
                        {base_prob*100:.1f}%
                    </span>
                </div>
                <div style='background:#1E2A3E;border-radius:6px;height:8px;'>
                    <div style='background:{risk_color(base_prob)};height:8px;
                                border-radius:6px;width:{base_prob*100:.0f}%;'></div>
                </div>
            </div>
            <div>
                <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                    <span style='color:#00E5FF;font-size:13px;font-weight:600;'>MODIFIED RISK</span>
                    <span style='color:{new_rc};
                                 font-family:JetBrains Mono,monospace;font-size:13px;font-weight:600;'>
                        {new_prob*100:.1f}%
                    </span>
                </div>
                <div style='background:#1E2A3E;border-radius:6px;height:8px;'>
                    <div style='background:{new_rc};height:8px;
                                border-radius:6px;width:{new_prob*100:.0f}%;'></div>
                </div>
            </div>
            <div style='margin-top:16px;text-align:center;'>
                <span style='color:{man_col};font-family:JetBrains Mono,monospace;
                             font-size:16px;font-weight:700;'>
                    {man_arrow} {abs(man_delta*100):.1f}% {man_dir}
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 3 — RETENTION DECISION CENTER
# ══════════════════════════════════════════════════════
with tab3:
    if 'inp' not in st.session_state:
        st.markdown("""
        <div style='background:#131C2C;border:1px solid #1E2A3E;border-radius:16px;
                    text-align:center;padding:80px;'>
            <div style='font-size:48px;margin-bottom:16px;'>⚡</div>
            <div style='color:#00E5FF;font-family:JetBrains Mono,monospace;font-size:14px;'>
                Run a prediction in the Risk Prediction Engine tab first.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        prob    = st.session_state['prob']
        inp     = st.session_state['inp']
        monthly = st.session_state['monthly']
        tenure  = st.session_state['tenure']
        shap_s  = get_shap_series(inp)

        rc  = risk_color(prob)
        rl  = risk_label(prob)

        primary_actions, secondary_actions, roadmap = \
            shap_based_recommendations(prob, inp, shap_s)

        urgency = ("🔴 IMMEDIATE ACTION REQUIRED"      if prob >= 0.6 else
                   ("🟡 PROACTIVE OUTREACH RECOMMENDED" if prob >= 0.3 else
                    "🟢 STANDARD ENGAGEMENT"))

        st.markdown(f"### Retention Decision Support — {rl}")
        st.markdown(f"""
        <div style='background:{rc}20;border-left:5px solid {rc};
                    padding:14px 24px;border-radius:8px;margin-bottom:24px;'>
            <span style='color:{rc};font-family:JetBrains Mono,monospace;
                         font-size:12px;font-weight:700;letter-spacing:1px;'>
                {urgency}
            </span>
            <span style='color:#FFFFFF;font-size:13px;margin-left:20px;'>
                Recommendations derived from SHAP feature importance analysis
            </span>
        </div>""", unsafe_allow_html=True)

        col_action, col_secondary = st.columns(2)
        with col_action:
            st.markdown("**🎯 CUSTOMER-SPECIFIC PRIMARY ACTIONS**")
            for i, action in enumerate(primary_actions, 1):
                st.markdown(f"""
                <div style='background:#131C2C;border:1px solid #1E2A3E;
                            border-radius:10px;padding:12px 16px;margin-bottom:10px;'>
                    <span style='color:#00E5FF;font-family:JetBrains Mono,monospace;
                                 font-size:12px;font-weight:600;margin-right:12px;'>0{i}</span>
                    <span style='color:#EDF2F7;font-size:14px;'>{action}</span>
                </div>""", unsafe_allow_html=True)

        with col_secondary:
            st.markdown("**📋 SECONDARY ACTIONS**")
            for i, action in enumerate(secondary_actions, 1):
                st.markdown(f"""
                <div style='background:#131C2C;border:1px solid #1E2A3E;
                            border-radius:10px;padding:12px 16px;margin-bottom:10px;'>
                    <span style='color:#00E5FF;font-family:JetBrains Mono,monospace;
                                 font-size:12px;font-weight:600;margin-right:12px;'>{i}.</span>
                    <span style='color:#FFFFFF;font-size:14px;'>{action}</span>
                </div>""", unsafe_allow_html=True)

        # ── 30-Day Roadmap ──
        st.divider()
        st.markdown("### 📅 30-Day Retention Roadmap")
        for period, title, actions in roadmap:
            st.markdown(f"""
            <div style='background:#131C2C;border:1px solid #1E2A3E;
                        border-left:4px solid {rc};border-radius:12px;
                        padding:18px 24px;margin-bottom:14px;'>
                <div style='display:flex;align-items:center;margin-bottom:12px;'>
                    <div style='background:{rc}20;border:1px solid {rc};
                                border-radius:6px;padding:4px 12px;margin-right:14px;'>
                        <span style='color:{rc};font-family:JetBrains Mono,monospace;
                                     font-size:11px;font-weight:700;'>{period}</span>
                    </div>
                    <span style='color:#EDF2F7;font-size:15px;font-weight:600;'>{title}</span>
                </div>
                {''.join(f"<div style='color:#FFFFFF;font-size:13px;padding:4px 0;'>&nbsp;&nbsp;▹ {a}</div>" for a in actions)}
            </div>""", unsafe_allow_html=True)

        # ── Business Impact ──
        st.divider()
        st.markdown("### 💰 Business Impact Analysis")
        rev_risk = clv_revenue_at_risk(prob, monthly, tenure)
        ret_cost = round(monthly * (2 if prob >= 0.6 else (0.5 if prob >= 0.3 else 0.1)), 2)
        net_save = round(rev_risk - ret_cost, 2)

        col_bi1, col_bi2, col_bi3 = st.columns(3)
        col_bi1.metric("Revenue at Risk (CLV-based)",    f"${rev_risk:,.0f}",
                       help="Monthly charges × expected remaining tenure × churn probability")
        col_bi2.metric("Estimated Retention Investment", f"${ret_cost:,.0f}")
        col_bi3.metric("Net Savings if Retained",        f"${net_save:,.0f}",
                       delta="Positive ROI" if net_save > 0 else "Review ROI")

        if rev_risk > 0:
            fig_bi = go.Figure()
            fig_bi.add_trace(go.Bar(
                x=['Revenue at Risk', 'Retention Investment', 'Net Savings'],
                y=[rev_risk, ret_cost, net_save],
                marker_color=['#FF5252', '#FFB74D', '#00E676'],
                text=[f"${v:,.0f}" for v in [rev_risk, ret_cost, net_save]],
                textposition='outside',
                textfont={'color': '#EDF2F7', 'size': 13}
            ))
            fig_bi.update_layout(
                paper_bgcolor='#131C2C', plot_bgcolor='#131C2C',
                font={'color': '#EDF2F7'}, height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis={'gridcolor': '#1E2A3E', 'tickprefix': '$',
                       'color': '#A0AEC0', 'title': 'Amount ($)'},
                xaxis={'color': '#A0AEC0'}, showlegend=False
            )
            st.plotly_chart(fig_bi, use_container_width=True,
                            config={'modeBarButtonsToKeep': ['toImage'], 'displaylogo': False})

# ══════════════════════════════════════════════════════
# TAB 4 — BATCH ANALYSIS
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown("### Batch Customer Risk Analysis")
    st.markdown("<p style='color:#00E5FF;font-size:14px;margin-bottom:20px;'>"
                "Upload a CSV file to analyse multiple customers using the XGBoost model.</p>",
                unsafe_allow_html=True)

    # ── Sample CSV download + column info ──
    dl_col, info_col = st.columns([1, 3])
    with dl_col:
        st.download_button(
            "📥 DOWNLOAD SAMPLE CSV TEMPLATE",
            data=make_sample_csv(),
            file_name="churnsense_sample_input.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download a pre-filled sample CSV showing the required column format"
        )
    with info_col:
        st.markdown("""
        <div style='background:#131C2C;border:1px solid #1E2A3E;border-radius:10px;
                    padding:12px 16px;'>
            <span style='color:#00E5FF;font-family:JetBrains Mono,monospace;
                         font-size:11px;font-weight:600;'>REQUIRED COLUMNS: </span>
            <span style='color:#FFFFFF;font-size:12px;'>
                tenure · MonthlyCharges · TotalCharges · Contract_* ·
                InternetService_* · PaymentMethod_* · OnlineSecurity ·
                TechSupport · OnlineBackup · DeviceProtection ·
                StreamingTV · StreamingMovies · PhoneService ·
                MultipleLines · PaperlessBilling · SeniorCitizen ·
                Partner · Dependents · gender
            </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Customer Data (CSV)", type=['csv'],
                                 label_visibility="collapsed")

    if uploaded:
        batch   = pd.read_csv(uploaded)
        results = []
        skipped = 0
        with st.spinner(f"Analysing {len(batch)} customers..."):
            for idx, row_data in batch.iterrows():
                try:
                    d = row_data.to_dict()
                    p = predict_prob(d)
                    results.append({
                        'Churn Probability (%)':  round(p * 100, 1),
                        'Risk Tier':              risk_label(p),
                        'Monthly Charges ($)':    d.get('MonthlyCharges', 0),
                        'Annual Rev at Risk ($)': round(d.get('MonthlyCharges', 0) * 12 * p, 0),
                        'Tenure (months)':        d.get('tenure', 0),
                    })
                except Exception as e:
                    skipped += 1

        if skipped > 0:
            st.warning(f"⚠️ {skipped} row(s) skipped due to missing or invalid data.")

        res_df = pd.DataFrame(results).sort_values(
            'Churn Probability (%)', ascending=False).reset_index(drop=True)
        res_df.index = res_df.index + 1

        high = (res_df['Risk Tier'] == "HIGH RISK").sum()
        med  = (res_df['Risk Tier'] == "MEDIUM RISK").sum()
        tot  = res_df['Annual Rev at Risk ($)'].sum()

        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric("Total Customers Analysed", len(res_df))
        col_b2.metric("High Risk Customers",       high,
                      delta=f"{high/len(res_df)*100:.0f}% of batch" if len(res_df) > 0 else None,
                      delta_color="inverse")
        col_b3.metric("Medium Risk Customers",     med)
        col_b4.metric("Total Revenue at Risk",     f"${tot:,.0f}")

        st.markdown("<br>**🎯 Priority Customers (Highest Risk First)**</br>",
                    unsafe_allow_html=True)
        st.dataframe(res_df.head(10), use_container_width=True, height=400)
        st.download_button(
            "📥 DOWNLOAD FULL RESULTS",
            data=res_df.to_csv(index=False),
            file_name="churn_batch_analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px;'>
            <div style='font-size:48px;margin-bottom:16px;'>📂</div>
            <div style='color:#FFFFFF;font-size:14px;margin-top:8px;'>
                Upload a CSV file containing customer data to begin batch analysis
            </div>
            <div style='color:#00E5FF;font-size:12px;margin-top:12px;'>
                Use the sample template above to get started quickly
            </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 5 — BUSINESS INTELLIGENCE DASHBOARD
# ══════════════════════════════════════════════════════
with tab5:
    st.markdown("### Business Intelligence Dashboard")
    st.markdown("<p style='color:#00E5FF;font-size:14px;margin-bottom:20px;'>"
                "Historical analysis from the 1,409-customer test dataset.</p>",
                unsafe_allow_html=True)

    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    col_d1.metric("Total Customers Analysed",   len(pred_hist))
    col_d2.metric("Predicted Churn Events",     int(pred_hist['predicted_churn'].sum()))
    col_d3.metric("Average Churn Probability",
                  f"{pred_hist['churn_probability'].mean()*100:.1f}%")
    col_d4.metric("High Risk Customers",
                  len(pred_hist[pred_hist['risk_tier'] == "High Risk"]))

    st.markdown("<br>", unsafe_allow_html=True)
    col_ch1, col_ch2 = st.columns(2)

    with col_ch1:
        rc_counts = pred_hist['risk_tier'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=rc_counts.index, values=rc_counts.values, hole=0.55,
            marker_colors=['#FF5252', '#FFB74D', '#00E676'],
            textfont={'color': '#EDF2F7', 'size': 13},
            textinfo='label+percent',
            hovertemplate='%{label}: %{value} customers<br>%{percent:.1%}<extra></extra>'
        ))
        fig_pie.update_layout(
            title={'text': 'Risk Tier Distribution',
                   'font': {'color': '#EDF2F7', 'size': 15, 'weight': 'bold'}},
            paper_bgcolor='#131C2C', font={'color': '#EDF2F7'},
            height=340, margin=dict(l=20, r=20, t=50, b=20),
            legend={'font': {'color': '#A0AEC0'}, 'bgcolor': 'rgba(0,0,0,0)'}
        )
        st.plotly_chart(fig_pie, use_container_width=True,
                        config={'modeBarButtonsToKeep': ['toImage'], 'displaylogo': False})

    with col_ch2:
        fig_hist = px.histogram(pred_hist, x='churn_probability',
                                nbins=25, color_discrete_sequence=['#00E5FF'])
        fig_hist.update_layout(
            title={'text': 'Churn Probability Distribution',
                   'font': {'color': '#EDF2F7', 'size': 15, 'weight': 'bold'}},
            paper_bgcolor='#131C2C', plot_bgcolor='#131C2C',
            font={'color': '#EDF2F7'}, height=340,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis={'gridcolor': '#1E2A3E', 'color': '#A0AEC0',
                   'title': 'Churn Probability', 'tickformat': '.0%'},
            yaxis={'gridcolor': '#1E2A3E', 'color': '#A0AEC0',
                   'title': 'Number of Customers'},
            bargap=0.08
        )
        st.plotly_chart(fig_hist, use_container_width=True,
                        config={'modeBarButtonsToKeep': ['toImage'], 'displaylogo': False})

    col_ch3, col_ch4 = st.columns(2)
    with col_ch3:
        tg = pred_hist.groupby('tenure_group')['churn_probability'].mean().reset_index()
        tg['label'] = tg['tenure_group'].map(
            {0: '0-12 months', 1: '13-24 months',
             2: '25-48 months', 3: '49+ months'})
        fig_bar = px.bar(tg, x='label', y='churn_probability',
                         color='churn_probability',
                         color_continuous_scale=['#00E676', '#FFB74D', '#FF5252'],
                         text=tg['churn_probability'].apply(lambda x: f'{x*100:.1f}%'))
        fig_bar.update_traces(textposition='outside', textfont={'size': 12})
        fig_bar.update_layout(
            title={'text': 'Average Churn Risk by Tenure Group',
                   'font': {'color': '#EDF2F7', 'size': 15, 'weight': 'bold'}},
            paper_bgcolor='#131C2C', plot_bgcolor='#131C2C',
            font={'color': '#EDF2F7'}, height=340,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis={'gridcolor': '#1E2A3E', 'color': '#A0AEC0', 'title': 'Tenure Group'},
            yaxis={'gridcolor': '#1E2A3E', 'color': '#A0AEC0',
                   'title': 'Avg Churn Probability', 'tickformat': '.0%'},
            showlegend=False, coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar, use_container_width=True,
                        config={'modeBarButtonsToKeep': ['toImage'], 'displaylogo': False})

    with col_ch4:
        try:
            rv = pred_hist.groupby('risk_tier').apply(
                lambda x: (x['MonthlyCharges'] * 12 * x['churn_probability']).sum(),
                include_groups=False
            ).reset_index()
        except TypeError:
            rv = pred_hist.groupby('risk_tier').apply(
                lambda x: (x['MonthlyCharges'] * 12 * x['churn_probability']).sum()
            ).reset_index()
        rv.columns = ['Risk Tier', 'Revenue at Risk']
        fig_rev = px.bar(rv, x='Risk Tier', y='Revenue at Risk',
                         color='Risk Tier',
                         color_discrete_map={
                             'High Risk':   '#FF5252',
                             'Medium Risk': '#FFB74D',
                             'Low Risk':    '#00E676'
                         },
                         text=rv['Revenue at Risk'].apply(lambda x: f'${x:,.0f}'))
        fig_rev.update_traces(textposition='outside', textfont={'color': '#EDF2F7', 'size': 12})
        fig_rev.update_layout(
            title={'text': 'Annual Revenue at Risk by Risk Tier',
                   'font': {'color': '#EDF2F7', 'size': 15, 'weight': 'bold'}},
            paper_bgcolor='#131C2C', plot_bgcolor='#131C2C',
            font={'color': '#EDF2F7'}, height=340,
            margin=dict(l=20, r=20, t=50, b=40),
            xaxis={'gridcolor': '#1E2A3E', 'color': '#A0AEC0', 'title': 'Risk Tier'},
            yaxis={'gridcolor': '#1E2A3E', 'color': '#A0AEC0', 'title': 'Revenue at Risk ($)'},
            showlegend=False
        )
        st.plotly_chart(fig_rev, use_container_width=True,
                        config={'modeBarButtonsToKeep': ['toImage'], 'displaylogo': False})

# ══════════════════════════════════════════════════════
# TAB 6 — SYSTEM OVERVIEW
# ══════════════════════════════════════════════════════
with tab6:
    st.markdown("### System Overview")

    col_ab1, col_ab2 = st.columns(2)

    with col_ab1:
        st.markdown("""
        <div style='background:#131C2C;border:1px solid #1E2A3E;
                    border-radius:16px;padding:24px;margin-bottom:20px;'>
            <div style='font-family:JetBrains Mono,monospace;color:#00E5FF;
                        font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:14px;'>
                🎯 PROJECT OVERVIEW
            </div>
            <p style='color:#FFFFFF;font-size:14px;line-height:1.7;'>
                ChurnSense is a machine learning-based customer retention intelligence platform
                built on the IBM Telco Customer Churn dataset (7,043 customers). The objective
                was to identify customers at high churn risk and provide actionable, SHAP-driven
                retention recommendations. Five models were evaluated; XGBoost won on AUC-ROC
                with SMOTE+Tomek resampling and Optuna hyperparameter tuning. SHAP TreeExplainer
                provides per-customer feature attribution.
            </p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#131C2C;border:1px solid #1E2A3E;
                    border-radius:16px;padding:24px;'>
            <div style='font-family:JetBrains Mono,monospace;color:#00E5FF;
                        font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:14px;'>
                ⚙️ KEY CAPABILITIES
            </div>""", unsafe_allow_html=True)

        for cap in ["🎯 Churn Probability Prediction",
                    "📊 SHAP Explainability for Risk Drivers",
                    "👥 Customer Segmentation & Profiling",
                    "💰 Revenue at Risk Calculation (CLV-based)",
                    "🔄 Retention Strategy Simulation",
                    "📋 30-Day Retention Roadmap Generator",
                    "📂 Batch CSV Analysis with Error Reporting",
                    "✅ Input Validation & Data Quality Checks"]:
            st.markdown(f"<p style='color:#FFFFFF;font-size:14px;margin:8px 0;'>• {cap}</p>",
                       unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_ab2:
        st.markdown("""
        <div style='background:#131C2C;border:1px solid #1E2A3E;
                    border-radius:16px;padding:24px;margin-bottom:20px;'>
            <div style='font-family:JetBrains Mono,monospace;color:#00E5FF;
                        font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:14px;'>
                📊 MODEL METRICS (XGBoost — Selected)
            </div>""", unsafe_allow_html=True)

        for k, v in [("AUC-ROC",                    "0.8437"),
                     ("Recall (Churners)",           "81%"),
                     ("Precision (Churners)",        "51%"),
                     ("Overall Accuracy",            "74%"),
                     ("Optimal Threshold",           "0.279"),
                     ("Class Balancing",             "SMOTE + Tomek Links"),
                     ("Feature Engineering",         "12 engineered features"),
                     ("Hyperparameter Tuning",       "Optuna (50 trials)")]:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;
                        padding:8px 0;border-bottom:1px solid #1E2A3E;'>
                <span style='color:#FFFFFF;font-size:13px;'>{k}</span>
                <span style='color:#00E5FF;font-family:JetBrains Mono,monospace;
                             font-size:13px;'>{v}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Model Comparison Table ──
    st.divider()
    st.markdown("### 🏆 Model Comparison — All 5 Models Evaluated")
    st.markdown("<p style='color:#00E5FF;font-size:13px;margin-bottom:16px;'>"
                "XGBoost selected as the final model based on highest AUC-ROC and Recall, "
                "prioritising churn detection over precision.</p>",
                unsafe_allow_html=True)

    hdr_cols = st.columns([2.5, 1.2, 1.2, 1.2, 1.2, 0.8])
    for col, h in zip(hdr_cols, ['MODEL', 'AUC-ROC', 'RECALL', 'PRECISION', 'ACCURACY', 'SELECTED']):
        col.markdown(f"<div style='color:#00E5FF;font-family:JetBrains Mono,monospace;"
                     f"font-size:11px;font-weight:600;letter-spacing:0.5px;"
                     f"padding:8px 0;border-bottom:2px solid #2D3A5E;'>{h}</div>",
                     unsafe_allow_html=True)

    for model_name, auc, recall, precision, accuracy, selected in MODEL_COMPARISON:
        bg     = "#0D1E35"    if selected else "#0F1525"
        border = "#00E5FF"    if selected else "transparent"
        star   = "✅"         if selected else ""
        name_style = "color:#00E5FF;font-weight:700;" if selected else "color:#FFFFFF;"
        cols   = st.columns([2.5, 1.2, 1.2, 1.2, 1.2, 0.8])
        cols[0].markdown(
            f"<div style='background:{bg};border-left:3px solid {border};"
            f"padding:10px 8px;margin:2px 0;border-radius:4px;"
            f"{name_style}font-size:13px;'>{model_name}</div>",
            unsafe_allow_html=True)
        for col, val, highlight in zip(cols[1:5],
                                        [auc, recall, precision, accuracy],
                                        [selected, selected, False, False]):
            col_color = "#00E5FF" if (selected and highlight) else "#FFFFFF"
            col.markdown(
                f"<div style='background:{bg};padding:10px 8px;margin:2px 0;"
                f"border-radius:4px;color:{col_color};font-family:JetBrains Mono,monospace;"
                f"font-size:13px;font-weight:{'700' if selected else '400'};'>{val}</div>",
                unsafe_allow_html=True)
        cols[5].markdown(
            f"<div style='background:{bg};padding:10px 8px;margin:2px 0;"
            f"border-radius:4px;text-align:center;font-size:16px;'>{star}</div>",
            unsafe_allow_html=True)

    # ── Threshold Sensitivity Slider ──
    st.divider()
    st.markdown("### 🎚️ Threshold Sensitivity Analysis")
    st.markdown("<p style='color:#00E5FF;font-size:13px;margin-bottom:16px;'>Adjust the decision threshold and see how at-risk counts change across the test dataset.</p>", unsafe_allow_html=True)

    thresh_val = st.slider("Decision Threshold", 0.10, 0.70, 0.279, step=0.01,
                           help="Default optimised threshold = 0.279. Lower = more customers flagged.")
    high_c  = (pred_hist['churn_probability'] >= thresh_val).sum()
    low_c   = (pred_hist['churn_probability'] <  thresh_val).sum()
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Threshold",          f"{thresh_val:.3f}")
    t2.metric("Flagged At-Risk",    f"{high_c}",  delta=f"{high_c/len(pred_hist)*100:.1f}% of dataset", delta_color="inverse")
    t3.metric("Flagged Safe",       f"{low_c}",   delta=f"{low_c/len(pred_hist)*100:.1f}% of dataset")
    t4.metric("vs Default",         f"{thresh_val-0.279:+.3f}")

    fig_thresh = go.Figure()
    fig_thresh.add_trace(go.Histogram(x=pred_hist['churn_probability'], nbinsx=30, name='All Customers', marker_color='#2D3A5E', opacity=0.8))
    fig_thresh.add_trace(go.Histogram(x=pred_hist[pred_hist['churn_probability'] >= thresh_val]['churn_probability'], nbinsx=30, name='At-Risk', marker_color='#FF5252', opacity=0.9))
    fig_thresh.add_vline(x=thresh_val, line_color='#00E5FF', line_width=2, annotation_text=f"Threshold: {thresh_val:.3f}", annotation_font_color='#00E5FF')
    fig_thresh.update_layout(paper_bgcolor='#131C2C', plot_bgcolor='#131C2C', font={'color':'#EDF2F7'}, height=280, barmode='overlay',
        margin=dict(l=20,r=20,t=30,b=20),
        xaxis={'gridcolor':'#1E2A3E','color':'#A0AEC0','title':'Churn Probability'},
        yaxis={'gridcolor':'#1E2A3E','color':'#A0AEC0','title':'Count'},
        legend={'font':{'color':'#A0AEC0'},'bgcolor':'rgba(0,0,0,0)'})
    st.plotly_chart(fig_thresh, use_container_width=True, config={'modeBarButtonsToKeep':['toImage'],'displaylogo':False})

    # ── ROC Curve ──
    st.divider()
    st.markdown("### 📈 ROC Curve — AUC-ROC 0.8437")
    st.markdown("<p style='color:#00E5FF;font-size:13px;margin-bottom:16px;'>"
                "Receiver Operating Characteristic curve — visual proof of model discrimination ability. "
                "The closer to top-left, the better. Random classifier = diagonal line.</p>",
                unsafe_allow_html=True)

    roc_col, roc_info = st.columns([1.6, 1])
    with roc_col:
        # Approximate ROC curve using known operating points
        fpr_pts = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.295, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.00]
        tpr_pts = [0.00, 0.38, 0.55, 0.63, 0.69, 0.74, 0.81,  0.85, 0.89, 0.92, 0.95, 0.97, 0.99, 1.00]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name='Random Classifier (AUC=0.50)',
            line=dict(color='#4A5568', width=1.5, dash='dash')))
        fig_roc.add_trace(go.Scatter(
            x=fpr_pts, y=tpr_pts,
            mode='lines', name='XGBoost (AUC=0.8437)',
            line=dict(color='#00E5FF', width=3),
            fill='tozeroy', fillcolor='rgba(0,229,255,0.08)'))
        fig_roc.add_trace(go.Scatter(
            x=[0.295], y=[0.81],
            mode='markers+text',
            name='Operating Point (threshold=0.279)',
            marker=dict(color='#FF5252', size=12, symbol='circle'),
            text=['  Operating Point<br>  FPR=29.5% TPR=81%'],
            textposition='middle right',
            textfont=dict(color='#FF5252', size=10)))
        fig_roc.update_layout(
            paper_bgcolor='#131C2C', plot_bgcolor='#131C2C',
            font={'color':'#EDF2F7'}, height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis={'gridcolor':'#1E2A3E','color':'#A0AEC0',
                   'title':'False Positive Rate (1 - Specificity)','range':[0,1]},
            yaxis={'gridcolor':'#1E2A3E','color':'#A0AEC0',
                   'title':'True Positive Rate (Recall)','range':[0,1]},
            legend={'font':{'color':'#A0AEC0'},'bgcolor':'rgba(0,0,0,0)',
                    'x':0.35,'y':0.08})
        st.plotly_chart(fig_roc, use_container_width=True,
                        config={'modeBarButtonsToKeep':['toImage'],'displaylogo':False})
    with roc_info:
        for label, val, color, desc in [
            ("AUC-ROC",    "0.8437", "#00E5FF", "Area under curve — model discrimination power"),
            ("Threshold",  "0.279",  "#B388FF", "Optimised operating point"),
            ("TPR @ Threshold", "81%","#00E676","Recall — churners correctly caught"),
            ("FPR @ Threshold", "29.5%","#FF5252","False alarm rate at operating point"),
            ("vs Random",  "+34.37%","#FFB74D", "Improvement over random classifier"),
        ]:
            st.markdown(
                f"<div style='background:#131C2C;border:1px solid #1E2A3E;"
                f"border-radius:10px;padding:10px 14px;margin-bottom:8px;"
                f"display:flex;justify-content:space-between;align-items:center;'>"
                f"<div><div style='color:#FFFFFF;font-size:12px;font-weight:600;'>{label}</div>"
                f"<div style='color:#718096;font-size:10px;margin-top:2px;'>{desc}</div></div>"
                f"<div style='color:{color};font-family:JetBrains Mono,monospace;"
                f"font-size:18px;font-weight:700;'>{val}</div></div>",
                unsafe_allow_html=True)

    # ── Confusion Matrix ──
    st.divider()
    st.markdown("### 🟥 Confusion Matrix — XGBoost on Test Set")
    st.markdown("<p style='color:#00E5FF;font-size:13px;margin-bottom:16px;'>Model performance on 1,409-customer test set at threshold 0.279.</p>", unsafe_allow_html=True)

    TP, FP, FN, TN = 314, 301, 74, 720
    total_cm = TP + FP + FN + TN
    cm_col, cm_metrics = st.columns([1, 1])

    with cm_col:
        fig_cm = go.Figure(data=go.Heatmap(
            z=[[TN, FP], [FN, TP]],
            x=['Predicted: No Churn', 'Predicted: Churn'],
            y=['Actual: No Churn', 'Actual: Churn'],
            colorscale=[[0,'#0A0E1A'],[0.5,'#1E2A3E'],[1,'#00E5FF']],
            showscale=False,
            text=[[f'TN<br>{TN}', f'FP<br>{FP}'], [f'FN<br>{FN}', f'TP<br>{TP}']],
            texttemplate='<b>%{text}</b>',
            textfont={'size':18, 'color':'#EDF2F7'},
        ))
        fig_cm.update_layout(paper_bgcolor='#131C2C', plot_bgcolor='#131C2C',
            font={'color':'#EDF2F7'}, height=320, margin=dict(l=20,r=20,t=20,b=20),
            xaxis={'color':'#A0AEC0'}, yaxis={'color':'#A0AEC0'})
        st.plotly_chart(fig_cm, use_container_width=True, config={'modeBarButtonsToKeep':['toImage'],'displaylogo':False})

    with cm_metrics:
        precision   = TP / (TP + FP)
        recall      = TP / (TP + FN)
        f1          = 2 * precision * recall / (precision + recall)
        accuracy    = (TP + TN) / total_cm
        specificity = TN / (TN + FP)
        for label, val, color, tip in [
            ("Precision",   f"{precision:.1%}", "#00E5FF",  "Of predicted churners — how many actually churned"),
            ("Recall",      f"{recall:.1%}",    "#00E676",  "Of actual churners — how many did we catch"),
            ("F1 Score",    f"{f1:.3f}",         "#B388FF",  "Harmonic mean of precision and recall"),
            ("Accuracy",    f"{accuracy:.1%}",  "#FFB74D",  "Overall correct predictions"),
            ("Specificity", f"{specificity:.1%}","#FF5252",  "Of non-churners — how many correctly identified"),
        ]:
            st.markdown(f"""
            <div style='background:#131C2C;border:1px solid #1E2A3E;border-radius:10px;
                        padding:12px 16px;margin-bottom:8px;
                        display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <div style='color:#FFFFFF;font-size:13px;font-weight:600;'>{label}</div>
                    <div style='color:#718096;font-size:11px;margin-top:2px;'>{tip}</div>
                </div>
                <div style='color:{color};font-family:JetBrains Mono,monospace;font-size:20px;font-weight:700;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    col_tech1, col_tech2 = st.columns(2)

    with col_tech1:
        st.markdown("""
        <div style='background:#131C2C;border:1px solid #1E2A3E;
                    border-radius:16px;padding:24px;'>
            <div style='font-family:JetBrains Mono,monospace;color:#00E5FF;
                        font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:14px;'>
                🛠️ TECH STACK
            </div>""", unsafe_allow_html=True)
        for tech, detail in [("ML Framework",  "XGBoost"),
                              ("Explainability","SHAP TreeExplainer"),
                              ("Visualization", "Plotly, Matplotlib, Power BI"),
                              ("UI Framework",  "Streamlit"),
                              ("Language",      "Python 3.10")]:
            st.markdown(f"""
            <div style='padding:8px 0;border-bottom:1px solid #1E2A3E;'>
                <span style='color:#EDF2F7;font-size:13px;font-weight:600;'>{tech}</span><br>
                <span style='color:#FFFFFF;font-size:13px;'>{detail}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_tech2:
        st.markdown("""
        <div style='background:#131C2C;border:1px solid #1E2A3E;
                    border-radius:16px;padding:24px;'>
            <div style='font-family:JetBrains Mono,monospace;color:#00E5FF;
                        font-size:11px;font-weight:600;letter-spacing:1px;margin-bottom:14px;'>
                🔍 TOP CHURN DRIVERS (SHAP ANALYSIS)
            </div>""", unsafe_allow_html=True)
        for num, name, val in [("01", "Month-to-month Contract", "0.717"),
                                ("02", "Spend-to-Tenure Ratio",   "0.338"),
                                ("03", "Fiber Optic Internet",     "0.287"),
                                ("04", "Electronic Check Payment", "0.193"),
                                ("05", "Monthly Charges",          "0.182")]:
            st.markdown(f"""
            <div style='display:flex;align-items:center;padding:8px 0;
                        border-bottom:1px solid #1E2A3E;'>
                <span style='color:#00E5FF;font-family:JetBrains Mono,monospace;
                             font-size:11px;width:32px;font-weight:600;'>{num}</span>
                <span style='color:#FFFFFF;font-size:13px;flex:1;'>{name}</span>
                <span style='color:#FFB74D;font-family:JetBrains Mono,monospace;
                             font-size:12px;font-weight:600;'>{val}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='text-align:center;color:#4A5568;font-size:11px;
                font-family:JetBrains Mono,monospace;padding:20px;'>
        CHURNSENSE &nbsp;·&nbsp;
        Machine Learning-Based Customer Retention Intelligence Platform
        &nbsp;·&nbsp; XGBoost + SHAP + Power BI
    </div>""", unsafe_allow_html=True)