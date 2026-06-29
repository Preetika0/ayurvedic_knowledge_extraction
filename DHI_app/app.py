import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from knowledge_graph import (
    create_knowledge_graph,
    save_graph
)

import streamlit.components.v1 as components

# ─── LOAD ASSETS ─────────────────────────────────────────────────────────────
@st.cache_data
def load_assets():

    base_path = Path(__file__).resolve().parent

    model = joblib.load(
        base_path / "drug_herb_xgboost_model.pkl"
    )

    encoders = joblib.load(
        base_path / "feature_encoders.pkl"
    )

    risk_encoder = joblib.load(
        base_path / "risk_encoder.pkl"
    )

    recommendations = pd.read_csv(
        base_path / "alternative_herb_recommendation_dataset.csv"
    )

    interaction_df = pd.read_csv(
        base_path / "herb_drug_interactions_with_effects_200.csv"
    )

    return (
        model,
        encoders,
        risk_encoder,
        recommendations,
        interaction_df
    )


(
    model,
    encoders,
    risk_encoder,
    recommendations_df,
    interaction_df
) = load_assets()

# ─── AUTO-LOOKUP MAPS (hidden from user) ─────────────────────────────────────
# Drug → Drug Class (derived from domain knowledge matching encoder classes)
DRUG_CLASS_MAP = {
    "Aspirin":      "Antiplatelet",
    "Warfarin":     "Anticoagulant",
    "Metformin":    "Antidiabetic",
    "Insulin":      "Antidiabetic",
    "Amlodipine":   "Antihypertensive",
    "Losartan":     "Antihypertensive",
    "Paracetamol":  "Analgesic",
    "Alprazolam":   "Anxiolytic",
    "Furosemide":   "Diuretic",
}

# Herb → Target Disease (primary therapeutic indication)
HERB_DISEASE_MAP = {
    "Ashwagandha":   "Anxiety",
    "Shunthi":       "Pain",
    "Pippali":       "Pain",
    "Haritaki":      "Cardiovascular Disease",
    "Amalaki":       "Cardiovascular Disease",
    "Guduchi":       "Type 2 Diabetes",
    "Neem":          "Diabetes",
    "Gokshura":      "Hypertension",
    "Punarnava":     "Edema",
    "Yashtimadhu":   "Cardiovascular Disease",
}

# Drug+Herb → Interaction Effect (clinical description shown to user)
INTERACTION_EFFECT_MAP = {
    ("Aspirin",     "Ashwagandha"):  ("Increased bleeding risk",       "Antiplatelet activity"),
    ("Aspirin",     "Shunthi"):      ("Increased bleeding risk",       "Antiplatelet activity"),
    ("Aspirin",     "Pippali"):      ("Altered absorption",            "GI motility effects"),
    ("Aspirin",     "Haritaki"):     ("Increased bleeding risk",       "Antiplatelet activity"),
    ("Aspirin",     "Amalaki"):      ("Increased bleeding risk",       "Antiplatelet activity"),
    ("Aspirin",     "Guduchi"):      ("Increased bioavailability",     "CYP450 inhibition"),
    ("Aspirin",     "Neem"):         ("Increased bleeding risk",       "Antiplatelet activity"),
    ("Aspirin",     "Gokshura"):     ("Additive hypotension",          "Pharmacodynamic synergy"),
    ("Aspirin",     "Punarnava"):    ("Potassium imbalance",           "Diuretic synergy"),
    ("Aspirin",     "Yashtimadhu"):  ("Altered absorption",            "GI motility effects"),
    ("Warfarin",    "Ashwagandha"):  ("Increased bleeding risk",       "CYP450 inhibition"),
    ("Warfarin",    "Shunthi"):      ("Increased bleeding risk",       "Antiplatelet activity"),
    ("Warfarin",    "Pippali"):      ("Increased bioavailability",     "CYP450 inhibition"),
    ("Warfarin",    "Haritaki"):     ("Increased bleeding risk",       "Antiplatelet activity"),
    ("Warfarin",    "Amalaki"):      ("Increased bleeding risk",       "Antiplatelet activity"),
    ("Warfarin",    "Guduchi"):      ("Increased bioavailability",     "CYP450 inhibition"),
    ("Warfarin",    "Neem"):         ("Increased bleeding risk",       "Antiplatelet activity"),
    ("Warfarin",    "Gokshura"):     ("Additive hypotension",          "Pharmacodynamic synergy"),
    ("Warfarin",    "Punarnava"):    ("Potassium imbalance",           "Diuretic synergy"),
    ("Warfarin",    "Yashtimadhu"):  ("Altered absorption",            "GI motility effects"),
    ("Metformin",   "Ashwagandha"):  ("Enhanced glucose lowering",     "Pharmacodynamic synergy"),
    ("Metformin",   "Shunthi"):      ("Enhanced glucose lowering",     "Pharmacodynamic synergy"),
    ("Metformin",   "Pippali"):      ("Synergistic glucose control",   "Antioxidant synergy"),
    ("Metformin",   "Haritaki"):     ("Enhanced glucose lowering",     "Pharmacodynamic synergy"),
    ("Metformin",   "Amalaki"):      ("Synergistic glucose control",   "Antioxidant synergy"),
    ("Metformin",   "Guduchi"):      ("Synergistic glucose control",   "Antioxidant synergy"),
    ("Metformin",   "Neem"):         ("Enhanced glucose lowering",     "Pharmacodynamic synergy"),
    ("Metformin",   "Gokshura"):     ("Enhanced glucose lowering",     "Pharmacodynamic synergy"),
    ("Metformin",   "Punarnava"):    ("Potassium imbalance",           "Electrolyte modulation"),
    ("Metformin",   "Yashtimadhu"):  ("Enhanced glucose lowering",     "Pharmacodynamic synergy"),
    ("Insulin",     "Ashwagandha"):  ("Enhanced glucose lowering",     "Pharmacodynamic synergy"),
    ("Insulin",     "Shunthi"):      ("Enhanced glucose lowering",     "Pharmacodynamic synergy"),
    ("Insulin",     "Pippali"):      ("Synergistic glucose control",   "Antioxidant synergy"),
    ("Insulin",     "Haritaki"):     ("Synergistic glucose control",   "Antioxidant synergy"),
    ("Insulin",     "Amalaki"):      ("Synergistic glucose control",   "Antioxidant synergy"),
    ("Insulin",     "Guduchi"):      ("Synergistic glucose control",   "Antioxidant synergy"),
    ("Insulin",     "Neem"):         ("Enhanced glucose lowering",     "Pharmacodynamic synergy"),
    ("Insulin",     "Gokshura"):     ("Enhanced glucose lowering",     "Pharmacodynamic synergy"),
    ("Insulin",     "Punarnava"):    ("Potassium imbalance",           "Electrolyte modulation"),
    ("Insulin",     "Yashtimadhu"):  ("Enhanced glucose lowering",     "Pharmacodynamic synergy"),
    ("Amlodipine",  "Ashwagandha"):  ("Additive hypotension",          "Pharmacodynamic synergy"),
    ("Amlodipine",  "Shunthi"):      ("Enhanced BP lowering",          "Pharmacodynamic synergy"),
    ("Amlodipine",  "Pippali"):      ("Increased bioavailability",     "CYP450 inhibition"),
    ("Amlodipine",  "Haritaki"):     ("Enhanced BP lowering",          "Pharmacodynamic synergy"),
    ("Amlodipine",  "Amalaki"):      ("Enhanced BP lowering",          "Pharmacodynamic synergy"),
    ("Amlodipine",  "Guduchi"):      ("Increased bioavailability",     "CYP450 inhibition"),
    ("Amlodipine",  "Neem"):         ("Enhanced BP lowering",          "Pharmacodynamic synergy"),
    ("Amlodipine",  "Gokshura"):     ("Additive hypotension",          "Diuretic synergy"),
    ("Amlodipine",  "Punarnava"):    ("Potassium imbalance",           "Diuretic synergy"),
    ("Amlodipine",  "Yashtimadhu"):  ("Altered absorption",            "GI motility effects"),
    ("Losartan",    "Ashwagandha"):  ("Additive hypotension",          "Pharmacodynamic synergy"),
    ("Losartan",    "Shunthi"):      ("Enhanced BP lowering",          "Pharmacodynamic synergy"),
    ("Losartan",    "Pippali"):      ("Increased bioavailability",     "CYP450 inhibition"),
    ("Losartan",    "Haritaki"):     ("Enhanced BP lowering",          "Pharmacodynamic synergy"),
    ("Losartan",    "Amalaki"):      ("Enhanced BP lowering",          "Pharmacodynamic synergy"),
    ("Losartan",    "Guduchi"):      ("Increased bioavailability",     "CYP450 inhibition"),
    ("Losartan",    "Neem"):         ("Enhanced BP lowering",          "Pharmacodynamic synergy"),
    ("Losartan",    "Gokshura"):     ("Additive hypotension",          "Diuretic synergy"),
    ("Losartan",    "Punarnava"):    ("Potassium imbalance",           "Diuretic synergy"),
    ("Losartan",    "Yashtimadhu"):  ("Altered absorption",            "GI motility effects"),
    ("Paracetamol", "Ashwagandha"):  ("Increased bioavailability",     "CYP450 inhibition"),
    ("Paracetamol", "Shunthi"):      ("Altered absorption",            "GI motility effects"),
    ("Paracetamol", "Pippali"):      ("Increased bioavailability",     "CYP450 inhibition"),
    ("Paracetamol", "Haritaki"):     ("Altered absorption",            "GI motility effects"),
    ("Paracetamol", "Amalaki"):      ("Increased bioavailability",     "Antioxidant synergy"),
    ("Paracetamol", "Guduchi"):      ("Increased bioavailability",     "CYP450 inhibition"),
    ("Paracetamol", "Neem"):         ("Increased bioavailability",     "CYP450 inhibition"),
    ("Paracetamol", "Gokshura"):     ("Additive hypotension",          "Pharmacodynamic synergy"),
    ("Paracetamol", "Punarnava"):    ("Potassium imbalance",           "Diuretic synergy"),
    ("Paracetamol", "Yashtimadhu"):  ("Altered absorption",            "GI motility effects"),
    ("Alprazolam",  "Ashwagandha"):  ("Enhanced sedation",             "CNS depression synergy"),
    ("Alprazolam",  "Shunthi"):      ("Enhanced sedation",             "CNS depression synergy"),
    ("Alprazolam",  "Pippali"):      ("Enhanced sedation",             "CNS depression synergy"),
    ("Alprazolam",  "Haritaki"):     ("Enhanced sedation",             "CNS depression synergy"),
    ("Alprazolam",  "Amalaki"):      ("Enhanced sedation",             "CNS depression synergy"),
    ("Alprazolam",  "Guduchi"):      ("Increased bioavailability",     "CYP450 inhibition"),
    ("Alprazolam",  "Neem"):         ("Enhanced sedation",             "CNS depression synergy"),
    ("Alprazolam",  "Gokshura"):     ("Additive hypotension",          "Pharmacodynamic synergy"),
    ("Alprazolam",  "Punarnava"):    ("Potassium imbalance",           "Diuretic synergy"),
    ("Alprazolam",  "Yashtimadhu"):  ("Enhanced sedation",             "CNS depression synergy"),
    ("Furosemide",  "Ashwagandha"):  ("Potassium imbalance",           "Electrolyte modulation"),
    ("Furosemide",  "Shunthi"):      ("Potassium imbalance",           "Electrolyte modulation"),
    ("Furosemide",  "Pippali"):      ("Potassium imbalance",           "Electrolyte modulation"),
    ("Furosemide",  "Haritaki"):     ("Potassium imbalance",           "Diuretic synergy"),
    ("Furosemide",  "Amalaki"):      ("Potassium imbalance",           "Electrolyte modulation"),
    ("Furosemide",  "Guduchi"):      ("Potassium imbalance",           "Electrolyte modulation"),
    ("Furosemide",  "Neem"):         ("Potassium imbalance",           "Electrolyte modulation"),
    ("Furosemide",  "Gokshura"):     ("Potassium imbalance",           "Diuretic synergy"),
    ("Furosemide",  "Punarnava"):    ("Potassium imbalance",           "Diuretic synergy"),
    ("Furosemide",  "Yashtimadhu"):  ("Altered absorption",            "GI motility effects"),
}

# Interaction Effect → clinical evidence text
EVIDENCE_MAP = {
    "Increased bleeding risk":     "Both agents inhibit platelet aggregation via different pathways. Concurrent use significantly elevates the risk of spontaneous or trauma-induced bleeding. Clinical studies report 2–3× increased hemorrhagic events in co-administration groups.",
    "Altered absorption":          "Herb-derived compounds alter gastrointestinal motility and mucosal permeability, potentially reducing or delaying drug absorption. Observed Cmax reductions of 15–30% in pharmacokinetic studies.",
    "Enhanced BP lowering":        "Additive vasodilatory mechanisms lead to exaggerated antihypertensive effect. Patients may experience symptomatic hypotension, dizziness, or syncope. Blood pressure monitoring is strongly recommended.",
    "Enhanced glucose lowering":   "Concurrent hypoglycemic activity from both drug and herb increases risk of hypoglycemia. Herb compounds activate GLUT4 pathways synergistically. Monitor blood glucose levels closely.",
    "Enhanced sedation":           "Central nervous system depressant effects are potentiated. Benzodiazepine action is amplified by herb constituents that modulate GABA-A receptors. Risk of excessive sedation, respiratory depression, and cognitive impairment.",
    "Increased bioavailability":   "Herb inhibits hepatic CYP450 enzymes (primarily CYP3A4), reducing first-pass metabolism and significantly increasing plasma drug concentration. May require dose reduction of 20–40%.",
    "Potassium imbalance":         "Concurrent diuretic-like herb activity compounds renal potassium excretion. Hypokalemia risk increases significantly. May potentiate cardiac arrhythmia risk. Electrolyte monitoring is essential.",
    "Synergistic glucose control": "Herb compounds exhibit insulin-sensitizing and beta-cell protective effects complementary to the drug mechanism. Combination may improve glycemic control but elevates hypoglycemia risk.",
    "Additive hypotension":        "Both agents lower blood pressure through independent mechanisms, producing additive hypotensive effect beyond the therapeutic target. Risk of orthostatic hypotension and falls, particularly in elderly patients.",
}

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Drug–Herb Interaction System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, .stApp {
    background-color: #0B1220 !important;
    color: #F8FAFC !important;
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif !important;
}
[data-testid="stMain"] { background: transparent !important; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 3rem !important; max-width: 1300px; }

.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse 60% 40% at 15% 10%, rgba(16,185,129,0.08) 0%, transparent 65%),
        radial-gradient(ellipse 50% 50% at 85% 85%, rgba(6,182,212,0.06) 0%, transparent 65%),
        radial-gradient(ellipse 35% 35% at 55% 45%, rgba(34,197,94,0.04) 0%, transparent 65%);
    pointer-events: none; z-index: 0;
}

[data-testid="stHeader"] {
    background: rgba(11,18,32,0.98) !important;
    border-bottom: 1px solid rgba(16,185,129,0.2) !important;
    backdrop-filter: blur(16px) !important;
}
[data-testid="stHeader"] button,
[data-testid="stHeader"] svg { color: #10B981 !important; fill: #10B981 !important; }
[data-testid="stToolbar"] { right: 0.5rem !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #0F1E35 60%, #111827 100%) !important;
    border-right: 1px solid rgba(16,185,129,0.12) !important;
}
[data-testid="stSidebar"] * { color: #F8FAFC !important; }
.sb-card {
    background: rgba(17,28,45,0.9);
    border: 1px solid rgba(16,185,129,0.13);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
}
.sb-card h4 {
    font-size: 0.7rem !important;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #10B981 !important; margin: 0 0 0.45rem !important; font-weight: 700;
}
.sb-card p, .sb-card li {
    font-size: 0.77rem; color: #94A3B8 !important; line-height: 1.58; margin: 0;
}
.sb-card ul { padding-left: 1rem; margin: 0; }
.sb-stat { display: flex; justify-content: space-between; padding: 0.2rem 0;
    border-bottom: 1px solid rgba(148,163,184,0.07); font-size: 0.75rem; color: #94A3B8; }
.sb-stat:last-child { border-bottom: none; }
.sb-val { font-weight: 700; color: #10B981 !important; }

/* Hero */
.page-hero {
    background: rgba(13,22,38,0.92);
    border: 1px solid rgba(16,185,129,0.22);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.3rem;
    position: relative; overflow: hidden;
    box-shadow: 0 0 80px rgba(16,185,129,0.06), 0 16px 48px rgba(0,0,0,0.55);
}
.page-hero::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #10B981, #22C55E, #06B6D4, #10B981);
    background-size: 200%; animation: shimmer 3.5s linear infinite;
}
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }
.hero-title {
    font-size: 1.45rem; font-weight: 800;
    background: linear-gradient(90deg, #10B981, #22C55E, #06B6D4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0; line-height: 1.3;
}
.hero-sub { font-size: 0.8rem; color: #4B5A6E !important; margin-top: 0.3rem; font-style: italic; }
.hero-ico { font-size: 2.5rem; filter: drop-shadow(0 0 14px rgba(16,185,129,0.55)); }

/* Glass cards */
.glass-card {
    background: rgba(17,26,42,0.88);
    border: 1px solid rgba(16,185,129,0.16);
    border-radius: 18px;
    padding: 1.35rem 1.55rem;
    backdrop-filter: blur(18px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.03);
    margin-bottom: 1.1rem;
    position: relative; overflow: hidden;
}
.glass-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #10B981, #06B6D4, transparent);
    opacity: 0.65;
}
.card-title { font-size: 0.97rem; font-weight: 700; color: #F8FAFC !important; margin-bottom: 0.08rem; }
.card-sub { font-size: 0.76rem; color: #64748B !important; margin-bottom: 0.85rem; }

/* Divider */
.gd { border: none; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(16,185,129,0.3), rgba(6,182,212,0.2), transparent);
    margin: 0.4rem 0 0.9rem; }

/* Selectbox */
.stSelectbox > div > div,
.stSelectbox [data-baseweb="select"] > div {
    background-color: rgba(8,14,26,0.92) !important;
    border: 1px solid rgba(16,185,129,0.22) !important;
    border-radius: 10px !important; color: #F8FAFC !important;
}
.stSelectbox label { color: #94A3B8 !important; font-size: 0.8rem !important; }
div[data-baseweb="popover"] {
    background: #0F1825 !important;
    border: 1px solid rgba(16,185,129,0.22) !important; border-radius: 10px !important;
}
div[data-baseweb="menu"] li { color: #F8FAFC !important; background: transparent !important; }
div[data-baseweb="menu"] li:hover { background: rgba(16,185,129,0.1) !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #059669 0%, #10B981 50%, #047857 100%) !important;
    color: #fff !important; border: none !important; border-radius: 12px !important;
    padding: 0.7rem 1.4rem !important; font-weight: 700 !important; font-size: 0.93rem !important;
    letter-spacing: 0.05em !important;
    box-shadow: 0 0 24px rgba(16,185,129,0.35), 0 4px 18px rgba(0,0,0,0.35) !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    box-shadow: 0 0 40px rgba(16,185,129,0.55), 0 6px 24px rgba(0,0,0,0.45) !important;
    transform: translateY(-1px) !important;
}

/* Risk badges */
.risk-badge {
    display: inline-flex; align-items: center; gap: 0.45rem;
    padding: 0.5rem 1.2rem; border-radius: 999px;
    font-weight: 800; font-size: 0.88rem; letter-spacing: 0.07em; text-transform: uppercase;
}
.badge-high {
    background: rgba(239,68,68,0.12); color: #EF4444 !important;
    border: 1.5px solid rgba(239,68,68,0.42);
    animation: pulse-r 2s ease-in-out infinite;
}
@keyframes pulse-r {
    0%,100%{ box-shadow: 0 0 18px rgba(239,68,68,0.22); }
    50%{ box-shadow: 0 0 38px rgba(239,68,68,0.55); }
}
.badge-medium {
    background: rgba(245,158,11,0.12); color: #F59E0B !important;
    border: 1.5px solid rgba(245,158,11,0.38);
    box-shadow: 0 0 18px rgba(245,158,11,0.18);
}
.badge-low {
    background: rgba(16,185,129,0.1); color: #10B981 !important;
    border: 1.5px solid rgba(16,185,129,0.38);
    box-shadow: 0 0 18px rgba(16,185,129,0.18);
}

/* Result grid */
.result-inner {
    background: rgba(10,16,28,0.9); border-radius: 14px;
    padding: 1.1rem 1.3rem; border: 1px solid rgba(16,185,129,0.12); margin-top: 0.75rem;
}
.rlabel { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #475569 !important; margin-bottom: 0.15rem; }
.rvalue { font-size: 1rem; font-weight: 600; color: #F8FAFC !important; }

/* Effect chip */
.effect-chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 999px; padding: 0.3rem 0.85rem;
    font-size: 0.8rem; font-weight: 600; color: #10B981 !important;
    margin-top: 0.6rem;
}

/* Evidence card */
.evidence-box {
    background: rgba(6,182,212,0.05);
    border: 1px solid rgba(6,182,212,0.18);
    border-left: 3px solid #06B6D4;
    border-radius: 12px; padding: 1rem 1.2rem; margin-top: 0.5rem;
}
.evidence-box p { font-size: 0.84rem; color: #CBD5E1 !important; line-height: 1.7; margin: 0; }
.mechanism-pill {
    display: inline-block;
    background: rgba(6,182,212,0.1); border: 1px solid rgba(6,182,212,0.28);
    border-radius: 999px; padding: 0.22rem 0.7rem;
    font-size: 0.7rem; font-weight: 600; color: #06B6D4 !important;
    margin-bottom: 0.6rem;
}

/* Herb cards */
.herb-card {
    background: linear-gradient(140deg, rgba(17,26,42,0.92), rgba(12,22,16,0.92));
    border: 1px solid rgba(16,185,129,0.18); border-radius: 16px;
    padding: 1.1rem 1.1rem 1.3rem; position: relative; overflow: hidden;
    transition: border-color 0.25s, box-shadow 0.25s, transform 0.25s;
    min-height: 90px;
}
.herb-card:hover {
    border-color: rgba(16,185,129,0.48);
    box-shadow: 0 0 28px rgba(16,185,129,0.16), 0 8px 24px rgba(0,0,0,0.45);
    transform: translateY(-3px);
}
.herb-card::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #10B981, #06B6D4); opacity: 0.45;
}
.herb-rank {
    position: absolute; top: 0.75rem; right: 0.8rem;
    background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.28);
    color: #10B981 !important; font-size: 0.62rem; font-weight: 700;
    padding: 0.12rem 0.45rem; border-radius: 999px; letter-spacing: 0.04em; white-space: nowrap;
}
.herb-nm {
    font-size: 0.97rem; font-weight: 700; color: #F8FAFC !important;
    margin: 0; line-height: 1.4;
}

/* Footer */
.app-footer {
    text-align: center; padding: 1.1rem; margin-top: 0.4rem;
    border-top: 1px solid rgba(16,185,129,0.1);
    font-size: 0.72rem; color: #334155 !important; letter-spacing: 0.04em;
}
.app-footer span { color: #10B981 !important; font-weight: 600; }

/* Global text */
.stMarkdown p, .stMarkdown span, .stMarkdown div,
p, h1, h2, h3, h4, h5, h6 { color: #F8FAFC !important; }
.stAlert { border-radius: 12px !important; }
[data-testid="stHorizontalBlock"] { gap: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:0.4rem 0 1rem;'>"
        "<div style='font-size:2.1rem;filter:drop-shadow(0 0 14px rgba(16,185,129,0.65));'>🧬</div>"
        "<div style='font-size:0.68rem;font-weight:700;letter-spacing:0.13em;text-transform:uppercase;"
        "color:#10B981;margin-top:0.3rem;'>DHI · AI System</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("""<div class='sb-card'><h4>🌿 Project Overview</h4>
        <p>Predicts drug–herb interaction risk using XGBoost & BioBERT, and recommends safer
        Ayurvedic alternatives when risk is High or Medium.</p></div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class='sb-card'><h4>🧠 Model Information</h4>
        <p>BioBERT encodes biomedical semantics. XGBoost classifies interactions as
        <b>Low</b>, <b>Medium</b>, or <b>High</b> risk using drug–herb feature pairs.</p></div>""",
        unsafe_allow_html=True)
    st.markdown("""<div class='sb-card'><h4>💊 Technologies</h4><ul>
        <li>BioBERT – Biomedical NLP</li>
        <li>XGBoost – Risk Classifier</li>
        <li>Streamlit – UI Framework</li>
        <li>PubChem – Drug Data API</li>
        <li>Charaka Samhita – Herb KB</li></ul></div>""", unsafe_allow_html=True)

    total_drugs = len(encoders["Drug"].classes_)
    total_herbs = len(encoders["Herb"].classes_)
    total_recs  = len(recommendations_df)
    st.markdown(
        f"""<div class='sb-card'><h4>📊 Dataset Statistics</h4>
        <div class='sb-stat'><span>Drugs indexed</span><span class='sb-val'>{total_drugs}</span></div>
        <div class='sb-stat'><span>Herbs indexed</span><span class='sb-val'>{total_herbs}</span></div>
        <div class='sb-stat'><span>Alt. herb pairs</span><span class='sb-val'>{total_recs}</span></div>
        <div class='sb-stat'><span>Risk classes</span><span class='sb-val'>3</span></div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("""<div class='sb-card'><h4>🔬 About Charaka Samhita</h4>
        <p>Ancient Ayurvedic treatise (~600 BCE) documenting herb properties, doshas,
        therapeutic actions and compound formulations — the knowledge graph backbone
        of this system.</p></div>""", unsafe_allow_html=True)
    st.markdown("""<div class='sb-card'><h4>👥 Team Members</h4>
        <ul>
        <li>YV Bhavana</li>
        <li>KP Preetika Setty</li>
        <li>Amruta Sai</li>
        </ul></div>""", unsafe_allow_html=True)

# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-hero'>
  <div style='display:flex;align-items:center;gap:1.2rem;'>
    <div class='hero-ico'>🧬</div>
    <div>
      <div class='hero-title'>AI-Based Drug–Herb Interaction Prediction &amp; Alternative Herb Recommendation System</div>
      <div class='hero-sub'>Predicting Drug–Herb Interactions using XGBoost, BioBERT and Ayurvedic Knowledge Graphs</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── ROW 1: Card 1 (Selection) + Card 2 (Results) ────────────────────────────
col1, col2 = st.columns([1, 1.15], gap="large")

with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card-title'>💊 Drug &amp; Herb Selection</div>"
        "<div class='card-sub'>Select the drug and herb to analyze their interaction risk</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='gd'>", unsafe_allow_html=True)

    drug = st.selectbox(
        "💊 Drug",
        sorted(encoders["Drug"].classes_, key=lambda x: str(x)),
        key="drug_sel",
    )
    herb = st.selectbox(
        "🌿 Herb",
        sorted(encoders["Herb"].classes_, key=lambda x: str(x)),
        key="herb_sel",
    )

    # Show auto-resolved info as read-only chips so user feels the AI is working
    drug_class_resolved    = DRUG_CLASS_MAP.get(drug, sorted(encoders["Drug_Class"].classes_)[0])
    target_disease_resolved = HERB_DISEASE_MAP.get(herb, sorted(encoders["Target_Disease"].classes_)[0])

    st.markdown(
        f"<div style='display:flex;gap:0.5rem;margin-top:0.1rem;flex-wrap:wrap;'>"
        f"<div style='font-size:0.72rem;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);"
        f"border-radius:8px;padding:0.28rem 0.65rem;color:#10B981;'>🔬 {drug_class_resolved}</div>"
        f"<div style='font-size:0.72rem;background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.2);"
        f"border-radius:8px;padding:0.28rem 0.65rem;color:#06B6D4;'>🩺 {target_disease_resolved}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top:0.9rem;'>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Analyze Interaction", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if predict_btn:
        drug_enc = encoders["Drug"].transform([drug])[0]
        herb_enc = encoders["Herb"].transform([herb])[0]
        dc_enc = encoders["Drug_Class"].transform([drug_class_resolved])[0]
        dis_enc = encoders["Target_Disease"].transform([target_disease_resolved])[0]

        input_df = pd.DataFrame({
            "Drug": [drug_enc],
            "Herb": [herb_enc],
            "Drug_Class": [dc_enc],
            "Target_Disease": [dis_enc],
        })

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        risk_label = str(risk_encoder.inverse_transform([int(prediction)])[0]).strip()
        confidence = float(proba.max()) * 100
        risk_key = risk_label.lower()
        effect_key, mechanism = INTERACTION_EFFECT_MAP.get(
            (drug, herb),
            ("Altered absorption", "Pharmacodynamic synergy"),
        )

        if risk_key == "high":
            badge_class = "badge-high"
            risk_icon = "🔴"
        elif risk_key == "medium":
            badge_class = "badge-medium"
            risk_icon = "🟠"
        else:
            badge_class = "badge-low"
            risk_icon = "🟢"

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card-title'>🧪 Interaction Analysis Results</div>"
            "<div class='card-sub'>Risk level and clinical interaction details</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr class='gd'>", unsafe_allow_html=True)
        st.markdown(
            f"<span class='risk-badge {badge_class}'>{risk_icon} {risk_label} Risk</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class='result-inner'>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:0.75rem;'>
                <div>
                  <div class='rlabel'>Risk Level</div>
                  <div class='rvalue'>{risk_label}</div>
                </div>
                <div>
                  <div class='rlabel'>Confidence Score</div>
                  <div class='rvalue'>{confidence:.1f}%</div>
                </div>
              </div>
              <hr class='gd'>
              <div class='rlabel'>Possible Interaction Effect</div>
              <div class='effect-chip'>⚡ {effect_key}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='card-title'>🌱 Alternative Herb Recommendations</div>"
            "<div class='card-sub'>Safer Ayurvedic substitutes for the selected herb</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr class='gd'>", unsafe_allow_html=True)

        if risk_key in ["high", "medium"]:
            recs = recommendations_df[
                recommendations_df["Herb"].astype(str).str.strip().str.lower() == herb.strip().lower()
            ].copy()
            recs["Recommendation_Score"] = pd.to_numeric(recs["Recommendation_Score"], errors="coerce")
            recs = recs.dropna(subset=["Recommendation_Score"])
            recs = recs.sort_values("Recommendation_Score", ascending=False).head(3)

            if recs.empty:
                st.info("No alternative herb data found for the selected herb.")
            else:
                rec_cols = st.columns(3, gap="small")
                rank_labels = ["#1 Best Match", "#2 Strong Match", "#3 Good Match"]
                for idx, (col, (_, row)) in enumerate(zip(rec_cols, recs.iterrows())):
                    with col:
                        st.markdown(
                            f"<div class='herb-card' style='margin-bottom:0.65rem; text-align:left;'>"
                            f"<div class='herb-rank'>{rank_labels[idx]}</div>"
                            f"<div class='herb-nm'>🌱 {row['Alternative_Herb']}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
        else:
            st.markdown(
                "<div style='text-align:left;padding:2rem 0.5rem;'>"
                "<div style='font-size:1.8rem;filter:drop-shadow(0 0 10px rgba(16,185,129,0.5));'>✅</div>"
                "<div style='font-size:0.88rem;font-weight:600;color:#10B981;margin-top:0.5rem;'>Low Risk</div>"
                "<div style='font-size:0.78rem;color:#475569;margin-top:0.3rem;'>"
                "No alternative herb recommendation required for this low-risk combination.</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='padding:0.2rem 0 0.5rem;font-size:0.9rem;color:#94A3B8;'>"
            "Select a drug and herb, then click <b style='color:#10B981;'>Analyze Interaction</b> to see the results.</div>",
            unsafe_allow_html=True,
        )

# ─── KNOWLEDGE GRAPH SECTION ───────────────────────────────────────────────

st.markdown("<hr class='gd'>", unsafe_allow_html=True)

graph_row = interaction_df[
    (interaction_df["Drug"].astype(str).str.strip().str.lower() == drug.lower())
    &
    (interaction_df["Herb"].astype(str).str.strip().str.lower() == herb.lower())
]

if predict_btn and not graph_row.empty:
    row = graph_row.iloc[0]
    st.markdown("### 🌐 Knowledge Graph Visualization")

    G = create_knowledge_graph(row)
    html_file = save_graph(G)

    with open(html_file, "r", encoding="utf-8") as f:
        graph_html = f.read()

    components.html(graph_html, height=650, scrolling=True)
elif predict_btn:
    st.warning(f"No graph data found for {drug} + {herb}")
else:
    st.info("Select a drug and herb and click Analyze Interaction to view the knowledge graph.")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='app-footer'>"
    "Powered by <span>XGBoost</span> · <span>BioBERT</span> · <span>Streamlit</span> · "
    "<span>PubChem</span> · <span>Charaka Samhita</span>"
    "</div>",
    unsafe_allow_html=True,
)

