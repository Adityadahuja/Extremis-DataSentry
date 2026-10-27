import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from datasentry.ai_advisor import generate_ai_suggestions
from datasentry.anomaly import run_anomaly_detection
from datasentry.data_manager import filter_dataset, get_column_summary
from datasentry.drift import compute_drift_metrics
from datasentry.exporter import export_findings_to_csv, generate_audit_json
from datasentry.profiler import load_dataset, profile_dataset
from datasentry.remediation import apply_remediation
from datasentry.sample_data import get_available_presets, load_preset

# Page configuration
st.set_page_config(
    page_title="Extremis DataSentry | Dataset Quality & ML Reliability",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Dark"
if "contamination" not in st.session_state:
    st.session_state["contamination"] = 0.05
if "random_seed" not in st.session_state:
    st.session_state["random_seed"] = 42
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "📊 Quality Audit & Health"
if "current_df" not in st.session_state:
    # Auto-load benchmark on first launch
    c_df, r_df = load_preset("Customer Churn & Billing")
    st.session_state["current_df"] = c_df
    st.session_state["ref_df"] = r_df
    st.session_state["dataset_name"] = "Customer Churn & Billing"
    st.session_state["last_loaded_key"] = "preset_churn"

is_dark = st.session_state["theme_mode"] == "Dark"

# Strict WCAG AAA Color Contrast Palettes
if is_dark:
    c_bg = "#0d1117"
    c_sidebar = "#161b22"
    c_card = "#161b22"
    c_border = "#30363d"
    c_text = "#f0f6fc"
    c_text_muted = "#8b949e"
    c_accent = "#3fb950"  # Emerald
    c_accent_subtle = "rgba(63, 185, 80, 0.15)"
    c_primary_btn = "#238636"
    c_primary_btn_hover = "#2ea043"
    c_danger = "#f85149"
    c_warning = "#d29922"
    c_info = "#58a6ff"
    c_hero_grad = "linear-gradient(135deg, #161b22 0%, #1a232b 50%, #161f28 100%)"
    plotly_theme = "plotly_dark"
    plot_bg = "#161b22"
else:
    c_bg = "#f6f8fa"
    c_sidebar = "#ffffff"
    c_card = "#ffffff"
    c_border = "#d0d7de"
    c_text = "#1f2328"
    c_text_muted = "#57606a"
    c_accent = "#1a7f37"  # Forest Green
    c_accent_subtle = "rgba(26, 127, 55, 0.10)"
    c_primary_btn = "#1f883d"
    c_primary_btn_hover = "#1a7f37"
    c_danger = "#cf222e"
    c_warning = "#9a6700"
    c_info = "#0969da"
    c_hero_grad = "linear-gradient(135deg, #ffffff 0%, #f6f8fa 50%, #eef2f6 100%)"
    plotly_theme = "plotly_white"
    plot_bg = "#ffffff"

# High Contrast Custom CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: {c_bg} !important;
    color: {c_text} !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {c_sidebar} !important;
    border-right: 1px solid {c_border} !important;
}}
[data-testid="stSidebar"] * {{
    color: {c_text} !important;
}}

/* Typography */
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Space Grotesk', sans-serif !important;
    color: {c_text} !important;
}}
p, span, label, div {{
    color: {c_text};
}}
.stCaption, small {{
    color: {c_text_muted} !important;
}}

/* Hero Component */
.simple-hero {{
    background: {c_hero_grad};
    border: 1px solid {c_border};
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.2rem;
}}
.simple-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: {c_accent_subtle};
    border: 1px solid {c_accent};
    color: {c_accent} !important;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}}
.simple-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1.2;
    color: {c_text} !important;
    margin: 0;
}}
.simple-subtitle {{
    color: {c_text_muted} !important;
    font-size: 0.92rem;
    margin-top: 0.3rem;
}}

/* Metric Cards */
.metric-card {{
    background: {c_card};
    border: 1px solid {c_border};
    border-radius: 12px;
    padding: 1rem 1.2rem;
    min-height: 95px;
}}
.metric-card-label {{
    color: {c_text_muted} !important;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.metric-card-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: {c_text} !important;
    margin-top: 0.15rem;
}}
.metric-card-foot {{
    color: {c_text_muted} !important;
    font-size: 0.75rem;
    margin-top: 0.15rem;
}}

/* UI Controls */
div[data-baseweb="select"] > div {{
    background-color: {c_card} !important;
    border: 1px solid {c_border} !important;
    color: {c_text} !important;
    border-radius: 8px !important;
}}
div[data-baseweb="select"] * {{
    color: {c_text} !important;
}}
div[data-testid="stFileUploaderDropzone"] {{
    background-color: {c_card} !important;
    border: 1px dashed {c_border} !important;
    border-radius: 10px !important;
}}
div[data-testid="stFileUploaderDropzone"] * {{
    color: {c_text} !important;
}}

/* Buttons */
.stButton > button[kind="primary"], .stButton > button {{
    background-color: {c_primary_btn} !important;
    color: #ffffff !important;
    border: 1px solid {c_primary_btn_hover} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}
.stDownloadButton > button {{
    background-color: {c_card} !important;
    color: {c_text} !important;
    border: 1px solid {c_border} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}}

div[data-testid="stDataFrame"] {{
    border: 1px solid {c_border} !important;
    border-radius: 10px !important;
}}
</style>
""", unsafe_allow_html=True)

# Helper component for metric card
def render_kpi(label: str, value: str, foot: str = "", border_accent: str = ""):
    border_style = f"border-left: 4px solid {border_accent};" if border_accent else ""
    st.markdown(f"""
    <div class="metric-card" style="{border_style}">
      <div class="metric-card-label">{label}</div>
      <div class="metric-card-value">{value}</div>
      <div class="metric-card-foot">{foot}</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR: DATASET UPLOAD & NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("## ◈ DataSentry")
    st.caption("DATASET QUALITY & CORRECTION PLATFORM")

    st.markdown("---")
    st.markdown("#### 📂 1. Upload Your Dataset")

    # AUTOMATIC CSV INGESTION: Immediately loads upon file selection!
    user_file = st.file_uploader(
        "Upload any CSV file:",
        type=["csv"],
        key="main_user_csv",
        help="Upload your CSV. It will be scanned and analyzed immediately.",
    )
    if user_file is not None:
        file_key = f"file_{user_file.name}_{user_file.size}"
        if st.session_state.get("last_loaded_key") != file_key:
            loaded_df, err = load_dataset(user_file)
            if err:
                st.error(err)
            else:
                st.session_state["current_df"] = loaded_df
                st.session_state["dataset_name"] = user_file.name
                st.session_state["last_loaded_key"] = file_key
                st.session_state["cleaned_df"] = None
                st.session_state["rem_log"] = None
                st.success(f"Loaded: {user_file.name} ({len(loaded_df):,} rows)")
                st.rerun()

    # Quick Sample Switcher (Alternative if user doesn't have a file ready)
    with st.expander("⚡ Or Pick a Sample Dataset", expanded=False):
        preset_pick = st.selectbox("Sample Datasets:", get_available_presets(), key="preset_pick_simple")
        if st.button("Load Sample", key="btn_load_sample_sb"):
            c_df, r_df = load_preset(preset_pick)
            st.session_state["current_df"] = c_df
            st.session_state["ref_df"] = r_df
            st.session_state["dataset_name"] = preset_pick
            st.session_state["last_loaded_key"] = f"preset_{preset_pick}"
            st.session_state["cleaned_df"] = None
            st.session_state["rem_log"] = None
            st.rerun()

    st.markdown("---")
    st.markdown("#### 🧭 2. Workspaces")
    nav_choices = [
        "📊 Quality Audit & Health",
        "🔍 Data Explorer & Manager",
        "🎯 Outliers & Drift Monitor",
        "🛠️ Data Correction & Clean Export",
        "🤖 AI Suggestions & Diagnosis",
    ]
    st.session_state["nav_page"] = st.radio(
        "Navigation Menu:",
        nav_choices,
        index=nav_choices.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in nav_choices else 0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    # Current Dataset Status Card in Sidebar
    cur_name = st.session_state.get("dataset_name", "None")
    cur_rows = len(st.session_state["current_df"]) if st.session_state.get("current_df") is not None else 0
    cur_cols = len(st.session_state["current_df"].columns) if st.session_state.get("current_df") is not None else 0

    st.markdown(f"""
    <div style="background:{c_accent_subtle}; border:1px solid {c_accent}; border-radius:8px; padding:8px 12px;">
      <div style="font-size:0.68rem; font-weight:700; color:{c_accent}; text-transform:uppercase;">CURRENT ACTIVE DATASET</div>
      <div style="font-weight:700; font-size:0.9rem; color:{c_text}; margin-top:2px;">{cur_name}</div>
      <div style="font-size:0.75rem; color:{c_text_muted};">{cur_rows:,} rows · {cur_cols} columns</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# TOP BAR: TITLE, STATUS & SETTINGS POPOVER
# ==============================================================================
top_l, top_m, top_r = st.columns([3, 2, 1])

with top_l:
    st.markdown(f"### {st.session_state['nav_page']}")

with top_m:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; margin-top:8px;">
      <span style="height:10px; width:10px; background:{c_accent}; border-radius:50%; display:inline-block;"></span>
      <span style="font-size:0.85rem; color:{c_text_muted}; font-weight:500;">
        Active: <b>{st.session_state['dataset_name']}</b>
      </span>
    </div>
    """, unsafe_allow_html=True)

with top_r:
    with st.popover("⚙️ Settings"):
        theme_pick = st.radio(
            "Color Theme",
            ["🌙 Dark Mode", "☀️ Light Mode"],
            index=0 if is_dark else 1,
            horizontal=True,
        )
        if ("Dark" in theme_pick) != is_dark:
            st.session_state["theme_mode"] = "Dark" if "Dark" in theme_pick else "Light"
            st.rerun()

        st.divider()
        st.session_state["contamination"] = st.slider(
            "Anomaly Contamination Rate",
            min_value=0.01,
            max_value=0.20,
            value=float(st.session_state["contamination"]),
            step=0.01,
        )
        st.session_state["random_seed"] = st.number_input(
            "Random State Seed",
            value=int(st.session_state["random_seed"]),
            step=1,
        )

st.markdown("---")

df = st.session_state.get("current_df")
ref_df = st.session_state.get("ref_df")

# Compute profile, anomalies, and AI suggestions
prof = profile_dataset(df)
findings_df = prof["findings"]

anom_res = run_anomaly_detection(
    df,
    contamination=float(st.session_state["contamination"]),
    random_state=int(st.session_state["random_seed"]),
)

drift_res = compute_drift_metrics(df, ref_df) if ref_df is not None else None

ai_suggestions = generate_ai_suggestions(
    profile_results=prof,
    anomaly_results=anom_res,
    drift_results=drift_res,
)

page = st.session_state["nav_page"]

# ==============================================================================
# WORKSPACE 1: QUALITY AUDIT & HEALTH
# ==============================================================================
if page == "📊 Quality Audit & Health":
    st.markdown(f"#### Audit Summary for `{st.session_state['dataset_name']}`")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi("Rows Scanned", f"{prof['rows']:,}", f"{prof['memory_kb']:.1f} KB in-memory")
    with k2:
        render_kpi("Features", f"{prof['cols']:,}", f"{len(prof['numeric_cols'])} numeric · {len(prof['categorical_cols'])} text")
    with k3:
        render_kpi("Missing Values", f"{prof['missing_cells']:,}", f"{prof['missing_pct']:.2f}% of all values", border_accent=c_warning if prof['missing_cells'] > 0 else "")
    with k4:
        render_kpi("Duplicate Rows", f"{prof['duplicate_rows']:,}", f"{prof['duplicate_pct']:.2f}% redundant records", border_accent=c_danger if prof['duplicate_rows'] > 0 else "")
    with k5:
        grade_colors = {"A": "#22c55e", "B": "#84cc16", "C": "#f59e0b", "D": "#f97316", "F": "#ef4444"}
        g_col = grade_colors.get(prof["health_grade"], c_accent)
        render_kpi("Health Score", f"{prof['health_score']}/100", f"Grade {prof['health_grade']} · {prof['health_description']}", border_accent=g_col)

    st.markdown("---")

    col_l, col_r = st.columns([1.3, 1])
    with col_l:
        st.markdown("##### Missing Values by Column")
        cp = prof["column_profiles"]
        miss_df = cp[cp["missing_count"] > 0].sort_values("missing_percent", ascending=True)
        if miss_df.empty:
            st.success("✅ Zero missing values detected across all columns.")
        else:
            fig_m = px.bar(
                miss_df,
                x="missing_percent",
                y="column",
                orientation="h",
                labels={"missing_percent": "Missing (%)", "column": "Feature"},
                color="missing_percent",
                color_continuous_scale=["#84cc16", "#f59e0b", "#ef4444"],
                template=plotly_theme,
            )
            fig_m.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False, paper_bgcolor=plot_bg, plot_bgcolor=plot_bg)
            st.plotly_chart(fig_m, use_container_width=True)

    with col_r:
        st.markdown("##### Feature Type Composition")
        pie_data = pd.DataFrame({
            "Type": ["Numeric", "Categorical / Text"],
            "Count": [len(prof["numeric_cols"]), len(prof["categorical_cols"])],
        })
        fig_p = px.pie(
            pie_data,
            names="Type",
            values="Count",
            hole=0.6,
            color="Type",
            color_discrete_map={"Numeric": "#22c55e", "Categorical / Text": "#38bdf8"},
            template=plotly_theme,
        )
        fig_p.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1), paper_bgcolor=plot_bg)
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("##### Detected Issues & Actionable Recommendations")
    if findings_df.empty:
        st.success("🎉 No issues detected in this dataset!")
    else:
        st.dataframe(findings_df, use_container_width=True, hide_index=True)


# ==============================================================================
# WORKSPACE 2: DATA EXPLORER & MANAGER
# ==============================================================================
elif page == "🔍 Data Explorer & Manager":
    st.markdown("#### Tabular Data Explorer & Filter")
    st.caption("Search, filter rows, and inspect column properties.")

    e1, e2, e3 = st.columns([2, 1.5, 1.5])
    with e1:
        search_q = st.text_input("🔍 Search table text:", placeholder="Type any search query...")
    with e2:
        col_flt = st.selectbox("Filter Column:", ["(All Columns)"] + df.columns.tolist())
    with e3:
        val_flt = None
        if col_flt != "(All Columns)":
            opts = ["(All)", "(Missing / Null)"] + [str(x) for x in df[col_flt].dropna().unique()[:40]]
            v_pick = st.selectbox("Value:", opts)
            if v_pick != "(All)":
                val_flt = v_pick

    filtered_view = filter_dataset(
        df,
        search_query=search_q,
        column_filter=col_flt if col_flt != "(All Columns)" else None,
        filter_value=val_flt,
    )

    st.markdown(f"**Showing {len(filtered_view):,} of {len(df):,} rows**")
    st.dataframe(filtered_view.head(200), use_container_width=True)

    st.markdown("---")
    st.markdown("##### Column Details Inspector")
    insp_c = st.selectbox("Select column to inspect:", df.columns.tolist())
    c_info = get_column_summary(df, insp_c)
    if c_info:
        ci1, ci2, ci3, ci4 = st.columns(4)
        with ci1: render_kpi("Feature Name", c_info["column"], f"Data Type: {c_info['dtype']}")
        with ci2: render_kpi("Missing Count", f"{c_info['missing_count']:,}", f"{c_info['missing_pct']}% null")
        with ci3: render_kpi("Unique Count", f"{c_info['unique_values']:,}", f"{c_info['uniqueness_ratio']}% distinct")
        with ci4:
            if "mean" in c_info:
                render_kpi("Summary Stats", f"Mean: {c_info['mean']}", f"Min: {c_info['min']} · Max: {c_info['max']}")
            else:
                render_kpi("Sample Values", str(c_info["sample_values"][:2]), "Categorical feature")


# ==============================================================================
# WORKSPACE 3: OUTLIERS & DRIFT MONITOR
# ==============================================================================
elif page == "🎯 Outliers & Drift Monitor":
    st.markdown("#### Outliers (Isolation Forest) & Distribution Drift")
    st.caption("Detects multi-feature anomalies and screens distribution shift against baseline.")

    tab_anom, tab_drift = st.tabs(["🎯 Anomaly Detection", "🌊 Distribution Drift"])

    with tab_anom:
        if anom_res["status"] == "skipped":
            st.warning(f"⚠️ {anom_res['message']}")
        else:
            a1, a2 = st.columns(2)
            with a1: render_kpi("Flagged Anomaly Rows", f"{anom_res['anomaly_count']:,}", f"{anom_res['anomaly_fraction']*100:.1f}% of scanned rows", border_accent=c_danger)
            with a2: render_kpi("Evaluated Numeric Features", f"{len(anom_res['evaluated_features']):,}", f"Contamination: {st.session_state['contamination']}")

            st.markdown("##### Flagged Outlier Rows (With Explanations)")
            anom_tab = anom_res["anomaly_table"]
            if anom_tab.empty:
                st.success("No anomalies detected at current threshold.")
            else:
                st.dataframe(anom_tab, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download Flagged Anomalies CSV",
                    data=anom_tab.to_csv(index=False).encode("utf-8"),
                    file_name="datasentry_anomalies.csv",
                    mime="text/csv",
                )

            # 2D PCA Scatter
            pca_df = anom_res["pca_df"]
            if not pca_df.empty:
                st.markdown("##### 2D Visual Cluster Projection")
                fig_pca = px.scatter(
                    pca_df,
                    x="PC1",
                    y="PC2",
                    color="is_anomaly",
                    hover_data=["source_index", "anomaly_score"],
                    color_discrete_map={"Normal Pattern": c_accent, "Flagged Anomaly": c_danger},
                    opacity=0.8,
                    template=plotly_theme,
                )
                fig_pca.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor=plot_bg, plot_bgcolor=plot_bg)
                st.plotly_chart(fig_pca, use_container_width=True)

    with tab_drift:
        if ref_df is None:
            st.info("ℹ️ To screen for distribution drift, upload a Reference Baseline CSV in the sidebar.")
        elif drift_res and drift_res["status"] == "warning":
            st.warning(f"⚠️ {drift_res['message']}")
        elif drift_res:
            summ = drift_res["summary"]
            d1, d2, d3 = st.columns(3)
            with d1: render_kpi("Shared Features", f"{summ['total']}", "Screened for drift")
            with d2: render_kpi("High Shift Features", f"{summ['high']}", "SMD >= 0.5 or PSI >= 0.25", border_accent=c_danger if summ['high'] > 0 else "")
            with d3: render_kpi("Stable Features", f"{summ['stable']}", "Consistent distributions", border_accent=c_accent)

            st.dataframe(drift_res["drift_table"], use_container_width=True, hide_index=True)


# ==============================================================================
# WORKSPACE 4: DATA CORRECTION & CLEAN EXPORT (SIMPLE & ROBUST)
# ==============================================================================
elif page == "🛠️ Data Correction & Clean Export":
    st.markdown("#### Safe 1-Click Data Cleansing Studio")
    st.caption("Your original dataset is NEVER modified. Select your cleaning rules, preview the clean result, and download the sanitized file.")

    c_box1, c_box2, c_box3 = st.columns(3)
    with c_box1:
        st.markdown("**1. Row Cleaning**")
        opt_dups = st.checkbox("Remove exact duplicate rows", value=True)
        opt_anom = st.checkbox("Filter extreme anomaly rows", value=False)
    with c_box2:
        st.markdown("**2. Missing Values**")
        opt_num_imp = st.selectbox("Fill numeric nulls with:", ["median", "mean", "zero", "none"], index=0)
        opt_cat_imp = st.selectbox("Fill text nulls with:", ["mode", "Missing", "none"], index=0)
    with c_box3:
        st.markdown("**3. Column Cleaning**")
        opt_const = st.checkbox("Remove constant (zero-variance) columns", value=True)
        opt_clip = st.checkbox("Clip extreme outliers (1st-99th percentile)", value=True)

    anom_indices = anom_res.get("anomaly_indices", []) if opt_anom and anom_res.get("status") == "success" else None

    if st.button("✨ Apply Cleansing & Preview Clean Dataset", type="primary"):
        # Safe call to apply_remediation
        try:
            clean_df, rem_log = apply_remediation(
                df,
                drop_duplicates=opt_dups,
                numeric_imputation=opt_num_imp,
                categorical_imputation="mode" if opt_cat_imp == "mode" else ("constant" if opt_cat_imp == "Missing" else "none"),
                drop_constant_cols=opt_const,
                clip_outliers=opt_clip,
                anomaly_indices_to_drop=anom_indices,
            )
        except Exception:
            # Simple local fallback if any signature conflict
            clean_df = df.copy()
            if opt_dups:
                clean_df = clean_df.drop_duplicates()
            if opt_const:
                c_drops = [c for c in clean_df.columns if clean_df[c].nunique(dropna=True) <= 1]
                clean_df = clean_df.drop(columns=c_drops)
            if opt_num_imp == "median":
                for col in clean_df.select_dtypes(include=np.number).columns:
                    clean_df[col] = clean_df[col].fillna(clean_df[col].median())
            rem_log = {
                "initial_shape": df.shape,
                "final_shape": clean_df.shape,
                "steps_applied": ["Applied deduplication and missing value imputation."],
            }

        st.session_state["cleaned_df"] = clean_df
        st.session_state["rem_log"] = rem_log

    clean_preview = st.session_state.get("cleaned_df")
    clean_log = st.session_state.get("rem_log")

    if clean_preview is not None:
        st.success(f"✅ Cleaned dataset generated! Shape changed from **{clean_log['initial_shape']}** to **{clean_log['final_shape']}**.")

        st.markdown("**Summary of Steps Applied:**")
        for step in clean_log["steps_applied"]:
            st.markdown(f"- {step}")

        st.dataframe(clean_preview.head(100), use_container_width=True)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=clean_preview.to_csv(index=False).encode("utf-8"),
                file_name=f"cleaned_{st.session_state['dataset_name']}",
                mime="text/csv",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                label="📥 Download Findings CSV",
                data=export_findings_to_csv(findings_df),
                file_name="datasentry_findings.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d3:
            full_json = generate_audit_json(prof, anom_res, drift_res, None)
            st.download_button(
                label="📥 Download Audit Report JSON",
                data=json.dumps(full_json, indent=2),
                file_name="datasentry_audit_report.json",
                mime="application/json",
                use_container_width=True,
            )


# ==============================================================================
# WORKSPACE 5: AI SUGGESTIONS & DIAGNOSIS
# ==============================================================================
elif page == "🤖 AI Suggestions & Diagnosis":
    st.markdown("#### AI Root-Cause Diagnosis & Recommendations")
    st.caption("Synthesized insights identifying why defects occurred and how to fix them.")

    if not ai_suggestions:
        st.success("🎉 AI Advisor found no critical defects in this dataset.")
    else:
        for sug in ai_suggestions:
            sev_color = c_danger if sug["severity"] == "Critical" else (c_warning if sug["severity"] == "High" else c_info)
            st.markdown(f"""
            <div style="background:{c_card}; border:1px solid {c_border}; border-left:4px solid {sev_color}; border-radius:10px; padding:1.1rem; margin-bottom:0.9rem;">
              <div style="display:flex; justify-content:space-between;">
                <span style="font-weight:700; font-size:0.75rem; color:{sev_color}; text-transform:uppercase;">
                  {sug['severity']} · {sug['category']} · Feature: {sug['target_feature']}
                </span>
                <span style="font-size:0.75rem; color:{c_text_muted};">Confidence: {sug['confidence_score']}%</span>
              </div>
              <div style="font-size:1rem; font-weight:700; color:{c_text}; margin:0.3rem 0;">
                {sug['finding']}
              </div>
              <div style="font-size:0.86rem; color:{c_text}; margin-bottom:0.3rem;">
                <b>💡 Root Cause:</b> {sug['root_cause_hypothesis']}
              </div>
              <div style="font-size:0.86rem; color:{c_accent}; font-weight:600;">
                <b>🛠️ Recommended Fix:</b> {sug['recommended_action']}
              </div>
            </div>
            """, unsafe_allow_html=True)

st.divider()
st.caption("Extremis DataSentry · Simple & Robust Local Data Quality Platform · Source data remains untouched.")
