from __future__ import annotations

import glob
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import streamlit as st
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss
from utils.chat_engine import run_chatbot


# -----------------------------
# Page config + theme
# -----------------------------
st.set_page_config(
    page_title="Integrated Thailand Stress + Global Country Network",
    page_icon="🌍",
    layout="wide",
)

st.markdown(
    """
<style>
header[data-testid="stHeader"] {
    background: linear-gradient(180deg, #0b1220 0%, #0a0f18 100%) !important;
    border-bottom: 1px solid rgba(251, 191, 36, 0.12);
}
[data-testid="stDecoration"] { display: none; }
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #1a2744 0%, #0d1526 45%, #070b12 100%);
    color: #e2e8f0;
}
.main .block-container, .block-container {
    max-width: 1500px;
    padding-top: 1.55rem !important;
    padding-bottom: 1.5rem;
}
.kicker {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.hero {
    background: linear-gradient(105deg, #fbbf24 0%, #f59e0b 55%, #ea580c 100%);
    border-radius: 14px;
    color: #0f172a;
    text-align: center;
    font-weight: 800;
    font-size: clamp(16px, 1.75vw, 30px);
    line-height: 1.25;
    padding: 16px 20px;
    margin-bottom: 10px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 4px 24px rgba(245, 158, 11, 0.15);
}
.tagline { text-align: center; color: #cbd5e1; font-size: 13px; margin-bottom: 14px; }
.section-label {
    color: #fbbf24;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 18px 0 8px 0;
}
.card {
    background: linear-gradient(160deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(251, 191, 36, 0.12);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.card h3, .card h4 { color: #fde68a; margin: 0 0 8px 0; }
.card p, .card li, .card div, .card span { color: #cbd5e1; }
.sig {
    border-left: 4px solid #f59e0b;
    background: rgba(245, 158, 11, 0.12);
    color: #fef3c7;
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 13px;
    margin-top: 8px;
}
.badge {
    display: inline-block;
    border-radius: 999px;
    padding: 4px 10px;
    margin-right: 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.badge-measured { background: rgba(34,197,94,.18); color: #bbf7d0; border: 1px solid rgba(34,197,94,.35); }
.badge-estimated { background: rgba(59,130,246,.18); color: #bfdbfe; border: 1px solid rgba(59,130,246,.35); }
.badge-note { background: rgba(245,158,11,.18); color: #fde68a; border: 1px solid rgba(245,158,11,.35); }
.badge-upload { background: rgba(236,72,153,.18); color: #fbcfe8; border: 1px solid rgba(236,72,153,.35); }
.small-muted { color: #94a3b8; font-size: 12px; }
.chat-fab {
    position: fixed;
    right: 22px;
    bottom: 22px;
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: linear-gradient(145deg, #fde047, #f59e0b);
    color: #111827 !important;
    font-size: 28px;
    text-decoration: none !important;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 10px 28px rgba(0,0,0,0.45);
    z-index: 9999;
    border: 3px solid #fff7ed;
}
.chat-fab:hover {
    transform: scale(1.04);
}

.assistant-shell {
    background:
        radial-gradient(circle at top right, rgba(245,158,11,0.18), transparent 28%),
        radial-gradient(circle at top left, rgba(251,191,36,0.10), transparent 24%),
        linear-gradient(180deg, #0a0f18 0%, #0b1220 100%);
    border: 1px solid rgba(251, 191, 36, 0.14);
    border-radius: 20px;
    padding: 18px 18px 10px 18px;
    margin-top: 6px;
}
.assistant-topline {
    text-align: center;
    color: #fbbf24;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.assistant-hero {
    text-align: center;
    margin-bottom: 16px;
}
.assistant-hero h1 {
    color: #fde68a;
    font-size: 34px;
    margin: 0 0 8px 0;
}
.assistant-hero h2 {
    color: #f59e0b;
    font-size: 24px;
    margin: 0 0 8px 0;
}
.assistant-hero p {
    color: #e2e8f0;
    font-size: 16px;
    margin: 0;
}
.assistant-panel {
    background: linear-gradient(160deg, rgba(17,24,39,0.94), rgba(15,23,42,0.96));
    border: 1px solid rgba(251,191,36,0.12);
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 12px;
}
.assistant-panel h3 {
    color: #fde68a;
    margin: 0 0 10px 0;
}
.assistant-robot {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    margin: 0 auto 10px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 52px;
    background: radial-gradient(circle at 30% 30%, #fde047, #f59e0b);
    border: 4px solid rgba(255,255,255,0.85);
    box-shadow: 0 10px 34px rgba(245,158,11,0.24);
}
.assistant-event {
    border: 1px solid rgba(251,191,36,0.10);
    border-radius: 12px;
    padding: 12px 12px;
    margin-bottom: 10px;
    background: rgba(15, 23, 42, 0.55);
}
.assistant-event-year {
    color: #fbbf24;
    font-weight: 800;
    font-size: 24px;
    line-height: 1;
    margin-bottom: 6px;
}
.assistant-chip-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}
.assistant-chip {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(30,41,59,0.9);
    border: 1px solid rgba(251,191,36,0.16);
    color: #e2e8f0;
    font-size: 13px;
}
.chat-wrap {
    background: rgba(2,6,23,0.55);
    border: 1px solid rgba(251,191,36,0.12);
    border-radius: 18px;
    padding: 14px;
}
.msg-user {
    background: linear-gradient(135deg, #f59e0b, #ea580c);
    color: #111827;
    padding: 12px 14px;
    border-radius: 16px 16px 4px 16px;
    margin: 10px 0 10px auto;
    max-width: 78%;
    font-weight: 700;
}
.msg-bot {
    background: linear-gradient(160deg, rgba(22,101,52,0.40), rgba(6,78,59,0.36));
    color: #dcfce7;
    padding: 12px 14px;
    border-radius: 16px 16px 16px 4px;
    margin: 10px auto 6px 0;
    max-width: 88%;
    border: 1px solid rgba(34,197,94,0.12);
}
.msg-meta {
    color: #94a3b8;
    font-size: 12px;
    margin-bottom: 8px;
}
.assistant-close {
    text-align: right;
    margin-bottom: 8px;
}
.assistant-close a {
    color: #fbbf24 !important;
    text-decoration: none !important;
    font-size: 14px;
    font-weight: 700;
}
</style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Constants
# -----------------------------
REGION_COLORS = {
    "EAP": "#3b82f6",
    "ECA": "#a855f7",
    "LAC": "#22c55e",
    "MENA": "#f97316",
    "NAC": "#8b5cf6",
    "SAS": "#ef4444",
    "SSA": "#06b6d4",
    "Other": "#64748b",
}

RELATION_COLORS = {
    "imports": "#38bdf8",
    "exports": "#22c55e",
    "smuggling": "#ef4444",
    "other reference": "#94a3b8",
    "payment constraint": "#a78bfa",
}

COUNTRY_ALIASES: Dict[str, Dict[str, str]] = {
    "Türkiye": {"code": "TUR", "region": "ECA"},
    "Kyrgyzstan": {"code": "KGZ", "region": "ECA"},
    "Korea, Rep.": {"code": "KOR", "region": "EAP"},
    "Iran, Islamic Rep.": {"code": "IRN", "region": "MENA"},
    "Egypt, Arab Rep.": {"code": "EGY", "region": "MENA"},
    "Brunei Darussalam": {"code": "BRN", "region": "EAP"},
    "Côte d'Ivoire": {"code": "CIV", "region": "SSA"},
    "Syrian Arab Republic": {"code": "SYR", "region": "MENA"},
    "Czechia": {"code": "CZE", "region": "ECA"},
}

DOEB_FILE_MAP = {
    "vw_opendata_045_i_fuel_sum_x_data_view.csv": "doeb_import_refined_qty",
    "vw_opendata_037_i_fuel_value_data_view.csv": "doeb_import_refined_value",
    "vw_opendata_038_i_crude_value_data_view.csv": "doeb_import_crude_value",
    "vw_opendata_039_e_fuel_sum_data_view.csv": "doeb_export_refined_qty",
}


# -----------------------------
# Generic helpers
# -----------------------------
def _kind_badge(kind: str) -> str:
    cls = {
        "measured": "badge-measured",
        "estimated": "badge-estimated",
        "note": "badge-note",
        "upload": "badge-upload",
    }.get(kind, "badge-estimated")
    return f'<span class="badge {cls}">{kind.upper()}</span>'


def _root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _clean_text(x: Any) -> str:
    return re.sub(r"\s+", " ", "" if pd.isna(x) else str(x)).strip()


def _safe_read_csv(path: Optional[str], **kwargs: Any) -> pd.DataFrame:
    if not path or not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _safe_read_excel(path: Optional[str], **kwargs: Any) -> pd.DataFrame:
    if not path or not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_excel(path, engine="openpyxl", **kwargs)
    except Exception:
        try:
            return pd.read_excel(path, **kwargs)
        except Exception:
            return pd.DataFrame()


def _excel_sheets(path: str) -> List[str]:
    try:
        return list(pd.ExcelFile(path, engine="openpyxl").sheet_names)
    except Exception:
        try:
            return list(pd.ExcelFile(path).sheet_names)
        except Exception:
            return []


def _value_box(label: str, value: str, delta: Optional[str] = None) -> None:
    st.metric(label, value, delta)


def _historical_percentile(s: pd.Series, min_periods: int = 5) -> pd.Series:
    vals: List[float] = []
    out: List[float] = []
    s = pd.to_numeric(s, errors="coerce")
    for x in s.tolist():
        if pd.isna(x):
            out.append(np.nan)
            continue
        vals.append(float(x))
        if len(vals) < min_periods:
            out.append(np.nan)
            continue
        rank = pd.Series(vals).rank(pct=True, method="average").iloc[-1]
        out.append(float(rank * 100.0))
    return pd.Series(out, index=s.index)


def _normalize_weights(weight_map: Dict[str, float], available: Sequence[str]) -> Dict[str, float]:
    picked = {k: float(v) for k, v in weight_map.items() if k in available and float(v) > 0}
    total = sum(picked.values())
    if total <= 0:
        if not available:
            return {}
        eq = 1.0 / len(available)
        return {k: eq for k in available}
    return {k: v / total for k, v in picked.items()}


def _winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    lo = float(s.quantile(lower_q))
    hi = float(s.quantile(upper_q))
    return s.clip(lower=lo, upper=hi)


def _rolling_past_zscore(s: pd.Series, window: int = 12, min_periods: int = 6, clip_upper: float = 6.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    past = s.shift(1)
    roll_mean = past.rolling(window=window, min_periods=min_periods).mean()
    roll_std = past.rolling(window=window, min_periods=min_periods).std(ddof=1)
    exp_mean = past.expanding(min_periods=max(3, min_periods // 2)).mean()
    exp_std = past.expanding(min_periods=max(3, min_periods // 2)).std(ddof=1)
    mean = roll_mean.fillna(exp_mean)
    std = roll_std.fillna(exp_std)
    z = (s - mean) / std.replace(0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z.clip(lower=-clip_upper, upper=clip_upper)


def _historical_quantile_threshold(s: pd.Series, q: float = 0.85, min_periods: int = 12) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    past = s.shift(1)
    thresholds: List[float] = []
    for i in range(len(s)):
        hist = past.iloc[: i + 1].dropna()
        if len(hist) < min_periods:
            thresholds.append(np.nan)
        else:
            thresholds.append(float(hist.quantile(q)))
    return pd.Series(thresholds, index=s.index)


def _trackb_future_signal(metric: pd.DataFrame, horizon_years: int = 1) -> pd.DataFrame:
    needed = {"year", "basket_price", "stress_index"}
    if metric.empty or not needed.issubset(metric.columns):
        return pd.DataFrame()
    yearly = metric.groupby("year", as_index=False).agg(
        basket_price=("basket_price", "mean"),
        stress_index=("stress_index", "mean"),
    ).sort_values("year")
    yearly[f"future_change_pct_{horizon_years}y"] = (yearly["basket_price"].shift(-horizon_years) / yearly["basket_price"] - 1.0) * 100.0
    yearly = yearly.dropna()
    if yearly.empty:
        return pd.DataFrame()
    q80 = yearly["stress_index"].quantile(0.80)
    q20 = yearly["stress_index"].quantile(0.20)
    high = yearly[yearly["stress_index"] >= q80]
    low = yearly[yearly["stress_index"] <= q20]
    return pd.DataFrame(
        [
            {
                "group": "High-stress years (top 20%)",
                f"avg_next_{horizon_years}y_basket_change_pct": float(high[f"future_change_pct_{horizon_years}y"].mean()) if not high.empty else np.nan,
                "n_years": int(len(high)),
            },
            {
                "group": "Low-stress years (bottom 20%)",
                f"avg_next_{horizon_years}y_basket_change_pct": float(low[f"future_change_pct_{horizon_years}y"].mean()) if not low.empty else np.nan,
                "n_years": int(len(low)),
            },
        ]
    )


@st.cache_data(show_spinner=False)
def load_trackb_bundle(roots: Tuple[str, ...]) -> Dict[str, Any]:
    files: Dict[str, str] = {}
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
            base = os.path.basename(path).lower()
            if base == "thailand_fuel_prices_cleaned.csv" and "fuel" not in files:
                files["fuel"] = path
            elif base == "owid-energy-data(clean).csv" and "owid_th" not in files:
                files["owid_th"] = path
            elif base == "owid_energy_data.csv" and "owid_global" not in files:
                files["owid_global"] = path

    out: Dict[str, Any] = {
        "files": files,
        "fuel": pd.DataFrame(),
        "metric": pd.DataFrame(),
        "yearly": pd.DataFrame(),
        "global_gap": pd.DataFrame(),
        "map_check": pd.DataFrame(),
        "future_signal": pd.DataFrame(),
        "weights": {},
        "components": [],
    }

    g = _safe_read_csv(files.get("owid_global"), low_memory=False)
    if not g.empty and {"country", "iso_code", "year", "oil_consumption"}.issubset(g.columns):
        gg = g[[c for c in ["country", "iso_code", "year", "oil_consumption", "oil_production"] if c in g.columns]].copy()
        gg["iso_code"] = gg["iso_code"].astype(str).str.upper().str.strip()
        gg = gg[gg["iso_code"].str.fullmatch(r"[A-Z]{3}", na=False)]
        gg["year"] = pd.to_numeric(gg["year"], errors="coerce")
        gg["oil_consumption"] = pd.to_numeric(gg["oil_consumption"], errors="coerce")
        gg["oil_production"] = pd.to_numeric(gg.get("oil_production", np.nan), errors="coerce").fillna(0.0)
        gg = gg.dropna(subset=["year", "oil_consumption"])
        gg["gap_dependency_ratio"] = (gg["oil_consumption"] - gg["oil_production"]).abs() / (gg["oil_consumption"].abs() + gg["oil_production"].abs() + 1e-9) * 100.0
        gg["gap_percentile_0_100"] = gg.groupby("year")["gap_dependency_ratio"].rank(method="average", pct=True).mul(100.0)
        out["global_gap"] = gg.sort_values(["year", "country"]).reset_index(drop=True)
        missing_prod = int(g["oil_production"].isna().sum()) if "oil_production" in g.columns else int(len(g))
        out["map_check"] = pd.DataFrame(
            [
                {"check": "Country-year rows", "value": int(len(gg)), "status": "INFO"},
                {"check": "Valid ISO-3 rows", "value": int(gg["iso_code"].str.fullmatch(r"[A-Z]{3}", na=False).sum()), "status": "PASS"},
                {"check": "Rows missing oil_production before fill", "value": missing_prod, "status": "WARN" if missing_prod else "PASS"},
            ]
        )

    fuel = _safe_read_csv(files.get("fuel"), low_memory=False)
    owid_th = _safe_read_csv(files.get("owid_th"), low_memory=False)
    if not fuel.empty and {"Date", "Price", "Description", "Unit"}.issubset(fuel.columns):
        fuel = fuel.copy()
        fuel["Date"] = pd.to_datetime(fuel["Date"], errors="coerce")
        fuel["Price"] = pd.to_numeric(fuel["Price"], errors="coerce")
        fuel = fuel.dropna(subset=["Date", "Price"])
        fuel = fuel[fuel["Description"].astype(str).str.contains("Thailand", case=False, na=False)]
        fuel = fuel[fuel["Unit"].astype(str).eq("LCU (Local Currency Unit)")]
        fuel = fuel[fuel["Description"].astype(str).str.contains("Retail Price", case=False, na=False)].copy()
        fuel["Month"] = fuel["Date"].dt.to_period("M")

        basket_weights = {
            "Thailand (Diesel HSD B7) - Retail Price": 0.35,
            "Thailand (Gasohol 95-E10) - Table 9 Retail Price": 0.30,
            "Thailand (LPG low income hh) - Retail Price": 0.25,
            "Thailand (Kerosene) - Retail Price": 0.10,
        }
        available = sorted(fuel["Description"].dropna().astype(str).unique().tolist())
        weights = _normalize_weights(basket_weights, available)
        if weights:
            keep = list(weights.keys())
            fb = fuel[fuel["Description"].isin(keep)].copy()
            fb["Price"] = fb.groupby("Description")["Price"].transform(_winsorize_series)
            monthly = fb.groupby(["Month", "Description"], as_index=False)["Price"].mean()
            monthly["Date"] = monthly["Month"].dt.to_timestamp()
            pivot = monthly.pivot(index="Date", columns="Description", values="Price").sort_index()
            for d in keep:
                if d not in pivot.columns:
                    pivot[d] = np.nan
            pivot = pivot[keep].sort_index().ffill()
            pivot["basket_price"] = sum(pivot[d] * weights[d] for d in keep)
            pivot = pivot.dropna(subset=["basket_price"]).copy()
            if not pivot.empty:
                pivot["mom_change_pct"] = pivot["basket_price"].pct_change() * 100.0
                trailing_median = pivot["basket_price"].shift(1).rolling(12, min_periods=6).median()
                pivot["price_gap_vs_12m_median_pct"] = ((pivot["basket_price"] / trailing_median) - 1.0) * 100.0
                pivot["upside_mom_pct"] = pivot["mom_change_pct"].clip(lower=0.0)
                pivot["positive_level_gap_pct"] = pivot["price_gap_vs_12m_median_pct"].clip(lower=0.0)
                pivot["vol_6m"] = pivot["mom_change_pct"].rolling(6, min_periods=3).std(ddof=1)
                metric = pd.DataFrame({"Date": pivot.index}).merge(
                    pivot[["basket_price", "mom_change_pct", "price_gap_vs_12m_median_pct", "upside_mom_pct", "positive_level_gap_pct", "vol_6m"]],
                    left_on="Date", right_index=True, how="left"
                )
                metric["year"] = metric["Date"].dt.year

                annual_energy = pd.DataFrame()
                if not owid_th.empty and "year" in owid_th.columns:
                    temp = owid_th.copy()
                    if "country" in temp.columns:
                        temp = temp[temp["country"].astype(str).str.lower().eq("thailand")]
                    temp["year"] = pd.to_numeric(temp["year"], errors="coerce")
                    use_cols = ["year"]
                    if "oil_consumption" in temp.columns:
                        temp["oil_consumption"] = pd.to_numeric(temp["oil_consumption"], errors="coerce")
                        use_cols.append("oil_consumption")
                    if "oil_production" in temp.columns:
                        temp["oil_production"] = pd.to_numeric(temp["oil_production"], errors="coerce")
                        use_cols.append("oil_production")
                    annual_energy = temp[use_cols].dropna(subset=["year"]).drop_duplicates("year").sort_values("year")
                    if "oil_consumption" in annual_energy.columns:
                        annual_energy["oil_consumption_yoy_pct"] = annual_energy["oil_consumption"].pct_change() * 100.0
                    if {"oil_consumption", "oil_production"}.issubset(annual_energy.columns):
                        annual_energy["import_dependency_pct"] = np.where(
                            annual_energy["oil_consumption"] > 0,
                            np.maximum(annual_energy["oil_consumption"] - annual_energy["oil_production"], 0.0) / annual_energy["oil_consumption"] * 100.0,
                            np.nan,
                        )
                if not annual_energy.empty:
                    metric = metric.merge(annual_energy[[c for c in ["year", "oil_consumption_yoy_pct", "import_dependency_pct"] if c in annual_energy.columns]], on="year", how="left")

                metric["comp_affordability_shock"] = _rolling_past_zscore(metric["upside_mom_pct"], window=12, min_periods=6).clip(lower=0.0)
                metric["comp_affordability_level"] = _rolling_past_zscore(metric["positive_level_gap_pct"], window=12, min_periods=6).clip(lower=0.0)
                metric["comp_volatility"] = _rolling_past_zscore(metric["vol_6m"], window=12, min_periods=6).clip(lower=0.0)
                if "import_dependency_pct" in metric.columns and metric["import_dependency_pct"].notna().any():
                    metric["comp_import_dependency"] = _rolling_past_zscore(metric["import_dependency_pct"], window=24, min_periods=6).clip(lower=0.0)
                else:
                    metric["comp_import_dependency"] = 0.0

                active = [c for c in ["comp_affordability_shock", "comp_affordability_level", "comp_volatility", "comp_import_dependency"] if metric[c].fillna(0.0).abs().sum() > 0]
                comp_weights = _normalize_weights(
                    {
                        "comp_affordability_shock": 0.35,
                        "comp_affordability_level": 0.25,
                        "comp_volatility": 0.25,
                        "comp_import_dependency": 0.15,
                    },
                    active,
                )
                metric["raw_stress_score"] = 0.0
                for c, w in comp_weights.items():
                    metric["raw_stress_score"] = metric["raw_stress_score"] + metric[c].fillna(0.0) * w
                metric["stress_index"] = _historical_percentile(metric["raw_stress_score"], min_periods=12)
                metric["alert_threshold_85"] = _historical_quantile_threshold(metric["stress_index"], q=0.85, min_periods=12)
                metric["alert_flag_85"] = metric["stress_index"].notna() & metric["alert_threshold_85"].notna() & (metric["stress_index"] >= metric["alert_threshold_85"])
                yearly = metric.groupby("year", as_index=False).agg(
                    year_avg_basket_thb=("basket_price", "mean"),
                    year_avg_stress=("stress_index", "mean"),
                    year_alert_share=("alert_flag_85", lambda s: float(pd.Series(s).mean() * 100.0) if len(s) else np.nan),
                    year_price_volatility=("mom_change_pct", lambda s: float(pd.Series(s).std(ddof=1)) if pd.Series(s).notna().sum() > 1 else np.nan),
                )
                out.update(
                    {
                        "fuel": fb.sort_values("Date").reset_index(drop=True),
                        "metric": metric.sort_values("Date").reset_index(drop=True),
                        "yearly": yearly.sort_values("year").reset_index(drop=True),
                        "future_signal": _trackb_future_signal(metric, horizon_years=1),
                        "weights": {"basket_weights": weights, "component_weights": comp_weights},
                        "components": active,
                    }
                )
    return out


def _stationarity_row(s: pd.Series) -> Dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    out: Dict[str, Any] = {"n": int(len(x)), "adf_pvalue": np.nan, "kpss_pvalue": np.nan, "stationary_decision": "insufficient_data"}
    if len(x) < 8:
        return out
    try:
        out["adf_pvalue"] = float(adfuller(x, autolag="AIC")[1])
    except Exception:
        pass
    try:
        out["kpss_pvalue"] = float(kpss(x, regression="c", nlags="auto")[1])
    except Exception:
        pass

    adf_ok = pd.notna(out["adf_pvalue"]) and out["adf_pvalue"] < 0.05
    kpss_ok = pd.notna(out["kpss_pvalue"]) and out["kpss_pvalue"] > 0.05
    if adf_ok and kpss_ok:
        out["stationary_decision"] = "likely_stationary"
    elif (not adf_ok) and (not kpss_ok):
        out["stationary_decision"] = "likely_nonstationary"
    else:
        out["stationary_decision"] = "mixed_signal"
    return out


def _compute_vif_table(X: pd.DataFrame) -> pd.DataFrame:
    cols = list(X.columns)
    if len(cols) <= 1:
        return pd.DataFrame({"variable": cols, "vif": [np.nan] * len(cols)})
    Xc = sm.add_constant(X, has_constant="add")
    rows = []
    for i, c in enumerate(Xc.columns):
        if c == "const":
            continue
        try:
            vif = float(variance_inflation_factor(Xc.values, i))
        except Exception:
            vif = np.nan
        rows.append({"variable": c, "vif": vif})
    return pd.DataFrame(rows)


def _iterative_vif_filter(X: pd.DataFrame, threshold: float = 10.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Xf = X.copy()
    if Xf.shape[1] <= 1:
        return Xf, _compute_vif_table(Xf)
    while Xf.shape[1] > 1:
        vif = _compute_vif_table(Xf)
        vmax = vif["vif"].max()
        if pd.isna(vmax) or vmax <= threshold:
            return Xf, vif.sort_values("vif", ascending=False).reset_index(drop=True)
        drop_var = str(vif.sort_values("vif", ascending=False).iloc[0]["variable"])
        Xf = Xf.drop(columns=[drop_var])
    return Xf, _compute_vif_table(Xf)


def _spearman_table(df: pd.DataFrame, y_cols: Iterable[str], x_cols: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for y in y_cols:
        for x in x_cols:
            d = df[[x, y]].dropna()
            if len(d) < 4:
                rows.append({"outcome": y, "predictor": x, "rho": np.nan, "pvalue": np.nan, "n": len(d)})
                continue
            rho, p = spearmanr(d[x], d[y], nan_policy="omit")
            rows.append({"outcome": y, "predictor": x, "rho": float(rho), "pvalue": float(p), "n": int(len(d))})
    return pd.DataFrame(rows)


def _sensitivity_from_index(df: pd.DataFrame, idx_cols: List[str]) -> pd.DataFrame:
    if len(idx_cols) < 2 or df.empty:
        return pd.DataFrame()
    schemes = {
        "Baseline": _normalize_weights({"poverty_rate_pct": 0.50, "poverty_severity": 0.30, "gini_consumption": 0.20}, idx_cols),
        "Equal": _normalize_weights({c: 1.0 for c in idx_cols}, idx_cols),
        "Poverty-heavy": _normalize_weights({"poverty_rate_pct": 0.65, "poverty_severity": 0.25, "gini_consumption": 0.10}, idx_cols),
        "Inequality-heavy": _normalize_weights({"poverty_rate_pct": 0.25, "poverty_severity": 0.20, "gini_consumption": 0.55}, idx_cols),
    }
    scores: Dict[str, pd.Series] = {}
    for name, weights in schemes.items():
        s = pd.Series(0.0, index=df.index)
        for c, w in weights.items():
            s = s + _historical_percentile(df[c], min_periods=5).fillna(0.0) * w
        scores[name] = s
    baseline = scores["Baseline"]
    rows = []
    for name, s in scores.items():
        common = pd.concat([baseline, s], axis=1).dropna()
        if common.empty:
            continue
        rows.append(
            {
                "scheme": name,
                "corr_with_baseline": float(common.corr(method="spearman").iloc[0, 1]),
                "mean_abs_diff": float((common.iloc[:, 1] - common.iloc[:, 0]).abs().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("scheme").reset_index(drop=True)


# -----------------------------
# Upload handling
# -----------------------------
def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def materialize_uploaded_files(uploaded_files: Sequence[Any]) -> Optional[str]:
    if not uploaded_files:
        return None

    if "_manual_upload_root" not in st.session_state:
        st.session_state["_manual_upload_root"] = tempfile.mkdtemp(prefix="integrated_dash_upload_")
    root = st.session_state["_manual_upload_root"]

    sig_parts: List[Tuple[str, int, str]] = []
    raw_files: List[Tuple[str, bytes]] = []
    for f in uploaded_files:
        data = f.getvalue()
        digest = hashlib.md5(data).hexdigest()
        sig_parts.append((f.name, len(data), digest))
        raw_files.append((f.name, data))

    sig = tuple(sorted(sig_parts))
    if sig == st.session_state.get("_manual_upload_sig") and os.path.isdir(root):
        return root

    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)

    used_names: Dict[str, int] = {}
    for name, data in raw_files:
        base = _sanitize_filename(name)
        stem, ext = os.path.splitext(base)
        count = used_names.get(base, 0)
        used_names[base] = count + 1
        if count:
            base = f"{stem}_{count}{ext}"
        out_path = os.path.join(root, base)
        with open(out_path, "wb") as fh:
            fh.write(data)

        if zipfile.is_zipfile(io.BytesIO(data)):
            extract_dir = os.path.join(root, Path(base).stem)
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(extract_dir)

    st.session_state["_manual_upload_sig"] = sig
    return root


# -----------------------------
# File discovery
# -----------------------------
@st.cache_data(show_spinner=False)
def discover_files(roots: Tuple[str, ...]) -> Dict[str, str]:
    files: Dict[str, str] = {}
    excel_candidates: List[str] = []
    csv_candidates: List[str] = []
    json_candidates: List[str] = []

    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext in {".xlsx", ".xls"}:
                excel_candidates.append(path)
            elif ext == ".csv":
                csv_candidates.append(path)
            elif ext == ".json":
                json_candidates.append(path)

    for path in excel_candidates:
        base = os.path.basename(path).lower()
        if base == "costtogdp.xlsx" and "cost_to_gdp" not in files:
            files["cost_to_gdp"] = path
        elif base == "logtable.xlsx" and "log_table" not in files:
            files["log_table"] = path
        elif base.endswith("260205.xlsx") and "poverty_book" not in files:
            files["poverty_book"] = path

    for path in csv_candidates:
        base = os.path.basename(path)
        if base in DOEB_FILE_MAP and DOEB_FILE_MAP[base] not in files:
            files[DOEB_FILE_MAP[base]] = path
        elif base == "edges_detailed.csv" and "network_edges" not in files:
            files["network_edges"] = path
        elif base == "country_nodes.csv" and "network_nodes" not in files:
            files["network_nodes"] = path

    for path in json_candidates:
        if os.path.basename(path) == "dataset_summary.json" and "network_summary" not in files:
            files["network_summary"] = path

    if "cost_to_gdp" not in files and "log_table" not in files:
        for path in excel_candidates:
            sheets = set(_excel_sheets(path))
            if "Cost to GDP" in sheets:
                files["cost_to_gdp"] = path
                break

    if "poverty_book" not in files:
        for path in excel_candidates:
            sheets = set(_excel_sheets(path))
            if {"1.2", "1.8"}.issubset(sheets):
                files["poverty_book"] = path
                break

    if "network_edges" not in files:
        for path in csv_candidates:
            preview = _safe_read_csv(path, nrows=5)
            if {"source", "target", "relation_type"}.issubset(preview.columns):
                files["network_edges"] = path
                break

    if "network_nodes" not in files:
        for path in csv_candidates:
            preview = _safe_read_csv(path, nrows=5)
            if {"country", "code"}.issubset(preview.columns):
                files["network_nodes"] = path
                break

    return files


# -----------------------------
# Thailand parsers
# -----------------------------
@st.cache_data(show_spinner=False)
def parse_cost_to_gdp(path: str) -> pd.DataFrame:
    df = _safe_read_excel(path, sheet_name="Cost to GDP", header=None)
    if df.empty:
        sheets = _excel_sheets(path)
        if sheets:
            df = _safe_read_excel(path, sheet_name=sheets[0], header=None)
    if df.empty:
        return pd.DataFrame()

    years = pd.to_numeric(df.iloc[4, 1:], errors="coerce")
    year_cols = years.dropna().index.tolist()
    out = pd.DataFrame({"year": years.loc[year_cols].astype(int).tolist()})

    targets = {
        "total_logistics_cost_to_gdp": lambda s: s == "สัดส่วนต้นทุนโลจิสติกส์ ต่อ GDP",
        "transport_cost_to_gdp": lambda s: s == "สัดส่วนต้นทุนการขนส่งสินค้า ต่อ GDP",
        "inventory_cost_to_gdp": lambda s: s == "สัดส่วนต้นทุนการเก็บรักษาสินค้าคงคลัง ต่อ GDP",
        "management_cost_to_gdp": lambda s: s == "สัดส่วนต้นทุนการบริหารจัดการ ต่อ GDP",
        "road_cost_to_gdp": lambda s: s == "ทางถนน",
        "rail_cost_to_gdp": lambda s: s == "ทางราง",
        "water_cost_to_gdp": lambda s: s == "ทางน้ำ",
        "air_cost_to_gdp": lambda s: s == "ทางอากาศ",
    }

    for i in range(len(df)):
        label = _clean_text(df.iloc[i, 0])
        for col, matcher in targets.items():
            if matcher(label):
                out[col] = pd.to_numeric(df.iloc[i, year_cols], errors="coerce").tolist()

    out["cost_method_break_2017"] = (out["year"] >= 2017).astype(int)
    return out.sort_values("year").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def parse_poverty_indicators(path: str) -> pd.DataFrame:
    df = _safe_read_excel(path, sheet_name="1.8", header=None)
    if df.empty:
        return pd.DataFrame()

    years_be = pd.to_numeric(df.iloc[2, 1:], errors="coerce")
    year_cols = years_be.dropna().index.tolist()
    out = pd.DataFrame({"year": (years_be.loc[year_cols] - 543).astype(int).tolist()})

    for i in range(len(df)):
        label = _clean_text(df.iloc[i, 0])
        vals = pd.to_numeric(df.iloc[i, year_cols], errors="coerce").tolist()

        if label == "ช่องว่างความยากจน":
            out["poverty_gap"] = vals
        elif label == "ความรุนแรงปัญหาความยากจน":
            out["poverty_severity"] = vals
        elif label.startswith("เส้นความยากจน"):
            out["poverty_line_baht_per_person_month"] = vals
        elif label == "สัดส่วนคนจน (ร้อยละ)":
            out["poverty_rate_pct"] = vals
        elif label == "จำนวนคนจน (ล้านคน)":
            out["poor_million_people"] = vals
        elif label.startswith("สัดส่วนครัวเรือนยากจน"):
            out["poor_household_rate_pct"] = vals
        elif "Gini coefficient" in label and "รายได้" in label:
            out["gini_income"] = vals
        elif "Gini coefficient" in label and "รายจ่าย" in label:
            out["gini_consumption"] = vals
        elif "กลุ่มรวยสุด" in label and "กลุ่มจนสุด" in label:
            out["income_ratio_top20_bottom20"] = vals

    return out.sort_values("year").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def parse_region_latest(path: str, latest_year: Optional[int] = None) -> pd.DataFrame:
    df = _safe_read_excel(path, sheet_name="1.2", header=None)
    if df.empty:
        return pd.DataFrame()

    years_be = pd.to_numeric(df.iloc[2, 2:], errors="coerce")
    year_map = {int(y - 543): idx for idx, y in zip(years_be.dropna().index.tolist(), years_be.dropna().tolist())}
    if not year_map:
        return pd.DataFrame()
    if latest_year is None:
        latest_year = max(year_map)
    col = year_map.get(latest_year)
    if col is None:
        return pd.DataFrame()

    rows = []
    current_region: Optional[str] = None
    for i in range(3, len(df)):
        region = df.iloc[i, 0]
        area = _clean_text(df.iloc[i, 1])
        if pd.notna(region):
            current_region = _clean_text(region)
        if current_region and area == "รวม":
            val = pd.to_numeric(df.iloc[i, col], errors="coerce")
            rows.append({"region_th": current_region, "year": latest_year, "poverty_rate_pct": val})

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def parse_doeb_annual(files: Dict[str, str]) -> pd.DataFrame:
    mapping = {
        "doeb_import_refined_qty": "import_refined_qty_k_liters",
        "doeb_import_refined_value": "import_refined_value_million_baht",
        "doeb_import_crude_value": "import_crude_value_million_baht",
        "doeb_export_refined_qty": "export_refined_qty_k_liters",
    }

    out: Optional[pd.DataFrame] = None
    for key, new_col in mapping.items():
        path = files.get(key)
        raw = _safe_read_csv(path)
        if raw.empty:
            continue
        value_col = "QTY" if "QTY" in raw.columns else ("BALANCE_VALUE" if "BALANCE_VALUE" in raw.columns else None)
        if value_col is None or "YEAR_ID" not in raw.columns:
            continue
        temp = pd.DataFrame(
            {
                "year": pd.to_numeric(raw["YEAR_ID"], errors="coerce") - 543,
                new_col: pd.to_numeric(raw[value_col], errors="coerce"),
            }
        ).dropna()
        temp["year"] = temp["year"].astype(int)
        temp = temp.groupby("year", as_index=False)[new_col].sum()
        out = temp if out is None else out.merge(temp, on="year", how="outer")

    if out is None:
        return pd.DataFrame(columns=["year"])

    if {"import_refined_qty_k_liters", "export_refined_qty_k_liters"}.issubset(out.columns):
        out["net_refined_import_qty_k_liters"] = out["import_refined_qty_k_liters"] - out["export_refined_qty_k_liters"]
        out["refined_import_export_ratio"] = out["import_refined_qty_k_liters"] / out["export_refined_qty_k_liters"].replace(0, np.nan)

    if {"import_refined_value_million_baht", "import_crude_value_million_baht"}.issubset(out.columns):
        out["total_energy_import_value_million_baht"] = out["import_refined_value_million_baht"] + out["import_crude_value_million_baht"]

    return out.sort_values("year").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_annual_dataset(roots: Tuple[str, ...]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    files = discover_files(roots)

    cost_path = files.get("log_table") or files.get("cost_to_gdp")
    poverty_path = files.get("poverty_book")

    cost = parse_cost_to_gdp(cost_path) if cost_path else pd.DataFrame()
    poverty = parse_poverty_indicators(poverty_path) if poverty_path else pd.DataFrame()
    doeb = parse_doeb_annual(files)
    region_latest = parse_region_latest(poverty_path) if poverty_path else pd.DataFrame()

    annual = pd.DataFrame()
    if not cost.empty:
        annual = cost.copy()
    if not poverty.empty:
        annual = poverty.copy() if annual.empty else annual.merge(poverty, on="year", how="outer")
    if not doeb.empty:
        annual = doeb.copy() if annual.empty else annual.merge(doeb, on="year", how="outer")

    if not annual.empty:
        annual = annual.sort_values("year").reset_index(drop=True)
        idx_cols = [c for c in ["poverty_rate_pct", "poverty_severity", "gini_consumption"] if c in annual.columns]
        weights = _normalize_weights(
            {"poverty_rate_pct": 0.50, "poverty_severity": 0.30, "gini_consumption": 0.20},
            idx_cols,
        )
        annual["vulnerability_index_0_100"] = 0.0
        for c, w in weights.items():
            annual["vulnerability_index_0_100"] = annual["vulnerability_index_0_100"] + _historical_percentile(annual[c], min_periods=5).fillna(0.0) * w

    return annual, region_latest, doeb, files


# -----------------------------
# Validation pipeline
# -----------------------------
@st.cache_data(show_spinner=False)
def regression_diagnostics(df: pd.DataFrame, y_col: str, x_cols: Tuple[str, ...], vif_threshold: float = 10.0) -> Dict[str, Any]:
    keep = [y_col] + list(x_cols)
    d = df[["year"] + keep].dropna().copy()
    if d.empty or len(d) < 8:
        return {}

    stationarity_rows = []
    nonstationary = False
    for c in keep:
        row = _stationarity_row(d[c])
        row["series"] = c
        stationarity_rows.append(row)
        if row["stationary_decision"] != "likely_stationary":
            nonstationary = True

    mode = "levels"
    if nonstationary:
        d[keep] = d[keep].diff()
        d = d.dropna().copy()
        mode = "first_difference"

    if len(d) < 8:
        return {}

    X = d[list(x_cols)].copy()
    X, vif_table = _iterative_vif_filter(X, threshold=vif_threshold)
    y = d[y_col].copy()

    X_sm = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(y, X_sm).fit()
    robust = fit.get_robustcov_results(cov_type="HAC", maxlags=max(1, min(2, len(d) // 5)))

    coef = pd.DataFrame(
        {
            "term": X_sm.columns,
            "coef": np.asarray(robust.params),
            "pvalue": np.asarray(robust.pvalues),
            "ci_low": np.asarray(robust.conf_int())[:, 0],
            "ci_high": np.asarray(robust.conf_int())[:, 1],
        }
    )

    try:
        lb = acorr_ljungbox(robust.resid, lags=[1], return_df=True)
        ljungbox_p = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        ljungbox_p = np.nan

    try:
        bp = het_breuschpagan(robust.resid, X_sm)
        bp_p = float(bp[1])
    except Exception:
        bp_p = np.nan

    cv_mae = np.nan
    cv_rmse = np.nan
    if len(d) >= 10:
        splits = min(5, max(2, len(d) // 4))
        tscv = TimeSeriesSplit(n_splits=splits)
        preds: List[float] = []
        actuals: List[float] = []
        for tr, te in tscv.split(d):
            train = d.iloc[tr]
            test = d.iloc[te]
            X_tr = sm.add_constant(train[X.columns], has_constant="add")
            X_te = sm.add_constant(test[X.columns], has_constant="add")
            y_tr = train[y_col]
            temp_fit = sm.OLS(y_tr, X_tr).fit()
            pred = temp_fit.predict(X_te)
            preds.extend(pred.tolist())
            actuals.extend(test[y_col].tolist())
        if preds and actuals:
            cv_mae = float(mean_absolute_error(actuals, preds))
            cv_rmse = float(np.sqrt(mean_squared_error(actuals, preds)))

    return {
        "mode": mode,
        "n": int(len(d)),
        "coef": coef,
        "vif": vif_table,
        "stationarity": pd.DataFrame(stationarity_rows).sort_values("series").reset_index(drop=True),
        "r2": float(robust.rsquared),
        "adj_r2": float(robust.rsquared_adj),
        "dw": float(durbin_watson(robust.resid)),
        "ljungbox_p": ljungbox_p,
        "bp_p": bp_p,
        "cv_mae": cv_mae,
        "cv_rmse": cv_rmse,
        "used_predictors": list(X.columns),
        "model_df": d,
    }


# -----------------------------
# Network parsers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_network_data(roots: Tuple[str, ...]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    files = discover_files(roots)
    edges_path = files.get("network_edges")
    nodes_path = files.get("network_nodes")
    summary_path = files.get("network_summary")

    edges = _safe_read_csv(edges_path)
    nodes = _safe_read_csv(nodes_path)
    summary: Dict[str, Any] = {}

    if summary_path and os.path.isfile(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
        except Exception:
            summary = {}

    if edges.empty:
        return pd.DataFrame(), pd.DataFrame(), summary

    if "country" not in nodes.columns:
        nodes = pd.DataFrame(columns=["country", "code", "region"])

    for col in ["country", "code", "region"]:
        if col not in nodes.columns:
            nodes[col] = np.nan
    nodes = nodes[["country", "code", "region"]].copy()
    nodes["country"] = nodes["country"].astype(str).str.strip()
    nodes["code"] = nodes["code"].astype(str).str.upper().str.strip()
    nodes["region"] = nodes["region"].astype(str).replace({"nan": "Other", "": "Other"}).str.strip()

    edge_countries = sorted(set(edges.get("source", pd.Series(dtype=str)).astype(str)).union(set(edges.get("target", pd.Series(dtype=str)).astype(str))))
    node_map = {
        row["country"]: {"code": row["code"] if re.fullmatch(r"[A-Z]{3}", str(row["code"])) else None, "region": row["region"] if row["region"] else "Other"}
        for _, row in nodes.iterrows()
        if str(row["country"]).strip()
    }

    rows = []
    unresolved = []
    for country in edge_countries:
        meta = node_map.get(country, {})
        alias = COUNTRY_ALIASES.get(country, {})
        code = meta.get("code") or alias.get("code")
        region = meta.get("region") or alias.get("region") or "Other"
        if not code or not re.fullmatch(r"[A-Z]{3}", str(code)):
            unresolved.append(country)
            code = np.nan
        rows.append({"country": country, "code": code, "region": region})

    nodes_aug = pd.DataFrame(rows)
    summary = dict(summary)
    summary["countries_in_edges"] = len(edge_countries)
    summary["countries_with_valid_iso3"] = int(nodes_aug["code"].notna().sum())
    summary["unresolved_countries"] = unresolved
    summary["source_edges_path"] = edges_path
    summary["source_nodes_path"] = nodes_path
    return edges, nodes_aug, summary


@st.cache_data(show_spinner=False)
def aggregate_edges(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "source", "target", "total_mentions", "dominant_relation",
                "relation_summary", "years", "sheets", "sample_evidence"
            ]
        )

    grouped = (
        df.groupby(["source", "target"], dropna=False)
        .agg(
            total_mentions=("sentence", "count"),
            years=("year_label", lambda s: ", ".join(sorted(set(map(str, s))))),
            sheets=("sheet", lambda s: "; ".join(sorted(set(map(str, s))))),
            sample_evidence=("sentence", lambda s: " || ".join(list(dict.fromkeys(map(str, s)))[:3])),
            relation_summary=("relation_type", lambda s: ", ".join(
                f"{k} ({v})" for k, v in pd.Series(list(s)).value_counts().to_dict().items()
            )),
            dominant_relation=("relation_type", lambda s: pd.Series(list(s)).value_counts().idxmax()),
        )
        .reset_index()
    )
    return grouped.sort_values(["total_mentions", "source", "target"], ascending=[False, True, True])


@st.cache_data(show_spinner=False)
def compute_node_metrics(edge_pairs: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    meta = nodes.copy()
    if meta.empty:
        return pd.DataFrame(columns=["country", "code", "region", "in_degree", "out_degree", "weighted_in", "weighted_out", "weighted_total", "degree_total"])
    meta["in_degree"] = 0.0
    meta["out_degree"] = 0.0
    meta["weighted_in"] = 0.0
    meta["weighted_out"] = 0.0

    if edge_pairs.empty:
        meta["weighted_total"] = 0.0
        meta["degree_total"] = 0.0
        return meta

    out_counts = edge_pairs.groupby("source").agg(
        out_degree=("target", "nunique"),
        weighted_out=("total_mentions", "sum"),
    )
    in_counts = edge_pairs.groupby("target").agg(
        in_degree=("source", "nunique"),
        weighted_in=("total_mentions", "sum"),
    )

    meta = meta.merge(out_counts, left_on="country", right_index=True, how="left", suffixes=("", "_new"))
    meta = meta.merge(in_counts, left_on="country", right_index=True, how="left", suffixes=("", "_new"))

    for col in ["in_degree", "out_degree", "weighted_in", "weighted_out"]:
        if f"{col}_new" in meta.columns:
            meta[col] = meta[f"{col}_new"].fillna(meta[col]).fillna(0).astype(float)
            meta.drop(columns=[f"{col}_new"], inplace=True)

    meta["weighted_total"] = meta["weighted_in"] + meta["weighted_out"]
    meta["degree_total"] = meta["in_degree"] + meta["out_degree"]
    return meta


@st.cache_data(show_spinner=False)
def verify_network_map(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty and edges.empty:
        return pd.DataFrame()
    edge_countries = set(edges.get("source", pd.Series(dtype=str)).astype(str)).union(set(edges.get("target", pd.Series(dtype=str)).astype(str)))
    node_countries = set(nodes.get("country", pd.Series(dtype=str)).astype(str))
    valid_iso_mask = nodes["code"].astype(str).str.fullmatch(r"[A-Z]{3}", na=False) if "code" in nodes.columns else pd.Series(dtype=bool)
    unresolved = sorted(c for c in edge_countries if c not in node_countries)
    invalid_code_rows = nodes.loc[~valid_iso_mask, ["country", "code"]] if not nodes.empty else pd.DataFrame(columns=["country", "code"])

    rows = [
        {"check": "Countries mentioned in edges", "value": len(edge_countries), "status": "INFO"},
        {"check": "Node rows available after repair", "value": len(nodes), "status": "PASS" if len(nodes) >= len(edge_countries) else "WARN"},
        {"check": "Countries with valid ISO-3 codes", "value": int(valid_iso_mask.sum()) if len(nodes) else 0, "status": "PASS" if int(valid_iso_mask.sum()) > 0 else "FAIL"},
        {"check": "Countries missing from repaired node table", "value": len(unresolved), "status": "PASS" if len(unresolved) == 0 else "WARN"},
        {"check": "Node rows with invalid ISO-3", "value": len(invalid_code_rows), "status": "PASS" if len(invalid_code_rows) == 0 else "WARN"},
    ]
    return pd.DataFrame(rows)


def build_choropleth_map(node_metrics: pd.DataFrame, metric_col: str) -> go.Figure:
    fig = go.Figure()
    if node_metrics.empty or metric_col not in node_metrics.columns:
        fig.update_layout(
            title="No mapable network data",
            xaxis={"visible": False},
            yaxis={"visible": False},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    map_df = node_metrics.copy()
    map_df = map_df[map_df["code"].astype(str).str.fullmatch(r"[A-Z]{3}", na=False)].copy()
    if map_df.empty:
        fig.update_layout(
            title="No valid ISO-3 codes for map rendering",
            xaxis={"visible": False},
            yaxis={"visible": False},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    fig.add_trace(
        go.Choropleth(
            locations=map_df["code"],
            z=map_df[metric_col],
            locationmode="ISO-3",
            colorscale="YlOrRd",
            marker_line_color="rgba(255,255,255,0.25)",
            colorbar_title=metric_col.replace("_", " ").title(),
            text=map_df["country"],
            customdata=np.stack(
                [
                    map_df["region"].astype(str),
                    map_df["weighted_in"].astype(float),
                    map_df["weighted_out"].astype(float),
                ],
                axis=1,
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Region: %{customdata[0]}<br>"
                f"{metric_col}: %{{z:.1f}}<br>"
                "Weighted in: %{customdata[1]:.0f}<br>"
                "Weighted out: %{customdata[2]:.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Global country map from the filtered network",
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth", bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        margin=dict(l=0, r=0, t=48, b=0),
        height=520,
    )
    return fig


def build_network_plot(edge_pairs: pd.DataFrame, node_metrics: pd.DataFrame, show_labels: bool = True) -> go.Figure:
    fig = go.Figure()
    if edge_pairs.empty:
        fig.update_layout(
            title="No links match the current filters.",
            xaxis={"visible": False},
            yaxis={"visible": False},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
        )
        return fig

    graph = nx.DiGraph()
    for _, row in edge_pairs.iterrows():
        graph.add_edge(row["source"], row["target"], weight=float(row["total_mentions"]))

    undirected = graph.to_undirected()
    pos = nx.spring_layout(
        undirected,
        seed=42,
        k=1.2 / max(1, math.sqrt(max(1, undirected.number_of_nodes()))),
        weight="weight",
    )

    for rel, group in edge_pairs.groupby("dominant_relation"):
        xs, ys = [], []
        hover_x, hover_y, hover_text = [], [], []
        for _, row in group.iterrows():
            if row["source"] not in pos or row["target"] not in pos:
                continue
            x0, y0 = pos[row["source"]]
            x1, y1 = pos[row["target"]]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
            hover_x.append((x0 + x1) / 2)
            hover_y.append((y0 + y1) / 2)
            hover_text.append(
                f"<b>{row['source']} → {row['target']}</b><br>"
                f"Mentions: {int(row['total_mentions'])}<br>"
                f"Relations: {row['relation_summary']}<br>"
                f"Years: {row['years']}<br>"
                f"Sheets: {row['sheets']}<br>"
                f"Evidence: {row['sample_evidence']}"
            )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"width": 1.0, "color": RELATION_COLORS.get(rel, "#94a3b8")},
                hoverinfo="skip",
                name=f"Edges: {rel}",
                opacity=0.55,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hover_x,
                y=hover_y,
                mode="markers",
                marker={"size": 10, "opacity": 0.0},
                hoverinfo="text",
                hovertext=hover_text,
                showlegend=False,
                name=f"Hover: {rel}",
            )
        )

    node_metrics = node_metrics[node_metrics["country"].isin(set(edge_pairs["source"]).union(edge_pairs["target"]))].copy()
    node_metrics["x"] = node_metrics["country"].map(lambda c: pos[c][0])
    node_metrics["y"] = node_metrics["country"].map(lambda c: pos[c][1])
    node_metrics["size"] = node_metrics["weighted_total"].apply(lambda v: 10 + 4 * math.sqrt(max(v, 0)))

    for region, group in node_metrics.groupby("region"):
        text_values = group["country"].tolist() if show_labels else None
        fig.add_trace(
            go.Scatter(
                x=group["x"],
                y=group["y"],
                mode="markers+text" if show_labels else "markers",
                text=text_values,
                textposition="top center",
                marker={
                    "size": group["size"],
                    "color": REGION_COLORS.get(region, "#94a3b8"),
                    "line": {"width": 0.5, "color": "#cbd5e1"},
                    "opacity": 0.95,
                },
                name=f"Region: {region}",
                hoverinfo="text",
                hovertext=[
                    (
                        f"<b>{row.country}</b><br>"
                        f"Code: {row.code}<br>"
                        f"Region: {row.region}<br>"
                        f"Outgoing links: {int(row.out_degree)}<br>"
                        f"Incoming links: {int(row.in_degree)}<br>"
                        f"Weighted outgoing: {int(row.weighted_out)}<br>"
                        f"Weighted incoming: {int(row.weighted_in)}"
                    )
                    for row in group.itertuples()
                ],
            )
        )

    fig.update_layout(
        title="Country network extracted from the workbook narratives",
        xaxis={"visible": False},
        yaxis={"visible": False},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        hovermode="closest",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        height=850,
    )
    return fig


# -----------------------------
# Sidebar: source selection
# -----------------------------
with st.sidebar:
    st.markdown("### Data input")
    source_mode = st.radio(
        "Data source mode",
        options=["Auto-discover near app", "Manual upload", "Auto + upload"],
        index=2,
    )
    uploaded_files = st.file_uploader(
        "Upload ZIP / XLSX / XLS / CSV / JSON files",
        type=["zip", "xlsx", "xls", "csv", "json"],
        accept_multiple_files=True,
        help="You can upload NESDC, DOEB, and the full_country_network_streamlit.zip file here.",
    )
    uploaded_root = materialize_uploaded_files(uploaded_files) if source_mode in {"Manual upload", "Auto + upload"} else None

    local_root = _root()
    if source_mode == "Auto-discover near app":
        data_roots = tuple(r for r in [local_root] if r)
    elif source_mode == "Manual upload":
        data_roots = tuple(r for r in [uploaded_root] if r)
    else:
        data_roots = tuple(r for r in [uploaded_root, local_root] if r)

    st.caption("Upload files are extracted to a temporary working folder for parsing.")


# -----------------------------
# Load datasets
# -----------------------------
annual, region_latest, doeb_annual, discovered_files = build_annual_dataset(data_roots)
edges_raw, network_nodes, network_summary = load_network_data(data_roots)
trackb_bundle = load_trackb_bundle(data_roots)
trackb_metric = trackb_bundle.get("metric", pd.DataFrame())
trackb_yearly = trackb_bundle.get("yearly", pd.DataFrame())
trackb_global_gap = trackb_bundle.get("global_gap", pd.DataFrame())
trackb_map_check = trackb_bundle.get("map_check", pd.DataFrame())
trackb_future_signal = trackb_bundle.get("future_signal", pd.DataFrame())

def _assistant_events_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"year": 2024, "event_th": "ติดตามแรงกดดันด้านต้นทุนพลังงานและการคุ้มครองผู้บริโภค", "topic": "policy"},
            {"year": 2023, "event_th": "เฝ้าระวังความผันผวนราคาเชื้อเพลิงและภาระครัวเรือน", "topic": "affordability"},
            {"year": 2022, "event_th": "ติดตามมาตรการพยุงราคาและผลกระทบเชิงระบบ", "topic": "system"},
            {"year": 2021, "event_th": "มองหาสัญญาณเปราะบางด้านพลังงานและการปรับตัว", "topic": "general"},
        ]
    )

# -----------------------------
# Assistant page helpers
# -----------------------------
def _build_event_df(osf_df: pd.DataFrame) -> pd.DataFrame:
    if osf_df.empty or "event_th" not in osf_df.columns:
        return pd.DataFrame(
            [
                {"year": 2024, "event_th": "ติดตามแรงกดดันด้านต้นทุนพลังงานและการคุ้มครองผู้บริโภค", "topic": "policy"},
                {"year": 2023, "event_th": "เฝ้าระวังความผันผวนราคาเชื้อเพลิงและภาระครัวเรือน", "topic": "affordability"},
                {"year": 2022, "event_th": "ติดตามมาตรการพยุงราคาและผลกระทบเชิงระบบ", "topic": "system"},
                {"year": 2021, "event_th": "มองหาสัญญาณเปราะบางด้านพลังงานและการปรับตัว", "topic": "general"},
            ]
        )

    out = osf_df.copy()
    if "topic" not in out.columns:
        out["topic"] = "policy"

    return out[["year", "event_th", "topic"]].sort_values("year", ascending=False)

def render_assistant_page(
    osf_df: pd.DataFrame,
    stress: float,
    latest: pd.Series,
    alerts: int,
    threshold_q: float,
) -> None:
    if "assistant_history" not in st.session_state:
        st.session_state.assistant_history = []

    if "assistant_prompt" not in st.session_state:
        st.session_state.assistant_prompt = ""

    if "assistant_mode" not in st.session_state:
        st.session_state.assistant_mode = "อธิบายข้อมูลให้หน่อย"

    events_df = _build_event_df(osf_df)
    stress_text = "NA" if pd.isna(stress) else f"{stress:.1f}"

    st.markdown('<div class="assistant-shell">', unsafe_allow_html=True)
    st.markdown('<div class="assistant-close"><a href="?assistant=0">← กลับไปหน้า dashboard</a></div>', unsafe_allow_html=True)
    st.markdown('<div class="assistant-topline">ASSISTANT FOR THE DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="assistant-hero">
            <h1>พลังงาน: จากความเครียดของระบบ สู่การปรับตัวของมนุษย์</h1>
            <h2>Energy Stress → Human Adaptation</h2>
            <p>การตรวจจับความเสี่ยงก่อนที่จะเกิดขึ้น</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.05, 2.5], gap="large")

    with left_col:
        st.markdown('<div class="assistant-panel"><h3>เหตุการณ์สำคัญ</h3></div>', unsafe_allow_html=True)

        year_options = ["ทั้งหมด"] + [str(y) for y in sorted(events_df["year"].dropna().unique(), reverse=True)]
        selected_year = st.selectbox("ปี", year_options, key="assistant_year")
        selected_topic = st.selectbox(
            "หัวข้อ",
            ["ทั้งหมด", "policy", "affordability", "system", "general"],
            key="assistant_topic"
        )

        filtered = events_df.copy()
        if selected_year != "ทั้งหมด":
            filtered = filtered[filtered["year"].astype(str) == selected_year]
        if selected_topic != "ทั้งหมด":
            filtered = filtered[filtered["topic"] == selected_topic]

        for _, row in filtered.head(6).iterrows():
            st.markdown(
                f"""
                <div class="assistant-event">
                    <div class="assistant-event-year">{int(row["year"])}</div>
                    <div>{row["event_th"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown('<div class="assistant-panel">', unsafe_allow_html=True)
        st.markdown('<div class="assistant-robot">🤖</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="assistant-hero" style="margin-bottom:10px;">
                <h2 style="margin-bottom:6px;">สวัสดีครับ/ค่ะ นี่คือผู้ช่วยของคุณ</h2>
                <p>ถามอะไรก็ได้เกี่ยวกับความเครียดพลังงาน การปรับตัว และข้อมูลใน dashboard</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        top_a, top_b, top_c = st.columns([1.2, 1.2, 0.35])
        with top_a:
            if st.button("💡 อธิบายข้อมูลให้หน่อย", use_container_width=True):
                st.session_state.assistant_mode = "อธิบายข้อมูลให้หน่อย"
        with top_b:
            if st.button("📊 แสดงข้อมูลเชิงสถิติ", use_container_width=True):
                st.session_state.assistant_mode = "แสดงข้อมูลเชิงสถิติ"
        with top_c:
            st.metric("⚠️", alerts)

        st.markdown(
            f"""
            <div class="assistant-chip-row">
                <span class="assistant-chip">Stress ล่าสุด: {stress_text}</span>
                <span class="assistant-chip">Threshold q = {threshold_q:.2f}</span>
                <span class="assistant-chip">เดือนเตือน: {alerts}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        preset_col1, preset_col2 = st.columns(2)
        with preset_col1:
            if st.button("ความเครียดของระบบพลังงานของไทยมีลักษณะอย่างไรบ้าง", use_container_width=True):
                st.session_state.assistant_prompt = "ความเครียดของระบบพลังงานของไทยมีลักษณะอย่างไรบ้าง"
        with preset_col2:
            if st.button("ประเทศไทยควรปรับตัวต่อวิกฤตพลังงานอย่างไร", use_container_width=True):
                st.session_state.assistant_prompt = "ประเทศไทยควรปรับตัวต่อวิกฤตพลังงานอย่างไร"

        if st.session_state.assistant_history:
            st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

        if not st.session_state.assistant_history:
            st.markdown(
                """
                <div class="msg-bot">
                    สวัสดีครับ/ค่ะ ผมพร้อมช่วยอธิบายข้อมูลใน dashboard นี้ ทั้งเชิงความหมาย เชิงสถิติ และเชิงการปรับตัว
                </div>
                """,
                unsafe_allow_html=True,
            )

        for item in st.session_state.assistant_history:
            st.markdown(f'<div class="msg-user">{item["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="msg-bot">{item["answer"]}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="msg-meta">📊 Topic: {item["topic"]} &nbsp;&nbsp; 🧠 Columns used: {", ".join(item["columns_used"]) if item["columns_used"] else "None"}</div>',
                unsafe_allow_html=True,
            )

        with st.form("assistant_form", clear_on_submit=True):
            user_q = st.text_input(
                "พิมพ์คำถามของคุณ",
                value=st.session_state.assistant_prompt,
                placeholder="พิมพ์คำถามหรือคำขอของคุณได้ที่นี่...",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("ส่ง")

        if submitted:
            q = user_q.strip()
            if q:
                if st.session_state.assistant_mode == "แสดงข้อมูลเชิงสถิติ":
                    q = f"ช่วยตอบเชิงสถิติและยกตัวชี้วัดที่เกี่ยวข้องจาก dashboard ด้วย: {q}"
                else:
                    q = f"ช่วยอธิบายให้เข้าใจง่ายโดยอิงจาก dashboard นี้: {q}"

                with st.spinner("กำลังวิเคราะห์..."):
                    try:
                        result = run_chatbot(q)
                        st.session_state.assistant_history.append(
                            {
                                "question": user_q,
                                "answer": result["answer"],
                                "topic": result["topic"],
                                "columns_used": result["columns_used"],
                            }
                        )
                        st.session_state.assistant_prompt = ""
                        st.rerun()
                    except Exception as e:
                        st.error(f"Assistant error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Assistant variables
# -----------------------------
osf_df = trackb_metric.copy()

if not trackb_metric.empty:
    latest = trackb_metric.iloc[-1]
    stress = latest.get("stress_index", np.nan)
    alerts = int(trackb_metric["alert_flag_85"].sum()) if "alert_flag_85" in trackb_metric.columns else 0
    threshold_q = 0.85
else:
    latest = pd.Series(dtype="object")
    stress = np.nan
    alerts = 0
    threshold_q = 0.85

# -----------------------------
# Assistant switch
# -----------------------------
assistant_open = st.query_params.get("assistant", "0") == "1"

if assistant_open:
    render_assistant_page(
        osf_df=osf_df,
        stress=stress,
        latest=latest,
        alerts=alerts,
        threshold_q=threshold_q,
    )
    st.stop()

has_thailand = not annual.empty
has_network = not edges_raw.empty and not network_nodes.empty
has_trackb = (not trackb_metric.empty) or (not trackb_global_gap.empty) or has_thailand

available_years = sorted(annual["year"].dropna().astype(int).unique().tolist()) if has_thailand else []
network_year_labels = sorted(edges_raw["year_label"].dropna().astype(str).unique().tolist(), key=lambda x: (x == "Trade Status", x)) if has_network and "year_label" in edges_raw.columns else []
network_relation_types = sorted(edges_raw["relation_type"].dropna().astype(str).unique().tolist()) if has_network and "relation_type" in edges_raw.columns else []
network_country_list = sorted(network_nodes["country"].dropna().astype(str).unique().tolist()) if has_network else []

with st.sidebar:
    st.divider()
    st.markdown("### Analysis controls")

    if has_thailand:
        y0, y1 = st.select_slider(
            "Thailand year range",
            options=available_years,
            value=(available_years[0], available_years[-1]),
        )
        outcome_options = [c for c in ["poverty_rate_pct", "poverty_severity", "poor_household_rate_pct", "gini_consumption"] if c in annual.columns]
        selected_outcome = st.selectbox("Outcome for regression", outcome_options, index=0) if outcome_options else None
        predictor_family = st.selectbox(
            "Predictor family",
            ["Aggregate", "Components", "Components + DOEB (recent years)"],
            index=1,
        )
        vif_threshold = st.slider("VIF cutoff", 5.0, 15.0, 10.0, 0.5)
    else:
        y0 = y1 = None
        selected_outcome = None
        predictor_family = "Components"
        vif_threshold = 10.0
        st.info("Upload or place NESDC / DOEB files to unlock Thailand analytics.")

    if has_network:
        st.divider()
        st.markdown("### Network controls")
        confidence_mode = st.radio(
            "Confidence",
            options=["High-confidence only", "Include lower-confidence mentions"],
            index=0,
        )
        min_conf = 0.75 if confidence_mode == "High-confidence only" else 0.0
        relation_types = st.multiselect(
            "Relation types",
            options=network_relation_types,
            default=[r for r in ["imports", "exports", "smuggling"] if r in network_relation_types] or network_relation_types,
        )
        year_filter = st.multiselect(
            "Years / trade-status layer",
            options=network_year_labels,
            default=network_year_labels,
        )
        focus_countries = st.multiselect(
            "Focus countries",
            options=network_country_list,
            default=["Thailand"] if "Thailand" in network_country_list else [],
        )
        focus_mode = st.radio(
            "Focus logic",
            options=["Show links touching any focus country", "Show only links between selected countries"],
            index=0,
        )
        min_mentions = st.slider("Minimum mentions per link", 1, 5, 1)
        max_nodes = st.slider("Maximum countries shown in graph", 20, max(20, len(network_country_list)), min(80, max(20, len(network_country_list))))
        show_labels = st.checkbox("Show country labels", value=True)
        map_metric = st.selectbox("Map metric", options=["weighted_total", "weighted_in", "weighted_out", "degree_total"], index=0)
    else:
        min_conf = 0.75
        relation_types = []
        year_filter = []
        focus_countries = []
        focus_mode = "Show links touching any focus country"
        min_mentions = 1
        max_nodes = 80
        show_labels = True
        map_metric = "weighted_total"
        st.info("Upload the country network ZIP or edges/nodes CSV files to unlock the map and network tabs.")

    st.divider()
    st.markdown("**Data badges**")
    st.markdown(
        _kind_badge("measured") + _kind_badge("estimated") + _kind_badge("note") + _kind_badge("upload"),
        unsafe_allow_html=True,
    )


# -----------------------------
# Derived Thailand views
# -----------------------------
view = annual[(annual["year"] >= y0) & (annual["year"] <= y1)].copy() if has_thailand else pd.DataFrame()

if has_thailand:
    if predictor_family == "Aggregate":
        selected_predictors = tuple(c for c in ["total_logistics_cost_to_gdp"] if c in annual.columns)
    elif predictor_family == "Components":
        selected_predictors = tuple(c for c in ["transport_cost_to_gdp", "inventory_cost_to_gdp", "management_cost_to_gdp"] if c in annual.columns)
    else:
        selected_predictors = tuple(
            c for c in [
                "transport_cost_to_gdp",
                "inventory_cost_to_gdp",
                "management_cost_to_gdp",
                "total_energy_import_value_million_baht",
            ] if c in annual.columns
        )

    model_result = regression_diagnostics(annual, selected_outcome, selected_predictors, vif_threshold=vif_threshold) if selected_outcome and selected_predictors else {}
    corr_predictors = [c for c in [
        "total_logistics_cost_to_gdp",
        "transport_cost_to_gdp",
        "inventory_cost_to_gdp",
        "management_cost_to_gdp",
    ] if c in annual.columns]
    corr_outcomes = [c for c in [
        "poverty_rate_pct",
        "poverty_severity",
        "poor_household_rate_pct",
        "gini_consumption",
    ] if c in annual.columns]
    correlation_table = _spearman_table(annual, corr_outcomes, corr_predictors)
    weight_sensitivity = _sensitivity_from_index(annual, [c for c in ["poverty_rate_pct", "poverty_severity", "gini_consumption"] if c in annual.columns])
    latest_row = view.dropna(subset=["vulnerability_index_0_100"]).iloc[-1] if not view.empty and view["vulnerability_index_0_100"].notna().any() else (view.iloc[-1] if not view.empty else annual.iloc[-1])
    latest_vuln = latest_row.get("vulnerability_index_0_100", np.nan)
    latest_poverty = latest_row.get("poverty_rate_pct", np.nan)
    latest_logistics = latest_row.get("total_logistics_cost_to_gdp", np.nan)
else:
    model_result = {}
    correlation_table = pd.DataFrame()
    weight_sensitivity = pd.DataFrame()
    latest_vuln = latest_poverty = latest_logistics = np.nan


# -----------------------------
# Derived network views
# -----------------------------
if has_network:
    filtered = edges_raw.copy()
    if "confidence" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["confidence"], errors="coerce") >= min_conf]
    if relation_types and "relation_type" in filtered.columns:
        filtered = filtered[filtered["relation_type"].isin(relation_types)]
    if year_filter and "year_label" in filtered.columns:
        filtered = filtered[filtered["year_label"].astype(str).isin(year_filter)]

    if focus_countries:
        focus_set = set(focus_countries)
        if focus_mode == "Show links touching any focus country":
            filtered = filtered[(filtered["source"].isin(focus_set)) | (filtered["target"].isin(focus_set))]
        else:
            filtered = filtered[(filtered["source"].isin(focus_set)) & (filtered["target"].isin(focus_set))]

    edge_pairs = aggregate_edges(filtered)
    edge_pairs = edge_pairs[edge_pairs["total_mentions"] >= min_mentions].copy()
    node_metrics = compute_node_metrics(edge_pairs, network_nodes)

    if focus_countries:
        selected_nodes = set(edge_pairs["source"]).union(edge_pairs["target"])
    else:
        ranked_nodes = node_metrics.sort_values(["weighted_total", "degree_total", "country"], ascending=[False, False, True])
        selected_nodes = set(ranked_nodes.head(max_nodes)["country"])

    edge_pairs_graph = edge_pairs[
        edge_pairs["source"].isin(selected_nodes) & edge_pairs["target"].isin(selected_nodes)
    ].copy()
    node_metrics_graph = compute_node_metrics(edge_pairs_graph, network_nodes)
    network_fig = build_network_plot(edge_pairs_graph, node_metrics_graph, show_labels=show_labels)
    map_fig = build_choropleth_map(node_metrics_graph, map_metric if map_metric in node_metrics_graph.columns else "weighted_total")
    network_validation = verify_network_map(network_nodes, edges_raw)
else:
    filtered = edge_pairs = edge_pairs_graph = node_metrics = node_metrics_graph = pd.DataFrame()
    network_fig = go.Figure()
    map_fig = go.Figure()
    network_validation = pd.DataFrame()

# -----------------------------
# Header
# -----------------------------
st.markdown('<p class="kicker">Thailand logistics stress + uploaded-data analytics + global country network</p>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero">Integrated dashboard: Thailand household vulnerability, DOEB energy flows, and global country network<br/>'
    '<span style="font-size:0.72em;font-weight:700;">Manual upload supported · auto-discovery still supported · world map restored and verified</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="tagline">This app can work from files next to the script, from manual uploads, or from both together. '
    'It now restores the country map, repairs missing node metadata for edge countries, and keeps the Thailand analytics and the network explorer in one interface.</p>',
    unsafe_allow_html=True,
)

if has_thailand:
    st.markdown(
        '<div class="sig"><b>Methodology note:</b> The NESDC logistics-cost table includes a methodology change starting in 2017, so before/after comparisons should be interpreted with caution.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------
# Tabs
# -----------------------------
main_tabs = st.tabs(
    [
        "A — Data manager",
        "B — Thailand trends",
        "C — Statistical validation",
        "D — Thailand regions & DOEB",
        "E — Global map & network",
        "F — Methods & files",
        "G — Track B lab",
    ]
)

# -----------------------------
# A: Data manager
# -----------------------------
with main_tabs[0]:
    st.markdown('<p class="section-label">A · Data loading, upload parsing, and source audit</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _value_box("Thailand dataset", "Loaded" if has_thailand else "Missing")
    with c2:
        _value_box("Network dataset", "Loaded" if has_network else "Missing")
    with c3:
        _value_box("Discovered files", str(len(discovered_files)))
    with c4:
        _value_box("Uploaded root", "Yes" if uploaded_root else "No")

    st.markdown(
        f'<div class="card">{_kind_badge("upload")}<h3>How data loading works now</h3>'
        '<p>The app first decides which roots to scan based on your sidebar choice. It can scan the script folder, a temporary folder built from uploaded files, or both. ZIP uploads are extracted automatically, then the parser searches recursively for NESDC workbooks, DOEB CSVs, and the network edges/nodes files.</p></div>',
        unsafe_allow_html=True,
    )

    roots_df = pd.DataFrame({"root": list(data_roots)}) if data_roots else pd.DataFrame(columns=["root"])
    if not roots_df.empty:
        st.markdown("**Active search roots**")
        st.dataframe(roots_df, use_container_width=True, hide_index=True)

    if discovered_files:
        st.markdown("**Detected source files**")
        det = pd.DataFrame(sorted(discovered_files.items()), columns=["role", "path"])
        st.dataframe(det, use_container_width=True, hide_index=True)
    else:
        st.warning("No recognized source files were found yet.")

    st.markdown("**Upload tips**")
    st.write(
        "You can upload individual Excel and CSV files or upload ZIP files directly. The parser understands the NESDC workbooks, the four DOEB CSVs used by the dashboard, and the existing country network ZIP or its CSV files."
    )


# -----------------------------
# B: Thailand trends
# -----------------------------
with main_tabs[1]:
    st.markdown('<p class="section-label">B · Thailand descriptive trends</p>', unsafe_allow_html=True)
    if not has_thailand or view.empty:
        st.warning("Thailand analytics are unavailable until NESDC and/or DOEB files are available.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            _value_box("Latest year", str(int(latest_row["year"])))
        with k2:
            _value_box("Vulnerability index", "NA" if pd.isna(latest_vuln) else f"{latest_vuln:.1f}")
        with k3:
            _value_box("Poverty rate (%)", "NA" if pd.isna(latest_poverty) else f"{latest_poverty:.2f}")
        with k4:
            _value_box("Logistics cost / GDP (%)", "NA" if pd.isna(latest_logistics) else f"{latest_logistics:.2f}")

        c1, c2 = st.columns((1.15, 0.85))
        with c1:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            if "vulnerability_index_0_100" in view.columns:
                fig.add_trace(
                    go.Scatter(
                        x=view["year"],
                        y=view["vulnerability_index_0_100"],
                        name="Vulnerability index (0–100)",
                        line=dict(color="#fb923c", width=3),
                    ),
                    secondary_y=False,
                )
            if "total_logistics_cost_to_gdp" in view.columns:
                fig.add_trace(
                    go.Scatter(
                        x=view["year"],
                        y=view["total_logistics_cost_to_gdp"],
                        name="Logistics cost / GDP (%)",
                        line=dict(color="#fde047", width=2),
                    ),
                    secondary_y=True,
                )
            fig.update_layout(
                template="plotly_dark",
                height=420,
                legend=dict(orientation="h", y=1.08),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=20, b=0),
            )
            fig.update_yaxes(title_text="Vulnerability index", secondary_y=False)
            fig.update_yaxes(title_text="Cost / GDP (%)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with c2:
            st.markdown(
                f'<div class="card">{_kind_badge("measured")}{_kind_badge("estimated")}<h3>Thailand source coverage</h3></div>',
                unsafe_allow_html=True,
            )
            cov_rows = []
            for c in [
                "total_logistics_cost_to_gdp",
                "transport_cost_to_gdp",
                "inventory_cost_to_gdp",
                "management_cost_to_gdp",
                "poverty_rate_pct",
                "poverty_severity",
                "gini_consumption",
                "total_energy_import_value_million_baht",
            ]:
                if c in annual.columns:
                    cov_rows.append({"field": c, "non_missing": int(annual[c].notna().sum()), "missing_pct": float(annual[c].isna().mean() * 100.0)})
            if cov_rows:
                st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No Thailand columns were parsed.")

        if {"total_logistics_cost_to_gdp", "poverty_rate_pct"}.issubset(view.columns):
            scatter = go.Figure(
                data=[
                    go.Scatter(
                        x=view["total_logistics_cost_to_gdp"],
                        y=view["poverty_rate_pct"],
                        mode="markers+text",
                        text=view["year"].astype(str),
                        textposition="top center",
                        marker=dict(size=12, color=view["vulnerability_index_0_100"], colorscale="YlOrRd", showscale=True, colorbar=dict(title="Vulnerability")),
                    )
                ]
            )
            scatter.update_layout(
                template="plotly_dark",
                title="Logistics cost / GDP vs poverty rate",
                xaxis_title="Logistics cost / GDP (%)",
                yaxis_title="Poverty rate (%)",
                height=420,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(scatter, use_container_width=True, config={"displayModeBar": False})


# -----------------------------
# C: Statistical validation
# -----------------------------
with main_tabs[2]:
    st.markdown('<p class="section-label">C · Statistical validation and model diagnostics</p>', unsafe_allow_html=True)
    if not has_thailand:
        st.warning("Thailand statistical validation requires NESDC / DOEB data.")
    else:
        c1, c2 = st.columns((1.05, 0.95))
        with c1:
            st.markdown(
                f'<div class="card">{_kind_badge("estimated")}<h3>Spearman association table</h3></div>',
                unsafe_allow_html=True,
            )
            if not correlation_table.empty:
                st.dataframe(correlation_table.sort_values(["outcome", "pvalue"], ascending=[True, True]), use_container_width=True, hide_index=True)
            else:
                st.info("Not enough parsed Thailand columns for the association table.")

            st.markdown(
                f'<div class="card">{_kind_badge("estimated")}<h3>Composite-index weight sensitivity</h3></div>',
                unsafe_allow_html=True,
            )
            if not weight_sensitivity.empty:
                st.dataframe(weight_sensitivity, use_container_width=True, hide_index=True)
            else:
                st.info("Need at least two component series to run sensitivity checks.")

        with c2:
            st.markdown(
                f'<div class="card">{_kind_badge("estimated")}<h3>Regression diagnostics</h3></div>',
                unsafe_allow_html=True,
            )
            if not model_result:
                st.info("Model diagnostics need enough overlapping years after the stationarity step.")
            else:
                meta = pd.DataFrame(
                    [
                        {"item": "Mode", "value": model_result["mode"]},
                        {"item": "Observations used", "value": model_result["n"]},
                        {"item": "R²", "value": round(model_result["r2"], 4)},
                        {"item": "Adj. R²", "value": round(model_result["adj_r2"], 4)},
                        {"item": "Durbin-Watson", "value": round(model_result["dw"], 4)},
                        {"item": "Ljung-Box p-value", "value": round(model_result["ljungbox_p"], 4) if pd.notna(model_result["ljungbox_p"]) else np.nan},
                        {"item": "Breusch-Pagan p-value", "value": round(model_result["bp_p"], 4) if pd.notna(model_result["bp_p"]) else np.nan},
                        {"item": "Backtest MAE", "value": round(model_result["cv_mae"], 4) if pd.notna(model_result["cv_mae"]) else np.nan},
                        {"item": "Backtest RMSE", "value": round(model_result["cv_rmse"], 4) if pd.notna(model_result["cv_rmse"]) else np.nan},
                    ]
                )
                st.dataframe(meta, use_container_width=True, hide_index=True)
                st.markdown("**Stationarity checks**")
                st.dataframe(model_result["stationarity"], use_container_width=True, hide_index=True)
                st.markdown("**VIF after filtering**")
                st.dataframe(model_result["vif"], use_container_width=True, hide_index=True)
                st.markdown("**HAC-robust coefficient table**")
                st.dataframe(model_result["coef"], use_container_width=True, hide_index=True)


# -----------------------------
# D: Thailand regions & DOEB
# -----------------------------
with main_tabs[3]:
    st.markdown('<p class="section-label">D · Thailand regional poverty and DOEB energy flow series</p>', unsafe_allow_html=True)
    if not has_thailand:
        st.warning("Thailand regional and DOEB views require uploaded or local NESDC / DOEB files.")
    else:
        c1, c2 = st.columns((1, 1))
        with c1:
            st.markdown(
                f'<div class="card">{_kind_badge("measured")}<h3>NESDC regional poverty view</h3></div>',
                unsafe_allow_html=True,
            )
            if not region_latest.empty:
                region_plot = go.Figure(
                    go.Bar(
                        x=region_latest["poverty_rate_pct"],
                        y=region_latest["region_th"],
                        orientation="h",
                        marker=dict(color=region_latest["poverty_rate_pct"], colorscale="YlOrRd"),
                    )
                )
                region_plot.update_layout(
                    template="plotly_dark",
                    xaxis_title="Poverty rate (%)",
                    yaxis_title="",
                    height=360,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(region_plot, use_container_width=True, config={"displayModeBar": False})
                st.dataframe(region_latest.sort_values("poverty_rate_pct", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.info("Regional poverty sheet was not parsed.")

        with c2:
            st.markdown(
                f'<div class="card">{_kind_badge("measured")}<h3>DOEB annual indexed series</h3></div>',
                unsafe_allow_html=True,
            )
            if not doeb_annual.empty:
                pick = st.multiselect(
                    "Choose DOEB annual series",
                    options=[c for c in doeb_annual.columns if c != "year"],
                    default=[c for c in ["import_refined_qty_k_liters", "export_refined_qty_k_liters", "total_energy_import_value_million_baht"] if c in doeb_annual.columns] or [c for c in doeb_annual.columns if c != "year"][:3],
                    key="doeb_pick",
                )
                fig_m = go.Figure()
                palette = ["#22d3ee", "#a78bfa", "#4ade80", "#fb7185", "#fde047"]
                for i, col in enumerate(pick):
                    temp = doeb_annual[["year", col]].dropna().copy()
                    if temp.empty:
                        continue
                    base = temp[col].iloc[0]
                    temp["index_100"] = temp[col] / base * 100.0 if base not in [0, np.nan] else np.nan
                    fig_m.add_trace(go.Scatter(x=temp["year"], y=temp["index_100"], mode="lines+markers", name=col, line=dict(color=palette[i % len(palette)], width=2)))
                fig_m.update_layout(
                    template="plotly_dark",
                    title="DOEB annual index (first available year = 100)",
                    yaxis_title="Index",
                    height=360,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=1.08),
                )
                st.plotly_chart(fig_m, use_container_width=True, config={"displayModeBar": False})
                st.dataframe(doeb_annual, use_container_width=True, hide_index=True)
            else:
                st.info("No DOEB annual series were parsed.")


# -----------------------------
# E: Global map & network
# -----------------------------
with main_tabs[4]:
    st.markdown('<p class="section-label">E · Restored world map + integrated country network explorer</p>', unsafe_allow_html=True)
    if not has_network:
        st.warning("Upload the country network ZIP or its CSV files to bring back the map and network views.")
    else:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Countries in repaired node table", len(network_nodes))
        metric_cols[1].metric("Detailed mentions in filter", len(filtered))
        metric_cols[2].metric("Pairwise links shown", len(edge_pairs_graph))
        metric_cols[3].metric("Countries with valid ISO-3", int(network_nodes["code"].astype(str).str.fullmatch(r"[A-Z]{3}", na=False).sum()))

        st.markdown(
            f'<div class="card">{_kind_badge("measured")}{_kind_badge("estimated")}<h3>Map verification</h3>'
            '<p>The world map uses ISO-3 country codes from the node table. Before plotting, the app repairs missing node metadata for countries present in the edge list, then checks how many countries remain unresolved.</p></div>',
            unsafe_allow_html=True,
        )
        if not network_validation.empty:
            st.dataframe(network_validation, use_container_width=True, hide_index=True)
        unresolved = network_summary.get("unresolved_countries", [])
        if unresolved:
            st.warning("Unresolved country metadata: " + ", ".join(map(str, unresolved)))
        else:
            st.success("All countries in the edge list were resolved into the repaired node table.")

        map_tab, graph_tab, edge_tab, country_tab = st.tabs(["World map", "Network graph", "Edge table", "Country explorer"])

        with map_tab:
            st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False})
            st.caption("Country shading is computed from the filtered network, not from static node metadata. The default metric is weighted total link volume.")

        with graph_tab:
            st.plotly_chart(network_fig, use_container_width=True)
            st.caption(
                "Node size reflects total weighted degree in the filtered network. Node color reflects region. Edge color reflects the dominant relation type for that country pair."
            )

        with edge_tab:
            st.subheader("Aggregated links")
            st.dataframe(edge_pairs, use_container_width=True, hide_index=True)
            st.download_button(
                "Download filtered links as CSV",
                data=edge_pairs.to_csv(index=False).encode("utf-8"),
                file_name="filtered_country_links.csv",
                mime="text/csv",
            )
            st.subheader("Country metrics")
            st.dataframe(
                node_metrics.sort_values(["weighted_total", "degree_total", "country"], ascending=[False, False, True]),
                use_container_width=True,
                hide_index=True,
            )

        with country_tab:
            country_choice = st.selectbox(
                "Choose a country",
                options=network_country_list,
                index=network_country_list.index("Thailand") if "Thailand" in network_country_list else 0,
                key="country_choice",
            )
            outgoing = edge_pairs[edge_pairs["source"] == country_choice].sort_values("total_mentions", ascending=False)
            incoming = edge_pairs[edge_pairs["target"] == country_choice].sort_values("total_mentions", ascending=False)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### Outgoing from {country_choice}")
                st.dataframe(outgoing, use_container_width=True, hide_index=True)
            with col2:
                st.markdown(f"### Incoming to {country_choice}")
                st.dataframe(incoming, use_container_width=True, hide_index=True)
            related_detail = filtered[(filtered["source"] == country_choice) | (filtered["target"] == country_choice)].copy()
            cols = [c for c in ["source", "target", "relation_type", "year_label", "sheet", "sentence"] if c in related_detail.columns]
            st.markdown("### Supporting evidence")
            st.dataframe(
                related_detail[cols].sort_values(cols[:3], ascending=True) if cols else related_detail,
                use_container_width=True,
                hide_index=True,
            )


# -----------------------------
# F: Methods & files
# -----------------------------
with main_tabs[5]:
    st.markdown('<p class="section-label">F · Methods, diagnostics, and file notes</p>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="card">{_kind_badge("note")}<h3>What changed in this integrated version</h3>'
        '<ul>'
        '<li>Manual file upload support was added using a single uploader that accepts raw files and ZIP archives.</li>'
        '<li>The world map is back and now uses repaired ISO-3 node metadata.</li>'
        '<li>The country network explorer was integrated into the same app instead of staying as a separate dashboard.</li>'
        '<li>The Thailand analytics still run from the parsed NESDC and DOEB files, whether they were auto-discovered or uploaded manually.</li>'
        '</ul></div>',
        unsafe_allow_html=True,
    )

    notes = [
        {
            "method": "Spearman association",
            "where_applied": "Thailand statistical validation tab",
            "why": "Used for monotonic association without relying on normality.",
        },
        {
            "method": "ADF + KPSS stationarity checks",
            "where_applied": "Regression diagnostics before OLS fitting",
            "why": "If series are not clearly stationary, the model moves to first differences.",
        },
        {
            "method": "VIF filtering",
            "where_applied": "Regression diagnostics",
            "why": "Reduces multicollinearity before fitting the multivariable model.",
        },
        {
            "method": "HAC robust covariance",
            "where_applied": "Coefficient table in regression diagnostics",
            "why": "Makes inference less fragile under autocorrelation and heteroskedasticity.",
        },
        {
            "method": "TimeSeriesSplit backtest",
            "where_applied": "Regression diagnostics",
            "why": "Provides a simple leakage-safe out-of-sample error estimate for time-ordered data.",
        },
        {
            "method": "Map verification",
            "where_applied": "Global map & network tab",
            "why": "Checks node/edge coverage and ISO-3 validity before the choropleth renders.",
        },
    ]
    st.dataframe(pd.DataFrame(notes), use_container_width=True, hide_index=True)

    file_rows = []
    for role, path in sorted(discovered_files.items()):
        try:
            size_kb = round(os.path.getsize(path) / 1024.0, 1)
        except OSError:
            size_kb = np.nan
        file_rows.append({"role": role, "path": path, "size_kb": size_kb})
    if file_rows:
        st.markdown("**Recognized files**")
        st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)

    if network_summary:
        st.markdown("**Network summary metadata**")
        st.json(network_summary)


# -----------------------------
# G: Track B lab
# -----------------------------
with main_tabs[6]:
    st.markdown('<p class="section-label">G · Track B money resilience lab (uploaded prototype, statistically hardened)</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="card">{_kind_badge("estimated")}{_kind_badge("upload")}<h3>What was fixed from the uploaded prototype</h3>'
        '<p>This page keeps the same storytelling goal as your Track B dashboard, but removes the main statistical weak points: no backward fill, no full-sample min-max stress scaling, no full-sample z-scores, and no fixed alert threshold. Instead it uses past-only normalization and historical quantile alerts.</p></div>',
        unsafe_allow_html=True,
    )

    fix_table = pd.DataFrame(
        [
            {"old_logic": "Full-sample z-score and min-max scaling", "fixed_logic": "Past-only rolling/expanding z-score + historical percentile", "why": "Reduces look-ahead leakage."},
            {"old_logic": "Backward fill after pivoting fuel prices", "fixed_logic": "Forward fill only", "why": "Does not borrow future values."},
            {"old_logic": "Single fixed alert cutoff", "fixed_logic": "Historical quantile threshold", "why": "Lets the alert adapt to regime shifts."},
            {"old_logic": "Raw multi-unit DOEB lines on one axis", "fixed_logic": "Base-100 indexed comparison", "why": "Avoids misleading scale comparisons."},
            {"old_logic": "Narrative and measured signals mixed together", "fixed_logic": "Narrative and measured series separated", "why": "Keeps interpretation cleaner."},
        ]
    )
    st.dataframe(fix_table, use_container_width=True, hide_index=True)

    local_sections = st.tabs(["1 — Global map", "2 — Thailand stress", "3 — Social protection & DOEB"])

    with local_sections[0]:
        if trackb_global_gap.empty:
            st.info("Upload or place OWID_Energy_Data.csv to bring back the Track B world dependency map on this page.")
        else:
            map_years = sorted(trackb_global_gap["year"].dropna().astype(int).unique().tolist())
            map_year_g = st.select_slider("Track B map year", options=map_years, value=map_years[-1], key="trackb_map_year")
            gmap = trackb_global_gap[trackb_global_gap["year"] == map_year_g].dropna(subset=["gap_percentile_0_100"])
            fig_map_tb = go.Figure(
                data=go.Choropleth(
                    locations=gmap["iso_code"],
                    z=gmap["gap_percentile_0_100"],
                    locationmode="ISO-3",
                    colorscale=[[0, "#0f172a"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                    colorbar_title="Percentile",
                    text=gmap["country"],
                    hovertemplate="<b>%{text}</b><br>Dependency percentile: %{z:.1f}<extra></extra>",
                )
            )
            fig_map_tb.update_layout(
                title=f"Track B global oil dependency percentile ({map_year_g})",
                geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth", bgcolor="rgba(0,0,0,0)"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                height=480,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_map_tb, use_container_width=True, config={"displayModeBar": False})
            if not trackb_map_check.empty:
                st.markdown("**Map verification**")
                st.dataframe(trackb_map_check, use_container_width=True, hide_index=True)

    with local_sections[1]:
        if trackb_metric.empty:
            st.warning("The uploaded Track B page needs thailand_fuel_prices_cleaned.csv and owid-energy-data(clean).csv for the monthly stress monitor. This page is ready for them, but they are not in the current data roots.")
        else:
            metric_years = sorted(trackb_metric["year"].dropna().astype(int).unique().tolist())
            default_range = (metric_years[0], metric_years[-1])
            if y0 is not None and y1 is not None:
                default_range = (max(metric_years[0], y0), min(metric_years[-1], y1))
            y0_tb, y1_tb = st.select_slider("Track B Thailand year range", options=metric_years, value=default_range, key="trackb_year_range")
            q_tb = st.slider("Track B historical alert quantile", 0.70, 0.95, 0.85, 0.01, key="trackb_alert_q")
            view_tb = trackb_metric[(trackb_metric["year"] >= y0_tb) & (trackb_metric["year"] <= y1_tb)].copy()
            view_tb["alert_threshold_q"] = _historical_quantile_threshold(view_tb["stress_index"], q=q_tb, min_periods=12)
            view_tb["alert_flag_q"] = view_tb["stress_index"].notna() & view_tb["alert_threshold_q"].notna() & (view_tb["stress_index"] >= view_tb["alert_threshold_q"])

            c1, c2, c3 = st.columns(3)
            latest_tb = view_tb.dropna(subset=["stress_index"]).iloc[-1] if view_tb["stress_index"].notna().any() else view_tb.iloc[-1]
            with c1:
                _value_box("Track B stress", "NA" if pd.isna(latest_tb.get("stress_index", np.nan)) else f"{float(latest_tb['stress_index']):.1f}")
            with c2:
                _value_box("Alert months", str(int(view_tb["alert_flag_q"].sum())), f"q = {q_tb:.2f}")
            with c3:
                _value_box("Basket series used", str(len(trackb_bundle.get("weights", {}).get("basket_weights", {}))))

            fig_tb = make_subplots(specs=[[{"secondary_y": True}]])
            fig_tb.add_trace(go.Scatter(x=view_tb["Date"], y=view_tb["stress_index"], name="Stress index", line=dict(color="#fb923c", width=3), fill="tozeroy", fillcolor="rgba(251,146,60,0.18)"), secondary_y=False)
            fig_tb.add_trace(go.Scatter(x=view_tb["Date"], y=view_tb["alert_threshold_q"], name=f"Alert threshold (q={q_tb:.2f})", line=dict(color="#f87171", width=1.5, dash="dash")), secondary_y=False)
            fig_tb.add_trace(go.Scatter(x=view_tb["Date"], y=view_tb["basket_price"], name="Retail basket (THB/L)", line=dict(color="#fde047", width=2)), secondary_y=True)
            fig_tb.update_layout(template="plotly_dark", height=420, legend=dict(orientation="h", y=1.12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            fig_tb.update_yaxes(title_text="Stress percentile (0–100)", secondary_y=False)
            fig_tb.update_yaxes(title_text="THB/L", secondary_y=True)
            st.plotly_chart(fig_tb, use_container_width=True, config={"displayModeBar": False})

            component_cols = [c for c in ["comp_affordability_shock", "comp_affordability_level", "comp_volatility", "comp_import_dependency"] if c in view_tb.columns]
            if component_cols:
                comp_fig = go.Figure()
                for c, color in zip(component_cols, ["#fb923c", "#fbbf24", "#22d3ee", "#a78bfa"]):
                    comp_fig.add_trace(go.Scatter(x=view_tb["Date"], y=view_tb[c], mode="lines", name=c.replace("comp_", ""), line=dict(width=2, color=color)))
                comp_fig.update_layout(template="plotly_dark", height=300, title="Stress components (past-only normalized)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(comp_fig, use_container_width=True, config={"displayModeBar": False})

            c4, c5 = st.columns((1, 1))
            with c4:
                yearly_tb = trackb_yearly[(trackb_yearly["year"] >= y0_tb) & (trackb_yearly["year"] <= y1_tb)].copy() if not trackb_yearly.empty else pd.DataFrame()
                if not yearly_tb.empty:
                    fig_trade_tb = go.Figure()
                    fig_trade_tb.add_trace(
                        go.Scatter(
                            x=yearly_tb["year_avg_basket_thb"], y=yearly_tb["year_alert_share"], mode="markers+text", text=yearly_tb["year"].astype(str), textposition="top center",
                            marker=dict(size=np.clip(yearly_tb["year_avg_stress"].fillna(0.0), 8, 22), color=yearly_tb["year_avg_stress"], colorscale="YlOrRd", showscale=True, colorbar=dict(title="Avg stress")), name="Year",
                        )
                    )
                    fig_trade_tb.update_layout(template="plotly_dark", title="Average basket vs share of alert months", xaxis_title="Average retail basket price (THB/L)", yaxis_title="Share of alert months (%)", height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_trade_tb, use_container_width=True, config={"displayModeBar": False})
            with c5:
                if not trackb_future_signal.empty:
                    st.markdown("**Face-validity check**")
                    st.caption("Compare next-year average basket change after high-stress years versus low-stress years.")
                    st.dataframe(trackb_future_signal, use_container_width=True, hide_index=True)
                weights_df = pd.DataFrame([{"group": k, "weights": json.dumps(v, ensure_ascii=False)} for k, v in trackb_bundle.get("weights", {}).items()]) if trackb_bundle.get("weights") else pd.DataFrame()
                if not weights_df.empty:
                    st.markdown("**Weights used**")
                    st.dataframe(weights_df, use_container_width=True, hide_index=True)

    with local_sections[2]:
        c1, c2 = st.columns((1, 1))
        with c1:
            if has_thailand and not view.empty:
                st.markdown(f'<div class="card">{_kind_badge("measured")}<h3>NESDC vulnerability and logistics</h3></div>', unsafe_allow_html=True)
                fig_social = make_subplots(specs=[[{"secondary_y": True}]])
                if "vulnerability_index_0_100" in view.columns:
                    fig_social.add_trace(go.Scatter(x=view["year"], y=view["vulnerability_index_0_100"], name="Vulnerability index", line=dict(color="#fb923c", width=3)), secondary_y=False)
                if "poverty_rate_pct" in view.columns:
                    fig_social.add_trace(go.Scatter(x=view["year"], y=view["poverty_rate_pct"], name="Poverty rate (%)", line=dict(color="#fde047", width=2)), secondary_y=False)
                if "total_logistics_cost_to_gdp" in view.columns:
                    fig_social.add_trace(go.Bar(x=view["year"], y=view["total_logistics_cost_to_gdp"], name="Logistics cost / GDP", marker_color="#22d3ee", opacity=0.55), secondary_y=True)
                fig_social.update_layout(template="plotly_dark", height=380, legend=dict(orientation="h", y=1.1), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                fig_social.update_yaxes(title_text="Vulnerability / poverty", secondary_y=False)
                fig_social.update_yaxes(title_text="% GDP", secondary_y=True)
                st.plotly_chart(fig_social, use_container_width=True, config={"displayModeBar": False})
                if not region_latest.empty:
                    region_label_col = next((c for c in ["region_th", "region_en", "region"] if c in region_latest.columns), None)
                    if region_label_col and "poverty_rate_pct" in region_latest.columns:
                        region_plot_df = region_latest[[region_label_col, "poverty_rate_pct"]].dropna().copy()
                        region_chart = go.Figure(
                            go.Bar(
                                x=region_plot_df["poverty_rate_pct"],
                                y=region_plot_df[region_label_col],
                                orientation="h",
                                marker=dict(color=region_plot_df["poverty_rate_pct"], colorscale="YlOrRd"),
                            )
                        )
                        region_chart.update_layout(template="plotly_dark", height=320, title="Latest NESDC regional poverty rates", xaxis_title="Poverty rate (%)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(region_chart, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.info("NESDC regional poverty data loaded, but no recognized region label column was found.")
        with c2:
            if not doeb_annual.empty:
                st.markdown(f'<div class="card">{_kind_badge("measured")}<h3>DOEB infrastructure series (indexed)</h3></div>', unsafe_allow_html=True)
                doeb_cols = [c for c in ["import_refined_qty_k_liters", "import_refined_value_million_baht", "import_crude_value_million_baht", "export_refined_qty_k_liters"] if c in doeb_annual.columns]
                selected_doeb = st.multiselect("Choose DOEB annual series", options=doeb_cols, default=doeb_cols[: min(3, len(doeb_cols))], key="trackb_doeb_pick")
                fig_doeb_idx = go.Figure()
                for c, color in zip(selected_doeb, ["#22d3ee", "#a78bfa", "#4ade80", "#fb7185"]):
                    temp = doeb_annual[["year", c]].dropna().sort_values("year")
                    if temp.empty:
                        continue
                    base = float(temp[c].iloc[0]) if pd.notna(temp[c].iloc[0]) else np.nan
                    temp["index_100"] = temp[c] / base * 100.0 if pd.notna(base) and base != 0 else np.nan
                    fig_doeb_idx.add_trace(go.Scatter(x=temp["year"], y=temp["index_100"], mode="lines+markers", name=c, line=dict(width=2, color=color)))
                fig_doeb_idx.update_layout(template="plotly_dark", height=360, title="DOEB annual index (base year = 100)", yaxis_title="Index (base = 100)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_doeb_idx, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Upload NESDC and DOEB files to unlock the Thailand social-protection and infrastructure panels on this page.")

if st.query_params.get("assistant", "0") != "1":
    st.markdown(
        """
        <a href="?assistant=1" class="chat-fab" title="Open Assistant">🤖</a>
        """,
        unsafe_allow_html=True,
    )                
