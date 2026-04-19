from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# -----------------------------
# Page config + theme
# -----------------------------
st.set_page_config(
    page_title="Energy Stress → Human Adaptation | Track B (Validated)",
    page_icon="⛽",
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
    max-width: 1450px;
    padding-top: 1.75rem !important;
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
.badge-narrative { background: rgba(245,158,11,.18); color: #fde68a; border: 1px solid rgba(245,158,11,.35); }
.badge-placeholder { background: rgba(239,68,68,.18); color: #fecaca; border: 1px solid rgba(239,68,68,.35); }
.small-muted { color: #94a3b8; font-size: 12px; }
.chat-fab {
    position: fixed; right: 20px; bottom: 22px; width: 56px; height: 56px;
    border-radius: 50%; background: linear-gradient(145deg, #fde047, #f59e0b);
    color: #0f172a; font-size: 26px; display: flex; align-items: center; justify-content: center;
    box-shadow: 0 8px 28px rgba(0,0,0,0.45); z-index: 9999; border: 2px solid #fffbeb;
}
</style>
    """,
    unsafe_allow_html=True,
)

TRACK_B = "Track_B_Adaptive_Infrastructures_Datasets"
_GLOBAL_OWID = os.path.join(TRACK_B, "OWID_Energy_Data.csv")

_DOEB_SERIES: Tuple[Tuple[str, str, str], ...] = (
    (
        "DOEB dataset/การนำเข้าน้ำมัน/files/ปริมาณการนำเข้าน้ำมันสำเร็จรูป/vw_opendata_045_i_fuel_sum_x_data_view.csv",
        "import_refined_qty",
        "นำเข้าน้ำมันสำเร็จรูป — ปริมาณ (พันลิตร/เดือน)",
    ),
    (
        "DOEB dataset/การนำเข้าน้ำมัน/files/มูลค่าการนำเข้าน้ำมันสำเร็จรูป/vw_opendata_037_i_fuel_value_data_view.csv",
        "import_refined_value_mthb",
        "นำเข้าน้ำมันสำเร็จรูป — มูลค่า (ล้านบาท/เดือน)",
    ),
    (
        "DOEB dataset/การนำเข้าน้ำมัน/files/มูลค่าการนำเข้าน้ำมันดิบ/vw_opendata_038_i_crude_value_data_view.csv",
        "import_crude_value_mthb",
        "นำเข้าน้ำมันดิบ — มูลค่า (ล้านบาท/เดือน)",
    ),
    (
        "DOEB dataset/การส่งออกน้ำมัน/files/ปริมาณการส่งออกน้ำมันสำเร็จรูป/vw_opendata_039_e_fuel_sum_data_view.csv",
        "export_refined_qty",
        "ส่งออกน้ำมันสำเร็จรูป — ปริมาณ (พันลิตร/เดือน)",
    ),
)

DEFAULT_BASKET_WEIGHTS = {
    "Thailand (Diesel HSD B7) - Retail Price": 0.35,
    "Thailand (Gasohol 95-E10) - Table 9 Retail Price": 0.30,
    "Thailand (LPG low income hh) - Retail Price": 0.25,
    "Thailand (Kerosene) - Retail Price": 0.10,
}


# -----------------------------
# Helper functions
# -----------------------------
def _kind_badge(kind: str) -> str:
    cls = {
        "measured": "badge-measured",
        "estimated": "badge-estimated",
        "narrative": "badge-narrative",
        "placeholder": "badge-placeholder",
    }.get(kind, "badge-estimated")
    return f'<span class="badge {cls}">{kind.upper()}</span>'


def _here(*parts: str) -> str:
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts))


def _track_b_root() -> str:
    return _here(TRACK_B)


def _safe_read_csv(path: str, **kwargs: Any) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _safe_read_excel(path: str, **kwargs: Any) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_excel(path, engine="openpyxl", **kwargs)
    except Exception:
        try:
            return pd.read_excel(path, **kwargs)
        except Exception:
            return pd.DataFrame()


def _winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    lo = float(s.quantile(lower_q))
    hi = float(s.quantile(upper_q))
    return s.clip(lower=lo, upper=hi)


def _normalize_weights(weight_map: Dict[str, float], available: List[str]) -> Dict[str, float]:
    picked = {k: float(v) for k, v in weight_map.items() if k in available and float(v) > 0}
    total = sum(picked.values())
    if total <= 0:
        if not available:
            return {}
        eq = 1.0 / len(available)
        return {k: eq for k in available}
    return {k: v / total for k, v in picked.items()}


def _rolling_past_zscore(s: pd.Series, window: int = 12, min_periods: int = 6, clip_upper: float = 6.0) -> pd.Series:
    """
    Leak-free z-score:
    each point is standardized using only *past* data (shift(1)).
    Fallbacks:
    - rolling past mean/std
    - expanding past mean/std if rolling history is too short
    """
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


def _historical_percentile(s: pd.Series, min_periods: int = 12) -> pd.Series:
    """
    Leak-free percentile score:
    percentile rank of current point against values observed up to current time.
    """
    s = pd.to_numeric(s, errors="coerce")
    vals: List[float] = []
    out: List[float] = []
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


def _historical_quantile_threshold(s: pd.Series, q: float = 0.85, min_periods: int = 12) -> pd.Series:
    """
    Leak-free rolling historical threshold based only on data available *before* current point.
    """
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


def _annual_to_monthly(month_df: pd.DataFrame, annual_df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if month_df.empty or annual_df.empty or "year" not in annual_df.columns:
        return month_df
    out = month_df.copy()
    out["year"] = out["Date"].dt.year
    keep = ["year"] + [c for c in cols if c in annual_df.columns]
    ann = annual_df[keep].drop_duplicates("year")
    return out.merge(ann, on="year", how="left")


def _value_box(label: str, value: str, delta: Optional[str] = None) -> None:
    st.metric(label, value, delta)


def _try_get_country_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["country", "Country", "COUNTRY"]:
        if col in df.columns:
            return col
    return None


def _try_get_year_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["year", "Year", "YEAR"]:
        if col in df.columns:
            return col
    return None


def _compute_weight_sensitivity(metric: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "comp_affordability_shock",
        "comp_affordability_level",
        "comp_volatility",
        "comp_import_dependency",
    ]
    present = [c for c in needed if c in metric.columns]
    if len(present) < 2 or metric.empty:
        return pd.DataFrame()

    schemes = {
        "Baseline": {
            "comp_affordability_shock": 0.35,
            "comp_affordability_level": 0.25,
            "comp_volatility": 0.25,
            "comp_import_dependency": 0.15,
        },
        "Equal": {c: 1.0 for c in present},
        "Affordability-heavy": {
            "comp_affordability_shock": 0.45,
            "comp_affordability_level": 0.30,
            "comp_volatility": 0.15,
            "comp_import_dependency": 0.10,
        },
        "Volatility-heavy": {
            "comp_affordability_shock": 0.20,
            "comp_affordability_level": 0.20,
            "comp_volatility": 0.45,
            "comp_import_dependency": 0.15,
        },
    }

    score_map: Dict[str, pd.Series] = {}
    for name, weights in schemes.items():
        weights = _normalize_weights(weights, present)
        raw = pd.Series(0.0, index=metric.index)
        for c, w in weights.items():
            raw = raw + metric[c].fillna(0.0) * w
        score_map[name] = _historical_percentile(raw, min_periods=12)

    baseline = score_map["Baseline"]
    rows = []
    for name, s in score_map.items():
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


def _diagnostic_future_signal(metric: pd.DataFrame, horizon_months: int = 3) -> pd.DataFrame:
    if metric.empty or "basket_price" not in metric.columns or "stress_index" not in metric.columns:
        return pd.DataFrame()

    df = metric[["Date", "basket_price", "stress_index"]].copy()
    df["future_change_pct"] = (df["basket_price"].shift(-horizon_months) / df["basket_price"] - 1.0) * 100.0
    df = df.dropna()

    if df.empty:
        return pd.DataFrame()

    q80 = df["stress_index"].quantile(0.80)
    q20 = df["stress_index"].quantile(0.20)

    high = df[df["stress_index"] >= q80]
    low = df[df["stress_index"] <= q20]

    return pd.DataFrame(
        [
            {
                "group": "High stress months (top 20%)",
                f"avg_next_{horizon_months}m_price_change_pct": float(high["future_change_pct"].mean()) if not high.empty else np.nan,
                "n_months": int(len(high)),
            },
            {
                "group": "Low stress months (bottom 20%)",
                f"avg_next_{horizon_months}m_price_change_pct": float(low["future_change_pct"].mean()) if not low.empty else np.nan,
                "n_months": int(len(low)),
            },
        ]
    )


def _render_data_status(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        st.warning(f"ไม่พบข้อมูล: {label}")
    else:
        st.caption(f"{label}: {len(df):,} rows")


# -----------------------------
# Data loaders
# -----------------------------
def _read_doeb_csv(rel_path: str) -> pd.DataFrame:
    path = os.path.normpath(os.path.join(_track_b_root(), *rel_path.split("/")))
    raw = _safe_read_csv(path, encoding="utf-8")
    if raw.empty:
        return pd.DataFrame()

    value_col = "QTY" if "QTY" in raw.columns else ("BALANCE_VALUE" if "BALANCE_VALUE" in raw.columns else None)
    if value_col is None or "YEAR_ID" not in raw.columns or "MONTH_ID" not in raw.columns:
        return pd.DataFrame()

    out = raw.copy()
    out["year_ce"] = pd.to_numeric(out["YEAR_ID"], errors="coerce") - 543
    out["month"] = pd.to_numeric(out["MONTH_ID"], errors="coerce").astype("Int64")
    out["value"] = pd.to_numeric(out[value_col], errors="coerce")
    out["unit"] = out.get("UNIT", "").astype(str)
    out["subject"] = out.get("SUBJECT", "").astype(str)

    out = out.dropna(subset=["year_ce", "month", "value"])
    out["Date"] = pd.to_datetime(
        out["year_ce"].astype(int).astype(str)
        + "-"
        + out["month"].astype(int).astype(str).str.zfill(2)
        + "-01",
        errors="coerce",
    )
    out = out.dropna(subset=["Date"])
    return out[["Date", "year_ce", "month", "value", "unit", "subject"]].sort_values("Date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_doeb_bundle_from_repo() -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    bundle: Dict[str, pd.DataFrame] = {}
    labels: Dict[str, str] = {}
    for rel, key, label_th in _DOEB_SERIES:
        df = _read_doeb_csv(rel)
        if not df.empty:
            bundle[key] = df
            labels[key] = label_th
    return bundle, labels


@st.cache_data(show_spinner=False)
def list_cloned_data_files() -> pd.DataFrame:
    root = _track_b_root()
    if not os.path.isdir(root):
        return pd.DataFrame(columns=["relative_path", "size_kb"])
    rows = []
    for dirpath, _, filenames in os.walk(root):
        depth = dirpath[len(root) :].count(os.sep)
        if depth > 12:
            continue
        for fn in filenames:
            if fn.lower().endswith((".csv", ".xlsx", ".ipynb")):
                fp = os.path.join(dirpath, fn)
                try:
                    sz = os.path.getsize(fp) / 1024.0
                except OSError:
                    sz = 0.0
                rows.append({"relative_path": os.path.relpath(fp, root), "size_kb": round(sz, 1)})
    if not rows:
        return pd.DataFrame(columns=["relative_path", "size_kb"])
    return pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def try_load_world_bank_tables() -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[str], Optional[pd.DataFrame]]:
    tb = _track_b_root()
    targets = [
        "Global_Fuel_Prices_Database.xlsx",
        "Global_Fuel_Subsidies_and_Price_Control_Measures_Database.xlsx",
    ]
    found_prices = None
    found_subsidy = None

    if not os.path.isdir(tb):
        return None, None, None, None

    for dirpath, _, filenames in os.walk(tb):
        for fn in filenames:
            if fn == targets[0]:
                found_prices = os.path.join(dirpath, fn)
            elif fn == targets[1]:
                found_subsidy = os.path.join(dirpath, fn)

    df_p = _safe_read_excel(found_prices, sheet_name=0, nrows=4000) if found_prices else pd.DataFrame()
    df_s = _safe_read_excel(found_subsidy, sheet_name=0, nrows=4000) if found_subsidy else pd.DataFrame()

    return found_prices, (df_p if not df_p.empty else None), found_subsidy, (df_s if not df_s.empty else None)


@st.cache_data(show_spinner=False)
def load_global_gap_stress() -> pd.DataFrame:
    path = _here(_GLOBAL_OWID)
    g = _safe_read_csv(path, low_memory=False)
    if g.empty:
        return pd.DataFrame()

    required = {"country", "iso_code", "year", "oil_consumption"}
    if not required.issubset(set(g.columns)):
        return pd.DataFrame()

    cols = ["country", "iso_code", "year", "oil_consumption"]
    if "oil_production" in g.columns:
        cols.append("oil_production")
    g = g[cols].copy()

    g["iso_code"] = g["iso_code"].astype(str).str.upper().str.strip()
    g = g[g["iso_code"].str.len() == 3]
    g["year"] = pd.to_numeric(g["year"], errors="coerce")
    g["oil_consumption"] = pd.to_numeric(g["oil_consumption"], errors="coerce")
    g["oil_production"] = pd.to_numeric(g.get("oil_production", np.nan), errors="coerce")

    g = g.dropna(subset=["year", "oil_consumption"])
    g["oil_production"] = g["oil_production"].fillna(0.0)

    g["gap_twh_abs"] = (g["oil_consumption"] - g["oil_production"]).abs()
    denom = g["oil_consumption"].abs() + g["oil_production"].abs() + 1e-9
    g["gap_dependency_ratio"] = g["gap_twh_abs"] / denom * 100.0

    g["gap_percentile_0_100"] = (
        g.groupby("year")["gap_dependency_ratio"]
        .rank(method="average", pct=True)
        .mul(100.0)
    )

    return g.sort_values(["year", "country"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_thailand_core() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    fuel_path = _here("thailand_fuel_prices_cleaned.csv")
    owid_th_path = _here("owid-energy-data(clean).csv")

    dq: Dict[str, Any] = {
        "fuel_rows_raw": 0,
        "fuel_rows_clean": 0,
        "basket_series": [],
        "basket_weights_used": {},
        "date_min": None,
        "date_max": None,
        "normalization_note": "Leak-free normalization: shift(1) + rolling/expanding history only; no backward fill.",
    }

    fuel = _safe_read_csv(fuel_path)
    owid_raw = _safe_read_csv(owid_th_path, low_memory=False)

    if fuel.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), dq

    dq["fuel_rows_raw"] = int(len(fuel))
    fuel["Date"] = pd.to_datetime(fuel.get("Date"), errors="coerce")
    fuel["Price"] = pd.to_numeric(fuel.get("Price"), errors="coerce")
    fuel["Description"] = fuel.get("Description", "").astype(str)
    fuel["Unit"] = fuel.get("Unit", "").astype(str)

    fuel = fuel.dropna(subset=["Date", "Price"])
    fuel = fuel[fuel["Description"].str.contains("Thailand", case=False, na=False)]
    fuel = fuel[fuel["Unit"].eq("LCU (Local Currency Unit)")]
    fuel = fuel[fuel["Description"].str.contains("Retail Price", case=False, na=False)].copy()
    fuel["Year"] = fuel["Date"].dt.year
    fuel["Month"] = fuel["Date"].dt.to_period("M")

    available_desc = sorted(fuel["Description"].dropna().unique().tolist())
    weights_used = _normalize_weights(DEFAULT_BASKET_WEIGHTS, available_desc)
    keep = list(weights_used.keys())

    if not keep:
        dq["error"] = "No configured Thailand fuel basket series found in thailand_fuel_prices_cleaned.csv"
        return fuel, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), dq

    fb = fuel[fuel["Description"].isin(keep)].copy()
    fb["Price"] = fb.groupby("Description")["Price"].transform(_winsorize_series)
    dq["fuel_rows_clean"] = int(len(fb))

    monthly = fb.groupby(["Month", "Description"], as_index=False)["Price"].mean()
    monthly["Date"] = monthly["Month"].dt.to_timestamp()

    pivot = monthly.pivot(index="Date", columns="Description", values="Price").sort_index()

    for d in keep:
        if d not in pivot.columns:
            pivot[d] = np.nan

    pivot = pivot[keep].sort_index()
    pivot = pivot.ffill()  # forward fill only; no backward fill to avoid future leakage
    pivot["basket_price"] = sum(pivot[d] * weights_used[d] for d in keep)
    pivot = pivot.dropna(subset=["basket_price"]).copy()

    if pivot.empty:
        dq["error"] = "Basket series exist but basket_price became empty after filtering."
        return fb, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), dq

    # Leak-free component construction
    pivot["mom_change_pct"] = pivot["basket_price"].pct_change() * 100.0
    trailing_median = pivot["basket_price"].shift(1).rolling(12, min_periods=6).median()
    pivot["price_gap_vs_12m_median_pct"] = ((pivot["basket_price"] / trailing_median) - 1.0) * 100.0
    pivot["upside_mom_pct"] = pivot["mom_change_pct"].clip(lower=0.0)
    pivot["positive_level_gap_pct"] = pivot["price_gap_vs_12m_median_pct"].clip(lower=0.0)
    pivot["vol_6m"] = pivot["mom_change_pct"].rolling(6, min_periods=3).std(ddof=1)

    # OWID Thailand data
    owid_th = pd.DataFrame()
    annual_energy = pd.DataFrame()

    if not owid_raw.empty:
        country_col = _try_get_country_column(owid_raw)
        year_col = _try_get_year_column(owid_raw)

        if country_col and year_col:
            owid_th = owid_raw.copy()
            owid_th = owid_th[owid_th[country_col].astype(str).str.lower().eq("thailand")].copy()
            owid_th["year"] = pd.to_numeric(owid_th[year_col], errors="coerce")
            if "oil_consumption" in owid_th.columns:
                owid_th["oil_consumption"] = pd.to_numeric(owid_th["oil_consumption"], errors="coerce")
            if "oil_production" in owid_th.columns:
                owid_th["oil_production"] = pd.to_numeric(owid_th["oil_production"], errors="coerce")
            use_cols = ["year"]
            for c in ["oil_consumption", "oil_production"]:
                if c in owid_th.columns:
                    use_cols.append(c)
            annual_energy = owid_th[use_cols].dropna(subset=["year"]).drop_duplicates("year").sort_values("year")
            if "oil_consumption" in annual_energy.columns:
                annual_energy["oil_consumption_yoy_pct"] = annual_energy["oil_consumption"].pct_change() * 100.0
            if {"oil_consumption", "oil_production"}.issubset(set(annual_energy.columns)):
                annual_energy["import_dependency_pct"] = np.where(
                    annual_energy["oil_consumption"] > 0,
                    np.maximum(annual_energy["oil_consumption"] - annual_energy["oil_production"], 0.0)
                    / annual_energy["oil_consumption"] * 100.0,
                    np.nan,
                )
            elif "oil_consumption" in annual_energy.columns:
                annual_energy["import_dependency_pct"] = annual_energy["oil_consumption_yoy_pct"].clip(lower=0.0)

    metric = pd.DataFrame({"Date": pivot.index}).merge(
        pivot[
            [
                "basket_price",
                "mom_change_pct",
                "price_gap_vs_12m_median_pct",
                "upside_mom_pct",
                "positive_level_gap_pct",
                "vol_6m",
            ]
        ],
        left_on="Date",
        right_index=True,
        how="left",
    )

    metric["year"] = metric["Date"].dt.year
    if not annual_energy.empty:
        metric = _annual_to_monthly(metric, annual_energy, ["oil_consumption_yoy_pct", "import_dependency_pct"])

    # Components
    metric["comp_affordability_shock"] = _rolling_past_zscore(metric["upside_mom_pct"], window=12, min_periods=6).clip(lower=0.0)
    metric["comp_affordability_level"] = _rolling_past_zscore(metric["positive_level_gap_pct"], window=12, min_periods=6).clip(lower=0.0)
    metric["comp_volatility"] = _rolling_past_zscore(metric["vol_6m"], window=12, min_periods=6).clip(lower=0.0)

    if "import_dependency_pct" in metric.columns and metric["import_dependency_pct"].notna().any():
        metric["comp_import_dependency"] = _rolling_past_zscore(metric["import_dependency_pct"], window=24, min_periods=6).clip(lower=0.0)
    else:
        metric["comp_import_dependency"] = 0.0

    component_cols = [
        "comp_affordability_shock",
        "comp_affordability_level",
        "comp_volatility",
        "comp_import_dependency",
    ]
    active_components = [c for c in component_cols if metric[c].fillna(0).abs().sum() > 0]
    comp_weights = _normalize_weights(
        {
            "comp_affordability_shock": 0.35,
            "comp_affordability_level": 0.25,
            "comp_volatility": 0.25,
            "comp_import_dependency": 0.15,
        },
        active_components,
    )

    metric["raw_stress_score"] = 0.0
    for c, w in comp_weights.items():
        metric["raw_stress_score"] = metric["raw_stress_score"] + metric[c].fillna(0.0) * w

    metric["stress_index"] = _historical_percentile(metric["raw_stress_score"], min_periods=12)
    metric["alert_threshold_85"] = _historical_quantile_threshold(metric["stress_index"], q=0.85, min_periods=12)
    metric["alert_flag_85"] = (
        metric["stress_index"].notna()
        & metric["alert_threshold_85"].notna()
        & (metric["stress_index"] >= metric["alert_threshold_85"])
    )

    yearly_rows = []
    for y, g in metric.groupby("year"):
        yearly_rows.append(
            {
                "year": int(y),
                "year_avg_basket_thb": float(g["basket_price"].mean()),
                "year_avg_stress": float(g["stress_index"].mean()) if g["stress_index"].notna().any() else np.nan,
                "year_alert_share": float(g["alert_flag_85"].mean()) * 100.0 if g["alert_flag_85"].notna().any() else np.nan,
                "year_price_volatility": float(g["mom_change_pct"].std(ddof=1)) if g["mom_change_pct"].notna().sum() > 1 else np.nan,
            }
        )
    yearly_fuel = pd.DataFrame(yearly_rows).sort_values("year").reset_index(drop=True)

    dq["basket_series"] = keep
    dq["basket_weights_used"] = weights_used
    dq["component_weights_used"] = comp_weights
    dq["date_min"] = str(metric["Date"].min().date()) if not metric.empty else None
    dq["date_max"] = str(metric["Date"].max().date()) if not metric.empty else None
    dq["missing_rate_basket_pct"] = float(metric["basket_price"].isna().mean() * 100.0)
    dq["active_components"] = active_components

    return fb, metric, yearly_fuel, owid_th, dq


@st.cache_data(show_spinner=False)
def _osf_narrative_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year": 2021,
                "osf_pressure_index": 62,
                "kind": "narrative",
                "event_th": "กู้ OSF ~20 พันล้านบาท / คุมดีเซล ~30 บ./ล. (ช่วงประท้วงรถบรรทุก)",
                "event_en": "OSF loan / diesel cap ~30 THB/L (trucker protests)",
            },
            {
                "year": 2022,
                "osf_pressure_index": 92,
                "kind": "narrative",
                "event_th": "ลดภาษีสรรพสามิต ~3 บ./ล. / หนี้กองทุนสูง (~129 พันล้านบาท ตามสรุปโครงการ)",
                "event_en": "Excise relief; OSF debt peak (project summary)",
            },
            {
                "year": 2023,
                "osf_pressure_index": 78,
                "kind": "narrative",
                "event_th": "กองทุนติดลบต่อเนื่อง (~78 พันล้านบาท ช่วงขาดทุน)",
                "event_en": "OSF deficit pressure",
            },
            {
                "year": 2024,
                "osf_pressure_index": 88,
                "kind": "narrative",
                "event_th": "ขยายเพดานกู้ 1.05 แสนล้านบาท / คุมดีเซล ~33 บ./ล.",
                "event_en": "Borrowing ceiling expansion; diesel cap ~33 THB/L",
            },
        ]
    )


@st.cache_data(show_spinner=False)
def _th_regions_placeholder() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region_th": ["ภาคกลาง", "ภาคเหนือ", "ภาคตะวันออก", "ภาคอีสาน", "ภาคใต้"],
            "region_en": ["Central", "North", "East", "Northeast", "South"],
            "vulnerability_index_0_100": [48, 55, 52, 62, 58],
            "kind": "placeholder",
        }
    )


@st.cache_data(show_spinner=False)
def build_validation_tables(metric: pd.DataFrame, dq: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    if metric.empty:
        return out

    component_cols = [c for c in dq.get("active_components", []) if c in metric.columns]
    miss_rows = []
    for c in ["basket_price", "mom_change_pct", "vol_6m", "stress_index"] + component_cols:
        if c in metric.columns:
            miss_rows.append(
                {
                    "field": c,
                    "missing_pct": float(metric[c].isna().mean() * 100.0),
                    "non_missing_n": int(metric[c].notna().sum()),
                }
            )
    out["missingness"] = pd.DataFrame(miss_rows)

    if component_cols:
        out["component_corr"] = metric[component_cols].corr(method="spearman").round(3)

    wt = _compute_weight_sensitivity(metric)
    if not wt.empty:
        out["weight_sensitivity"] = wt

    fut = _diagnostic_future_signal(metric, horizon_months=3)
    if not fut.empty:
        out["future_signal"] = fut

    summary = pd.DataFrame(
        [
            {
                "check": "Leak-free normalization",
                "result": "PASS",
                "details": dq.get("normalization_note", ""),
            },
            {
                "check": "Backward fill removed",
                "result": "PASS",
                "details": "Only forward fill is used on price basket components.",
            },
            {
                "check": "Threshold calibration",
                "result": "PARTIAL",
                "details": "Historical quantile threshold is leak-free, but still not outcome-calibrated against labeled events.",
            },
            {
                "check": "Causal interpretation",
                "result": "FAIL",
                "details": "Dashboard is descriptive/exploratory only; no treatment-control identification strategy implemented.",
            },
            {
                "check": "Placeholder separation",
                "result": "PASS",
                "details": "Placeholder and narrative series are explicitly labeled and visually separated.",
            },
        ]
    )
    out["summary"] = summary
    return out


# -----------------------------
# Load data
# -----------------------------
global_gap = load_global_gap_stress()
fuel, metric, yearly_fuel, owid_th, dq = load_thailand_core()
doeb_bundle, doeb_labels = load_doeb_bundle_from_repo()
data_inventory = list_cloned_data_files()
wb_price_path, wb_prices_df, wb_sub_path, wb_sub_df = try_load_world_bank_tables()
osf_df = _osf_narrative_df()
regions_df = _th_regions_placeholder()
validation = build_validation_tables(metric, dq)

dq["doeb_series_loaded"] = list(doeb_bundle.keys())
dq["doeb_total_rows"] = int(sum(len(v) for v in doeb_bundle.values()))

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### Monitor & Guide")
    st.caption("Validated prototype — descriptive monitoring, not causal proof")

    if metric.empty:
        st.error("ไม่พบ Thailand core data")
        st.stop()

    years = sorted(metric["Date"].dt.year.dropna().astype(int).unique().tolist())
    y_min = years[0]
    y_max = years[-1]
    y0, y1 = st.select_slider("ช่วงปี (Thailand charts)", options=years, value=(max(2021, y_min), y_max))

    if not global_gap.empty:
        map_year = st.slider(
            "ปีแผนที่โลก (Global dependency percentile)",
            int(global_gap["year"].min()),
            int(global_gap["year"].max()),
            int(min(max(2019, int(global_gap["year"].min())), int(global_gap["year"].max()))),
        )
    else:
        map_year = 2019

    threshold_q = st.slider("เกณฑ์เตือนแบบ historical quantile", 0.70, 0.95, 0.85, 0.01)

    st.divider()
    st.markdown("**ประเภทข้อมูลใน dashboard**")
    st.markdown(
        _kind_badge("measured")
        + _kind_badge("estimated")
        + _kind_badge("narrative")
        + _kind_badge("placeholder"),
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Statistical guardrails**")
    st.markdown(
        "- ใช้ normalization แบบไม่มองอนาคต\n"
        "- ใช้ threshold จากประวัติย้อนหลัง ไม่ใช่ค่าคงที่\n"
        "- ไม่อ้าง causal effect จากกราฟ proxy"
    )

# -----------------------------
# Derived views
# -----------------------------
view = metric[(metric["Date"].dt.year >= y0) & (metric["Date"].dt.year <= y1)].copy()
if view.empty:
    st.error("ไม่มีข้อมูลในช่วงปีที่เลือก")
    st.stop()

view["alert_threshold_q"] = _historical_quantile_threshold(view["stress_index"], q=threshold_q, min_periods=12)
view["alert_flag_q"] = (
    view["stress_index"].notna()
    & view["alert_threshold_q"].notna()
    & (view["stress_index"] >= view["alert_threshold_q"])
)

latest = view.iloc[-1]
stress = float(latest["stress_index"]) if pd.notna(latest["stress_index"]) else np.nan
delta_st = float(view["stress_index"].diff().iloc[-1]) if len(view) > 1 and pd.notna(view["stress_index"].diff().iloc[-1]) else np.nan
alerts = int(view["alert_flag_q"].sum()) if "alert_flag_q" in view.columns else 0
yearly_v = yearly_fuel[(yearly_fuel["year"] >= y0) & (yearly_fuel["year"] <= y1)].copy()

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<p class="kicker">From Energy Stress to Human Adaptation · Track B · Statistically safer prototype</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero">พลังงาน: จากความเครียดของระบบ สู่การปรับตัวของมนุษย์<br/><span style="font-size:0.72em;font-weight:700;">Energy Stress → Human Adaptation</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="tagline">ปรับปรุงแล้วให้แยก <b>Measured / Estimated / Narrative / Placeholder</b>, '
    "เลิกใช้ backward fill, ใช้ normalization แบบไม่มองข้อมูลอนาคต, และเปลี่ยนจาก fixed threshold เป็น historical quantile threshold</p>",
    unsafe_allow_html=True,
)

tab_a, tab_b, tab_c, tab_d = st.tabs(
    [
        "A — Global dependency monitor",
        "B — Validation & policy explorer",
        "C — Thailand & social protection",
        "D — AI assistant",
    ]
)

# -----------------------------
# A: Global
# -----------------------------
with tab_a:
    st.markdown('<p class="section-label">A · Global energy dependency monitor</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="card">{_kind_badge("measured")}{_kind_badge("estimated")}'
        "<h3>แนวคิดของแผนที่โลก</h3>"
        "<p>ใช้ <b>gap dependency ratio</b> = |oil consumption − oil production| / (consumption + production) "
        "แล้วแปลงเป็น percentile ภายในปีเดียวกัน เพื่อให้เปรียบเทียบข้ามประเทศในปีเดียวกันได้ดีขึ้น "
        "(แต่ยังไม่ควรตีความเป็น welfare stress โดยตรง)</p></div>",
        unsafe_allow_html=True,
    )

    if global_gap.empty:
        st.warning("ไม่พบไฟล์ OWID หลายประเทศ — ใส่ `Track_B_Adaptive_Infrastructures_Datasets/OWID_Energy_Data.csv`")
    else:
        gmap = global_gap[global_gap["year"] == map_year].dropna(subset=["gap_percentile_0_100"])
        fig_map = go.Figure(
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
        fig_map.update_layout(
            title=f"Global oil dependency percentile ({map_year})",
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth", bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=480,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<p class="section-label">Thailand stress monitor — measured + estimated components only</p>', unsafe_allow_html=True)

    fig_ew = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ew.add_trace(
        go.Scatter(
            x=view["Date"],
            y=view["stress_index"],
            name="Stress index",
            line=dict(color="#fb923c", width=3),
            fill="tozeroy",
            fillcolor="rgba(251,146,60,0.18)",
        ),
        secondary_y=False,
    )
    fig_ew.add_trace(
        go.Scatter(
            x=view["Date"],
            y=view["alert_threshold_q"],
            name=f"Alert threshold (q={threshold_q:.2f})",
            line=dict(color="#f87171", width=1.5, dash="dash"),
        ),
        secondary_y=False,
    )
    fig_ew.add_trace(
        go.Scatter(
            x=view["Date"],
            y=view["basket_price"],
            name="Retail basket (THB/L)",
            line=dict(color="#fde047", width=2),
        ),
        secondary_y=True,
    )
    fig_ew.update_layout(
        template="plotly_dark",
        height=410,
        legend=dict(orientation="h", y=1.12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig_ew.update_yaxes(title_text="Stress percentile (0–100)", secondary_y=False)
    fig_ew.update_yaxes(title_text="THB/L", secondary_y=True)
    st.plotly_chart(fig_ew, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        '<div class="sig"><b>ข้อสำคัญ:</b> เส้น Stress นี้เป็น composite monitoring score ไม่ใช่ causal estimate และยังไม่ใช่ early-warning system ที่ผ่านการเทียบกับ labeled events อย่างเป็นทางการ</div>',
        unsafe_allow_html=True,
    )

# -----------------------------
# B: Validation & policy explorer
# -----------------------------
with tab_b:
    st.markdown('<p class="section-label">B · Statistical validation + descriptive policy explorer</p>', unsafe_allow_html=True)

    c1, c2 = st.columns((1.15, 0.85))
    with c1:
        st.markdown(
            f'<div class="card">{_kind_badge("estimated")}<h3>Validation summary</h3></div>',
            unsafe_allow_html=True,
        )
        if "summary" in validation:
            st.dataframe(validation["summary"], use_container_width=True, hide_index=True)

        if "missingness" in validation:
            st.markdown("**Missingness / usable sample**")
            st.dataframe(validation["missingness"], use_container_width=True, hide_index=True)

        if "weight_sensitivity" in validation:
            st.markdown("**Weight sensitivity check**")
            st.caption("ค่าที่ดีคือ correlation กับ baseline สูง และ mean_abs_diff ไม่สูงมาก")
            st.dataframe(validation["weight_sensitivity"], use_container_width=True, hide_index=True)

    with c2:
        st.markdown(
            f'<div class="card">{_kind_badge("estimated")}<h3>Component correlation</h3></div>',
            unsafe_allow_html=True,
        )
        if "component_corr" in validation and not validation["component_corr"].empty:
            corr = validation["component_corr"]
            heat = go.Figure(
                data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1,
                    text=corr.round(2).astype(str).values,
                    texttemplate="%{text}",
                )
            )
            heat.update_layout(
                template="plotly_dark",
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(heat, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("ยังไม่มี component เพียงพอสำหรับ correlation matrix")

        if "future_signal" in validation and not validation["future_signal"].empty:
            st.markdown("**Face-validity diagnostic**")
            st.caption("เปรียบเทียบค่าเฉลี่ยการเปลี่ยนแปลงราคาใน 3 เดือนข้างหน้า ระหว่างเดือนที่ stress สูงมากกับต่ำมาก")
            st.dataframe(validation["future_signal"], use_container_width=True, hide_index=True)

    st.markdown('<p class="section-label">Descriptive policy explorer (not causal)</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="card">{_kind_badge("estimated")}<h3>Trade-off explorer</h3>'
        "<p>กราฟนี้เป็น descriptive only: แสดงความสัมพันธ์ระหว่างระดับราคาตะกร้าน้ำมันเฉลี่ยรายปี กับสัดส่วนเดือนที่เกิน threshold "
        "ไม่ควรตีความเป็นผลของนโยบายโดยตรง</p></div>",
        unsafe_allow_html=True,
    )

    if not yearly_v.empty:
        fig_trade = go.Figure()
        fig_trade.add_trace(
            go.Scatter(
                x=yearly_v["year_avg_basket_thb"],
                y=yearly_v["year_alert_share"],
                mode="markers+text",
                text=yearly_v["year"].astype(str),
                textposition="top center",
                marker=dict(
                    size=np.clip(yearly_v["year_avg_stress"].fillna(0.0), 8, 22),
                    color=yearly_v["year_avg_stress"],
                    colorscale="YlOrRd",
                    showscale=True,
                    colorbar=dict(title="Avg stress"),
                ),
                name="Year",
            )
        )
        fig_trade.update_layout(
            template="plotly_dark",
            title="Average retail basket vs share of alert months",
            xaxis_title="Average retail basket price (THB/L)",
            yaxis_title="Share of alert months (%)",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_trade, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("ยังไม่มี yearly summary เพียงพอสำหรับ policy explorer")

    st.markdown('<p class="section-label">World Bank source preview</p>', unsafe_allow_html=True)
    if wb_sub_df is not None and wb_sub_path:
        st.success(f"โหลด World Bank subsidy database แล้ว: `{os.path.basename(wb_sub_path)}`")
        st.dataframe(wb_sub_df.head(20), use_container_width=True, hide_index=True)
    elif wb_prices_df is not None and wb_price_path:
        st.info(f"พบ World Bank fuel price file: `{os.path.basename(wb_price_path)}`")
        st.dataframe(wb_prices_df.head(15), use_container_width=True, hide_index=True)
    else:
        st.warning("ยังไม่พบไฟล์ World Bank .xlsx ใน repo clone")

# -----------------------------
# C: Thailand
# -----------------------------
with tab_c:
    st.markdown('<p class="section-label">C · Thailand focus & social protection</p>', unsafe_allow_html=True)

    c1, c2 = st.columns((1, 1))
    with c1:
        st.markdown(
            f'<div class="card">{_kind_badge("measured")}{_kind_badge("narrative")}'
            "<h3>OSF narrative vs measured market pressure</h3></div>",
            unsafe_allow_html=True,
        )

        if not yearly_fuel.empty:
            yf = yearly_fuel.merge(osf_df, on="year", how="left")
            yf = yf[(yf["year"] >= y0) & (yf["year"] <= y1)]

            fig_osf = make_subplots(specs=[[{"secondary_y": True}]])
            fig_osf.add_trace(
                go.Bar(
                    x=yf["year"],
                    y=yf["year_avg_stress"],
                    name="Measured avg stress",
                    marker_color="#f97316",
                ),
                secondary_y=False,
            )
            if "osf_pressure_index" in yf.columns:
                fig_osf.add_trace(
                    go.Scatter(
                        x=yf["year"],
                        y=yf["osf_pressure_index"],
                        name="Narrative OSF pressure index",
                        line=dict(color="#fde047", dash="dash"),
                    ),
                    secondary_y=True,
                )
            fig_osf.update_layout(
                template="plotly_dark",
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=1.08),
            )
            fig_osf.update_yaxes(title_text="Measured stress", secondary_y=False)
            fig_osf.update_yaxes(title_text="Narrative index", secondary_y=True)
            st.plotly_chart(fig_osf, use_container_width=True, config={"displayModeBar": False})

            show_cols = [c for c in ["year", "year_avg_stress", "year_avg_basket_thb", "osf_pressure_index", "event_th"] if c in yf.columns]
            st.dataframe(yf[show_cols], use_container_width=True, hide_index=True)
        else:
            st.warning("ไม่มี yearly Thailand data")

    with c2:
        st.markdown(
            f'<div class="card">{_kind_badge("measured")}<h3>DOEB infrastructure series</h3>'
            "<p>เพราะแต่ละ series มีคนละหน่วย จึงแสดงเป็น <b>indexed series (base = 100 at first available year)</b> "
            "เพื่อไม่ให้แกนเดียวหลอกสายตา</p></div>",
            unsafe_allow_html=True,
        )

        if doeb_bundle:
            pick = st.multiselect(
                "เลือกชุดข้อมูล DOEB",
                options=list(doeb_bundle.keys()),
                format_func=lambda k: doeb_labels.get(k, k),
                default=list(doeb_bundle.keys())[: min(3, len(doeb_bundle))],
            )

            fig_m = go.Figure()
            colors = ["#22d3ee", "#a78bfa", "#4ade80", "#fb7185"]

            for i, key in enumerate(pick):
                d = doeb_bundle.get(key, pd.DataFrame()).copy()
                d = d[(d["year_ce"] >= y0) & (d["year_ce"] <= y1)]
                if d.empty:
                    continue
                agg = d.groupby("year_ce", as_index=False)["value"].sum().sort_values("year_ce")
                if agg.empty:
                    continue
                base = agg["value"].iloc[0]
                if pd.isna(base) or base == 0:
                    agg["index_100"] = np.nan
                else:
                    agg["index_100"] = agg["value"] / base * 100.0
                fig_m.add_trace(
                    go.Scatter(
                        x=agg["year_ce"],
                        y=agg["index_100"],
                        mode="lines+markers",
                        name=(doeb_labels.get(key, key))[:52],
                        line=dict(width=2, color=colors[i % len(colors)]),
                    )
                )

            fig_m.update_layout(
                template="plotly_dark",
                height=340,
                title="DOEB annual index (base year = 100)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.05),
                yaxis_title="Index (base = 100)",
            )
            st.plotly_chart(fig_m, use_container_width=True, config={"displayModeBar": False})
            st.caption(f"โหลด {len(doeb_bundle)} ชุดข้อมูลจริงจาก repo · รวม {dq['doeb_total_rows']:,} แถว")
        else:
            st.warning("ไม่พบไฟล์ DOEB CSV ตาม path ใน repo")

    st.markdown(
        f'<div class="card">{_kind_badge("placeholder")}<h3>Household vulnerability (placeholder only)</h3>'
        "<p>ส่วนนี้ยังเป็น mock-up จนกว่าจะใส่ไฟล์ NESDC ของจริงเข้ามา</p></div>",
        unsafe_allow_html=True,
    )
    fig_r = go.Figure(
        go.Bar(
            x=regions_df["vulnerability_index_0_100"],
            y=regions_df["region_th"],
            orientation="h",
            marker=dict(color=regions_df["vulnerability_index_0_100"], colorscale=[[0, "#1e3a5f"], [1, "#ea580c"]]),
        )
    )
    fig_r.update_layout(
        template="plotly_dark",
        height=280,
        xaxis_title="Vulnerability index (placeholder)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        f'<div class="sig"><b>Monitor:</b> Stress ล่าสุด = <b>{stress:.1f}</b> · '
        f'เดือนที่เกิน threshold = <b>{alerts}</b> · threshold ใช้ historical quantile = <b>{threshold_q:.2f}</b></div>',
        unsafe_allow_html=True,
    )

# -----------------------------
# D: Assistant
# -----------------------------
with tab_d:
    st.markdown('<p class="section-label">D · AI personal assistant</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="card">{_kind_badge("estimated")}<h3>Rule-guided assistant</h3>'
        "<p>ตัวนี้ยังเป็น heuristic assistant ไม่ได้ใช้ model inference จริง และจะตอบโดยอิงระดับ stress + keywords แบบง่าย</p></div>",
        unsafe_allow_html=True,
    )

    q = st.text_area("ถามผู้ช่วย (ภาษาไทยหรืออังกฤษ)", "ช่วงนี้ควรเน้นความสามารถในการจ่ายหรือเสถียรภาพระบบ?", height=78)

    if st.button("รับคำแนะนำ", type="primary"):
        q_low = q.lower().strip()
        answer_bits = []

        if pd.notna(stress):
            if pd.notna(latest["alert_threshold_q"]) and stress >= float(latest["alert_threshold_q"]):
                answer_bits.append("ระดับ stress ตอนนี้สูงกว่าระดับเตือนจากประวัติย้อนหลัง จึงควรให้น้ำหนักกับการคุ้มครองระยะสั้นสำหรับกลุ่มเปราะบางก่อน")
            else:
                answer_bits.append("ระดับ stress ตอนนี้ยังไม่เกินระดับเตือนจากประวัติย้อนหลัง จึงสามารถเน้นการเสริม resilience ระยะกลางและการสื่อสารเชิงนโยบายได้มากขึ้น")

        if any(k in q_low for k in ["afford", "จ่าย", "affordability", "ค่าใช้จ่าย"]):
            answer_bits.append("หากโจทย์หลักคือ affordability ควรดูทั้งราคาเฉลี่ย ระดับการเปลี่ยนแปลงรายเดือน และกลุ่มเชื้อเพลิงที่กระทบครัวเรือนจริงมากที่สุด")
        if any(k in q_low for k in ["stability", "เสถียร", "system"]):
            answer_bits.append("หากโจทย์หลักคือ system stability ควรติดตาม volatility, import dependence และสัดส่วนเดือนที่เกิน threshold พร้อมกัน")
        if any(k in q_low for k in ["policy", "นโยบาย", "subsidy", "กองทุน"]):
            answer_bits.append("กราฟใน dashboard ตอนนี้ใช้ได้แค่ descriptive exploration ยังไม่เพียงพอที่จะสรุปผลของนโยบายแบบ causal")

        if not answer_bits:
            answer_bits.append("dashboard นี้เหมาะกับการ monitor และตั้งคำถามต่อข้อมูล แต่ยังไม่ใช่เครื่องมือพิสูจน์เชิงสาเหตุ")

        st.success(" ".join(answer_bits))

# -----------------------------
# KPI strip + details
# -----------------------------
st.divider()
k1, k2, k3, k4 = st.columns(4)
_value_box("Stress (ล่าสุด)", "NA" if pd.isna(stress) else f"{stress:.1f}", None if pd.isna(delta_st) else f"{delta_st:+.1f}")
_value_box("เดือนเตือน", str(alerts), f"q = {threshold_q:.2f}")
_value_box("แถวราคาที่ใช้", f"{dq.get('fuel_rows_clean', 0):,}", f"DOEB {len(doeb_bundle)} ชุด")
if dq.get("date_min") and dq.get("date_max"):
    _value_box("ช่วงวันที่", f"{dq['date_min'][:4]}–{dq['date_max'][:4]}")
else:
    _value_box("ช่วงวันที่", "NA")

with st.expander("Method notes / quality notes"):
    st.markdown("### Basket weights used")
    st.json(dq.get("basket_weights_used", {}))

    st.markdown("### Component weights used")
    st.json(dq.get("component_weights_used", {}))

    st.markdown("### Data quality notes")
    st.write(dq)

with st.expander("รายการไฟล์ข้อมูลใน `Track_B_Adaptive_Infrastructures_Datasets/`"):
    if data_inventory.empty:
        st.write("ไม่พบโฟลเดอร์ Track_B")
    else:
        st.dataframe(data_inventory, use_container_width=True, hide_index=True)

st.markdown('<div class="chat-fab" title="Assistant">🤖</div>', unsafe_allow_html=True)