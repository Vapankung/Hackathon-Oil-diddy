"""
From Energy Stress to Human Adaptation — Track B (Money Resilience Lab) dashboard.
Theme: Monitor & Guide | Dark mode + yellow/orange signals.
Sections A–D per structured brief; data: Thailand fuel basket, OWID (TH + global file), DOEB sample, OSF narrative points.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


st.set_page_config(
    page_title="Energy Stress → Human Adaptation | Track B",
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
    max-width: 1400px;
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
.card h3 { color: #fde68a; font-size: 15px; margin: 0 0 8px 0; }
.sig {
    border-left: 4px solid #f59e0b;
    background: rgba(245, 158, 11, 0.12);
    color: #fef3c7;
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 13px;
    margin-top: 8px;
}
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

# DOEB CSV paths (relative to Track_B) — all from the cloned GitHub repo
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


def _z(s: pd.Series) -> pd.Series:
    std = float(s.std())
    if np.isnan(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _minmax100(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) * 100.0 / (hi - lo)


def _track_b_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), TRACK_B))


def _read_doeb_csv(rel_path: str) -> pd.DataFrame:
    """Parse standard DOEB open-data CSV (YEAR_ID = Buddhist Era monthlies)."""
    path = os.path.normpath(os.path.join(_track_b_root(), *rel_path.split("/")))
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path, encoding="utf-8")
    except Exception:
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
    return out[["Date", "year_ce", "month", "value", "unit", "subject"]]


@st.cache_data
def load_doeb_bundle_from_repo() -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Load every DOEB series shipped in the cloned repo."""
    bundle: Dict[str, pd.DataFrame] = {}
    labels: Dict[str, str] = {}
    for rel, key, label_th in _DOEB_SERIES:
        df = _read_doeb_csv(rel)
        if not df.empty:
            bundle[key] = df
            labels[key] = label_th
    return bundle, labels


@st.cache_data
def list_cloned_data_files() -> pd.DataFrame:
    """Inventory CSV/XLSX/IPYNB under Track_B (cloned folder)."""
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
                rows.append({"relative_path": os.path.relpath(fp, _track_b_root()), "size_kb": round(sz, 1)})
    return pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)


@st.cache_data
def try_load_world_bank_tables() -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[str], Optional[pd.DataFrame]]:
    """
    If `Global_Fuel_*.xlsx` exists under Track_B (per README/manifest), load first sheet preview.
    Repo clone may omit xlsx — then returns None.
    """
    tb = _track_b_root()
    targets = [
        "Global_Fuel_Prices_Database.xlsx",
        "Global_Fuel_Subsidies_and_Price_Control_Measures_Database.xlsx",
    ]
    found_prices = found_subsidy = None
    df_p = df_s = None
    if not os.path.isdir(tb):
        return None, None, None, None
    for dirpath, _, filenames in os.walk(tb):
        for fn in filenames:
            if fn == targets[0]:
                found_prices = os.path.join(dirpath, fn)
            elif fn == targets[1]:
                found_subsidy = os.path.join(dirpath, fn)
    for path, slot in [(found_prices, "p"), (found_subsidy, "s")]:
        if path and os.path.isfile(path):
            try:
                prev = pd.read_excel(path, sheet_name=0, engine="openpyxl", nrows=4000)
            except Exception:
                try:
                    prev = pd.read_excel(path, sheet_name=0, nrows=4000)
                except Exception:
                    prev = None
            if slot == "p":
                df_p = prev
            else:
                df_s = prev
    return found_prices, df_p, found_subsidy, df_s


@st.cache_data
def load_global_gap_stress() -> pd.DataFrame:
    """Country-year gap stress: |oil consumption − oil production| (TWh), min–max within each year for map."""
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, _GLOBAL_OWID)
    if not os.path.isfile(path):
        return pd.DataFrame()
    use = ["country", "iso_code", "year", "oil_consumption", "oil_production"]
    g = pd.read_csv(path, usecols=use, low_memory=False)
    g = g[g["iso_code"].notna()].copy()
    g["iso_code"] = g["iso_code"].astype(str).str.upper().str.strip()
    g = g[g["iso_code"].str.len() == 3]
    g["oil_consumption"] = pd.to_numeric(g["oil_consumption"], errors="coerce")
    g["oil_production"] = pd.to_numeric(g["oil_production"], errors="coerce")
    g["gap_twh"] = (g["oil_consumption"].fillna(0) - g["oil_production"].fillna(0)).abs()

    def _mm_year(s: pd.Series) -> pd.Series:
        lo, hi = float(s.min()), float(s.max())
        if hi <= lo:
            return pd.Series(0.0, index=s.index)
        return (s - lo) * 100.0 / (hi - lo)

    g["gap_stress_0_100"] = g.groupby("year")["gap_twh"].transform(_mm_year)
    return g


@st.cache_data
def load_thailand_core() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    base = os.path.dirname(os.path.abspath(__file__))
    fuel_path = os.path.join(base, "thailand_fuel_prices_cleaned.csv")
    owid_th_path = os.path.join(base, "owid-energy-data(clean).csv")

    fuel = pd.read_csv(fuel_path)
    n_raw = len(fuel)
    fuel["Date"] = pd.to_datetime(fuel["Date"], errors="coerce")
    fuel["Price"] = pd.to_numeric(fuel["Price"], errors="coerce")
    fuel = fuel.dropna(subset=["Date", "Price"])
    fuel = fuel[fuel["Description"].str.contains("Thailand", case=False, na=False)]
    fuel = fuel[fuel["Unit"].eq("LCU (Local Currency Unit)")]
    fuel = fuel[fuel["Description"].str.contains("Retail Price", case=False, na=False)].copy()
    fuel["Year"] = fuel["Date"].dt.year
    fuel["Month"] = fuel["Date"].dt.to_period("M")

    basket = {
        "Thailand (Diesel HSD B7) - Retail Price": 0.35,
        "Thailand (Gasohol 95-E10) - Table 9 Retail Price": 0.30,
        "Thailand (LPG low income hh) - Retail Price": 0.25,
        "Thailand (Kerosene) - Retail Price": 0.10,
    }
    keep = [k for k in basket if k in set(fuel["Description"])]
    fb = fuel[fuel["Description"].isin(keep)].copy()
    parts = []
    for _, grp in fb.groupby("Description"):
        q1, q3 = grp["Price"].quantile(0.25), grp["Price"].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        parts.append(grp[(grp["Price"] >= lo) & (grp["Price"] <= hi)])
    fb = pd.concat(parts, ignore_index=True).sort_values("Date")

    monthly = fb.groupby(["Month", "Description"], as_index=False)["Price"].mean()
    monthly["Date"] = monthly["Month"].dt.to_timestamp()
    pivot = monthly.pivot(index="Date", columns="Description", values="Price").sort_index()
    for d in keep:
        if d not in pivot.columns:
            pivot[d] = np.nan
    pivot = pivot.ffill().bfill()
    pivot["basket_price"] = sum(pivot[d] * basket[d] for d in keep)
    pivot["basket_index"] = pivot["basket_price"] / pivot["basket_price"].iloc[0] * 100.0
    pivot["mom_change_pct"] = pivot["basket_price"].pct_change() * 100.0
    pivot["vol_6m"] = pivot["mom_change_pct"].rolling(6, min_periods=3).std()

    owid_th = pd.read_csv(owid_th_path)
    owid_th = owid_th[owid_th["country"].eq("Thailand")].sort_values("year")
    oil = owid_th[["year", "oil_consumption"]].dropna().drop_duplicates("year").sort_values("year")
    oil["oil_yoy_pct"] = oil["oil_consumption"].pct_change() * 100.0

    metric = pd.DataFrame({"Date": pivot.index})
    metric["year"] = metric["Date"].dt.year
    metric = metric.merge(
        pivot[["basket_price", "basket_index", "mom_change_pct", "vol_6m"]],
        left_on="Date",
        right_index=True,
        how="left",
    )
    metric = metric.merge(oil[["year", "oil_yoy_pct"]], on="year", how="left")
    metric["oil_yoy_pct"] = metric["oil_yoy_pct"].fillna(0.0)
    raw = (
        0.50 * _z(metric["mom_change_pct"].fillna(0)).abs()
        + 0.30 * _z(metric["vol_6m"].fillna(0)).abs()
        + 0.20 * _z(metric["oil_yoy_pct"].fillna(0)).abs()
    )
    metric["stress_index"] = _minmax100(raw)

    yrows = []
    for y, g in metric.groupby("year"):
        mc = g["mom_change_pct"].dropna()
        yrows.append(
            {
                "year": y,
                "year_avg_basket_thb": float(g["basket_price"].mean()),
                "year_price_volatility": float(mc.std()) if len(mc) > 1 else 0.0,
            }
        )
    yearly_fuel = pd.DataFrame(yrows).sort_values("year")

    dq: Dict[str, Any] = {
        "fuel_rows_raw": n_raw,
        "fuel_rows_clean": len(fb),
        "basket_series": keep,
        "date_min": str(metric["Date"].min().date()),
        "date_max": str(metric["Date"].max().date()),
    }
    return fb, metric, yearly_fuel, owid_th, dq


def _osf_narrative_df() -> pd.DataFrame:
    """Policy narrative 2021–2024 (illustrative magnitudes — verify against official sources)."""
    return pd.DataFrame(
        [
            {"year": 2021, "osf_pressure_index": 62, "event_th": "กู้ OSF ~20 พันล้านบาท / คุมดีเซล ~30 บ./ล. (ช่วงประท้วงรถบรรทุก)", "event_en": "OSF loan / diesel cap ~30 THB/L (trucker protests)"},
            {"year": 2022, "osf_pressure_index": 92, "event_th": "ลดภาษีสรรพสามิต ~3 บ./ล. / หนี้กองทุนสูง (~129 พันล้านบาท ตามสรุปโครงการ)", "event_en": "Excise relief; OSF debt peak (project summary)"},
            {"year": 2023, "osf_pressure_index": 78, "event_th": "กองทุนติดลบต่อเนื่อง (~78 พันล้านบาท ช่วงขาดทุน)", "event_en": "OSF deficit pressure"},
            {"year": 2024, "osf_pressure_index": 88, "event_th": "ขยายเพดานกู้ 1.05 แสนล้านบาท / คุมดีเซล ~33 บ./ล.", "event_en": "Borrowing ceiling expansion; diesel cap ~33 THB/L"},
        ]
    )


def _th_regions_placeholder() -> pd.DataFrame:
    """Placeholder until NESDC Excel is wired — relative vulnerability index by macro-region."""
    return pd.DataFrame(
        {
            "region_th": ["ภาคกลาง", "ภาคเหนือ", "ภาคตะวันออก", "ภาคอีสาน", "ภาคใต้"],
            "region_en": ["Central", "North", "East", "Northeast", "South"],
            "vulnerability_index_0_100": [48, 55, 52, 62, 58],
        }
    )


global_gap = load_global_gap_stress()
fuel, metric, yearly_fuel, owid_th, dq = load_thailand_core()
doeb_bundle, doeb_labels = load_doeb_bundle_from_repo()
data_inventory = list_cloned_data_files()
wb_price_path, wb_prices_df, wb_sub_path, wb_sub_df = try_load_world_bank_tables()
dq["doeb_series_loaded"] = list(doeb_bundle.keys())
dq["doeb_total_rows"] = int(sum(len(v) for v in doeb_bundle.values()))
osf_df = _osf_narrative_df()
regions_df = _th_regions_placeholder()

with st.sidebar:
    st.markdown("### Monitor & Guide")
    st.caption("Track B — Causal inference · ABM-style signals · Optimization")
    years = sorted(metric["Date"].dt.year.unique().tolist())
    y0, y1 = st.select_slider(
        "ช่วงปี (Thailand charts)",
        options=years,
        value=(max(2021, min(years)), max(years)),
    )
    map_year = st.slider("ปีแผนที่โลก (Global gap stress)", int(global_gap["year"].min()) if len(global_gap) else 2000, int(global_gap["year"].max()) if len(global_gap) else 2024, 2019)
    alert_th = st.slider("เกณฑ์เตือน (Stress)", 55, 90, 70)
    st.divider()
    st.markdown("**ทีม / Alignment**")
    st.markdown(
        "- **Seniors:** causal inference + statistical validity (หน้า A–B)\n"
        "- **Fin-Eng:** optimization & trade-off math (หน้า B)\n"
        "- **คุณ (Lead):** storytelling + human adaptation + NESDC soul (หน้า C)"
    )

view = metric[(metric["Date"].dt.year >= y0) & (metric["Date"].dt.year <= y1)].copy()
if view.empty:
    st.error("ไม่มีข้อมูลในช่วงปีที่เลือก")
    st.stop()

latest = view.iloc[-1]
stress = float(latest["stress_index"])
delta_st = float(view["stress_index"].diff().iloc[-1]) if len(view) > 1 else 0.0
alerts = int((view["stress_index"] >= alert_th).sum())
yearly_v = yearly_fuel[(yearly_fuel["year"] >= y0) & (yearly_fuel["year"] <= y1)]

st.markdown('<p class="kicker">From Energy Stress to Human Adaptation · Track B · Money Resilience Lab</p>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero">พลังงาน: จากความเครียดของระบบ สู่การปรับตัวของมนุษย์<br/><span style="font-size:0.72em;font-weight:700;">Energy Stress → Human Adaptation</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="tagline">ธีม <b>Monitor & Guide</b> · ข้อมูลจากโฟลเดอร์ที่ clone: '
    "<code>thailand_fuel_prices_cleaned.csv</code>, <code>owid-energy-data(clean).csv</code>, "
    f"<code>{TRACK_B}/</code> (OWID global + DOEB CSV) · OSF narrative · NESDC: มีเฉพาะ <code>CostToGDP.ipynb</code> ใน repo</p>",
    unsafe_allow_html=True,
)

tab_a, tab_b, tab_c, tab_d = st.tabs(
    [
        "A — Global stress monitor",
        "B — Policy & optimization",
        "C — Thailand & social protection",
        "D — AI assistant",
    ]
)

# --- A: Global ---
with tab_a:
    st.markdown('<p class="section-label">A · Global Energy Stress Monitor (OWID + World Bank context)</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card"><h3>แนวคิด</h3><p style="color:#cbd5e1;font-size:13px;line-height:1.55;">'
        "ดัชนี Gap = |การบริโภคน้ำมัน − การผลิตน้ำมัน| (TWh) ต่อประเทศ-ปี จาก OWID แบบหลายประเทศ; "
        "จัดสเกล 0–100 ภายในปีเดียวกันเพื่อเปรียบเทียบบนแผนที่ได้</p></div>",
        unsafe_allow_html=True,
    )

    if global_gap.empty:
        st.warning("ไม่พบไฟล์ OWID หลายประเทศ — ใส่ `Track_B_Adaptive_Infrastructures_Datasets/OWID_Energy_Data.csv`")
    else:
        gmap = global_gap[global_gap["year"] == map_year].dropna(subset=["gap_stress_0_100"])
        fig_map = go.Figure(
            data=go.Choropleth(
                locations=gmap["iso_code"],
                z=gmap["gap_stress_0_100"],
                locationmode="ISO-3",
                colorscale=[[0, "#0f172a"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                colorbar_title="Gap stress",
                text=gmap["country"],
                hovertemplate="<b>%{text}</b><br>Gap stress: %{z:.1f}<extra></extra>",
            )
        )
        fig_map.update_layout(
            title=f"Global energy gap stress (normalized within {map_year})",
            geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth", bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=480,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_map, width="stretch", config={"displayModeBar": False})

    st.markdown('<p class="section-label">Early warning — Thailand: Stress spike vs retail basket (proxy for price pressure)</p>', unsafe_allow_html=True)
    fig_ew = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ew.add_trace(
        go.Scatter(x=view["Date"], y=view["stress_index"], name="Stress index", line=dict(color="#fb923c", width=3), fill="tozeroy", fillcolor="rgba(251,146,60,0.2)"),
        secondary_y=False,
    )
    fig_ew.add_trace(
        go.Scatter(x=view["Date"], y=view["basket_price"], name="Retail basket (THB/L)", line=dict(color="#fde047", width=2)),
        secondary_y=True,
    )
    # Social disruption markers (case study)
    for x0, x1, label in [
        ("2021-01-01", "2022-12-31", "Trucker protests (2021–2022)"),
        ("2023-10-01", "2023-12-31", "Retailer stress / hoarding window (late 2023)"),
    ]:
        fig_ew.add_vrect(x0=x0, x1=x1, fillcolor="rgba(239,68,68,0.12)", line_width=0, layer="below")
    fig_ew.add_annotation(x="2021-08-01", y=1, yref="paper", text="▲ " + "Trucker protests", showarrow=False, font=dict(color="#fecaca", size=11), yshift=120)
    fig_ew.update_layout(template="plotly_dark", height=400, legend=dict(orientation="h", y=1.12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig_ew.update_yaxes(title_text="Stress (0–100)", secondary_y=False)
    fig_ew.update_yaxes(title_text="THB/L", secondary_y=True)
    st.plotly_chart(fig_ew, width="stretch", config={"displayModeBar": False})
    st.markdown(
        '<div class="sig"><b>สัญญาณเตือน:</b> ถ้าเส้น Stress ขยับแรงก่อนระดับราคาปลีกจะพุ่งชัด — ใช้เป็น early signal ตาม New Assumption '
        "(วิกฤตเริ่มจากความไม่สมดุลที่ซ่อน)</div>",
        unsafe_allow_html=True,
    )

# --- B: Policy ---
with tab_b:
    st.markdown('<p class="section-label">B · Policy impact & optimization (World Bank subsidy + Fin-Eng)</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card"><h3>Causal layer — World Bank (จากโฟลเดอร์ clone ถ้ามีไฟล์ .xlsx)</h3>'
        "<p style='color:#cbd5e1;font-size:13px;'>ตาม <code>source_manifest.csv</code> ควรมี "
        "<code>Global_Fuel_Subsidies_and_Price_Control_Measures_Database.xlsx</code> — "
        "ถ้ายังไม่อัปโหลดใน repo จะแสดง proxy ด้านล่าง</p></div>",
        unsafe_allow_html=True,
    )
    if wb_sub_df is not None and wb_sub_path:
        st.success(f"โหลดนโยบาย World Bank แล้ว: `{os.path.basename(wb_sub_path)}`")
        st.dataframe(wb_sub_df.head(25), width="stretch", hide_index=True)
    elif wb_prices_df is not None and wb_price_path:
        st.info(f"พบราคา World Bank แต่ยังไม่พบ subsidy DB — `{os.path.basename(wb_price_path)}`")
        st.dataframe(wb_prices_df.head(15), width="stretch", hide_index=True)
    else:
        st.warning(
            "ยังไม่พบไฟล์ .xlsx ของ World Bank ในโฟลเดอร์ที่ clone (มีเฉพาะ manifest) — "
            "วาง `Global_Fuel_Subsidies_and_Price_Control_Measures_Database.xlsx` ใต้ `Track_B_Adaptive_Infrastructures_Datasets/` แล้วรีเฟรช"
        )
    ann = view.groupby("year").agg(mean_stress=("stress_index", "mean")).reset_index()
    ann = ann.merge(yearly_fuel[["year", "year_avg_basket_thb"]], on="year", how="inner")
    ann = ann[(ann["year"] >= y0) & (ann["year"] <= y1)]
    ann["subsidy_proxy"] = np.clip(100 - _minmax100(ann["year_avg_basket_thb"]), 0, 100)
    ann["stability_proxy"] = 100 - ann["mean_stress"]

    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=ann["subsidy_proxy"], y=ann["stability_proxy"], mode="markers+text", text=ann["year"].astype(str), textposition="top center", marker=dict(size=14, color="#38bdf8"), name="Year"))
    fig_c.add_shape(type="rect", x0=40, x1=70, y0=40, y1=75, line=dict(color="#fbbf24", width=2, dash="dash"), fillcolor="rgba(251,191,36,0.08)", layer="below")
    fig_c.add_annotation(x=55, y=57, text="Optimal policy region<br>(illustrative)", showarrow=False, font=dict(color="#fde68a", size=11))
    fig_c.update_layout(
        template="plotly_dark",
        title="Trade-off proxy: subsidy pressure (inverse price) vs system stability (inverse stress)",
        xaxis_title="Subsidy / affordability proxy (↑ = more support)",
        yaxis_title="Stability proxy (↑ = calmer system)",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_c, width="stretch", config={"displayModeBar": False})
    st.caption("Fin-Eng: replace proxies with fiscal cost vs household affordability from your optimization model.")

# --- C: Thailand ---
with tab_c:
    st.markdown('<p class="section-label">C · Thailand focus & social protection (NESDC + DOEB + insights)</p>', unsafe_allow_html=True)

    c1, c2 = st.columns((1, 1))
    with c1:
        st.markdown('<div class="card"><h3>Financial stress — OSF narrative vs market pressure (proxy)</h3>', unsafe_allow_html=True)
        yf = yearly_fuel.merge(osf_df, on="year", how="inner")
        yf = yf[(yf["year"] >= 2021) & (yf["year"] <= 2025)]
        fig_osf = make_subplots(specs=[[{"secondary_y": True}]])
        fig_osf.add_trace(go.Bar(x=yf["year"], y=yf["osf_pressure_index"], name="OSF pressure (index)", marker_color="#f97316"), secondary_y=False)
        fig_osf.add_trace(go.Scatter(x=yf["year"], y=yf["year_avg_basket_thb"], name="Avg retail basket THB/L", line=dict(color="#fde047")), secondary_y=True)
        fig_osf.update_layout(template="plotly_dark", height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.08))
        fig_osf.update_yaxes(title_text="OSF narrative index", secondary_y=False)
        fig_osf.update_yaxes(title_text="THB/L", secondary_y=True)
        st.plotly_chart(fig_osf, width="stretch", config={"displayModeBar": False})
        st.dataframe(yf[["year", "osf_pressure_index", "year_avg_basket_thb", "event_th"]], width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><h3>Infrastructure — DOEB (ไฟล์จริงจาก repo)</h3>', unsafe_allow_html=True)
        st.markdown(
            "<ul style='color:#cbd5e1;font-size:13px;line-height:1.6;'>"
            "<li>โรงกลั่นไทย ~6 แห่ง (~1.24M bpd ความจุรวม — ตรวจกับแหล่งทางการ)</li>"
            "<li>นำเข้าน้ำมันดิบ / ส่งออกผลิตภัณฑ์สำเร็จรูป → ความเปราะบางต่อราคาน้ำมันโลก</li></ul>",
            unsafe_allow_html=True,
        )
        if doeb_bundle:
            pick = st.multiselect(
                "เลือกชุดข้อมูล DOEB (รายเดือน → สรุปผลรวมรายปี ค.ศ.)",
                options=list(doeb_bundle.keys()),
                format_func=lambda k: doeb_labels.get(k, k),
                default=list(doeb_bundle.keys())[: min(3, len(doeb_bundle))],
            )
            fig_m = go.Figure()
            colors = ["#22d3ee", "#a78bfa", "#4ade80", "#fb7185"]
            for i, key in enumerate(pick):
                d = doeb_bundle[key]
                d = d[(d["year_ce"] >= y0) & (d["year_ce"] <= y1)]
                if d.empty:
                    continue
                agg = d.groupby("year_ce", as_index=False)["value"].sum().sort_values("year_ce")
                fig_m.add_trace(
                    go.Scatter(
                        x=agg["year_ce"],
                        y=agg["value"],
                        mode="lines+markers",
                        name=(doeb_labels.get(key, key))[:46],
                        line=dict(width=2, color=colors[i % len(colors)]),
                    )
                )
            fig_m.update_layout(
                template="plotly_dark",
                height=320,
                title="ผลรวมรายปีจากข้อมูลรายเดือน DOEB (หน่วยตามคอลัมน์ต้นทาง)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.05),
            )
            st.plotly_chart(fig_m, width="stretch", config={"displayModeBar": False})
            st.caption(f"โหลด {len(doeb_bundle)} ชุดจาก `{TRACK_B}/DOEB dataset/` · รวม {dq['doeb_total_rows']:,} แถว")
        else:
            st.warning("ไม่พบไฟล์ DOEB CSV ตาม path ใน repo")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>Household vulnerability (NESDC)</h3>', unsafe_allow_html=True)
    st.warning(
        "ในโฟลเดอร์ที่ clone มีเฉพาะ `NESDC/CostToGDP.ipynb` — ยังไม่มีไฟล์ Excel สถิติความยากจน/รายได้ใน repo "
        "(เพิ่มไฟล์ตามชื่อใน brief แล้วจะโหลดแทนตัวอย่างได้)"
    )
    fig_r = go.Figure(go.Bar(x=regions_df["vulnerability_index_0_100"], y=regions_df["region_th"], orientation="h", marker=dict(color=regions_df["vulnerability_index_0_100"], colorscale=[[0, "#1e3a5f"], [1, "#ea580c"]])))
    fig_r.update_layout(template="plotly_dark", height=280, xaxis_title="Vulnerability index (placeholder)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_r, width="stretch", config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="sig"><b>Monitor:</b> Stress ล่าสุด = <b>{stress:.0f}</b> (Δ {delta_st:+.1f}) · '
        f"เดือนที่แจ้งเตือน = <b>{alerts}</b> (เกณฑ์ {alert_th})</div>",
        unsafe_allow_html=True,
    )

# --- D: AI ---
with tab_d:
    st.markdown('<p class="section-label">D · AI personal assistant (Interactive Adaptation Assistant)</p>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card"><h3>บทบาท</h3>'
        "<p style='color:#cbd5e1;font-size:13px;'>เชื่อม Foundry / OpenAI ภายหลัง — ตอนนี้ให้คำตอบตามระดับ Stress และข้อความคำถาม</p></div>",
        unsafe_allow_html=True,
    )
    q = st.text_area("ถามผู้ช่วย (ภาษาไทยหรืออังกฤษ)", "ช่วงนี้ควรเน้นความสามารถในการจ่ายหรือเสถียรภาพระบบ?", height=68)
    if st.button("รับคำแนะปรับตัว (Adaptation guidance)", type="primary"):
        if stress >= alert_th:
            st.success(
                "ระดับความเครียดสูง: แนะนำมาตรการคุ้มครองเฉพาะกลุ่มเปราะบางก่อน + แจ้งเตือนการใช้พลังงานในบ้าน (ลดการขับรถสั้นๆ / จัดเที่ยวรถ) "
                "และเตรียมแผนสื่อสารนโยบายแบบค่อยเป็นค่อยไป."
            )
        else:
            st.info(
                "ระดับความเครียดยังไม่วิกฤต: เน้นการลงทุนใน resilience ระยะกลาง (ประสิทธิภาพ / โครงข่าย) และเก็บ stress monitor ไว้ติดตามต่อเนื่อง."
            )

# --- KPI strip ---
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Stress (ล่าสุด)", f"{stress:.0f}", f"{delta_st:+.1f}")
k2.metric("เดือนเตือน", str(alerts), f"≥ {alert_th}")
k3.metric("แถวราคาที่ใช้", f"{dq['fuel_rows_clean']:,}", f"DOEB {len(doeb_bundle)} ชุด")
k4.metric("ช่วงวันที่", f"{dq['date_min'][:4]}–{dq['date_max'][:4]}")

with st.expander("รายการไฟล์ข้อมูลใน `Track_B_Adaptive_Infrastructures_Datasets/` (จาก GitHub clone)"):
    st.caption("ใช้ยืนยันว่าโฟลเดอร์มีครบและทีมแบ่งงานตาม path ได้")
    if data_inventory.empty:
        st.write("ไม่พบโฟลเดอร์ Track_B")
    else:
        st.dataframe(data_inventory, width="stretch", hide_index=True)

st.markdown('<div class="chat-fab" title="Assistant">🤖</div>', unsafe_allow_html=True)
