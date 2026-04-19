from __future__ import annotations

import io
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

THAI_BE_OFFSET = 543


def be_to_ce(year: int) -> int:
    return year - THAI_BE_OFFSET if year > 2400 else year


def detect_schema(df: pd.DataFrame) -> Dict[str, Any]:
    cols = {str(c).upper(): c for c in df.columns}
    schema: Dict[str, Any] = {
        "date_cols": ("YEAR_ID" in cols and "MONTH_ID" in cols),
        "year_col": cols.get("YEAR_ID"),
        "month_col": cols.get("MONTH_ID"),
        "value_col": None,
        "series_col": None,
        "subject_col": cols.get("SUBJECT"),
        "unit_col": cols.get("UNIT"),
    }

    for candidate in ["PRICE", "QTY", "BALANCE_VALUE"]:
        if candidate in cols:
            schema["value_col"] = cols[candidate]
            break

    for candidate in ["OIL_NAME_ENG", "OIL_NAME_TH", "OIL_TYPE", "FUEL_TYPE", "SUBJECT"]:
        if candidate in cols and candidate != "SUBJECT":
            schema["series_col"] = cols[candidate]
            break

    return schema


def build_date_column(df: pd.DataFrame, year_col: str, month_col: str) -> pd.Series:
    ce_year = pd.to_numeric(df[year_col], errors="coerce").fillna(0).astype(int).apply(be_to_ce)
    month = pd.to_numeric(df[month_col], errors="coerce").fillna(1).astype(int).clip(1, 12)
    return pd.to_datetime(ce_year.astype(str) + "-" + month.astype(str).str.zfill(2) + "-01", errors="coerce")


@st.cache_data(show_spinner=False)
def load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = filename.rsplit(".", 1)[-1].lower()
    buf = io.BytesIO(file_bytes)
    if ext == "csv":
        for enc in ("utf-8", "utf-8-sig", "tis-620", "cp874", "latin-1"):
            try:
                buf.seek(0)
                return pd.read_csv(buf, encoding=enc)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise ValueError("Could not decode CSV with supported encodings")
    if ext in ("xlsx", "xls"):
        return pd.read_excel(buf, sheet_name=0)
    raise ValueError(f"Unsupported file type: .{ext}")


def safe_median(series: pd.Series) -> float:
    return float(series.median()) if not series.dropna().empty else 0.0


def _compute_indicators(df_in: pd.DataFrame, value_col: str, threshold_pct: float, vol_window: int, rsi_window: int, z_window: int, z_threshold: float) -> pd.DataFrame:
    df = df_in.copy().sort_values("Date").reset_index(drop=True)
    df["Pct_Change"] = df[value_col].pct_change() * 100.0
    df["Abs_Change"] = df[value_col].diff()
    df["Is_Inc_Spike"] = df["Pct_Change"] >= threshold_pct
    df["Is_Dec_Spike"] = df["Pct_Change"] <= -threshold_pct
    df["Volatility"] = df["Pct_Change"].rolling(vol_window).std()

    delta = df[value_col].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_window).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    mean_roll = df["Pct_Change"].rolling(z_window).mean()
    std_roll = df["Pct_Change"].rolling(z_window).std()
    df["Z_Score"] = (df["Pct_Change"] - mean_roll) / std_roll.replace(0, np.nan)

    df["Composite_Event"] = (
        (df["Is_Inc_Spike"] | df["Is_Dec_Spike"])
        & (df["Z_Score"].abs() >= z_threshold)
        & (df["Volatility"] > safe_median(df["Volatility"]))
    )

    df["AVWAP_Group"] = df["Composite_Event"].cumsum()
    df["AVWAP"] = df.groupby("AVWAP_Group")[value_col].transform(lambda x: x.expanding().mean())
    df["AVWAP_Std"] = df.groupby("AVWAP_Group")[value_col].transform(lambda x: x.expanding().std())
    return df


def render_doeb_macro_explorer_tab() -> None:
    st.markdown('<p class="section-label">I · DOEB macro explorer (file-level, upload-first)</p>', unsafe_allow_html=True)
    st.markdown(
        "Upload any **DOEB CSV / Excel** file to inspect price, quantity, or value trends with volatility/RSI/Z-score/composite-event overlays."
    )

    up = st.file_uploader(
        "Upload a DOEB CSV/XLSX/XLS file",
        type=["csv", "xlsx", "xls"],
        key="doeb_macro_upload",
        help="Examples: vw_opendata_045*.csv, vw_opendata_037*.csv, 34-2566.xlsx",
    )

    if up is None:
        st.info("Upload a file to start this explorer.")
        return

    try:
        raw_df = load_file(up.getvalue(), up.name)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return

    schema = detect_schema(raw_df)
    missing: List[str] = []
    if not schema["date_cols"]:
        missing.append("YEAR_ID and/or MONTH_ID")
    if schema["value_col"] is None:
        missing.append("PRICE/QTY/BALANCE_VALUE")

    if missing:
        st.error(f"Unsupported DOEB schema, missing: {', '.join(missing)}")
        st.write("Detected columns:", list(raw_df.columns))
        return

    raw_df["Date"] = build_date_column(raw_df, schema["year_col"], schema["month_col"])
    raw_df = raw_df.dropna(subset=["Date"]).copy()
    raw_df[schema["value_col"]] = pd.to_numeric(raw_df[schema["value_col"]], errors="coerce")

    unit_label = str(raw_df[schema["unit_col"]].iloc[0]) if schema["unit_col"] and not raw_df.empty else "Value"
    if schema["subject_col"] and raw_df[schema["subject_col"]].nunique() == 1:
        dataset_title = str(raw_df[schema["subject_col"]].iloc[0])
    else:
        dataset_title = up.name.rsplit(".", 1)[0]

    cset1, cset2 = st.columns([1.2, 1])
    with cset1:
        if schema["series_col"]:
            options = sorted(raw_df[schema["series_col"]].dropna().astype(str).unique().tolist())
            selected_series = st.multiselect("Series", options=options, default=options[:1], key="doeb_series_pick")
            if not selected_series:
                st.warning("Select at least one series.")
                return
            primary_series = selected_series[0]
            view_df = raw_df[raw_df[schema["series_col"]].astype(str).isin(selected_series)].copy()
        else:
            primary_series = dataset_title
            selected_series = [dataset_title]
            view_df = raw_df.copy()

    with cset2:
        threshold_pct = st.slider("Spike threshold (%)", 1.0, 50.0, 5.0, 0.5, key="doeb_spike_threshold")
        vol_window = st.slider("Volatility window", 3, 24, 6, key="doeb_vol_window")
        rsi_window = st.slider("RSI window", 3, 24, 14, key="doeb_rsi_window")
        z_window = st.slider("Z-score window", 6, 48, 12, key="doeb_z_window")
        z_threshold = st.slider("Z-score threshold", 1.0, 5.0, 2.0, 0.1, key="doeb_z_threshold")

    val = schema["value_col"]
    if schema["series_col"]:
        primary_df = view_df[view_df[schema["series_col"]].astype(str).eq(primary_series)].copy()
    else:
        primary_df = view_df.copy()

    primary_df = _compute_indicators(primary_df, val, threshold_pct, vol_window, rsi_window, z_window, z_threshold)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dataset", dataset_title[:42])
    m2.metric("Range", f"{view_df['Date'].min().year}–{view_df['Date'].max().year}")
    m3.metric("Records", f"{len(primary_df):,}")
    m4.metric("Unit", unit_label)

    show_vol = st.checkbox("Show Volatility", True, key="doeb_show_vol")
    show_rsi = st.checkbox("Show RSI", True, key="doeb_show_rsi")
    show_z = st.checkbox("Show Z-score", True, key="doeb_show_z")
    show_spikes = st.checkbox("Show spikes", True, key="doeb_show_spikes")
    show_composite = st.checkbox("Show composite events", True, key="doeb_show_comp")

    subplots = []
    if show_vol:
        subplots.append("Volatility")
    if show_rsi:
        subplots.append("RSI")
    if show_z:
        subplots.append("Z-Score")

    total_rows = 1 + len(subplots)
    row_heights = [0.62] + ([0.38 / max(1, len(subplots))] * len(subplots))

    fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=row_heights)

    palette = ["#3B8BEB", "#FF6B6B", "#2ED9C3", "#FFA94D", "#9C7AFF"]
    if schema["series_col"]:
        for i, s in enumerate(selected_series):
            s_df = view_df[view_df[schema["series_col"]].astype(str).eq(s)].sort_values("Date").dropna(subset=[val])
            fig.add_trace(go.Scatter(x=s_df["Date"], y=s_df[val], mode="lines", name=s, line=dict(color=palette[i % len(palette)], width=2 if s == primary_series else 1.4), opacity=1.0 if s == primary_series else 0.55), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=primary_df["Date"], y=primary_df[val], mode="lines", name="Value", line=dict(color="#3B8BEB", width=2)), row=1, col=1)

    if show_spikes:
        inc = primary_df[primary_df["Is_Inc_Spike"]]
        dec = primary_df[primary_df["Is_Dec_Spike"]]
        if not inc.empty:
            fig.add_trace(go.Scatter(x=inc["Date"], y=inc[val], mode="markers", marker=dict(color="#FF4444", size=8, symbol="triangle-up"), name=f"▲ >{threshold_pct:.0f}%"), row=1, col=1)
        if not dec.empty:
            fig.add_trace(go.Scatter(x=dec["Date"], y=dec[val], mode="markers", marker=dict(color="#00CC88", size=8, symbol="triangle-down"), name=f"▼ <-{threshold_pct:.0f}%"), row=1, col=1)

    if show_composite:
        comp = primary_df[primary_df["Composite_Event"]]
        if not comp.empty:
            fig.add_trace(go.Scatter(x=comp["Date"], y=comp[val], mode="markers", marker=dict(color="orange", size=14, symbol="star", line=dict(color="black", width=1)), name="Composite events"), row=1, col=1)

    r = 2
    for name in subplots:
        if name == "Volatility":
            fig.add_trace(go.Scatter(x=primary_df["Date"], y=primary_df["Volatility"], mode="lines", fill="tozeroy", line=dict(color="#FF9E00"), name="Volatility"), row=r, col=1)
        elif name == "RSI":
            fig.add_trace(go.Scatter(x=primary_df["Date"], y=primary_df["RSI"], mode="lines", line=dict(color="#C855FF"), name="RSI"), row=r, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,80,80,0.6)", row=r, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(80,200,80,0.6)", row=r, col=1)
        elif name == "Z-Score":
            fig.add_trace(go.Scatter(x=primary_df["Date"], y=primary_df["Z_Score"], mode="lines", fill="tozeroy", line=dict(color="#00D1FF"), name="Z-Score"), row=r, col=1)
            fig.add_hline(y=z_threshold, line_dash="dash", line_color="rgba(255,80,80,0.6)", row=r, col=1)
            fig.add_hline(y=-z_threshold, line_dash="dash", line_color="rgba(80,200,80,0.6)", row=r, col=1)
        r += 1

    fig.update_layout(
        title=f"{dataset_title} ({unit_label})",
        hovermode="x unified",
        height=460 + max(0, len(subplots)) * 190,
        legend=dict(orientation="h", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    if show_composite:
        st.markdown("### Composite shock events")
        out_cols = ["Date", val, "Pct_Change", "Z_Score", "Volatility"]
        table_df = primary_df[primary_df["Composite_Event"]][out_cols].copy()
        if not table_df.empty:
            table_df["Date"] = table_df["Date"].dt.date
            st.dataframe(table_df.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No composite events for current thresholds.")

    with st.expander("Raw preview"):
        prev = view_df.copy()
        prev["Date"] = prev["Date"].dt.date
        st.dataframe(prev.head(100), use_container_width=True, hide_index=True)
