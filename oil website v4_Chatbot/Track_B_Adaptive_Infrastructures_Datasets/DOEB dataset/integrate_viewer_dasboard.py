import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="DOEB Energy Explorer", layout="wide")
st.title("🛢️ Thailand Energy Data: DOEB Macro Explorer")
st.markdown(
    "Upload any **CSV or Excel file** from the DOEB dataset to explore price and "
    "volume trends with statistical overlays."
)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
THAI_BE_OFFSET = 543  # Thai Buddhist Era = CE + 543


def be_to_ce(year: int) -> int:
    """Convert Thai Buddhist Era year to Common Era. Auto-detects BE vs CE."""
    return year - THAI_BE_OFFSET if year > 2400 else year


def detect_schema(df: pd.DataFrame) -> dict:
    """
    Returns a schema dict describing:
      - date_cols   : ('YEAR_ID', 'MONTH_ID') or None
      - value_col   : 'PRICE' | 'QTY' | 'BALANCE_VALUE' | None
      - series_col  : column name that differentiates series (e.g. OIL_NAME_ENG), or None
      - subject_col : 'SUBJECT' if present
      - unit_col    : 'UNIT' if present
    """
    cols = {c.upper(): c for c in df.columns}
    schema = {
        "date_cols": ("YEAR_ID" in cols and "MONTH_ID" in cols),
        "year_col":  cols.get("YEAR_ID"),
        "month_col": cols.get("MONTH_ID"),
        "value_col": None,
        "series_col": None,
        "subject_col": cols.get("SUBJECT"),
        "unit_col": cols.get("UNIT"),
    }

    # Detect value column (priority order)
    for candidate in ["PRICE", "QTY", "BALANCE_VALUE"]:
        if candidate in cols:
            schema["value_col"] = cols[candidate]
            break

    # Detect series differentiator
    for candidate in ["OIL_NAME_ENG", "OIL_NAME_TH", "OIL_TYPE", "FUEL_TYPE", "SUBJECT"]:
        if candidate in cols and candidate != "SUBJECT":
            schema["series_col"] = cols[candidate]
            break

    return schema


def build_date_column(df: pd.DataFrame, year_col: str, month_col: str) -> pd.Series:
    ce_year = df[year_col].apply(be_to_ce)
    return pd.to_datetime(
        ce_year.astype(str) + "-" + df[month_col].astype(str).str.zfill(2) + "-01"
    )


@st.cache_data(show_spinner="Loading file…")
def load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = filename.rsplit(".", 1)[-1].lower()
    buf = io.BytesIO(file_bytes)
    if ext == "csv":
        # Try common Thai encodings
        for enc in ("utf-8", "utf-8-sig", "tis-620", "cp874", "latin-1"):
            try:
                buf.seek(0)
                return pd.read_csv(buf, encoding=enc)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise ValueError("Could not decode CSV with any supported encoding.")
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(buf, sheet_name=0)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


def safe_median(series: pd.Series) -> float:
    return series.median() if not series.isna().all() else 0.0


# ─────────────────────────────────────────────
#  SIDEBAR — FILE UPLOAD
# ─────────────────────────────────────────────
st.sidebar.header("📂 1. Load Data")
uploaded = st.sidebar.file_uploader(
    "Upload a DOEB CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Any file from the DOEB dataset — price files (xlsx), quantity/value CSVs, etc.",
)

if uploaded is None:
    st.info(
        "👆 Upload a file from the DOEB dataset to get started.\n\n"
        "**Supported formats:**\n"
        "- Price XLSX files: `34-256x.xlsx`, `untitled.xlsx` (import prices by oil type)\n"
        "- Quantity CSVs: `vw_opendata_045_*.csv`, `vw_opendata_039_*.csv` …\n"
        "- Value CSVs: `vw_opendata_037_*.csv`, `vw_opendata_038_*.csv` …"
    )
    st.stop()

# ─────────────────────────────────────────────
#  LOAD & PARSE
# ─────────────────────────────────────────────
try:
    raw_df = load_file(uploaded.read(), uploaded.name)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

schema = detect_schema(raw_df)

# Validate we have at least date + value columns
missing = []
if not schema["date_cols"]:
    missing.append("YEAR_ID and/or MONTH_ID")
if schema["value_col"] is None:
    missing.append("a value column (PRICE / QTY / BALANCE_VALUE)")

if missing:
    st.error(
        f"This file doesn't look like a DOEB data file — missing: {', '.join(missing)}.\n\n"
        f"Detected columns: `{list(raw_df.columns)}`"
    )
    st.stop()

# Build Date column
raw_df["Date"] = build_date_column(raw_df, schema["year_col"], schema["month_col"])

# Coerce value column to numeric
raw_df[schema["value_col"]] = pd.to_numeric(raw_df[schema["value_col"]], errors="coerce")

# Unit label for axis
unit_label = raw_df[schema["unit_col"]].iloc[0] if schema["unit_col"] else "Value"

# Dataset title from SUBJECT if available, else filename
if schema["subject_col"] and raw_df[schema["subject_col"]].nunique() == 1:
    dataset_title = raw_df[schema["subject_col"]].iloc[0]
else:
    dataset_title = uploaded.name.rsplit(".", 1)[0]

# ─────────────────────────────────────────────
#  SIDEBAR — SERIES SELECTOR
# ─────────────────────────────────────────────
st.sidebar.header("📊 2. Select Series")

if schema["series_col"]:
    series_options = sorted(raw_df[schema["series_col"]].dropna().unique().tolist())
    selected_series = st.sidebar.multiselect(
        "Oil / Fuel Type",
        options=series_options,
        default=series_options[:1],
        help="Select one or more product types to compare.",
    )
    if not selected_series:
        st.warning("Select at least one fuel type from the sidebar.")
        st.stop()
    # Build a display name for the primary series used in spike analysis
    primary_series = selected_series[0]
    view_df = (
        raw_df[raw_df[schema["series_col"]].isin(selected_series)]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )
else:
    # Single-series file
    view_df = raw_df.copy().sort_values("Date").reset_index(drop=True)
    selected_series = [dataset_title]
    primary_series = dataset_title

val = schema["value_col"]

# ─────────────────────────────────────────────
#  SIDEBAR — OVERLAYS & INDICATORS
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🔬 3. Layer Indicators")
st.sidebar.caption("Active only on the primary series (first selection).")

show_spikes = st.sidebar.checkbox("Price/Value Spikes (Absolute %)", value=True)
show_volatility = st.sidebar.checkbox("Volatility (Rolling Std)", value=False)
show_rsi = st.sidebar.checkbox("Relative Strength Index (RSI)", value=False)
show_zscore = st.sidebar.checkbox("Statistical Outliers (Z-Score)", value=False)
show_avwap = st.sidebar.checkbox("Anchored VWAP & SD Bands", value=False)
show_composite = st.sidebar.checkbox("🔥 Composite High-Likelihood Events", value=True)

with st.sidebar.expander("⚙️ Indicator Sensitivities"):
    threshold_pct = st.slider("Spike Threshold (%)", 1.0, 50.0, 5.0, 0.5)
    vol_window = st.slider("Volatility Window", 3, 24, 6)
    rsi_window = st.slider("RSI Window", 3, 24, 14)
    z_window = st.slider("Z-Score Window", 6, 48, 12)
    z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 2.0, 0.1)
    vwap_sd_mult = st.slider("AVWAP SD Band Multiplier", 0.5, 4.0, 1.5, 0.1)

# ─────────────────────────────────────────────
#  PER-SERIES DATA PREP
# ─────────────────────────────────────────────
def compute_indicators(df_in: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators on a single time series."""
    df = df_in.copy().sort_values("Date").reset_index(drop=True)
    df["Pct_Change"] = df[val].pct_change() * 100
    df["Abs_Change"] = df[val].diff()
    df["Is_Inc_Spike"] = df["Pct_Change"] >= threshold_pct
    df["Is_Dec_Spike"] = df["Pct_Change"] <= -threshold_pct

    # Volatility
    df["Volatility"] = df["Pct_Change"].rolling(vol_window).std()

    # RSI
    delta = df[val].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_window).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Z-Score
    rm = df["Pct_Change"].rolling(z_window).mean()
    rs_std = df["Pct_Change"].rolling(z_window).std()
    df["Z_Score"] = (df["Pct_Change"] - rm) / rs_std.replace(0, np.nan)

    # Composite event
    df["Composite_Event"] = (
        (df["Is_Inc_Spike"] | df["Is_Dec_Spike"])
        & (df["Z_Score"].abs() >= z_threshold)
        & (df["Volatility"] > safe_median(df["Volatility"]))
    )

    # Anchored VWAP (volume-unweighted here since no volume data)
    df["AVWAP_Group"] = df["Composite_Event"].cumsum()
    df["Price_Volume"] = df[val]  # treat each point as unit weight
    df["Cum_PV"] = df.groupby("AVWAP_Group")["Price_Volume"].cumsum()
    df["Cum_V"] = df.groupby("AVWAP_Group")["Price_Volume"].transform("count").cumsum()
    # Recalculate properly: group-level expanding mean = AVWAP
    df["AVWAP"] = df.groupby("AVWAP_Group")[val].transform(
        lambda x: x.expanding().mean()
    )
    df["AVWAP_Std"] = df.groupby("AVWAP_Group")[val].transform(
        lambda x: x.expanding().std()
    )
    return df


# Only compute indicators on primary series (first selected)
if schema["series_col"]:
    primary_df = view_df[view_df[schema["series_col"]] == primary_series].copy()
else:
    primary_df = view_df.copy()

primary_df = compute_indicators(primary_df)

# ─────────────────────────────────────────────
#  METADATA BAR
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
min_date = view_df["Date"].min()
max_date = view_df["Date"].max()
col1.metric("Dataset", dataset_title[:40] + ("…" if len(dataset_title) > 40 else ""))
col2.metric("Date Range", f"{min_date.year}–{max_date.year}")
col3.metric("Records", f"{len(primary_df):,}")
col4.metric("Unit", unit_label)

st.divider()

# ─────────────────────────────────────────────
#  BUILD DYNAMIC SUBPLOTS
# ─────────────────────────────────────────────
active_subplots: list[str] = []
if show_volatility:
    active_subplots.append("Volatility")
if show_rsi:
    active_subplots.append("RSI")
if show_zscore:
    active_subplots.append("Z-Score")
active_subplots.append("Timeline")  # always last, thin

total_rows = 1 + len(active_subplots)
titles = [f"{dataset_title}  ({unit_label})"] + active_subplots

# Row height ratios: main=0.55, each indicator=0.4/n, timeline=0.06
n_ind = max(1, len(active_subplots) - 1)
ind_h = 0.35 / n_ind
row_heights = [0.55] + [ind_h] * n_ind + [0.06]
# Remove duplicate — timeline added separately
row_heights = [0.55] + [ind_h] * n_ind + [0.06]
# Correct: total_rows = 1 (main) + len(active_subplots)
# active_subplots already includes Timeline
row_heights = []
row_heights.append(0.55)
remaining = 0.45
if len(active_subplots) > 1:
    ind_each = 0.39 / (len(active_subplots) - 1)
    for _ in range(len(active_subplots) - 1):
        row_heights.append(ind_each)
row_heights.append(0.06)  # Timeline

fig = make_subplots(
    rows=total_rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    subplot_titles=titles,
    row_heights=row_heights,
)

# ── COLOUR PALETTE ──
PALETTE = [
    "#3B8BEB", "#FF6B6B", "#2ED9C3", "#FFA94D",
    "#9C7AFF", "#63E6BE", "#FF85C2", "#A9E34B",
]

# ── MAIN TRACES ──
# Multi-series lines
if schema["series_col"]:
    for i, series_name in enumerate(selected_series):
        s_df = view_df[view_df[schema["series_col"]] == series_name].sort_values("Date")
        # Drop NaN value rows
        s_df = s_df.dropna(subset=[val])
        color = PALETTE[i % len(PALETTE)]
        opacity = 1.0 if series_name == primary_series else 0.6
        fig.add_trace(
            go.Scatter(
                x=s_df["Date"],
                y=s_df[val],
                mode="lines",
                name=series_name,
                line=dict(color=color, width=2 if series_name == primary_series else 1.5),
                opacity=opacity,
            ),
            row=1, col=1,
        )
else:
    fig.add_trace(
        go.Scatter(
            x=primary_df["Date"],
            y=primary_df[val],
            mode="lines",
            name="Value",
            line=dict(color="#3B8BEB", width=2),
        ),
        row=1, col=1,
    )

# ── ANCHORED VWAP ──
if show_avwap:
    fig.add_trace(
        go.Scatter(
            x=primary_df["Date"],
            y=primary_df["AVWAP"],
            mode="lines",
            name="Anchored VWAP",
            line=dict(color="#FFBB00", width=2, dash="dash"),
        ),
        row=1, col=1,
    )
    upper = primary_df["AVWAP"] + vwap_sd_mult * primary_df["AVWAP_Std"].fillna(0)
    lower = primary_df["AVWAP"] - vwap_sd_mult * primary_df["AVWAP_Std"].fillna(0)
    fig.add_trace(
        go.Scatter(
            x=primary_df["Date"],
            y=upper,
            mode="lines",
            name=f"+{vwap_sd_mult}σ",
            line=dict(color="rgba(255,187,0,0.3)", width=1),
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=primary_df["Date"],
            y=lower,
            mode="lines",
            name=f"−{vwap_sd_mult}σ",
            line=dict(color="rgba(255,187,0,0.3)", width=1),
            fill="tonexty",
            fillcolor="rgba(255,187,0,0.06)",
            showlegend=False,
        ),
        row=1, col=1,
    )
    for reset_date in primary_df[primary_df["Composite_Event"]]["Date"]:
        fig.add_vline(
            x=reset_date,
            line_width=1,
            line_dash="solid",
            line_color="rgba(255,170,0,0.4)",
        )

# ── SPIKE MARKERS ──
if show_spikes:
    inc = primary_df[primary_df["Is_Inc_Spike"]].dropna(subset=[val])
    dec = primary_df[primary_df["Is_Dec_Spike"]].dropna(subset=[val])
    if not inc.empty:
        fig.add_trace(
            go.Scatter(
                x=inc["Date"], y=inc[val], mode="markers",
                marker=dict(color="#FF4444", size=8, symbol="triangle-up"),
                name=f"▲ Spike >{threshold_pct:.0f}%",
            ),
            row=1, col=1,
        )
    if not dec.empty:
        fig.add_trace(
            go.Scatter(
                x=dec["Date"], y=dec[val], mode="markers",
                marker=dict(color="#00CC88", size=8, symbol="triangle-down"),
                name=f"▼ Spike <−{threshold_pct:.0f}%",
            ),
            row=1, col=1,
        )

# ── COMPOSITE EVENT STARS ──
if show_composite:
    comp = primary_df[primary_df["Composite_Event"]].dropna(subset=[val])
    if not comp.empty:
        fig.add_trace(
            go.Scatter(
                x=comp["Date"], y=comp[val], mode="markers",
                marker=dict(
                    color="orange", size=16, symbol="star",
                    line=dict(color="black", width=1),
                ),
                name="🔥 Likely Shock",
            ),
            row=1, col=1,
        )

# ── INDICATOR SUBPLOTS ──
current_row = 2
for subplot_name in active_subplots:
    if subplot_name == "Volatility":
        fig.add_trace(
            go.Scatter(
                x=primary_df["Date"],
                y=primary_df["Volatility"],
                mode="lines",
                name="Volatility",
                fill="tozeroy",
                fillcolor="rgba(255,158,0,0.2)",
                line=dict(color="#FF9E00"),
            ),
            row=current_row, col=1,
        )

    elif subplot_name == "RSI":
        fig.add_trace(
            go.Scatter(
                x=primary_df["Date"],
                y=primary_df["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="#C855FF"),
            ),
            row=current_row, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,80,80,0.6)", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(80,200,80,0.6)", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)

    elif subplot_name == "Z-Score":
        fig.add_trace(
            go.Scatter(
                x=primary_df["Date"],
                y=primary_df["Z_Score"],
                mode="lines",
                name="Z-Score",
                fill="tozeroy",
                fillcolor="rgba(0,209,255,0.1)",
                line=dict(color="#00D1FF"),
            ),
            row=current_row, col=1,
        )
        fig.add_hline(y=z_threshold, line_dash="dash", line_color="rgba(255,80,80,0.6)", row=current_row, col=1)
        fig.add_hline(y=-z_threshold, line_dash="dash", line_color="rgba(80,200,80,0.6)", row=current_row, col=1)

    elif subplot_name == "Timeline":
        fig.add_trace(
            go.Bar(
                x=primary_df["Date"],
                y=[1] * len(primary_df),
                marker=dict(
                    color=primary_df["Date"].astype("int64"),
                    colorscale=[[0, "#1A1A2E"], [1, "#E8E8F0"]],
                ),
                name="Timeline",
                hoverinfo="none",
                showlegend=False,
            ),
            row=current_row, col=1,
        )
        fig.update_yaxes(visible=False, row=current_row, col=1)

    current_row += 1

# ── LAYOUT POLISH ──
fig.update_layout(
    hovermode="x unified",
    height=420 + max(0, (len(active_subplots) - 1)) * 200,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="rgba(18,18,30,0.95)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#CCCCDD"),
    margin=dict(l=10, r=10, t=60, b=10),
)
fig.update_xaxes(rangeslider_visible=False, gridcolor="rgba(80,80,100,0.2)")
fig.update_yaxes(gridcolor="rgba(80,80,100,0.2)")
fig.update_xaxes(
    rangeslider=dict(visible=True, thickness=0.06, bgcolor="rgba(40,40,60,0.8)"),
    row=total_rows, col=1,
)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
#  COMPOSITE EVENTS TABLE
# ─────────────────────────────────────────────
if show_composite:
    st.subheader("🔥 High-Likelihood Shock Events")
    st.caption(
        "Events flagged as statistically extreme — simultaneous spike, high Z-score, and above-median volatility."
    )
    cols_to_show = ["Date", val, "Pct_Change", "Z_Score", "Volatility"]
    if schema["series_col"]:
        # Reattach series column for display
        primary_df_disp = primary_df.copy()
        primary_df_disp[schema["series_col"]] = primary_series
        cols_to_show = [schema["series_col"]] + cols_to_show
    else:
        primary_df_disp = primary_df.copy()

    table_df = primary_df_disp[primary_df_disp["Composite_Event"]][cols_to_show].copy()
    if not table_df.empty:
        table_df["Date"] = table_df["Date"].dt.date
        rename_map = {
            val: f"Value ({unit_label})",
            "Pct_Change": "% Change",
            "Z_Score": "Z-Score",
            "Volatility": "Volatility (std)",
        }
        table_df = table_df.rename(columns=rename_map)
        fmt = {
            f"Value ({unit_label})": "{:.4f}",
            "% Change": "{:.2f}%",
            "Z-Score": "{:.2f}",
            "Volatility (std)": "{:.4f}",
        }
        st.dataframe(
            table_df.sort_values("Date", ascending=False).style.format(fmt),
            use_container_width=True,
        )
    else:
        st.info("No composite shock events detected. Try loosening the thresholds in the sidebar.")

# ─────────────────────────────────────────────
#  DATA PREVIEW
# ─────────────────────────────────────────────
with st.expander("🗂️ Raw Data Preview"):
    preview = view_df.copy()
    preview["Date"] = preview["Date"].dt.date
    st.dataframe(preview.head(100), use_container_width=True)
    st.caption(f"Showing first 100 of {len(view_df):,} rows.")