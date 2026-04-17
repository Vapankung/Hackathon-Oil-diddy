import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np

st.set_page_config(page_title="Fuel Price Explorer", layout="wide")
st.title("🛢️ Thailand Fuel Prices: Advanced Macro Explorer")
st.markdown("Toggle massive data overlays to trace exactly which macroeconomic shocks caused these fuel events.")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        # Load Monthly Prices
        price_df = pd.read_csv(os.path.join(base_dir, 'thailand_fuel_prices_cleaned.csv'))
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        price_df['Price'] = pd.to_numeric(price_df['Price'], errors='coerce')
        price_df = price_df.dropna(subset=['Price']).sort_values('Date')
        
        # Load OWID macro
        owid_df = pd.read_csv(os.path.join(base_dir, 'owid-energy-data(clean).csv'))
        owid_th = owid_df[owid_df['country'].str.contains('Thailand', case=False, na=False)].copy()
        # Convert OWID year to Datetime for Plotly shared axis alignment
        owid_th['Date'] = pd.to_datetime(owid_th['year'].astype(str) + '-01-01')
        owid_th = owid_th[owid_th['Date'].dt.year >= 2015]
        
        return price_df, owid_th
    except Exception as e:
        st.error(f"Initialization Failed: {e}")
        return pd.DataFrame(), pd.DataFrame()

df, owid_df = load_data()

if df.empty:
    st.stop()

# --- SIDEBAR: ASSET SELECTION ---
st.sidebar.header("1. Target Asset")
categories = df['Category'].unique()
selected_cat = st.sidebar.selectbox("Fuel Category", categories)
descriptions = df[df['Category'] == selected_cat]['Description'].unique()
selected_desc = st.sidebar.selectbox("Specific Product", descriptions)

view_df = df[(df['Category'] == selected_cat) & (df['Description'] == selected_desc)].copy()
view_df = view_df.sort_values('Date').reset_index(drop=True)

# --- SIDEBAR: OVERLAYS ---
st.sidebar.markdown("---")
st.sidebar.header("2. Layer Indicators")
st.sidebar.caption("Toggle any combinations to build your thesis.")

show_spikes = st.sidebar.checkbox("Price Spikes (Absolute %)", value=True)
show_volatility = st.sidebar.checkbox("Volatility (Rolling Std)", value=False)
show_rsi = st.sidebar.checkbox("Relative Strength Index (RSI)", value=False)
show_zscore = st.sidebar.checkbox("Statistical Outliers (Z-Score)", value=False)
show_avwap = st.sidebar.checkbox("Anchored VWAP & SD Bands", value=False)
show_composite = st.sidebar.checkbox("🔥 Composite 'High-Likelihood Event'", value=True)

# --- SIDEBAR: MACRO OWID ---
st.sidebar.markdown("---")
st.sidebar.header("3. OWID Macro Environment")
show_owid = st.sidebar.checkbox("Overlay Yearly OWID Data", value=False)
if show_owid:
    # Filter to only numeric columns for charting
    numeric_cols = [c for c in owid_df.columns if owid_df[c].dtype in ['float64', 'int64'] and c not in ['year']]
    # Give priority to 'gdp' or 'oil_consumption' if they exist
    def_idx = numeric_cols.index('oil_consumption') if 'oil_consumption' in numeric_cols else 0
    owid_metric = st.sidebar.selectbox("Select Macro Metric", numeric_cols, index=def_idx)

# --- SIDEBAR: PARAMETERS ---
with st.sidebar.expander("Indicator Sensitivities Configuration"):
    threshold_pct = st.slider("Absolute Spike Threshold (%)", 1.0, 50.0, 5.0, 0.5)
    vol_window = st.slider("Volatility Window", 3, 24, 6)
    rsi_window = st.slider("RSI Window", 3, 24, 14)
    z_window = st.slider("Z-Score Window", 6, 48, 12)
    z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 2.0, 0.1)
    vwap_sd_mult = st.slider("AVWAP SD Band Multiplier", 0.5, 4.0, 1.5, 0.1)

# --- CORE CALCULATIONS ---
if len(view_df) > 1:
    view_df['Price_Change_Pct'] = view_df['Price'].pct_change() * 100
    view_df['Price_Change_Abs'] = view_df['Price'].diff()
    
    # Precompute metrics so composite can use them regardless of subplot toggles
    view_df['Is_Inc_Spike'] = view_df['Price_Change_Pct'] >= threshold_pct
    view_df['Is_Dec_Spike'] = view_df['Price_Change_Pct'] <= -threshold_pct
    
    view_df['Volatility'] = view_df['Price_Change_Pct'].rolling(vol_window).std()
    
    delta = view_df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    view_df['RSI'] = 100 - (100 / (1 + (gain/loss)))

    roll_m = view_df['Price_Change_Pct'].rolling(z_window).mean()
    roll_s = view_df['Price_Change_Pct'].rolling(z_window).std()
    view_df['Z_Score'] = (view_df['Price_Change_Pct'] - roll_m) / roll_s

    # Create composite logic: Absolute spike that also breaches the Z-Score outlier parameter
    view_df['Composite_Event'] = (
        (view_df['Is_Inc_Spike'] | view_df['Is_Dec_Spike']) & 
        (view_df['Z_Score'].abs() >= z_threshold) & 
        (view_df['Volatility'] > getattr(view_df['Volatility'], "median")())
    )

    # --- AVWAP CALCULATION ---
    if 'oil_consumption' in owid_df.columns:
        yearly_vol = owid_df.set_index(owid_df['Date'].dt.year)['oil_consumption']
        view_df['Volume'] = view_df['Date'].dt.year.map(yearly_vol).fillna(1.0)
    else:
        view_df['Volume'] = 1.0

    # Cumsum uniquely resets the VWAP 'group' every time a composite event evaluates to True
    view_df['AVWAP_Group'] = view_df['Composite_Event'].cumsum()
    view_df['Price_Volume'] = view_df['Price'] * view_df['Volume']
    view_df['Cum_PV'] = view_df.groupby('AVWAP_Group')['Price_Volume'].cumsum()
    view_df['Cum_V'] = view_df.groupby('AVWAP_Group')['Volume'].cumsum()
    view_df['AVWAP'] = view_df['Cum_PV'] / view_df['Cum_V']
    
    # Calculate rolling standard deviation anchored natively down to the VWAP reset event
    view_df['AVWAP_Std'] = view_df.groupby('AVWAP_Group')['Price'].expanding().std().reset_index(level=0, drop=True)

    # --- BUILD DYNAMIC SUBPLOTS ---
    active_plots = []
    if show_volatility: active_plots.append("Volatility")
    if show_rsi: active_plots.append("Relative Strength Index")
    if show_zscore: active_plots.append("Z-Score")
    if show_owid: active_plots.append(f"OWID: {owid_metric}")
    
    # Unconditionally add the Timeline gradient as the very last subplot
    active_plots.append("Timeline")

    total_rows = 1 + len(active_plots)
    titles = [f"Price History: {selected_desc}"] + active_plots
    
    # We want the timeline to be very thin, so calculate row_heights correctly.
    main_heights = [0.5] + [0.4/max(1, len(active_plots)-1)] * (len(active_plots)-1)
    main_heights.append(0.1) # Timeline gets 10% height
    
    fig = make_subplots(
        rows=total_rows, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.04, 
        subplot_titles=titles,
        row_heights=main_heights
    )

    # Main Price Trace
    fig.add_trace(go.Scatter(x=view_df['Date'], y=view_df['Price'], mode='lines', name='Price', line=dict(color='#0078FF')), row=1, col=1)

    # Toggled Price Overlays
    if show_avwap:
        fig.add_trace(go.Scatter(x=view_df['Date'], y=view_df['AVWAP'], mode='lines', name='Anchored VWAP', line=dict(color='#FFBB00', width=2, dash='dash')), row=1, col=1)
        # Upper SD Band
        fig.add_trace(go.Scatter(x=view_df['Date'], y=(view_df['AVWAP'] + (vwap_sd_mult * view_df['AVWAP_Std'])), mode='lines', name=f'+{vwap_sd_mult} SD Band', line=dict(color='rgba(255, 187, 0, 0.2)', width=1)), row=1, col=1)
        # Lower SD Band
        fig.add_trace(go.Scatter(x=view_df['Date'], y=(view_df['AVWAP'] - (vwap_sd_mult * view_df['AVWAP_Std'])), mode='lines', name=f'-{vwap_sd_mult} SD Band', line=dict(color='rgba(255, 187, 0, 0.2)', width=1)), row=1, col=1)
        
        # Add stark vertical lines indicating exact reset epochs
        reset_dates = view_df[view_df['Composite_Event']]['Date']
        for reset_date in reset_dates:
            fig.add_vline(x=reset_date, line_width=1.5, line_dash="solid", line_color="rgba(255, 170, 0, 0.6)", row=1, col=1)
    if show_spikes:
        inc = view_df[view_df['Is_Inc_Spike']]
        dec = view_df[view_df['Is_Dec_Spike']]
        if not inc.empty: fig.add_trace(go.Scatter(x=inc['Date'], y=inc['Price'], mode='markers', marker=dict(color='#FF2842', size=8, symbol='triangle-up'), name=f"Spike (>{threshold_pct}%)"), row=1, col=1)
        if not dec.empty: fig.add_trace(go.Scatter(x=dec['Date'], y=dec['Price'], mode='markers', marker=dict(color='#00E88F', size=8, symbol='triangle-down'), name=f"Spike (<-{threshold_pct}%)"), row=1, col=1)

    if show_composite:
        cex = view_df[view_df['Composite_Event']]
        if not cex.empty:
            fig.add_trace(go.Scatter(x=cex['Date'], y=cex['Price'], mode='markers', marker=dict(color='orange', size=16, symbol='star', line=dict(color='black', width=1)), name='🔥 LIKELY EVENT'), row=1, col=1)

    # Iterate through dynamic subplots
    current_row = 2
    for plot_name in active_plots:
        if plot_name == "Volatility":
            fig.add_trace(go.Scatter(x=view_df['Date'], y=view_df['Volatility'], mode='lines', name='Volatility', fill='tozeroy', line=dict(color='#FF9E00')), row=current_row, col=1)
            
        elif plot_name == "Relative Strength Index":
            fig.add_trace(go.Scatter(x=view_df['Date'], y=view_df['RSI'], mode='lines', name='RSI', line=dict(color='#D700FF')), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
            
        elif plot_name == "Z-Score":
            fig.add_trace(go.Scatter(x=view_df['Date'], y=view_df['Z_Score'], mode='lines', name='Z-Score', fill='tozeroy', line=dict(color='#00D1FF')), row=current_row, col=1)
            fig.add_hline(y=z_threshold, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=-z_threshold, line_dash="dash", line_color="green", row=current_row, col=1)
            
        elif plot_name.startswith("OWID"):
            owid_data = owid_df[['Date', owid_metric]].dropna().sort_values('Date')
            # Line chart with markers is much cleaner than Bar for sparse yearly data on a monthly axis
            fig.add_trace(go.Scatter(
                x=owid_data['Date'], 
                y=owid_data[owid_metric], 
                mode='lines+markers', 
                name=owid_metric, 
                line=dict(color='#A8A8A8', width=2, dash='dot'),
                marker=dict(color='#FFA500', size=8)
            ), row=current_row, col=1)

        elif plot_name == "Timeline":
            # Draw a solid horizontal block of bars mapped to a chronological gray-to-white gradient
            fig.add_trace(go.Bar(
                x=view_df['Date'],
                y=[1] * len(view_df),
                marker=dict(
                    color=view_df['Date'].astype('int64'),
                    colorscale=[[0, '#2B2B2B'], [1, 'white']],
                ),
                name="Timeline",
                hoverinfo='none',
                showlegend=False
            ), row=current_row, col=1)
            # Hide the Y axis of the timeline completely so it just looks like a color bar
            fig.update_yaxes(visible=False, row=current_row, col=1)

        current_row += 1

    fig.update_layout(
        hovermode="x unified",
        height=400 + (250 * (len(active_plots) - 1)) + 50, # Add 50px for timeline
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Force X-axis range slider only on the absolute bottom subplot (the chronological gradient)
    fig.update_xaxes(rangeslider=dict(visible=False))
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.15), row=total_rows, col=1)

    st.plotly_chart(fig, width="stretch")

    # Composite Event DataFrame
    st.subheader("🔥 High-Likelihood Shocks")
    st.markdown("These are the most violent, statistically verified events calculated across multiple parameters.")
    table_df = view_df[view_df['Composite_Event']][['Date', 'Price', 'Price_Change_Pct', 'Z_Score', 'Volatility']].copy()
    if not table_df.empty:
        table_df['Date'] = table_df['Date'].dt.date
        
        # Format the percentage column to include the % suffix while keeping others to 2 decimals
        formatter = {
            'Price': '{:.2f}',
            'Price_Change_Pct': '{:.2f}%',
            'Z_Score': '{:.2f}',
            'Volatility': '{:.4f}'
        }
        st.dataframe(table_df.sort_values('Date', ascending=False).style.format(formatter))
    else:
        st.info("No extreme clusters found. Try widening your sensitivity thresholds under 'Configuration'.")

else:
    st.warning("Not enough data to calculate spikes for this product.")
