# oil website v4_Chatbot

This folder is a Track B-specific evolution of `integrated_dashboard_chatbot`.

## Kept from prototype
- Streamlit integrated dashboard structure
- Gemini chatbot flow (`utils/chat_engine.py`, `utils/gemini_client.py`)
- Manual upload + auto-discovery flow in `integrated_dashboard_with_trackb.py`

## Added for Track B objective alignment
- `utils/trackb_policy_models.py`
  - causal inference (TWFE)
  - agent-based modeling (household budget stress)
  - optimization (policy mix under budget)
- New dashboard tab: **H — Causal/ABM/Optimization**
- Policy scenario controls in sidebar

## Target data scope
Only Track B datasets:
- World Bank Global Fuel Prices
- World Bank Fuel Subsidies & Price Control
- OWID
- DOEB Open Data
- NESDC
