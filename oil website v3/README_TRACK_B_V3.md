# oil website v3

Track B dedicated implementation using **only**:
- World Bank Global Fuel Prices
- World Bank Fuel Subsidies & Price Control Measures
- OWID Energy Data
- DOEB Open Data
- NESDC Socio-Economic Data

## What is implemented
1. **Causal inference**: two-way fixed effects model on World Bank panel (`fuel_price ~ treatment + subsidy_size + country FE + year FE`).
2. **Agent-based modeling**: household-group budget simulation under fuel-price shock, subsidy, and cash transfer policy.
3. **Optimization**: grid-search policy mix (subsidy + transfer) under budget constraint.

## Run
```bash
cd "oil website v3"
python app.py
```
Then open `http://127.0.0.1:5000`.
