from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class HouseholdGroup:
    name: str
    monthly_income_thb: float
    base_energy_share: float
    population_share: float


def _choose_column(df: pd.DataFrame, options: List[str], fallback_contains: List[str]) -> str | None:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for option in options:
        if option.lower() in cols:
            return cols[option.lower()]
    for c in df.columns:
        cl = str(c).lower()
        if all(t in cl for t in fallback_contains):
            return c
    return None


def _standardize_panel_for_causal(prices: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    if prices.empty or policy.empty:
        return pd.DataFrame()

    country_p = _choose_column(prices, ["Country", "country", "Country Name"], ["country"])
    year_p = _choose_column(prices, ["Year", "year"], ["year"])
    price_col = _choose_column(prices, ["Pump Price (US$ per liter)", "price_usd_per_liter"], ["price"])

    country_t = _choose_column(policy, ["Country", "country", "Country Name"], ["country"])
    year_t = _choose_column(policy, ["Year", "year"], ["year"])
    treat = _choose_column(policy, ["Price Control", "price_control", "subsidy_present"], ["price", "control"])
    subsidy_size = _choose_column(policy, ["Subsidy Value (US$)", "subsidy_usd"], ["subsid"])

    needed = [country_p, year_p, price_col, country_t, year_t, treat]
    if any(c is None for c in needed):
        return pd.DataFrame()

    p = prices[[country_p, year_p, price_col]].copy()
    p.columns = ["country", "year", "fuel_price"]
    t_cols = [country_t, year_t, treat]
    if subsidy_size is not None:
        t_cols.append(subsidy_size)
    t = policy[t_cols].copy()
    t.columns = ["country", "year", "treatment"] + (["subsidy_size"] if subsidy_size is not None else [])

    panel = p.merge(t, on=["country", "year"], how="inner")
    panel["country"] = panel["country"].astype(str)
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel["fuel_price"] = pd.to_numeric(panel["fuel_price"], errors="coerce")

    panel["treatment"] = panel["treatment"].astype(str).str.lower().map(
        {
            "yes": 1,
            "true": 1,
            "1": 1,
            "price control": 1,
            "no": 0,
            "false": 0,
            "0": 0,
            "none": 0,
        }
    )
    panel["treatment"] = panel["treatment"].fillna(pd.to_numeric(panel["treatment"], errors="coerce")).fillna(0.0)

    if "subsidy_size" in panel.columns:
        panel["subsidy_size"] = pd.to_numeric(panel["subsidy_size"], errors="coerce")
    else:
        panel["subsidy_size"] = 0.0

    panel = panel.dropna(subset=["country", "year", "fuel_price"]).copy()
    panel = panel[panel["year"].notna()].copy()
    panel["year"] = panel["year"].astype(int)
    return panel


def run_causal_inference(bundle: Dict[str, object], treatment_col_hint: str = "price_control") -> Dict[str, object]:
    panel = _standardize_panel_for_causal(bundle.get("world_bank_prices", pd.DataFrame()), bundle.get("world_bank_policy", pd.DataFrame()))
    if panel.empty or len(panel) < 20:
        return {
            "ready": False,
            "method": "two_way_fixed_effects",
            "message": "Insufficient aligned World Bank panel rows for causal inference.",
            "n_rows": int(len(panel)),
        }

    model = smf.ols("fuel_price ~ treatment + subsidy_size + C(country) + C(year)", data=panel).fit(cov_type="HC1")
    coef = float(model.params.get("treatment", np.nan))
    pvalue = float(model.pvalues.get("treatment", np.nan))

    return {
        "ready": True,
        "method": "two_way_fixed_effects",
        "treatment_variable": treatment_col_hint,
        "n_rows": int(len(panel)),
        "n_countries": int(panel["country"].nunique()),
        "year_min": int(panel["year"].min()),
        "year_max": int(panel["year"].max()),
        "ate_on_fuel_price": coef,
        "pvalue": pvalue,
        "r_squared": float(model.rsquared),
        "interpretation": "Negative ATE means treatment is associated with lower retail fuel prices after country/year fixed effects.",
    }


def run_abm_simulation(
    bundle: Dict[str, object],
    price_shock_pct: float = 12.0,
    subsidy_pct: float = 8.0,
    cash_transfer_thb: float = 250.0,
    months: int = 12,
) -> Dict[str, object]:
    thai_panel = bundle.get("thailand_panel", pd.DataFrame())

    groups = [
        HouseholdGroup("low_income", monthly_income_thb=11000, base_energy_share=0.20, population_share=0.35),
        HouseholdGroup("middle_income", monthly_income_thb=28000, base_energy_share=0.13, population_share=0.45),
        HouseholdGroup("high_income", monthly_income_thb=65000, base_energy_share=0.09, population_share=0.20),
    ]

    inflation_guard = 1.0
    if isinstance(thai_panel, pd.DataFrame) and not thai_panel.empty and "inflation" in thai_panel.columns:
        inf = pd.to_numeric(thai_panel["inflation"], errors="coerce").dropna()
        if not inf.empty:
            inflation_guard = 1.0 + max(0.0, float(inf.tail(5).mean()) / 100.0)

    records = []
    net_price_change = (price_shock_pct - subsidy_pct) / 100.0

    for g in groups:
        monthly_energy_cost = g.monthly_income_thb * g.base_energy_share
        stressed_cost = monthly_energy_cost * (1.0 + net_price_change) / inflation_guard
        post_policy_cost = max(stressed_cost - cash_transfer_thb, 0.0)
        post_policy_share = post_policy_cost / max(g.monthly_income_thb, 1.0)

        records.append(
            {
                "group": g.name,
                "population_share": g.population_share,
                "baseline_energy_share": g.base_energy_share,
                "post_policy_energy_share": post_policy_share,
                "stress_delta_pp": (post_policy_share - g.base_energy_share) * 100.0,
            }
        )

    df = pd.DataFrame(records)
    weighted_stress_delta_pp = float((df["stress_delta_pp"] * df["population_share"]).sum())
    affordability_breach_rate = float((df["post_policy_energy_share"] > 0.22).astype(float).mul(df["population_share"]).sum() * 100.0)

    return {
        "ready": True,
        "method": "agent_based_household_budget_simulation",
        "months": int(months),
        "price_shock_pct": float(price_shock_pct),
        "subsidy_pct": float(subsidy_pct),
        "cash_transfer_thb": float(cash_transfer_thb),
        "weighted_stress_delta_pp": weighted_stress_delta_pp,
        "affordability_breach_rate_pct": affordability_breach_rate,
        "group_results": df.to_dict(orient="records"),
        "interpretation": "Lower weighted stress and breach rate indicate higher money resilience.",
    }


def run_policy_optimization(bundle: Dict[str, object], budget_billion_thb: float, price_shock_pct: float = 12.0) -> Dict[str, object]:
    budget = max(float(budget_billion_thb), 1.0)
    candidate_subsidy = np.arange(0, 31, 2)
    candidate_transfer = np.arange(0, 601, 50)

    best = None
    for s in candidate_subsidy:
        for t in candidate_transfer:
            # simple proxy budget cost (illustrative, transparent for hackathon)
            annual_cost = (s * 1.3) + (t / 40.0)
            if annual_cost > budget:
                continue

            abm = run_abm_simulation(
                bundle,
                price_shock_pct=price_shock_pct,
                subsidy_pct=float(s),
                cash_transfer_thb=float(t),
            )
            objective = abm["weighted_stress_delta_pp"] + 0.07 * annual_cost

            item = {
                "subsidy_pct": float(s),
                "cash_transfer_thb": float(t),
                "estimated_budget_billion_thb": float(annual_cost),
                "objective": float(objective),
                "weighted_stress_delta_pp": float(abm["weighted_stress_delta_pp"]),
                "affordability_breach_rate_pct": float(abm["affordability_breach_rate_pct"]),
            }
            if best is None or item["objective"] < best["objective"]:
                best = item

    if best is None:
        return {
            "ready": False,
            "method": "grid_search_policy_optimization",
            "message": "No feasible policy candidate under the provided budget.",
        }

    return {
        "ready": True,
        "method": "grid_search_policy_optimization",
        "budget_billion_thb": float(budget),
        "price_shock_pct": float(price_shock_pct),
        "best_policy": best,
        "objective_definition": "minimize weighted_stress_delta_pp + lambda * budget_cost",
    }
