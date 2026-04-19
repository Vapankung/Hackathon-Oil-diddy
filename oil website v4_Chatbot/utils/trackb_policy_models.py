from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _safe_read_csv(path: Optional[str], **kwargs: Any) -> pd.DataFrame:
    if not path or not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except Exception:
        return pd.DataFrame()


def _safe_read_excel(path: Optional[str], **kwargs: Any) -> pd.DataFrame:
    if not path or not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_excel(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _first_col(df: pd.DataFrame, preferred: Sequence[str], contains_all: Sequence[str]) -> Optional[str]:
    low_map = {str(c).strip().lower(): c for c in df.columns}
    for p in preferred:
        if p.lower() in low_map:
            return low_map[p.lower()]
    for c in df.columns:
        cc = str(c).strip().lower()
        if all(k.lower() in cc for k in contains_all):
            return c
    return None


def discover_trackb_policy_files(roots: Tuple[str, ...]) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if not os.path.isfile(path):
                continue
            base = os.path.basename(path)
            low = base.lower()
            if base == "Global_Fuel_Prices_Database.xlsx" and "wb_prices" not in files:
                files["wb_prices"] = path
            elif "global_fuel_subsidies" in low and low.endswith(".xlsx") and "wb_policy" not in files:
                files["wb_policy"] = path
            elif base == "OWID_Energy_Data.csv" and "owid_global" not in files:
                files["owid_global"] = path
            elif base == "vw_opendata_045_i_fuel_sum_x_data_view.csv" and "doeb_import_qty" not in files:
                files["doeb_import_qty"] = path
            elif base == "vw_opendata_037_i_fuel_value_data_view.csv" and "doeb_import_value" not in files:
                files["doeb_import_value"] = path
            elif base == "vw_opendata_038_i_crude_value_data_view.csv" and "doeb_crude_value" not in files:
                files["doeb_crude_value"] = path
            elif base == "vw_opendata_039_e_fuel_sum_data_view.csv" and "doeb_export_qty" not in files:
                files["doeb_export_qty"] = path
            elif base.endswith("260205.xlsx") and "nesdc_poverty" not in files:
                files["nesdc_poverty"] = path
    return files


def build_worldbank_policy_panel(wb_prices: pd.DataFrame, wb_policy: pd.DataFrame) -> pd.DataFrame:
    if wb_prices.empty or wb_policy.empty:
        return pd.DataFrame()

    cp = _first_col(wb_prices, ["country", "country name"], ["country"])
    yp = _first_col(wb_prices, ["year", "date"], ["year"])
    mp = _first_col(wb_prices, ["month"], ["month"])
    fp = _first_col(wb_prices, ["pump price (us$ per liter)", "price_usd_per_liter"], ["price"])

    ct = _first_col(wb_policy, ["country", "country name"], ["country"])
    yt = _first_col(wb_policy, ["year", "date"], ["year"])
    tr = _first_col(wb_policy, ["price control", "subsidy present", "treatment"], ["price", "control"])
    sz = _first_col(wb_policy, ["subsidy value (us$)", "subsidy_usd", "subsidy amount"], ["subsid"])

    needed = [cp, yp, fp, ct, yt, tr]
    if any(c is None for c in needed):
        return pd.DataFrame()

    pcols = [cp, yp, fp] + ([mp] if mp else [])
    prices = wb_prices[pcols].copy()
    prices.columns = ["country", "year", "fuel_price"] + (["month"] if mp else [])

    tcols = [ct, yt, tr] + ([sz] if sz else [])
    policy = wb_policy[tcols].copy()
    policy.columns = ["country", "year", "treatment"] + (["subsidy_size"] if sz else [])

    prices["year"] = pd.to_numeric(prices["year"], errors="coerce")
    prices["fuel_price"] = pd.to_numeric(prices["fuel_price"], errors="coerce")
    if "month" in prices.columns:
        prices["month"] = pd.to_numeric(prices["month"], errors="coerce")

    policy["year"] = pd.to_numeric(policy["year"], errors="coerce")
    policy["treatment"] = (
        policy["treatment"]
        .astype(str)
        .str.lower()
        .map({"yes": 1, "true": 1, "1": 1, "price control": 1, "no": 0, "false": 0, "0": 0, "none": 0})
        .fillna(pd.to_numeric(policy["treatment"], errors="coerce"))
        .fillna(0.0)
    )
    if "subsidy_size" not in policy.columns:
        policy["subsidy_size"] = 0.0
    else:
        policy["subsidy_size"] = pd.to_numeric(policy["subsidy_size"], errors="coerce").fillna(0.0)

    panel = prices.merge(policy[["country", "year", "treatment", "subsidy_size"]], on=["country", "year"], how="inner")
    panel = panel.dropna(subset=["country", "year", "fuel_price"]).copy()
    panel["year"] = panel["year"].astype(int)
    panel["country"] = panel["country"].astype(str)
    if "month" in panel.columns:
        panel["month"] = panel["month"].fillna(1).astype(int)
    else:
        panel["month"] = 1
    return panel


def causal_twfe(panel: pd.DataFrame) -> Dict[str, Any]:
    if panel.empty or len(panel) < 30:
        return {"ready": False, "message": "Insufficient panel rows for TWFE.", "n_rows": int(len(panel))}

    model = smf.ols("fuel_price ~ treatment + subsidy_size + C(country) + C(year)", data=panel).fit(cov_type="HC1")
    return {
        "ready": True,
        "method": "Two-way Fixed Effects (country/year)",
        "n_rows": int(len(panel)),
        "n_countries": int(panel["country"].nunique()),
        "year_min": int(panel["year"].min()),
        "year_max": int(panel["year"].max()),
        "ate_treatment_on_price": float(model.params.get("treatment", np.nan)),
        "pvalue": float(model.pvalues.get("treatment", np.nan)),
        "r2": float(model.rsquared),
    }


@dataclass
class HouseholdAgent:
    group: str
    income: float
    energy_share: float
    pop_share: float


def agent_based_money_resilience(price_shock_pct: float, subsidy_pct: float, cash_transfer_thb: float) -> Dict[str, Any]:
    groups = [
        HouseholdAgent("low_income", 11000.0, 0.20, 0.35),
        HouseholdAgent("middle_income", 28000.0, 0.13, 0.45),
        HouseholdAgent("high_income", 65000.0, 0.09, 0.20),
    ]
    net_change = (price_shock_pct - subsidy_pct) / 100.0
    rows: List[Dict[str, Any]] = []
    for g in groups:
        base_cost = g.income * g.energy_share
        post_cost = max(base_cost * (1.0 + net_change) - cash_transfer_thb, 0.0)
        post_share = post_cost / max(g.income, 1.0)
        rows.append(
            {
                "group": g.group,
                "population_share": g.pop_share,
                "baseline_energy_share": g.energy_share,
                "post_policy_energy_share": post_share,
                "stress_delta_pp": (post_share - g.energy_share) * 100.0,
            }
        )

    df = pd.DataFrame(rows)
    weighted_stress_delta_pp = float((df["stress_delta_pp"] * df["population_share"]).sum())
    affordability_breach_pct = float((df["post_policy_energy_share"] > 0.22).astype(float).mul(df["population_share"]).sum() * 100.0)
    return {
        "ready": True,
        "method": "Household-agent budget simulator",
        "weighted_stress_delta_pp": weighted_stress_delta_pp,
        "affordability_breach_rate_pct": affordability_breach_pct,
        "group_results": df.to_dict(orient="records"),
    }


def optimize_policy_mix(price_shock_pct: float, budget_billion_thb: float) -> Dict[str, Any]:
    budget_billion_thb = max(float(budget_billion_thb), 1.0)
    best: Optional[Dict[str, Any]] = None
    for subsidy in np.arange(0, 31, 2):
        for transfer in np.arange(0, 701, 50):
            estimated_cost = subsidy * 1.2 + transfer / 38.0
            if estimated_cost > budget_billion_thb:
                continue
            sim = agent_based_money_resilience(price_shock_pct=price_shock_pct, subsidy_pct=float(subsidy), cash_transfer_thb=float(transfer))
            objective = sim["weighted_stress_delta_pp"] + 0.08 * estimated_cost
            cand = {
                "subsidy_pct": float(subsidy),
                "cash_transfer_thb": float(transfer),
                "estimated_budget_billion_thb": float(estimated_cost),
                "objective": float(objective),
                "weighted_stress_delta_pp": float(sim["weighted_stress_delta_pp"]),
                "affordability_breach_rate_pct": float(sim["affordability_breach_rate_pct"]),
            }
            if best is None or cand["objective"] < best["objective"]:
                best = cand

    if best is None:
        return {"ready": False, "message": "No feasible policy under current budget."}
    return {
        "ready": True,
        "method": "Grid-search optimization",
        "objective": "min(weighted_stress_delta_pp + lambda*budget_cost)",
        "best_policy": best,
    }


def build_thai_energy_context(files: Dict[str, str], owid_global: pd.DataFrame) -> pd.DataFrame:
    doeb_parts: List[pd.DataFrame] = []
    for key in ["doeb_import_qty", "doeb_import_value", "doeb_crude_value", "doeb_export_qty"]:
        path = files.get(key)
        df = _safe_read_csv(path)
        if df.empty or "YEAR_ID" not in df.columns:
            continue
        value_col = "QTY" if "QTY" in df.columns else ("BALANCE_VALUE" if "BALANCE_VALUE" in df.columns else None)
        if not value_col:
            continue
        temp = pd.DataFrame({"year": pd.to_numeric(df["YEAR_ID"], errors="coerce") - 543, key: pd.to_numeric(df[value_col], errors="coerce")}).dropna()
        temp["year"] = temp["year"].astype(int)
        temp = temp.groupby("year", as_index=False)[key].sum()
        doeb_parts.append(temp)

    doeb = pd.DataFrame()
    for p in doeb_parts:
        doeb = p if doeb.empty else doeb.merge(p, on="year", how="outer")

    ow = pd.DataFrame()
    if not owid_global.empty and {"country", "year"}.issubset(owid_global.columns):
        ow = owid_global[owid_global["country"].astype(str).str.lower().eq("thailand")].copy()
        ow["year"] = pd.to_numeric(ow["year"], errors="coerce")
        keep = [c for c in ["year", "oil_consumption", "oil_production", "gdp", "population"] if c in ow.columns]
        ow = ow[keep].dropna(subset=["year"]).groupby("year", as_index=False).mean(numeric_only=True)
        ow["year"] = ow["year"].astype(int)

    out = doeb.copy() if not doeb.empty else ow.copy()
    if not doeb.empty and not ow.empty:
        out = ow.merge(doeb, on="year", how="outer")
    return out.sort_values("year").reset_index(drop=True) if not out.empty else pd.DataFrame()


def run_trackb_policy_suite(roots: Tuple[str, ...]) -> Dict[str, Any]:
    files = discover_trackb_policy_files(roots)
    wb_prices = _safe_read_excel(files.get("wb_prices"))
    wb_policy = _safe_read_excel(files.get("wb_policy"))
    owid_global = _safe_read_csv(files.get("owid_global"))

    panel = build_worldbank_policy_panel(wb_prices, wb_policy)
    thai_context = build_thai_energy_context(files, owid_global)

    return {
        "files": files,
        "wb_prices": wb_prices,
        "wb_policy": wb_policy,
        "owid_global": owid_global,
        "worldbank_panel": panel,
        "causal": causal_twfe(panel),
        "thai_context": thai_context,
    }
