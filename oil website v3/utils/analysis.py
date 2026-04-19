from __future__ import annotations

from typing import Dict


def summarize_trackb_results(causal: Dict, abm: Dict, optimize: Dict) -> str:
    lines = [
        "### Track B: Money Resilience Lab (Economic)",
        "This run is constrained to World Bank fuel-price/policy data, OWID, DOEB, and NESDC inputs.",
        "",
    ]

    if causal.get("ready"):
        ate = causal.get("ate_on_fuel_price")
        pvalue = causal.get("pvalue")
        lines.append(
            f"- **Causal inference (TWFE):** treatment ATE on fuel price = {ate:.4f} (p={pvalue:.4f})."
        )
    else:
        lines.append(f"- **Causal inference:** {causal.get('message', 'Not ready')}")

    if abm.get("ready"):
        lines.append(
            "- **Agent-based modeling:** "
            f"weighted stress change = {abm.get('weighted_stress_delta_pp', 0.0):.2f} pp, "
            f"affordability breach = {abm.get('affordability_breach_rate_pct', 0.0):.2f}% of households."
        )
    else:
        lines.append("- **Agent-based modeling:** Not ready.")

    if optimize.get("ready"):
        best = optimize.get("best_policy", {})
        lines.append(
            "- **Optimization:** best policy under budget suggests "
            f"subsidy {best.get('subsidy_pct', 0.0):.1f}% + transfer {best.get('cash_transfer_thb', 0.0):.1f} THB/month."
        )
    else:
        lines.append(f"- **Optimization:** {optimize.get('message', 'Not ready')}")

    lines.append(
        "- Policy interpretation should combine international signals (World Bank/OWID) with Thai context (DOEB/NESDC)."
    )
    return "\n".join(lines)
