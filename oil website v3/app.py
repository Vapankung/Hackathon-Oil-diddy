from __future__ import annotations

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

from utils.analysis import summarize_trackb_results
from utils.data_loader import load_trackb_datasets
from utils.modeling import (
    run_abm_simulation,
    run_causal_inference,
    run_policy_optimization,
)

load_dotenv()

app = Flask(__name__)

DATA_BUNDLE = load_trackb_datasets()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "datasets": DATA_BUNDLE["status"],
            "allowed_sources": [
                "World Bank Global Fuel Prices",
                "World Bank Fuel Subsidies & Price Control",
                "OWID",
                "DOEB Open Data",
                "NESDC",
            ],
        }
    )


@app.route("/api/trackb/evaluate", methods=["POST"])
def evaluate_trackb():
    payload = request.get_json(silent=True) or {}

    policy_price_shock_pct = float(payload.get("policy_price_shock_pct", 12.0))
    policy_subsidy_pct = float(payload.get("policy_subsidy_pct", 8.0))
    policy_cash_transfer_thb = float(payload.get("policy_cash_transfer_thb", 250.0))
    optimize_budget_billion_thb = float(payload.get("optimize_budget_billion_thb", 40.0))

    causal = run_causal_inference(
        DATA_BUNDLE,
        treatment_col_hint=payload.get("treatment_col_hint", "price_control"),
    )
    abm = run_abm_simulation(
        DATA_BUNDLE,
        price_shock_pct=policy_price_shock_pct,
        subsidy_pct=policy_subsidy_pct,
        cash_transfer_thb=policy_cash_transfer_thb,
    )
    optimize = run_policy_optimization(
        DATA_BUNDLE,
        budget_billion_thb=optimize_budget_billion_thb,
        price_shock_pct=policy_price_shock_pct,
    )

    narrative = summarize_trackb_results(causal=causal, abm=abm, optimize=optimize)

    return jsonify(
        {
            "causal_inference": causal,
            "agent_based_model": abm,
            "optimization": optimize,
            "summary": narrative,
            "dataset_status": DATA_BUNDLE["status"],
        }
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
