document.addEventListener("DOMContentLoaded", () => {
  const runBtn = document.getElementById("run-btn");
  const summary = document.getElementById("summary");
  const jsonOutput = document.getElementById("json-output");

  const getNumber = (id) => {
    const v = Number(document.getElementById(id).value);
    return Number.isFinite(v) ? v : 0;
  };

  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    summary.textContent = "Running models...";

    const payload = {
      policy_price_shock_pct: getNumber("price_shock_pct"),
      policy_subsidy_pct: getNumber("subsidy_pct"),
      policy_cash_transfer_thb: getNumber("cash_transfer_thb"),
      optimize_budget_billion_thb: getNumber("budget_billion_thb")
    };

    try {
      const res = await fetch("/api/trackb/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      summary.textContent = data.summary || "No summary returned.";
      jsonOutput.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      summary.textContent = "Failed to run Track B evaluation.";
      jsonOutput.textContent = String(err);
    } finally {
      runBtn.disabled = false;
    }
  });
});
