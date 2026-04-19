from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DOEB_FILE_NAMES = {
    "vw_opendata_045_i_fuel_sum_x_data_view.csv": "doeb_import_refined_qty",
    "vw_opendata_037_i_fuel_value_data_view.csv": "doeb_import_refined_value",
    "vw_opendata_038_i_crude_value_data_view.csv": "doeb_import_crude_value",
    "vw_opendata_039_e_fuel_sum_data_view.csv": "doeb_export_refined_qty",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except Exception:
        return pd.DataFrame()


def _safe_read_excel(path: Path, sheet_name=0) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        return pd.DataFrame()


def _find_first(root: Path, name: str) -> Optional[Path]:
    matches = list(root.rglob(name))
    return matches[0] if matches else None


def _load_doeb_frames(dataset_root: Path) -> Dict[str, pd.DataFrame]:
    doeb_root = dataset_root / "DOEB dataset"
    out: Dict[str, pd.DataFrame] = {}
    if not doeb_root.exists():
        return out

    for path in doeb_root.rglob("*.csv"):
        key = DOEB_FILE_NAMES.get(path.name)
        if key and key not in out:
            out[key] = _safe_read_csv(path)
    return out


def _load_nesdc_frames(dataset_root: Path) -> Dict[str, pd.DataFrame]:
    nesdc_root = dataset_root / "NESDC"
    if not nesdc_root.exists():
        return {}

    out = {
        "cost_to_gdp": _safe_read_excel(nesdc_root / "CostToGDP.xlsx", sheet_name="Cost to GDP"),
        "log_table": _safe_read_excel(nesdc_root / "LogTable.xlsx"),
        "poverty_income": _safe_read_excel(nesdc_root / "สถิติความยากจนและการกระจายรายได้_260205.xlsx", sheet_name="1.8"),
    }
    return out


def _extract_year_column(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    for col in df.columns:
        c = str(col).strip().lower()
        if c in {"year", "year_id", "time", "date", "month"}:
            return col
    for col in df.columns:
        c = str(col).lower()
        if "year" in c:
            return col
    return None


def _make_thailand_panel(owid: pd.DataFrame, doeb: Dict[str, pd.DataFrame], nesdc: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if owid.empty:
        return pd.DataFrame()

    panel = owid.copy()
    if "country" in panel.columns:
        panel = panel[panel["country"].astype(str).str.lower().eq("thailand")].copy()

    if panel.empty:
        return pd.DataFrame()

    if "year" in panel.columns:
        panel["year"] = pd.to_numeric(panel["year"], errors="coerce")

    annual = panel.groupby("year", as_index=False).mean(numeric_only=True)

    # merge DOEB annual totals
    doeb_annual: Optional[pd.DataFrame] = None
    for key, frame in doeb.items():
        if frame.empty:
            continue
        if "YEAR_ID" not in frame.columns:
            continue
        value_col = "QTY" if "QTY" in frame.columns else ("BALANCE_VALUE" if "BALANCE_VALUE" in frame.columns else None)
        if not value_col:
            continue
        d = pd.DataFrame(
            {
                "year": pd.to_numeric(frame["YEAR_ID"], errors="coerce") - 543,
                key: pd.to_numeric(frame[value_col], errors="coerce"),
            }
        ).dropna()
        d["year"] = d["year"].astype(int)
        d = d.groupby("year", as_index=False)[key].sum()
        doeb_annual = d if doeb_annual is None else doeb_annual.merge(d, on="year", how="outer")

    if doeb_annual is not None:
        annual = annual.merge(doeb_annual, on="year", how="left")

    # NESDC poverty proxy from sheet 1.8 (year in BE row index 2)
    pov = nesdc.get("poverty_income", pd.DataFrame())
    if not pov.empty and len(pov) > 3:
        try:
            years_be = pd.to_numeric(pov.iloc[2, 1:], errors="coerce")
            year_cols = years_be.dropna().index.tolist()
            if year_cols:
                out = pd.DataFrame({"year": (years_be.loc[year_cols] - 543).astype(int).tolist()})
                for i in range(len(pov)):
                    label = str(pov.iloc[i, 0])
                    vals = pd.to_numeric(pov.iloc[i, year_cols], errors="coerce").tolist()
                    if "สัดส่วนคนจน" in label:
                        out["poverty_rate_pct"] = vals
                    elif "ความรุนแรงปัญหาความยากจน" in label:
                        out["poverty_severity"] = vals
                annual = annual.merge(out, on="year", how="left")
        except Exception:
            pass

    annual = annual.sort_values("year").reset_index(drop=True)
    return annual


def load_trackb_datasets() -> Dict[str, object]:
    root = _repo_root()
    dataset_root = root / "Track_B_Adaptive_Infrastructures_Datasets"

    wb_prices_path = dataset_root / "Global_Fuel_Prices_Database.xlsx"
    wb_policy_path = dataset_root / "Global_Fuel_Subsidies_and_Price_Control_Measures_Database.xlsx"
    owid_path = dataset_root / "OWID_Energy_Data.csv"

    wb_prices = _safe_read_excel(wb_prices_path)
    wb_policy = _safe_read_excel(wb_policy_path)
    owid = _safe_read_csv(owid_path)
    doeb = _load_doeb_frames(dataset_root)
    nesdc = _load_nesdc_frames(dataset_root)

    thai_panel = _make_thailand_panel(owid, doeb, nesdc)

    status = {
        "world_bank_prices": not wb_prices.empty,
        "world_bank_subsidy_price_control": not wb_policy.empty,
        "owid": not owid.empty,
        "doeb_any": any(not df.empty for df in doeb.values()),
        "nesdc_any": any(not df.empty for df in nesdc.values()),
        "thailand_panel_ready": not thai_panel.empty,
    }

    return {
        "paths": {
            "dataset_root": str(dataset_root),
            "world_bank_prices": str(wb_prices_path),
            "world_bank_subsidy_price_control": str(wb_policy_path),
            "owid": str(owid_path),
        },
        "status": status,
        "world_bank_prices": wb_prices,
        "world_bank_policy": wb_policy,
        "owid": owid,
        "doeb": doeb,
        "nesdc": nesdc,
        "thailand_panel": thai_panel,
    }
