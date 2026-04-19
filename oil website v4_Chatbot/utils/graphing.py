import os
import uuid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_graph(df, spec):
    os.makedirs("static/graphs", exist_ok=True)

    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join("static", "graphs", filename)

    chart_type = spec.get("chart_type", "line")
    x_col = spec.get("x_col")
    y_cols = spec.get("y_cols", [])
    title = spec.get("title", "Generated Graph")
    comparison_mode = spec.get("comparison_mode", "none")
    layout = spec.get("layout", "single")
    colors = spec.get("colors", [])

    plt.figure(figsize=(8.5, 4.8))

    if comparison_mode == "compare_years":
        _make_compare_years_graph(df, x_col, y_cols, title, chart_type, layout, colors)
    else:
        _make_standard_graph(df, x_col, y_cols, title, chart_type, colors)

    plt.tight_layout()
    plt.savefig(filepath, dpi=140, bbox_inches="tight")
    plt.close()

    return f"/static/graphs/{filename}"


def _make_standard_graph(df, x_col, y_cols, title, chart_type, colors):
    for i, y in enumerate(y_cols):
        color = colors[i] if i < len(colors) else None

        if chart_type == "line":
            plt.plot(df[x_col], df[y], marker="o", linewidth=2, markersize=4, label=y, color=color)

        elif chart_type == "bar":
            plt.bar(df[x_col], df[y], label=y, color=color)

        elif chart_type == "scatter":
            plt.scatter(df[x_col], df[y], label=y, color=color)

        else:
            plt.plot(df[x_col], df[y], marker="o", linewidth=2, markersize=4, label=y, color=color)

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(", ".join(y_cols))
    if len(y_cols) > 1:
        plt.legend()
    plt.grid(alpha=0.25)


def _make_compare_years_graph(df, x_col, y_cols, title, chart_type, layout, colors):
    if x_col != "year":
        raise ValueError("compare_years mode currently requires x_col='year'")

    if not y_cols:
        raise ValueError("No y columns provided")

    years = sorted(df["year"].unique().tolist())
    if len(years) != 2:
        raise ValueError("compare_years mode requires exactly two years after filtering")

    metrics = y_cols
    year_1, year_2 = years

    row1 = df[df["year"] == year_1].iloc[0]
    row2 = df[df["year"] == year_2].iloc[0]

    vals1 = [row1[m] for m in metrics]
    vals2 = [row2[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    c1 = colors[0] if len(colors) > 0 else None
    c2 = colors[1] if len(colors) > 1 else None

    if chart_type == "bar":
        plt.bar(x - width / 2, vals1, width, label=str(year_1), color=c1)
        plt.bar(x + width / 2, vals2, width, label=str(year_2), color=c2)
        plt.xticks(x, metrics, rotation=15)

    elif chart_type == "line":
        plt.plot(metrics, vals1, marker="o", linewidth=2, label=str(year_1), color=c1)
        plt.plot(metrics, vals2, marker="o", linewidth=2, label=str(year_2), color=c2)

    elif chart_type == "scatter":
        plt.scatter(metrics, vals1, label=str(year_1), color=c1)
        plt.scatter(metrics, vals2, label=str(year_2), color=c2)

    else:
        plt.bar(x - width / 2, vals1, width, label=str(year_1), color=c1)
        plt.bar(x + width / 2, vals2, width, label=str(year_2), color=c2)
        plt.xticks(x, metrics, rotation=15)

    plt.title(title)
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(alpha=0.25)


def normalize_graph_spec(spec, df_columns):
    available = set(df_columns)

    x_col = spec.get("x_col", "")
    y_cols = [c for c in spec.get("y_cols", []) if c in available]

    spec["x_col"] = x_col if x_col in available else ""
    spec["y_cols"] = y_cols

    return spec


def detect_column_aliases(question, df_columns):
    q = question.lower()
    mapping = {
        "oil consumption": "oil_consumption",
        "oil use": "oil_consumption",
        "renewable share": "renewables_share_energy",
        "renewables share": "renewables_share_energy",
        "electricity generation": "electricity_generation",
        "population": "population",
        "gdp": "gdp",
        "oil share": "oil_share_energy",
    }

    found = []
    for phrase, col in mapping.items():
        if phrase in q and col in df_columns:
            found.append(col)

    return list(dict.fromkeys(found))