import os
import json
from google import genai


def ask_gemini(question, context):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = f"""
You are an AI assistant for Thailand oil and energy statistics.

Use the previous conversation when it is relevant.
Use the developer-provided dataset as the factual source of truth.
Do not invent dataset facts that are not supported by the provided context.
If the answer is not supported by the data, say so clearly.

{context}

Current user question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


def extract_graph_request(question, available_columns):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = f"""
You convert user graph requests into JSON instructions.

Available dataset columns:
{available_columns}

Return ONLY valid JSON.

Allowed chart_type:
- line
- bar
- scatter

Allowed comparison_mode:
- none
- multi_series
- compare_years

Allowed layout:
- single
- grouped

Allowed colors:
Any CSS color names if user explicitly requests them, otherwise empty list.

JSON format:
{{
  "make_graph": true,
  "chart_type": "line",
  "x_col": "year",
  "y_cols": ["oil_consumption"],
  "title": "Oil Consumption Over Time",
  "start_year": null,
  "end_year": null,
  "years": [],
  "comparison_mode": "none",
  "layout": "single",
  "colors": []
}}

Rules:
- Use ONLY columns from available dataset columns.
- If user asks for a graph, set make_graph = true.
- If user does not ask for a graph, return make_graph = false.
- If user mentions a year range like 2011 to 2023, fill start_year and end_year.
- If user mentions exact comparison years like 2011 and 2023, fill years = [2011, 2023].
- If user wants two variables on one graph, use y_cols with both columns.
- If user wants side-by-side comparison of years, use comparison_mode = "compare_years".
- If user wants both series in one graph, use layout = "grouped" for bar or "single" for line.
- If user mentions colors, include them in order.
- Do not add markdown fences.
- Do not explain anything.

User request:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    text = response.text.strip().replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(text)
        return {
            "make_graph": parsed.get("make_graph", False),
            "chart_type": parsed.get("chart_type", "line"),
            "x_col": parsed.get("x_col", ""),
            "y_cols": parsed.get("y_cols", []),
            "title": parsed.get("title", "Generated Graph"),
            "start_year": parsed.get("start_year"),
            "end_year": parsed.get("end_year"),
            "years": parsed.get("years", []),
            "comparison_mode": parsed.get("comparison_mode", "none"),
            "layout": parsed.get("layout", "single"),
            "colors": parsed.get("colors", []),
        }
    except Exception:
        return {
            "make_graph": False,
            "chart_type": "line",
            "x_col": "",
            "y_cols": [],
            "title": "",
            "start_year": None,
            "end_year": None,
            "years": [],
            "comparison_mode": "none",
            "layout": "single",
            "colors": [],
        }