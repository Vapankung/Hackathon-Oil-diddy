from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import re

from utils.data_loader import load_data
from utils.router import route_question
from utils.analysis import build_context
from utils.gemini_client import ask_gemini, extract_graph_request
from utils.graphing import make_graph, normalize_graph_spec, detect_column_aliases

load_dotenv()

app = Flask(__name__)

df = load_data("data/owid-energy-data(clean).csv")

# Runtime memory only: cleared when app stops
chat_history = []


def extract_year_range_fallback(question):
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", question)
    years = [int(y) for y in years]

    if len(years) >= 2:
        return min(years), max(years)
    if len(years) == 1:
        return years[0], years[0]
    return None, None


def extract_explicit_years(question):
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", question)
    years = [int(y) for y in years]
    return sorted(list(dict.fromkeys(years)))


def get_recent_history_text(limit=10):
    if not chat_history:
        return "No previous conversation."

    recent = chat_history[-limit:]
    lines = []
    for item in recent:
        role = item["role"].capitalize()
        content = item["content"]
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    global df, chat_history

    if df is None:
        return jsonify({"answer": "Dataset failed to load."}), 500

    data = request.get_json()
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"answer": "Please enter a question."}), 400

    # Save user message into memory
    chat_history.append({
        "role": "user",
        "content": question
    })

    graph_keywords = [
        "graph", "plot", "chart", "visualize", "draw",
        "bar chart", "line graph", "scatter",
        "compare visually", "same graph", "two graph",
        "side by side", "orange", "blue"
    ]
    wants_graph = any(word in question.lower() for word in graph_keywords)

    if wants_graph:
        available_columns = list(df.columns)
        spec = extract_graph_request(question, available_columns)
        spec = normalize_graph_spec(spec, available_columns)

        if not spec["y_cols"]:
            alias_cols = detect_column_aliases(question, available_columns)
            if alias_cols:
                spec["y_cols"] = alias_cols

        if not spec["x_col"] and "year" in df.columns:
            spec["x_col"] = "year"

        if not spec.get("years"):
            explicit_years = extract_explicit_years(question)
            if len(explicit_years) == 2 and any(word in question.lower() for word in ["and", "compare", "represent"]):
                spec["years"] = explicit_years

        if spec.get("start_year") is None and spec.get("end_year") is None:
            fy_start, fy_end = extract_year_range_fallback(question)
            if fy_start is not None and fy_end is not None and not spec.get("years"):
                spec["start_year"] = fy_start
                spec["end_year"] = fy_end

        if spec.get("make_graph"):
            x_col = spec.get("x_col")
            y_cols = spec.get("y_cols", [])

            if x_col not in df.columns or not y_cols:
                answer = "I understood that you want a graph, but I could not match the requested columns to the dataset."
                chat_history.append({"role": "assistant", "content": answer})
                return jsonify({"answer": answer})

            needed_cols = [x_col] + y_cols
            subset = df[needed_cols].dropna().copy()

            if x_col == "year" and spec.get("years"):
                years = spec["years"]
                subset = subset[subset["year"].isin(years)].copy()
                spec["comparison_mode"] = "compare_years"

            elif x_col == "year":
                if spec.get("start_year") is not None:
                    subset = subset[subset["year"] >= spec["start_year"]]
                if spec.get("end_year") is not None:
                    subset = subset[subset["year"] <= spec["end_year"]]

            if x_col in subset.columns:
                try:
                    subset = subset.sort_values(x_col)
                except Exception:
                    pass

            if subset.empty:
                answer = "I could not generate a graph because there is no usable data for that request."
                chat_history.append({"role": "assistant", "content": answer})
                return jsonify({"answer": answer})

            try:
                graph_url = make_graph(subset, spec)
            except Exception as e:
                answer = f"I understood the graph request, but I could not build that graph format yet. Details: {str(e)}"
                chat_history.append({"role": "assistant", "content": answer})
                return jsonify({"answer": answer})

            answer = f"Here is your **{spec.get('chart_type', 'line')} graph** for **{spec.get('title', 'Generated Graph')}**."
            chat_history.append({"role": "assistant", "content": answer})

            return jsonify({
                "answer": answer,
                "graph_url": graph_url
            })

    # Normal QA mode with memory
    routed = route_question(question)
    context = build_context(df, question, routed)
    history_text = get_recent_history_text(limit=10)

    full_context = f"""
Previous conversation:
{history_text}

{context}
"""

    answer = ask_gemini(question, full_context)

    # Save assistant response into memory
    chat_history.append({
        "role": "assistant",
        "content": answer
    })

    return jsonify({
        "answer": answer,
        "topic": routed["topic"],
        "columns_used": routed["columns"]
    })


@app.route("/memory", methods=["GET"])
def memory():
    return jsonify({
        "history": chat_history
    })


@app.route("/clear-memory", methods=["POST"])
def clear_memory():
    global chat_history
    chat_history = []
    return jsonify({"message": "Memory cleared."})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)