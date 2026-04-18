from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from utils.data_loader import load_data
from utils.router import route_question
from utils.analysis import build_context
from utils.gemini_client import ask_gemini, extract_graph_request
from utils.graphing import make_graph, normalize_graph_spec, detect_column_aliases

load_dotenv()

# Load dataset once
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "owid-energy-data(clean).csv"
df_chat = load_data(str(DATA_PATH))

# Runtime memory only
chat_history: List[Dict[str, str]] = []


def extract_year_range_fallback(question: str) -> Tuple[Optional[int], Optional[int]]:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", question)
    years = [int(y) for y in years]

    if len(years) >= 2:
        return min(years), max(years)
    if len(years) == 1:
        return years[0], years[0]
    return None, None


def extract_explicit_years(question: str) -> List[int]:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", question)
    years = [int(y) for y in years]
    return sorted(list(dict.fromkeys(years)))


def get_recent_history_text(limit: int = 10) -> str:
    if not chat_history:
        return "No previous conversation."

    recent = chat_history[-limit:]
    lines: List[str] = []
    for item in recent:
        role = item.get("role", "unknown").capitalize()
        content = item.get("content", "")
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def clear_chat_history() -> None:
    global chat_history
    chat_history = []


def get_chat_history() -> List[Dict[str, str]]:
    return chat_history.copy()


def _question_wants_graph(question: str) -> bool:
    q = question.lower()

    graph_keywords = [
        "graph",
        "plot",
        "chart",
        "visualize",
        "draw",
        "bar chart",
        "line graph",
        "scatter",
        "compare visually",
        "same graph",
        "two graph",
        "side by side",
        "orange",
        "blue",
        "กราฟ",
        "พล็อต",
        "แผนภูมิ",
        "แสดงกราฟ",
        "วาดกราฟ",
    ]

    return any(word in q for word in graph_keywords)


def run_chatbot(question: str) -> Dict[str, Any]:
    global df_chat, chat_history

    if df_chat is None:
        return {
            "answer": "Dataset failed to load.",
            "topic": None,
            "columns_used": [],
            "graph_url": None,
            "mode": "error",
        }

    question = question.strip()
    if not question:
        return {
            "answer": "Please enter a question.",
            "topic": None,
            "columns_used": [],
            "graph_url": None,
            "mode": "error",
        }

    # Save user message into memory
    chat_history.append({
        "role": "user",
        "content": question
    })

    wants_graph = _question_wants_graph(question)

    if wants_graph:
        available_columns = list(df_chat.columns)

        spec = extract_graph_request(question, available_columns)
        spec = normalize_graph_spec(spec, available_columns)

        if not spec.get("y_cols"):
            alias_cols = detect_column_aliases(question, available_columns)
            if alias_cols:
                spec["y_cols"] = alias_cols

        if not spec.get("x_col") and "year" in df_chat.columns:
            spec["x_col"] = "year"

        if not spec.get("years"):
            explicit_years = extract_explicit_years(question)
            if len(explicit_years) == 2 and any(
                word in question.lower() for word in ["and", "compare", "represent", "เทียบ", "เปรียบเทียบ"]
            ):
                spec["years"] = explicit_years

        if spec.get("start_year") is None and spec.get("end_year") is None:
            fy_start, fy_end = extract_year_range_fallback(question)
            if fy_start is not None and fy_end is not None and not spec.get("years"):
                spec["start_year"] = fy_start
                spec["end_year"] = fy_end

        if spec.get("make_graph"):
            x_col = spec.get("x_col")
            y_cols = spec.get("y_cols", [])

            if x_col not in df_chat.columns or not y_cols:
                answer = "I understood that you want a graph, but I could not match the requested columns to the dataset."
                chat_history.append({"role": "assistant", "content": answer})
                return {
                    "answer": answer,
                    "topic": None,
                    "columns_used": [],
                    "graph_url": None,
                    "mode": "graph",
                    "graph_spec": spec,
                }

            needed_cols = [x_col] + y_cols
            subset = df_chat[needed_cols].dropna().copy()

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
                return {
                    "answer": answer,
                    "topic": None,
                    "columns_used": needed_cols,
                    "graph_url": None,
                    "mode": "graph",
                    "graph_spec": spec,
                }

            try:
                graph_url = make_graph(subset, spec)
            except Exception as e:
                answer = f"I understood the graph request, but I could not build that graph format yet. Details: {str(e)}"
                chat_history.append({"role": "assistant", "content": answer})
                return {
                    "answer": answer,
                    "topic": None,
                    "columns_used": needed_cols,
                    "graph_url": None,
                    "mode": "graph",
                    "graph_spec": spec,
                }

            answer = f"Here is your {spec.get('chart_type', 'line')} graph for {spec.get('title', 'Generated Graph')}."
            chat_history.append({"role": "assistant", "content": answer})

            return {
                "answer": answer,
                "topic": None,
                "columns_used": needed_cols,
                "graph_url": graph_url,
                "mode": "graph",
                "graph_spec": spec,
            }

    # Normal QA mode with memory
    routed = route_question(question)
    context = build_context(df_chat, question, routed)
    history_text = get_recent_history_text(limit=10)

    full_context = f"""
Previous conversation:
{history_text}

{context}
"""

    answer = ask_gemini(question, full_context)

    chat_history.append({
        "role": "assistant",
        "content": answer
    })

    return {
        "answer": answer,
        "topic": routed.get("topic"),
        "columns_used": routed.get("columns", []),
        "graph_url": None,
        "mode": "qa",
    }