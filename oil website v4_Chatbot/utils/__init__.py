from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from utils.data_loader import load_data
from utils.router import route_question
from utils.analysis import build_context
from utils.gemini_client import ask_gemini

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "owid-energy-data(clean).csv"

df_chat = load_data(str(DATA_PATH))

def run_chatbot(question: str):
    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "topic": None,
            "columns_used": []
        }

    routed = route_question(question)
    context = build_context(df_chat, question, routed)
    answer = ask_gemini(question, context)

    return {
        "answer": answer,
        "topic": routed["topic"],
        "columns_used": routed["columns"]
    }