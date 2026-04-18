from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
load_dotenv()

from utils.data_loader import load_data
from utils.router import route_question
from utils.analysis import build_context
from utils.gemini_client import ask_gemini

app = Flask(__name__)

df = load_data("data/owid-energy-data(clean).csv")
print(type(df))
print(df.head())

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "").strip()

    if not question:
        return jsonify({"answer": "Please enter a question."}), 400

    routed = route_question(question)
    context = build_context(df, question, routed)
    answer = ask_gemini(question, context)

    return jsonify({
        "answer": answer,
        "topic": routed["topic"],
        "columns_used": routed["columns"]
    })

if __name__ == "__main__":
    app.run(debug=True)