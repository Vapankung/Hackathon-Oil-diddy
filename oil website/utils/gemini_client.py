import os
from google import genai

def ask_gemini(question, context):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = f"""{context}

Respond to the user's question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text