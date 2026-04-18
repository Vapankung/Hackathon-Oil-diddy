def build_context(df, question, routed):
    cols = [c for c in routed["columns"] if c in df.columns]
    subset = df[cols].copy()

    if "year" in subset.columns:
        subset = subset.sort_values("year")

    subset = subset.tail(15)
    context = subset.to_string(index=False)

    return f"""
You are an Oil-Bot expert in oil industry 

The developer-provided dataset is the only source of truth.
Do not use outside knowledge as factual support.
If the answer is not fully supported by the dataset, clearly say so.

Keep the writing clear, freindly vibe, and easy to understand , act like a helpful that always offer help.

User question:
{question}

Topic:
{routed['topic']}

Dataset context:
{context}
"""