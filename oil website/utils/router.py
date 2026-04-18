def route_question(question):
    q = question.lower()

    if any(word in q for word in ["oil", "petroleum"]):
        return {
            "topic": "oil_energy",
            "columns": ["year", "country", "oil_consumption", "oil_share_energy", "gdp", "population"]
        }

    if any(word in q for word in ["electricity", "power"]):
        return {
            "topic": "electricity_mix",
            "columns": ["year", "country", "electricity_generation"]
        }

    if any(word in q for word in ["renewable", "solar", "wind"]):
        return {
            "topic": "renewables",
            "columns": ["year", "country", "renewables_share_energy"]
        }

    return {
        "topic": "general",
        "columns": ["year", "country", "gdp", "population"]
    }