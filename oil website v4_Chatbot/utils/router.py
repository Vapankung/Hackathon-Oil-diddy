def route_question(question):
    q = question.lower()

    wants_graph = any(word in q for word in ["graph", "chart", "plot", "visualize"])

    if any(word in q for word in ["oil", "petroleum"]):
        return {
            "topic": "oil_energy",
            "columns": ["year", "country", "oil_consumption", "oil_share_energy", "gdp", "population"],
            "graph": wants_graph,
            "graph_x": "year",
            "graph_y": "oil_consumption",
            "graph_title": "Thailand Oil Consumption Over Time"
        }

    if any(word in q for word in ["electricity", "power"]):
        return {
            "topic": "electricity_mix",
            "columns": ["year", "country", "electricity_generation"],
            "graph": wants_graph,
            "graph_x": "year",
            "graph_y": "electricity_generation",
            "graph_title": "Thailand Electricity Generation Over Time"
        }

    if any(word in q for word in ["renewable", "solar", "wind"]):
        return {
            "topic": "renewables",
            "columns": ["year", "country", "renewables_share_energy"],
            "graph": wants_graph,
            "graph_x": "year",
            "graph_y": "renewables_share_energy",
            "graph_title": "Thailand Renewables Share Over Time"
        }

    if any(word in q for word in ["gdp", "economy", "economic"]):
        return {
            "topic": "macro_economy",
            "columns": ["year", "country", "gdp", "population"],
            "graph": wants_graph,
            "graph_x": "year",
            "graph_y": "gdp",
            "graph_title": "Thailand GDP Over Time"
        }

    return {
        "topic": "general",
        "columns": ["year", "country", "gdp", "population", "oil_consumption"],
        "graph": wants_graph,
        "graph_x": "year",
        "graph_y": "oil_consumption",
        "graph_title": "Thailand Oil Consumption Over Time"
    }