import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    if "country" in df.columns:
        df = df[df["country"].astype(str).str.lower() == "thailand"].copy()

    return df