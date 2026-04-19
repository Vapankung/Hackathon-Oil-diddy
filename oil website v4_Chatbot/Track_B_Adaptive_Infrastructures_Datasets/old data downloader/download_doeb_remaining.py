from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

FILES = {
    "DOEB_Imported_Refined_Oil_Prices_2566.xlsx": "https://data.doeb.go.th/dataset/7aa42839-cfe4-4383-a1ed-f85fb42ff9ad/resource/515eb811-873b-4862-93b5-f90b5b6f68dc/download/34-2566.xlsx",
    "DOEB_Imported_Crude_Oil_Prices_2566.xlsx": "https://data.doeb.go.th/dataset/d28d9622-91ad-468f-adaa-b275cf57d336/resource/e87efa41-6b36-413b-9728-b4dac3a022ee/download/35-2566.xlsx",
    "DOEB_Fuel_Supply_and_Distribution_Report_2565.pdf": "https://data.doeb.go.th/dataset/78091a3d-840a-4d68-8aa5-c0de1d70e9e1/resource/b6cbb7eb-9b17-4327-afb3-425425c3ae58/download/2.report_year-2565.pdf",
    "DOEB_Fuel_Supply_and_Distribution_Report_2564.pdf": "https://data.doeb.go.th/dataset/78091a3d-840a-4d68-8aa5-c0de1d70e9e1/resource/7d78321c-bcfd-4f8d-bde5-8f1879985b65/download/2.report_year-2564.pdf",
    "DOEB_Gasoline_Sales_Monthly_2568.xlsx": "https://data.doeb.go.th/dataset/a332e7c8-0bab-4639-a3e7-ad06a5cfa910/resource/0c4d7ca4-46c4-4dc1-96f8-35cb288a6d45/download/untitled.xlsx",
    "DOEB_Gasoline_Sales_Monthly_2564.csv": "https://data.doeb.go.th/dataset/a332e7c8-0bab-4639-a3e7-ad06a5cfa910/resource/4d83b0ab-69e9-46f6-814b-edcc1a07f6db/download/003_diesel_ulg_2564.csv",
}

def download(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as r:
        dest.write_bytes(r.read())

outdir = Path(__file__).resolve().parent / 'doeb_downloads'
outdir.mkdir(exist_ok=True)

for filename, url in FILES.items():
    path = outdir / filename
    try:
        print(f"Downloading {filename} ...")
        download(url, path)
        print(f"Saved: {path}")
    except HTTPError as e:
        print(f"HTTP error for {filename}: {e.code} {e.reason}")
    except URLError as e:
        print(f"URL error for {filename}: {e.reason}")
    except Exception as e:
        print(f"Failed {filename}: {e}")

print("Done.")
