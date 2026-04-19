#!/usr/bin/env python3
"""
Download all dataset resource files reachable from the provided DOEB search/listing page.

Updated to target: 
Crude Oil (น้ำมันดิบ) search results.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

# Updated to your new search URL
DEFAULT_URL = (
    "https://data.doeb.go.th/organization/trade_data"
    "?q=%E0%B8%99%E0%B9%89%E0%B8%B3%E0%B8%A1%E0%B8%B1%E0%B8%99%E0%B8%94%E0%B8%B4%E0%B8%9A"
    "&sort=score+desc%2C+metadata_modified+desc"
)
BASE_URL = "https://data.doeb.go.th"
TIMEOUT = 60

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "th-TH,th;q=0.9,en;q=0.8",
}


def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def safe_name(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|\r\n]+", "_", name).strip()
    return name or "unnamed"


def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    resp = session.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def extract_dataset_links(soup: BeautifulSoup, start_url: str) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []

    # Primary check: Standard CKAN dataset blocks
    for block in soup.select("div.dataset-content, li.dataset-item"):
        a = block.select_one("h2.dataset-heading a[href], h3.dataset-heading a[href], a[href*='/dataset/']")
        if not a:
            continue
        href = a.get("href", "").strip()
        if not href:
            continue
        full_url = urljoin(start_url, href)
        if full_url in seen:
            continue
        seen.add(full_url)
        
        title = " ".join(a.get_text(" ", strip=True).split())
        desc = ""
        # Try to find a description snippet nearby
        desc_el = block.select_one(".dataset-description, div:nth-of-type(2)")
        if desc_el:
            desc = " ".join(desc_el.get_text(" ", strip=True).split())
            
        out.append({"title": title, "dataset_url": full_url, "description": desc})

    if out:
        return out

    # Fallback: Scrape any link that looks like a dataset page
    # Updated to be locale-agnostic (removes strict /th/ requirement)
    for a in soup.select('a[href*="/dataset/"]'):
        href = a.get("href", "").strip()
        if not href or "/groups/" in href or "/organizations/" in href:
            continue
        full_url = urljoin(start_url, href)
        if full_url in seen:
            continue
        text = " ".join(a.get_text(" ", strip=True).split())
        if not text:
            continue
        seen.add(full_url)
        out.append({"title": text, "dataset_url": full_url, "description": ""})

    return out


def extract_download_links(dataset_soup: BeautifulSoup, dataset_url: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()

    # Look for the specific analytics-tracked download buttons
    for a in dataset_soup.select("a.resource-url-analytics[href], a.dropdown-item[href*='/download/']"):
        href = a.get("href", "").strip()
        if not href:
            continue
        full_url = urljoin(dataset_url, href)
        if full_url in seen:
            continue
        seen.add(full_url)

        label = ""
        # Try to find the resource title (usually in a parent list item)
        parent = a.find_parent(["li", "div", "section"], class_=re.compile("resource-item|resource-list"))
        if parent:
            label_el = parent.select_one(".heading, a.title")
            if label_el:
                label = label_el.get_text(strip=True)
        
        if not label:
            label = " ".join(a.get_text(" ", strip=True).split())

        filename = infer_filename_from_url(full_url)
        items.append(
            {
                "download_url": full_url,
                "resource_label": label,
                "filename": filename,
            }
        )

    return items


def infer_filename_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    name = Path(path).name.strip()
    # Clean up common URL params if they got stuck to the filename
    name = name.split('?')[0].split('#')[0]
    return safe_name(name or "downloaded_file")


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 2
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def download_file(session: requests.Session, url: str, dest: Path) -> tuple[int | None, int]:
    with session.get(url, stream=True, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        written = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 128):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
        return resp.status_code, written


def write_manifest_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    rows = list(rows)
    fieldnames = [
        "dataset_title",
        "dataset_url",
        "dataset_description",
        "resource_label",
        "download_url",
        "saved_path",
        "http_status",
        "bytes",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="First page URL to scrape")
    parser.add_argument("--output", default="doeb_crude_oil_downloads", help="Output folder")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests")
    args = parser.parse_args()

    out_dir = Path(args.output).resolve()
    files_dir = out_dir / "files"
    out_dir.mkdir(parents=True, exist_ok=True)
    files_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()

    print(f"Opening start page: {args.url}")
    try:
        start_soup = get_soup(session, args.url)
    except Exception as e:
        print(f"ERROR: cannot open start page: {e}", file=sys.stderr)
        return 1

    datasets = extract_dataset_links(start_soup, args.url)
    print(f"Found {len(datasets)} dataset links on the results page")
    
    if not datasets:
        print("No dataset links found. Check if the search returned results.", file=sys.stderr)
        return 2

    manifest: list[dict[str, object]] = []

    for i, ds in enumerate(datasets, start=1):
        title = ds["title"]
        dataset_url = ds["dataset_url"]
        description = ds["description"]
        print(f"[{i}/{len(datasets)}] Processing: {title}")

        try:
            time.sleep(args.delay)
            soup = get_soup(session, dataset_url)
            resources = extract_download_links(soup, dataset_url)
        except Exception as e:
            manifest.append({
                "dataset_title": title, "dataset_url": dataset_url, "error": f"Page error: {e}",
                "dataset_description": description, "resource_label": "", "download_url": "", 
                "saved_path": "", "http_status": "", "bytes": 0
            })
            print(f"    ! Failed to read dataset page: {e}")
            continue

        if not resources:
            print("    ! No download links found on this page")
            continue

        # Create a subfolder for each dataset to keep files organized
        dataset_folder = files_dir / safe_name(title)
        dataset_folder.mkdir(parents=True, exist_ok=True)

        for resource in resources:
            dl_url = resource["download_url"]
            filename = resource["filename"]
            dest = unique_path(dataset_folder / filename)
            
            print(f"    -> Downloading: {filename}")
            try:
                time.sleep(args.delay)
                status, size = download_file(session, dl_url, dest)
                err = ""
            except Exception as e:
                status, size, err = None, 0, str(e)
                print(f"       ! Download failed: {e}")

            manifest.append({
                "dataset_title": title, "dataset_url": dataset_url, "dataset_description": description,
                "resource_label": resource["resource_label"], "download_url": dl_url,
                "saved_path": str(dest) if not err else "", "http_status": status or "",
                "bytes": size, "error": err
            })

    # Save manifest files
    csv_path = out_dir / "manifest.csv"
    json_path = out_dir / "manifest.json"
    write_manifest_csv(manifest, csv_path)
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nFinished! Files saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())