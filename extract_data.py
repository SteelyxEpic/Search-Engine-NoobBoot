"""Read and search the document catalog in data.csv.

data.csv is produced by ingest.py, which scans the `information/` folder
(PDFs, Excel spreadsheets, Word docs, meeting minutes, and images via OCR)
and writes one row per file with a one-sentence summary and AI-extracted
keywords. This module reads that catalog back and offers a simple keyword
search over the summary, keywords, and metadata, turning the organizational
knowledge into a searchable resource.
"""

import csv
import os
import sys

CSV_FILE = "data.csv"


def extract_data(path=CSV_FILE):
    """Read the CSV file and return its rows as a list of dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}' in {os.getcwd()}. Run ingest.py first."
        )

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows):
    """Print a short summary of the extracted documents."""
    if not rows:
        print("No documents found. Add files to information/ and run ingest.py.")
        return

    headers = list(rows[0].keys())
    print(f"Columns ({len(headers)}): {', '.join(headers)}")
    print(f"Documents: {len(rows)}")

    print("-" * 40)
    print("Count by document type:")
    counts = {}
    for row in rows:
        doc_type = row.get("doc_type", "Unknown")
        counts[doc_type] = counts.get(doc_type, 0) + 1
    for doc_type, count in sorted(counts.items()):
        print(f"  {doc_type}: {count}")


def search(rows, keyword):
    """Return documents whose file name, type, summary, or keywords match."""
    keyword = keyword.lower()
    hits = []
    for row in rows:
        haystack = " ".join(
            [
                row.get("file_name", ""),
                row.get("doc_type", ""),
                row.get("summary", ""),
                row.get("keywords", ""),
            ]
        ).lower()
        if keyword in haystack:
            hits.append(row)
    return hits


def main():
    rows = extract_data()

    # Optional: `python extract_data.py <keyword>` to search the catalog.
    if len(sys.argv) > 1:
        keyword = " ".join(sys.argv[1:])
        hits = search(rows, keyword)
        print(f"Search '{keyword}': {len(hits)} match(es)")
        for r in hits:
            print(f"  [{r['doc_type']}] {r['file_name']}: {r.get('summary', '')}")
            if r.get("keywords"):
                print(f"      keywords: {r['keywords']}")
        return

    summarize(rows)


if __name__ == "__main__":
    main()
