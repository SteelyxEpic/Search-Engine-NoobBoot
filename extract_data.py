#!/usr/bin/env python3
"""Command-line search over the document catalog in data.csv.

A terminal companion to the index.py GUI. Both share core.py, so the CLI uses
the exact same ranked search and relevance scoring as the app.

Usage:
  python extract_data.py                 # summarize the whole catalog
  python extract_data.py linear algebra  # ranked search for a query
  python extract_data.py -n 5 gaming     # limit to the top 5 results
"""
from __future__ import annotations

import argparse
import collections

from core import human_size, load_catalog, search_catalog, trunc


def summarize(rows: list[dict]) -> None:
    """Print an overview of the whole catalog."""
    if not rows:
        print("No documents found. Add files to information/ and run: python ingest.py")
        return
    print(f"Columns: {', '.join(rows[0].keys() - {'_dt', '_date_s', '_kw_list'})}")
    print(f"Documents: {len(rows)}")
    print("-" * 48)
    print("Count by document type:")
    counts = collections.Counter(r.get("doc_type", "Unknown") for r in rows)
    for doc_type, count in sorted(counts.items()):
        print(f"  {doc_type}: {count}")


def print_results(query: str, results: list[dict]) -> None:
    print(f'Search "{query}": {len(results)} match(es)')
    print("-" * 48)
    for i, r in enumerate(results, 1):
        meta = " · ".join(x for x in (r.get("doc_type", ""), human_size(r.get("size_bytes")), r.get("_date_s", "")) if x)
        print(f"{i}. {r.get('file_name', '')}  [{meta}]")
        if r.get("summary"):
            print(f"   {trunc(r['summary'], 180)}")
        if r.get("keywords"):
            print(f"   keywords: {r['keywords']}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Search the data.csv document catalog.")
    parser.add_argument("query", nargs="*", help="search terms (omit to summarize the catalog)")
    parser.add_argument("-n", "--limit", type=int, default=None, help="max results to show")
    args = parser.parse_args(argv)

    catalog = load_catalog()
    if not catalog:
        print("No catalog found. Add files to information/ and run: python ingest.py")
        return

    query = " ".join(args.query).strip()
    if not query:
        summarize(catalog)
        return

    results = search_catalog(catalog, query, limit=args.limit)
    print_results(query, results)


if __name__ == "__main__":
    main()
