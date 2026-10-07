"""Shared core for the document-catalog pipeline.

This module is the single source of truth for things that ingest.py, index.py
and extract_data.py all need:

  - Project paths (resolved relative to this file, so tools work from any CWD)
  - The data.csv schema (FIELDNAMES)
  - Stopwords + tokenization
  - LLM resolution (Groq / Gemini) with a cached client
  - Catalog loading, relevance scoring and ranked search

Keeping these here removes the duplication that previously existed between
ingest.py and index.py (two copies of the LLM resolver, two stopword lists,
two tokenizers, two search implementations).
"""

from __future__ import annotations

import csv
import datetime
import os
import re

# --- Paths -----------------------------------------------------------------
# Resolve everything against this file's directory so the tools behave the
# same no matter what the current working directory is.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFORMATION_DIR = os.path.join(BASE_DIR, "information")
CSV_FILE = os.path.join(BASE_DIR, "data.csv")

# --- Schema ----------------------------------------------------------------
FIELDNAMES = [
    "doc_id",
    "file_name",
    "relative_path",
    "doc_type",
    "size_bytes",
    "modified",
    "status",
    "summary",
    "keywords",
    "text_excerpt",   # trimmed full text, used to ground Q&A answers
]

DATE_FMT = "%Y-%m-%d %H:%M:%S"

# --- LLM models (central so both producers/consumers agree) ----------------
GROQ_MODEL = "llama-3.1-8b-instant"
GEMINI_MODEL = "gemini-1.5-flash"

# --- Stopwords & tokenization ----------------------------------------------
STOPWORDS = set(
    """
    a an the and or but if then else for to of in on at by with from as is are was were
    be been being this that these those it its it's into over under about above below
    not no nor so than too very can will just don should now i you he she we they them
    me my your our their his her out up down off again further once here there all any
    both each few more most other some such only own same out which who whom what
    when where why how do does did doing have has had having would could may might must
    page version date name number total based using use used shall per via etc vs best
    guide
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (len >= 3) with stopwords removed."""
    return [w for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower()) if w not in STOPWORDS]


def trunc(s: str, n: int) -> str:
    """Collapse whitespace and truncate to n chars with an ellipsis."""
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def top_terms(texts, exclude=(), k: int = 8) -> list[str]:
    """The k most frequent content tokens across the given texts."""
    import collections

    c = collections.Counter(w for t in texts for w in tokenize(t) if w not in exclude)
    return [w for w, _ in c.most_common(k)]


def clean_ocr(text: str) -> str:
    """Repair the common OCR artefact of doubled letters (e.g. 'RReeggiisstteerr').

    Many stylized PDFs/scans produce each glyph twice. When a token is made
    almost entirely of adjacent duplicated letters, collapse the pairs.
    """
    if not text:
        return text

    def fix_token(tok: str) -> str:
        if len(tok) < 6 or not tok.isalpha():
            return tok
        # If collapsing adjacent duplicate letters roughly halves the token,
        # it was almost certainly doubled by the OCR engine - use the collapse.
        collapsed = re.sub(r"(.)\1+", r"\1", tok)
        if len(collapsed) <= len(tok) * 0.7:
            return collapsed
        return tok

    return " ".join(fix_token(t) for t in text.split())


# --- LLM resolution --------------------------------------------------------
_LLM = None  # cached (provider, client) or False once resolved


def resolve_llm():
    """Return (provider, client) if an API key is configured, else None.

    Loads a .env file if python-dotenv is present. Supports Groq and Google
    Gemini. Cached so the client is built once per process.
    """
    global _LLM
    if _LLM is not None:
        return _LLM or None

    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(BASE_DIR, ".env"))
    except Exception:
        pass

    groq_key = os.environ.get("GROQ_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if groq_key:
        try:
            from groq import Groq

            _LLM = ("groq", Groq(api_key=groq_key))
            return _LLM
        except Exception:
            pass

    if google_key:
        try:
            from google import genai

            _LLM = ("gemini", genai.Client(api_key=google_key))
            return _LLM
        except Exception:
            pass

    _LLM = False
    return None


def llm_complete(prompt: str, max_tokens: int = 300) -> str:
    """Run a single prompt through the configured LLM. '' if none/failure."""
    llm = resolve_llm()
    if not llm:
        return ""
    provider, client = llm
    try:
        if provider == "groq":
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (getattr(resp, "text", "") or "").strip()
    except Exception:
        return ""


# --- Date / size helpers ---------------------------------------------------
def parse_dt(v):
    if not v:
        return None
    try:
        return datetime.datetime.strptime(str(v), DATE_FMT)
    except Exception:
        try:
            return datetime.datetime.fromisoformat(str(v))
        except Exception:
            return None


def fmt_dt(d) -> str:
    return d.strftime("%Y-%m-%d") if d else ""


def human_size(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return ""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


# --- Catalog load / search -------------------------------------------------
def load_catalog(path: str = CSV_FILE) -> list[dict]:
    """Read data.csv into document dicts. Returns [] if missing/locked/empty."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        # Locked (open in Excel), partially written, or corrupt - fail soft.
        return []
    for r in rows:
        r["_dt"] = parse_dt(r.get("modified"))
        r["_date_s"] = fmt_dt(r["_dt"])
        r["_kw_list"] = [k.strip() for k in (r.get("keywords") or "").split(",") if k.strip()]
    return rows


def score(row: dict, query_terms: list[str], now: datetime.datetime | None = None) -> float:
    """Relevance of a document for query terms.

    Field weighting: keywords > file name > summary > excerpt, plus a mild
    recency bonus. An empty query gives every document a base score of 1.
    """
    now = now or datetime.datetime.now()
    if not query_terms:
        base = 1.0
    else:
        name = (row.get("file_name") or "").lower()
        summary = (row.get("summary") or "").lower()
        excerpt = (row.get("text_excerpt") or "").lower()
        kws = [k.lower() for k in row.get("_kw_list", [])]
        kw_blob = " ".join(kws)

        s = 0.0
        for term in query_terms:
            if term in kws:
                s += 12
            elif term in kw_blob:
                s += 6
            if term in name:
                s += 8
            if term in summary:
                s += 4
            if term in excerpt:
                s += 2
        if s == 0:
            return 0.0
        base = s

    dt = row.get("_dt")
    if dt:
        base += max(0.0, 5 - (now - dt).days / 90)
    return base


def search_catalog(catalog: list[dict], query: str, limit: int | None = None) -> list[dict]:
    """Rank catalog rows against the query. Empty query lists everything."""
    terms = tokenize(query)
    now = datetime.datetime.now()
    scored = []
    for row in catalog:
        sc = score(row, terms, now)
        if sc > 0:
            r = dict(row)
            r["_score"] = sc
            scored.append(r)
    scored.sort(key=lambda r: (-r["_score"], r.get("file_name", "")))
    return scored[:limit] if limit else scored
