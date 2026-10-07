"""Environment probe for the document-ingestion pipeline.

Reports which optional libraries and external tools are available so you can
tell at a glance what ingest.py / index.py can do in this environment. Nothing
here is required to run the probe itself - it only inspects, never installs.

Run:
    python probe.py
Output is printed and also written to probe.txt.
"""

import importlib.util
import os
import shutil
import sys

# (import name, pip package, what it enables)
LIBRARIES = [
    ("pdfplumber", "pdfplumber", "PDF text extraction"),
    ("openpyxl", "openpyxl", "Excel (.xlsx) extraction"),
    ("docx", "python-docx", "Word (.docx) extraction"),
    ("PIL", "Pillow", "image handling"),
    ("easyocr", "easyocr", "image OCR (current engine)"),
    ("torch", "torch", "EasyOCR backend"),
    ("pytesseract", "pytesseract", "legacy OCR wrapper (optional)"),
    ("groq", "groq", "AI keywords via Groq"),
    ("google.genai", "google-genai", "AI keywords via Gemini"),
    ("dotenv", "python-dotenv", "loading a .env file"),
    ("googleapiclient", "google-api-python-client", "Google Drive sync"),
    ("google.oauth2", "google-auth", "Google auth"),
    ("google_auth_oauthlib", "google-auth-oauthlib", "Google Drive OAuth flow"),
    ("customtkinter", "customtkinter", "search GUI (index.py)"),
]

# External (non-pip) tools worth knowing about.
EXECUTABLES = [
    ("tesseract", "Tesseract OCR engine (only needed for the legacy pytesseract path)"),
]

# Environment variables the pipeline looks for.
ENV_VARS = ["GROQ_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]

# Credential files Google Drive sync looks for (see ingest.py).
CRED_FILES = ["service_account.json", "credentials.json", "token.json"]


def has_module(name):
    """True if an import spec exists for `name` (supports dotted names)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def main():
    lines = []
    lines.append(f"Python: {sys.version.splitlines()[0]}")
    lines.append(f"Executable: {sys.executable}")
    lines.append(f"Working dir: {os.getcwd()}")
    lines.append("")

    lines.append("Libraries")
    lines.append("-" * 60)
    missing = []
    for import_name, pip_name, purpose in LIBRARIES:
        ok = has_module(import_name)
        mark = "OK " if ok else "-- "
        lines.append(f"  [{mark}] {import_name:<22} {purpose}")
        if not ok:
            missing.append(pip_name)

    lines.append("")
    lines.append("External tools")
    lines.append("-" * 60)
    for exe, purpose in EXECUTABLES:
        present = shutil.which(exe) is not None
        mark = "OK " if present else "-- "
        lines.append(f"  [{mark}] {exe:<22} {purpose}")

    lines.append("")
    lines.append("API keys (environment)")
    lines.append("-" * 60)
    any_key = False
    for var in ENV_VARS:
        present = bool(os.environ.get(var))
        any_key = any_key or present
        mark = "set" if present else "unset"
        lines.append(f"  {var:<16} {mark}")
    lines.append(
        f"  -> keyword extraction will use: {'AI (LLM)' if any_key else 'local fallback'}"
    )

    lines.append("")
    lines.append("Google Drive credential files")
    lines.append("-" * 60)
    for name in CRED_FILES:
        present = os.path.exists(name)
        mark = "found" if present else "missing"
        lines.append(f"  {name:<24} {mark}")

    if missing:
        lines.append("")
        lines.append("To enable everything that's missing:")
        lines.append("  pip install " + " ".join(sorted(set(missing))))

    report = "\n".join(lines)
    print(report)
    with open("probe.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
