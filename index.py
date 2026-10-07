#!/usr/bin/env python3
"""
PySearch GUI (CustomTkinter) - keyword search + Q&A over your LOCAL catalog.

Searches data.csv, the catalog produced by ingest.py from the files in the
information/ folder. Each row is a document with a one-sentence summary,
extracted keywords and a text excerpt.

Tabs:
  - Ask       : ask a natural-language question; get an answer grounded in the
                documents, with clickable source links
  - Documents : ranked search results; click a result to open the file
  - By Type   : results grouped by document type (PDF, Spreadsheet, Image, ...)
  - Keywords  : the most common keywords across matching documents
  - SIMPLE    : a short plain-text answer
  - ADVANCED  : a detailed breakdown by type, keyword themes and recency

Shared logic (catalog load/search, LLM access, tokenizing) lives in core.py.

Install:   pip install customtkinter
           (Linux only, if tkinter is missing: sudo apt install python3-tk)
Run:       python index.py
"""
from __future__ import annotations

import collections
import os
import queue
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import customtkinter as ctk

import core
from core import (
    INFORMATION_DIR,
    human_size,
    llm_complete,
    load_catalog,
    resolve_llm,
    search_catalog,
    top_terms,
    trunc,
)


# ----------------------------------------------------------------------------- file opening
def resolve_path(row: dict, root: str = INFORMATION_DIR) -> str:
    """Absolute path to the underlying file, for opening it."""
    rel = row.get("relative_path") or row.get("file_name") or ""
    return os.path.abspath(os.path.join(root, rel))


def open_file(row: dict) -> bool:
    """Open the document with the OS default application."""
    path = resolve_path(row)
    if not os.path.exists(path):
        return False
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return True
    except Exception:
        try:
            webbrowser.open(Path(path).as_uri())  # correct file:// URI, handles spaces
            return True
        except Exception:
            return False


# ----------------------------------------------------------------------------- Q&A over the catalog
def answer_question(catalog: list[dict], question: str, top_k: int = 5):
    """Answer a question grounded in the catalog.

    Returns (answer_text, sources). Uses the LLM over each source's text
    excerpt when a key is configured; otherwise an extractive fallback built
    from summaries. Sources are always returned for clickable linking.
    """
    results = search_catalog(catalog, question)
    if not results:
        return (
            "I couldn't find anything in your documents related to that question.\n"
            "Try different wording, or add more files and re-run ingest.py.",
            [],
        )
    sources = results[:top_k]

    if resolve_llm():
        answer = _answer_via_llm(question, sources)
        if answer:
            return answer, sources

    # Extractive fallback: summary (or excerpt) from each top document.
    lines = ["Based on your documents, here is what is most relevant:", ""]
    for i, r in enumerate(sources, 1):
        snippet = r.get("summary") or r.get("text_excerpt") or "(no text available)"
        lines.append(f"[{i}] {r.get('file_name', '')}: {trunc(snippet, 240)}")
    lines += ["", "Open the sources below to read the full documents."]
    return "\n".join(lines), sources


def _answer_via_llm(question: str, sources: list[dict]) -> str:
    """Compose a grounded answer from the sources' text excerpts via the LLM."""
    context = "\n\n".join(
        f"[{i}] {r.get('file_name','')}:\n"
        f"{r.get('text_excerpt') or r.get('summary') or ''}"
        for i, r in enumerate(sources, 1)
    )
    prompt = (
        "Answer the question using ONLY the document excerpts below. Give a "
        "concise answer (2-5 sentences). Cite the documents you used by their "
        "[number]. If the excerpts don't contain the answer, say so plainly.\n\n"
        f"QUESTION: {question}\n\nDOCUMENTS:\n{context[:12000]}"
    )
    return llm_complete(prompt, max_tokens=400)


# ----------------------------------------------------------------------------- SIMPLE / ADVANCED answers
def build_simple(query: str, results: list[dict], total: int) -> str:
    if not results:
        if total == 0:
            return (
                "No catalog found.\n\n"
                "data.csv is empty or missing. Add files to the "
                f"'{INFORMATION_DIR}/' folder and run:  python ingest.py"
            )
        return f'No documents matched "{query}". Try a different keyword.'

    types = collections.Counter(r.get("doc_type", "Other") for r in results)
    top_type, cnt = types.most_common(1)[0]
    header = "all documents" if not query.strip() else f'"{query}"'
    lines = [f"SIMPLE ANSWER for {header}", "=" * 60, ""]
    for i, r in enumerate(results[:5], 1):
        lines.append(f"{i}. {r.get('file_name', '')}  ({r.get('doc_type', '')})")
        lines.append(f"   {trunc(r.get('summary', ''), 160)}")
        if r.get("_kw_list"):
            lines.append(f"   keywords: {', '.join(r['_kw_list'][:6])}")
    lines += [
        "",
        f"Quick take: {len(results)} matching document(s); most are "
        f"'{top_type}' ({cnt}). Open #1 to start, or use the 'By Type' and "
        "'Keywords' tabs to explore.",
    ]
    return "\n".join(lines)


def build_advanced(query: str, results: list[dict], total: int) -> str:
    if not results:
        return build_simple(query, results, total)

    header = "all documents" if not query.strip() else f'"{query}"'
    groups = collections.defaultdict(list)
    for r in results:
        groups[r.get("doc_type", "Other")].append(r)

    L = [f"ADVANCED ANSWER for {header}", "=" * 60, "", "DOCUMENTS BY TYPE", "-" * 60]
    for t, xs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        themes = ", ".join(top_terms([f"{x.get('summary','')} {x.get('keywords','')}" for x in xs])) or "n/a"
        L.append(f"\n## {t}  ({len(xs)} document(s)) - recurring terms: {themes}")
        for x in xs[:4]:
            L.append(f"  • {x.get('file_name','')}  [{human_size(x.get('size_bytes'))}"
                     f"{' · ' + x['_date_s'] if x.get('_date_s') else ''}]")
            L.append(f"    {trunc(x.get('summary',''), 200)}")
            if x.get("_kw_list"):
                L.append(f"    keywords: {', '.join(x['_kw_list'][:8])}")

    kw_counter = collections.Counter()
    for r in results:
        kw_counter.update(k.lower() for k in r.get("_kw_list", []))
    L += ["", "TOP KEYWORDS ACROSS MATCHES", "-" * 60]
    if kw_counter:
        L.append(", ".join(f"{k} ({c})" for k, c in kw_counter.most_common(15)))
    else:
        L.append("No keywords recorded. Re-run: python ingest.py --rebuild")

    dated = [r for r in results if r.get("_dt")]
    L += ["", "CATALOG FACTS", "-" * 60]
    L.append(f"Matching documents: {len(results)} of {total} in catalog")
    if dated:
        newest = max(dated, key=lambda r: r["_dt"])
        oldest = min(dated, key=lambda r: r["_dt"])
        L.append(f"Newest: {newest.get('file_name','')} ({newest['_date_s']})")
        L.append(f"Oldest: {oldest.get('file_name','')} ({oldest['_date_s']})")
    sizes = []
    for r in results:
        try:
            sizes.append(float(r.get("size_bytes")))
        except (TypeError, ValueError):
            pass
    if sizes:
        L.append(f"Total size: {human_size(sum(sizes))} · largest {human_size(max(sizes))}")
    return "\n".join(L)


# ----------------------------------------------------------------------------- GUI (CustomTkinter)
# Colours are (light_mode, dark_mode) tuples so contrast holds in both themes.
LINK = ("#1a0dab", "#8ab4f8")
PATH_GREEN = ("#1a7f37", "#81c995")
MUTED = ("#5f6368", "#9aa0a6")
BODY = ("#1f1f1f", "#e8eaed")
WRAP = 880


def hand(widget):
    for target in (widget, getattr(widget, "_label", None), getattr(widget, "_canvas", None)):
        if target is not None:
            try:
                target.configure(cursor="hand2")
            except Exception:
                pass


def clickable(parent, text, callback, size=15, color=LINK):
    lb = ctk.CTkLabel(parent, text=text, text_color=color, anchor="w", justify="left", wraplength=WRAP,
                      font=ctk.CTkFont(size=size, underline=True))
    lb.bind("<Button-1>", lambda e: callback())
    hand(lb)
    return lb


def label(parent, text, size=13, color=None, bold=False, wrap=WRAP, **kw):
    return ctk.CTkLabel(parent, text=text, text_color=color or BODY, anchor="w", justify="left", wraplength=wrap,
                        font=ctk.CTkFont(size=size, weight="bold" if bold else "normal"), **kw)


def clear(frame):
    for w in frame.winfo_children():
        w.destroy()


class App:
    # Cap how many result cards we build at once, so a huge catalog stays snappy.
    MAX_CARDS = 200

    def __init__(self, root):
        self.root = root
        root.title("PySearch - local document catalog (data.csv)")
        root.geometry("1150x820")
        root.minsize(820, 520)
        self.q = queue.Queue()

        self.catalog = load_catalog()

        # --- top bar
        top = ctk.CTkFrame(root, fg_color="transparent")
        top.pack(fill="x", padx=12, pady=(12, 4))
        self.entry = ctk.CTkEntry(top, height=40, font=ctk.CTkFont(size=16),
                                  placeholder_text="Search your documents by keyword, name or topic…")
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda e: self.start())
        ctk.CTkLabel(top, text="Max:").pack(side="left", padx=(12, 4))
        self.n = ctk.StringVar(value="All")
        ctk.CTkOptionMenu(top, values=["10", "25", "50", "All"], variable=self.n, width=80, height=40).pack(side="left")
        ctk.CTkButton(top, text="Search", width=90, height=40, command=self.start,
                      font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=(12, 0))
        ctk.CTkButton(top, text="Re-ingest", width=100, height=40, command=self.reingest).pack(side="left", padx=(8, 0))
        # Plain-button theme toggle: we own the boolean, so the appearance and
        # the button label always stay in sync (no switch-variable quirks).
        self.is_dark = False
        ctk.set_appearance_mode("Light")
        self.theme_btn = ctk.CTkButton(top, text="Dark mode", width=100, height=40,
                                       command=self.toggle_theme)
        self.theme_btn.pack(side="left", padx=(8, 0))

        self.status = ctk.CTkLabel(root, text="", anchor="w", text_color=MUTED)
        self.status.pack(fill="x", padx=16)

        # --- tabs
        self.tabs = ctk.CTkTabview(root)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=(4, 12))
        self.frames, self.texts = {}, {}

        # Ask tab: a question box, an answer, and clickable source links.
        ask_tab = self.tabs.add("Ask")
        ask_bar = ctk.CTkFrame(ask_tab, fg_color="transparent")
        ask_bar.pack(fill="x", padx=4, pady=(4, 6))
        self.ask_entry = ctk.CTkEntry(ask_bar, height=38, font=ctk.CTkFont(size=15),
                                      placeholder_text="Ask a question about your documents…")
        self.ask_entry.pack(side="left", fill="x", expand=True)
        self.ask_entry.bind("<Return>", lambda e: self.ask())
        ctk.CTkButton(ask_bar, text="Ask", width=90, height=38, command=self.ask,
                      font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=(8, 0))
        self.answer_box = ctk.CTkTextbox(ask_tab, wrap="word", height=200, font=ctk.CTkFont(size=14))
        self.answer_box.pack(fill="x", padx=4, pady=(0, 6))
        self.answer_box.configure(state="disabled")
        label(ask_tab, "Sources", 14, bold=True).pack(anchor="w", padx=6, pady=(2, 0))
        self.ask_sources = ctk.CTkScrollableFrame(ask_tab, fg_color="transparent")
        self.ask_sources.pack(fill="both", expand=True, padx=2)

        for name in ("Documents", "By Type", "Keywords"):
            tab = self.tabs.add(name)
            f = ctk.CTkScrollableFrame(tab, fg_color="transparent")
            f.pack(fill="both", expand=True)
            self.frames[name] = f
        for name in ("SIMPLE", "ADVANCED"):
            tab = self.tabs.add(name)
            t = ctk.CTkTextbox(tab, wrap="word", font=ctk.CTkFont(family="Consolas", size=13))
            t.pack(fill="both", expand=True)
            t.configure(state="disabled")
            self.texts[name] = t
        self.tabs.set("Documents")

        root.protocol("WM_DELETE_WINDOW", self.close)
        self._after = None
        self.poll()

        if not self.catalog:
            self.status.configure(
                text=f"No catalog loaded. Add files to '{INFORMATION_DIR}/' and run: python ingest.py"
            )
        else:
            self.status.configure(text=f"{len(self.catalog)} documents in catalog. Enter a keyword or press Search to list all.")
            self.render("", search_catalog(self.catalog, ""))

    # --- plumbing
    def toggle_theme(self):
        self.is_dark = not self.is_dark
        ctk.set_appearance_mode("Dark" if self.is_dark else "Light")
        self.theme_btn.configure(text="Light mode" if self.is_dark else "Dark mode")

    def reload(self):
        self.catalog = load_catalog()
        self.status.configure(text=f"Reloaded: {len(self.catalog)} documents in catalog.")
        self.start()

    def reingest(self):
        """Run ingest.py in a worker thread, then reload the catalog."""
        self.status.configure(text="Re-ingesting… (this can take a while on first OCR)")
        threading.Thread(target=self._run_reingest, daemon=True).start()

    def _run_reingest(self):
        try:
            import ingest

            ingest.ingest()
            self.q.put(lambda: self.reload())
        except Exception as e:  # keep the GUI alive
            self.q.put(lambda: self.status.configure(text=f"Re-ingest failed: {e}"))

    def close(self):
        if self._after:
            self.root.after_cancel(self._after)
        self.root.destroy()

    def poll(self):
        try:
            while True:
                self.q.get_nowait()()
        except queue.Empty:
            pass
        self._after = self.root.after(100, self.poll)

    def put(self, name, text):
        w = self.texts[name]
        w.configure(state="normal")
        w.delete("1.0", "end")
        w.insert("end", text)
        w.configure(state="disabled")

    def _limit(self):
        v = self.n.get()
        return None if v == "All" else int(v)

    def start(self):
        if not self.catalog:
            self.status.configure(text=f"No catalog. Run: python ingest.py (reads {INFORMATION_DIR}/).")
            return
        kw = self.entry.get().strip()
        threading.Thread(target=self.run, args=(kw, self._limit()), daemon=True).start()

    def run(self, kw, limit):
        results = search_catalog(self.catalog, kw, limit)
        self.q.put(lambda: self.render(kw, results))

    # --- Q&A
    def ask(self):
        if not self.catalog:
            self._set_answer(f"No catalog. Run: python ingest.py (reads {INFORMATION_DIR}/).", [])
            return
        question = self.ask_entry.get().strip()
        if not question:
            return
        self.tabs.set("Ask")
        self._set_answer("Thinking…", [])
        threading.Thread(target=self._run_ask, args=(question,), daemon=True).start()

    def _run_ask(self, question):
        answer, sources = answer_question(self.catalog, question)
        self.q.put(lambda: self._set_answer(answer, sources))

    def _set_answer(self, text, sources):
        self.answer_box.configure(state="normal")
        self.answer_box.delete("1.0", "end")
        self.answer_box.insert("end", text)
        self.answer_box.configure(state="disabled")
        clear(self.ask_sources)
        if not sources:
            label(self.ask_sources, "No sources.", 12, MUTED).pack(anchor="w", padx=6, pady=6)
            return
        for i, r in enumerate(sources, 1):
            self._doc_card(self.ask_sources, r, prefix=f"[{i}] ")

    # --- rendering
    def render(self, kw, results):
        total = len(self.catalog)
        header = "all documents" if not kw else f'"{kw}"'
        shown = min(len(results), self.MAX_CARDS)
        extra = f" (showing first {shown})" if len(results) > self.MAX_CARDS else ""
        self.status.configure(text=f"{len(results)} result(s) for {header} · {total} in catalog{extra}")
        for f in self.frames.values():
            clear(f)
        self.render_documents(results[: self.MAX_CARDS])
        self.render_by_type(results[: self.MAX_CARDS])
        self.render_keywords(results)
        self.put("SIMPLE", build_simple(kw, results, total))
        self.put("ADVANCED", build_advanced(kw, results, total))
        self.tabs.set("Documents")

    def _doc_card(self, parent, r, prefix=""):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", anchor="w", padx=8, pady=6)
        meta_top = " · ".join(x for x in (r.get("doc_type", ""), human_size(r.get("size_bytes")), r.get("_date_s", "")) if x)
        label(f, meta_top, 11, PATH_GREEN).pack(anchor="w")
        clickable(f, prefix + r.get("file_name", "(unnamed)"), lambda row=r: self._open(row)).pack(anchor="w")
        if r.get("summary"):
            label(f, trunc(r["summary"], 300), 13).pack(anchor="w")
        if r.get("_kw_list"):
            label(f, "keywords: " + ", ".join(r["_kw_list"][:8]), 11, MUTED).pack(anchor="w")
        if r.get("status") and r["status"] != "ok":
            label(f, f"status: {r['status']}", 11, MUTED).pack(anchor="w")

    def _open(self, row):
        if not open_file(row):
            self.status.configure(text=f"Could not open: {resolve_path(row)}")

    def render_documents(self, results):
        p = self.frames["Documents"]
        if not results:
            label(p, "No matching documents.", 14).pack(pady=20)
            return
        for r in results:
            self._doc_card(p, r)

    def render_by_type(self, results):
        p = self.frames["By Type"]
        if not results:
            label(p, "No matching documents.", 14).pack(pady=20)
            return
        groups = collections.defaultdict(list)
        for r in results:
            groups[r.get("doc_type", "Other")].append(r)
        for t, xs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            label(p, f"{t}  ({len(xs)})", 16, bold=True).pack(anchor="w", padx=8, pady=(14, 2))
            for r in xs:
                self._doc_card(p, r)

    def render_keywords(self, results):
        p = self.frames["Keywords"]
        counter = collections.Counter()
        for r in results:
            for k in r.get("_kw_list", []):
                counter[k.lower()] += 1
        if not counter:
            label(p, "No keywords recorded. Run: python ingest.py --rebuild", 14).pack(pady=20)
            return
        label(p, "Click a keyword to filter the search to it.", 12, MUTED).pack(anchor="w", padx=8, pady=(6, 8))
        for kw, c in counter.most_common(60):
            row = ctk.CTkFrame(p, fg_color="transparent")
            row.pack(fill="x", anchor="w", padx=8, pady=2)
            clickable(row, f"{kw}  ({c})", lambda k=kw: self._search_for(k), size=14).pack(side="left")

    def _search_for(self, kw):
        self.entry.delete(0, "end")
        self.entry.insert(0, kw)
        self.start()


if __name__ == "__main__":
    ctk.set_appearance_mode("Light")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    App(root)
    root.mainloop()
