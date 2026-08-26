#!/usr/bin/env python3
"""Convert Drive PDFs to SARA training corpus.

Extracts text from every PDF in a folder, cleans it, and writes:
  - data/drive_corpus.txt   (one doc per line-ish plain text)
  - stats printed at the end
"""
import sys
from pathlib import Path

import fitz  # pymupdf

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/sara_drive_data")
OUT = Path("/root/SARA/data/drive_corpus.txt")

MIN_CHARS_PER_PAGE = 200   # skip scanned/image-only pages
MAX_CHARS_PER_DOC = 500_000

total_docs, total_chars, skipped = 0, 0, []
parts = []
for pdf in sorted(SRC.glob("*.pdf")):
    try:
        doc = fitz.open(pdf)
        pages = []
        for page in doc:
            t = page.get_text("text").strip()
            if len(t) >= MIN_CHARS_PER_PAGE:
                pages.append(t)
        doc.close()
        text = "\n\n".join(pages)[:MAX_CHARS_PER_DOC]
        if len(text) < 1000:
            skipped.append(pdf.name)
            continue
        parts.append(text)
        total_chars += len(text)
        total_docs += 1
        print(f"ok {pdf.name}: {len(pages)}p {len(text)//1000}k chars")
    except Exception as e:
        skipped.append(f"{pdf.name} ({e})")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n\n<|doc|>\n\n".join(parts), encoding="utf-8")
print(f"\n=== {total_docs} docs, {total_chars/1e6:.1f}M chars -> {OUT}")
if skipped:
    print(f"skipped {len(skipped)}:")
    for s in skipped:
        print("  -", s)
