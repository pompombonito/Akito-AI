import csv
import re
import shutil
from pathlib import Path
import os
import pymupdf
import pytesseract

from PIL import Image


DOCUMENTS_FOLDER = Path("documents")

TESSERACT_PATH = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
if not TESSERACT_PATH:
    bundled_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if bundled_path.is_file():
        TESSERACT_PATH = str(bundled_path)
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_normal_text(page):
    """
    Extract selectable text directly from a PDF page.
    """
    text = page.get_text()

    return text.strip()


def extract_ocr_text(page):
    """
    Convert a PDF page into an image and run Tesseract OCR.
    """

    if not TESSERACT_PATH:
        print("Tesseract is not installed; scanned page text is unavailable.")
        return ""

    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        alpha=False
    )

    image = Image.frombytes(
        "RGB",
        [pixmap.width, pixmap.height],
        pixmap.samples
    )

    try:
        text = pytesseract.image_to_string(image)
    except (pytesseract.TesseractNotFoundError, OSError, RuntimeError) as exc:
        print(f"OCR skipped ({exc})")
        text = ""

    return text.strip()


def read_pdf(file_path):
    """
    Read a PDF page-by-page.

    Normal text is extracted directly.

    If little/no text is found, OCR is used.
    """

    document = pymupdf.open(file_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        print(f"  Processing page {page_number}...", end=" ")

        text = extract_normal_text(page)

        if len(text) < 50:

            print("OCR")

            try:
                text = extract_ocr_text(page)
            except (pytesseract.TesseractNotFoundError, OSError, RuntimeError) as exc:
                print(f"OCR skipped ({exc})")
                text = ""

        else:

            print("text")

        if text:

            pages.append({
                "page": page_number,
                "text": text
            })

    document.close()

    return pages


def read_csv(file_path, max_rows=20000, max_chars=600000):
    """Turn a bounded CSV sample into labeled rows so local retrieval can cite it."""
    pages = []
    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            headers = [str(name or "").strip() for name in (reader.fieldnames or [])[:60]]
            if not headers:
                return pages
            lines = ["CSV columns: " + ", ".join(headers)]
            used = len(lines[0])
            for row_number, row in enumerate(reader, start=1):
                if row_number > max_rows:
                    lines.append(f"[Indexing stopped after {max_rows} data rows.]")
                    break
                values = [f"{name}: {str(row.get(name, '')).strip()}" for name in headers if str(row.get(name, "")).strip()]
                if not values:
                    continue
                line = f"Row {row_number}: " + " | ".join(values)
                if used + len(line) + 1 > max_chars:
                    lines.append("[Indexing stopped at the text-size limit; upload a smaller or filtered CSV for full retrieval.]")
                    break
                lines.append(line)
                used += len(line) + 1
    except (OSError, UnicodeError, csv.Error) as exc:
        print(f"Skipping unreadable CSV {file_path}: {exc}")
        return pages
    text = "\n".join(lines).strip()
    if text:
        pages.append({"page": 1, "text": text})
    return pages


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Split text into smaller overlapping chunks.

    chunk_size:
        Approximate number of characters per chunk.

    overlap:
        Number of characters repeated between chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# CREATE DOCUMENT CHUNKS
# ============================================================

def create_chunks(documents):

    all_chunks = []

    for document in documents:

        filename = document["filename"]

        for page in document["pages"]:

            page_number = page["page"]
            text = page["text"]

            chunks = chunk_text(text)

            for chunk_number, chunk in enumerate(chunks, start=1):

                all_chunks.append({
                    "filename": filename,
                    "page": page_number,
                    "chunk": chunk_number,
                    "text": chunk,
                    "source_url": document.get("source_url", ""),
                })

    return all_chunks


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    documents = []

    if not DOCUMENTS_FOLDER.exists():

        print(f"ERROR: Folder '{DOCUMENTS_FOLDER}' does not exist.")

        return documents

    for file_path in sorted(DOCUMENTS_FOLDER.rglob("*")):
        if not file_path.is_file():
            continue
        filename = str(file_path.relative_to(DOCUMENTS_FOLDER))

        if file_path.suffix.lower() == ".pdf":
            print()
            print("=" * 60)
            print(f"Reading: {filename}")
            print("=" * 60)

            try:
                pages = read_pdf(str(file_path))
            except Exception as exc:
                print(f"Skipping unreadable PDF {filename}: {exc}")
                continue

            documents.append({
                "filename": filename,
                "pages": pages
            })

        elif file_path.suffix.lower() in {".txt", ".md"}:
            try:
                text = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                print(f"Skipping {filename}: {exc}")
                continue
            if text.strip():
                source_match = re.search(r"^Source URL:\s*(\S+)", text, re.MULTILINE)
                documents.append({
                    "filename": filename,
                    "pages": [{"page": 1, "text": text.strip()}],
                    "source_url": source_match.group(1) if source_match else "",
                })

        elif file_path.suffix.lower() == ".csv":
            pages = read_csv(file_path)
            if pages:
                documents.append({"filename": filename, "pages": pages})

    return documents


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    documents = load_documents()

    chunks = create_chunks(documents)

    print()
    print("=" * 60)
    print("RAG CHUNKING TEST")
    print("=" * 60)

    print(f"PDF documents: {len(documents)}")
    print(f"Total chunks:  {len(chunks)}")

    if chunks:

        print()
        print("First chunk:")
        print("-" * 60)

        print(f"Source: {chunks[0]['filename']}")
        print(f"Page:   {chunks[0]['page']}")
        print(f"Chunk:  {chunks[0]['chunk']}")

        print()
        print(chunks[0]["text"])

        print("-" * 60)

        print()
        print("Second chunk:")

        if len(chunks) > 1:

            print("-" * 60)

            print(f"Source: {chunks[1]['filename']}")
            print(f"Page:   {chunks[1]['page']}")
            print(f"Chunk:  {chunks[1]['chunk']}")

            print()
            print(chunks[1]["text"])

            print("-" * 60)
