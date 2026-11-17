from dataclasses import dataclass
from pathlib import Path
import pymupdf


@dataclass(frozen=True)
class Chunk:
    id: str
    document: str
    page: int
    text: str


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((page_number, text))

    return pages


def split_text(text: str, chunk_size: int = 200,
               overlap: int = 40) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be bewteen 0 and chunk_size - 1")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= len(words):
            break

    return chunks


def ingest_pdf(pdf_path: Path, chunk_size: int = 200,
               overlap: int = 40) -> list[Chunk]:
    chunks = []

    pages = extract_pdf_pages(pdf_path)

    for page_number, text in pages:
        page_chunks = split_text(text, chunk_size, overlap)

        for index, chunk_text in enumerate(page_chunks):
            chunk = Chunk(
                id=f"{pdf_path.stem}_p{page_number}_c{index}",
                document=pdf_path.name,
                page=page_number,
                text=chunk_text
            )
            chunks.append(chunk)

    return chunks


def ingest_directory(directory: Path, chunk_size: int = 200,
                     overlap: int = 40) -> list[Chunk]:
    chunks = []

    for pdf_path in sorted(directory.glob("*.pdf")):
        pdf_chunks = ingest_pdf(pdf_path, chunk_size, overlap)
        chunks.extend(pdf_chunks)

    return chunks
