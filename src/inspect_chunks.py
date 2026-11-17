from pathlib import Path

from reliability_lab.ingestion import ingest_directory

chunks = ingest_directory(Path("data/documents"))

keyword = "recurrence"

for chunk in chunks:
    if keyword.lower() in chunk.text.lower():
        print(f"\nID: {chunk.id}")
        print(f"Document: {chunk.document}")
        print(f"Page: {chunk.page}")
        print(f"Text:\n{chunk.text}")
        print("-" * 80)
