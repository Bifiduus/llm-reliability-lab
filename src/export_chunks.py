import json
from pathlib import Path
from dataclasses import asdict

from reliability_lab.ingestion import ingest_directory


def main():
    input_dir = Path("data/documents")
    output_path = Path("data/benchmark/chunks.json")

    chunks = ingest_directory(input_dir)

    data = [asdict(chunk) for chunk in chunks]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"{len(chunks)} chunks exported to {output_path}")


if __name__ == "__main__":
    main()
