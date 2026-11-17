from pathlib import Path
from urllib.request import urlretrieve

DOCUMENTS = {
    "attention_is_all_you_need.pdf": "1706.03762",
    "rag.pdf": "2005.11401",
    "dense_retrieval.pdf": "2004.04906",
    "sentence_bert.pdf": "1908.10084",
    "lost_in_the_middle.pdf": "2307.03172",
}


def main():
    output_dir = Path("data/documents")
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, arxiv_id in DOCUMENTS.items():
        output_path = output_dir / filename

        if output_path.exists():
            print(f"Already exists: {filename}")
            continue

        url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"Downloading {filename}...")

        try:
            urlretrieve(url, output_path)

            # Verify that the downloaded file is a PDF
            with output_path.open("rb") as f:
                if f.read(5) != b"%PDF-":
                    raise ValueError("Downloaded file is not a PDF")

        except Exception:
            output_path.unlink(missing_ok=True)
            raise

    print("All documents downloaded.")


if __name__ == "__main__":
    main()
