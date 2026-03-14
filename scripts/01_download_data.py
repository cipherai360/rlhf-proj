from pathlib import Path
from datasets import load_dataset


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Spider dataset from Hugging Face...")
    spider = load_dataset("spider")

    print("\nDataset loaded successfully.")
    print(f"Train split size: {len(spider['train'])}")
    print(f"Validation split size: {len(spider['validation'])}")

    sample = spider["train"][0]
    print("\nSample keys:")
    print(list(sample.keys()))

    print("\nSample example:")
    print(f"db_id: {sample['db_id']}")
    print(f"question: {sample['question']}")
    print(f"query: {sample['query']}")

    marker_file = raw_dir / "spider_download_check.txt"
    marker_file.write_text(
        "Spider dataset loaded successfully via Hugging Face.\n",
        encoding="utf-8"
    )

    print(f"\nCreated marker file: {marker_file}")
    print("Step 2A complete.")


if __name__ == "__main__":
    main()