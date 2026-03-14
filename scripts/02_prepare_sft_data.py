import json
import random
from pathlib import Path
from typing import Any


SEED = 42
SFT_RATIO = 0.75


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_schema_lookup(tables_json: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Build a lookup: db_id -> schema metadata
    """
    lookup = {}
    for db in tables_json:
        lookup[db["db_id"]] = db
    return lookup


def build_schema_string(schema: dict[str, Any]) -> str:
    """
    Convert Spider tables.json schema into readable text.
    Includes tables, columns, primary keys, and foreign keys.
    """
    db_id = schema["db_id"]
    table_names = schema["table_names_original"]
    column_names = schema["column_names_original"]   # list of [table_idx, column_name]
    column_types = schema["column_types"]
    primary_keys = set(schema["primary_keys"])
    foreign_keys = schema["foreign_keys"]            # list of [src_col_idx, tgt_col_idx]

    lines = []
    lines.append(f"Database: {db_id}")
    lines.append("Tables:")

    for table_idx, table_name in enumerate(table_names):
        cols = []
        for col_idx, ((tbl_id, col_name), col_type) in enumerate(zip(column_names, column_types)):
            if tbl_id == table_idx and col_name != "*":
                marker = " [PK]" if col_idx in primary_keys else ""
                cols.append(f"{col_name} ({col_type}){marker}")
        lines.append(f"- {table_name}: " + ", ".join(cols))

    if foreign_keys:
        lines.append("Foreign Keys:")
        for src_idx, tgt_idx in foreign_keys:
            src_tbl_idx, src_col_name = column_names[src_idx]
            tgt_tbl_idx, tgt_col_name = column_names[tgt_idx]

            src_table = table_names[src_tbl_idx] if src_tbl_idx != -1 else "UNKNOWN"
            tgt_table = table_names[tgt_tbl_idx] if tgt_tbl_idx != -1 else "UNKNOWN"

            lines.append(f"- {src_table}.{src_col_name} -> {tgt_table}.{tgt_col_name}")

    return "\n".join(lines)


def build_prompt(schema_text: str, question: str) -> str:
    return f"""You are an NL2SQL assistant.
Generate a safe SQL query that answers the question.

Rules:
- Output only SQL
- Use only SELECT statements
- Do not use DROP, DELETE, INSERT, UPDATE, ALTER, TRUNCATE, CREATE, REPLACE
- Use only the tables and columns shown in the schema

Schema:
{schema_text}

Question:
{question}

SQL:"""


def format_example(
    example: dict[str, Any],
    schema_lookup: dict[str, dict[str, Any]],
    split_name: str,
    idx: int,
) -> dict[str, Any]:
    db_id = example["db_id"]
    schema = schema_lookup[db_id]
    schema_text = build_schema_string(schema)
    question = example["question"]
    gold_sql = example["query"]
    prompt = build_prompt(schema_text, question)

    return {
        "id": f"spider_{split_name}_{idx:06d}",
        "source": "spider",
        "db_id": db_id,
        "question": question,
        "schema_text": schema_text,
        "prompt": prompt,
        "gold_sql": gold_sql,
        "text": prompt + "\n" + gold_sql,
    }


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    random.seed(SEED)

    root = Path(__file__).resolve().parents[1]
    spider_dir = root / "data" / "raw" / "spider"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    tables_path = spider_dir / "tables.json"
    train_path = spider_dir / "train_spider.json"
    dev_path = spider_dir / "dev.json"

    if not tables_path.exists():
        raise FileNotFoundError(f"Missing file: {tables_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not dev_path.exists():
        raise FileNotFoundError(f"Missing file: {dev_path}")

    print("Loading Spider local files...")
    tables_data = load_json(tables_path)
    train_data = load_json(train_path)
    dev_data = load_json(dev_path)

    print(f"Loaded tables.json: {len(tables_data)} schemas")
    print(f"Loaded train_spider.json: {len(train_data)} examples")
    print(f"Loaded dev.json: {len(dev_data)} examples")

    schema_lookup = build_schema_lookup(tables_data)

    print("\nFormatting train examples...")
    formatted_train = [
        format_example(ex, schema_lookup, "train", i)
        for i, ex in enumerate(train_data)
    ]

    print("Formatting dev examples...")
    formatted_dev = [
        format_example(ex, schema_lookup, "dev", i)
        for i, ex in enumerate(dev_data)
    ]

    indices = list(range(len(formatted_train)))
    random.shuffle(indices)

    sft_size = int(SFT_RATIO * len(formatted_train))
    sft_idx = set(indices[:sft_size])

    sft_train = [formatted_train[i] for i in range(len(formatted_train)) if i in sft_idx]
    rlhf_train = [formatted_train[i] for i in range(len(formatted_train)) if i not in sft_idx]
    eval_set = formatted_dev

    sft_path = processed_dir / "sft_train.jsonl"
    rlhf_path = processed_dir / "rlhf_train.jsonl"
    eval_path = processed_dir / "eval.jsonl"

    save_jsonl(sft_path, sft_train)
    save_jsonl(rlhf_path, rlhf_train)
    save_jsonl(eval_path, eval_set)

    print("\nSaved files:")
    print(f"- {sft_path} ({len(sft_train)} rows)")
    print(f"- {rlhf_path} ({len(rlhf_train)} rows)")
    print(f"- {eval_path} ({len(eval_set)} rows)")

    print("\nSample processed example:")
    print(json.dumps(sft_train[0], indent=2)[:2000])

    print("\nStep 2B complete.")


if __name__ == "__main__":
    main()