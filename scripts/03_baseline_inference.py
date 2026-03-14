import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_EXAMPLES = 5
MAX_NEW_TOKENS = 128


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def clean_sql(output: str) -> str:
    """
    Remove explanations, markdown fences, and extra text.
    Return only the SQL statement.
    """

    # Remove markdown fences
    output = re.sub(r"```sql", "", output, flags=re.IGNORECASE)
    output = re.sub(r"```", "", output)

    # Keep text until first semicolon
    if ";" in output:
        output = output.split(";")[0] + ";"

    # Remove explanation sections
    output = re.split(r"\bExplanation\b", output, flags=re.IGNORECASE)[0]

    return output.strip()


def main():

    root = Path(__file__).resolve().parents[1]

    eval_path = root / "data" / "processed" / "eval.jsonl"
    output_dir = root / "evaluation"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "baseline_outputs.json"

    print("Loading evaluation dataset...")
    examples = load_jsonl(eval_path)[:NUM_EXAMPLES]

    print("Loading model:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )
    # Disable sampling parameters to silence warnings
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []

    for i, ex in enumerate(examples):

        print(f"\nExample {i+1}")

        prompt = ex["prompt"]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        predicted_sql = generated_text[len(prompt):].strip()
        predicted_sql = clean_sql(predicted_sql)

        print("Question:", ex["question"])
        print("Gold SQL:", ex["gold_sql"])
        print("Predicted:", predicted_sql)

        results.append({
            "id": ex["id"],
            "question": ex["question"],
            "gold_sql": ex["gold_sql"],
            "predicted_sql": predicted_sql
        })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved outputs to:", output_file)


if __name__ == "__main__":
    main()