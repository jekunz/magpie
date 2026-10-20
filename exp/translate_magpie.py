"""
Translate a Magpie English JSONL file into a target language using a Gemma instruction-tuned model.

Translates the 'instruction' and 'response' fields; output JSONL preserves all original columns
and adds 'instruction_{lang_code}' and 'response_{lang_code}'.

Supports CUDA (bf16), batched inference, and resume (safe to Ctrl+C and restart).

Usage — English → Swedish:
    python translate_magpie.py \\
        --input magpie_eng_for_swe_2000.jsonl \\
        --output magpie_eng_for_swe_sv_2000.jsonl \\
        --tgt-lang Swedish --lang-code sv

Usage — English → Norwegian Bokmål:
    python translate_magpie.py \\
        --input magpie_eng_for_nob_2000.jsonl \\
        --output magpie_eng_for_nob_nb_2000.jsonl \\
        --tgt-lang "Norwegian Bokmål" --lang-code nb

Reduce --batch-size if you hit OOM (responses can be several hundred tokens).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Max new tokens per field. Instructions are short; responses can be long.
MAX_NEW_TOKENS = {
    "instruction": 512,
    "response":    2048,
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_done(output_path: Path) -> int:
    """Count complete rows already written (for resume)."""
    if not output_path.exists():
        return 0
    with open(output_path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def make_prompt(text: str, tgt_lang: str, tokenizer) -> str:
    messages = [{"role": "user", "content": (
        f"Translate the following English text into {tgt_lang}. "
        "Preserve all formatting, markdown, code blocks, lists, and technical terms exactly. "
        "Respond with only the translation, nothing else.\n\n"
        + text
    )}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def translate_batch(
    texts: list[str],
    tgt_lang: str,
    tokenizer,
    model,
    device: torch.device,
    max_src_length: int,
    max_new_tokens: int,
) -> list[str]:
    prompts = [make_prompt(t, tgt_lang, tokenizer) for t in texts]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_src_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[:, prompt_len:]
    return [t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",     required=True, help="JSONL input file")
    parser.add_argument("--output",    required=True, help="JSONL output file")
    parser.add_argument("--tgt-lang",  required=True,
                        help="Target language name used in the prompt, e.g. 'Swedish'")
    parser.add_argument("--lang-code", required=True,
                        help="Short code appended to output column names, e.g. 'sv'")
    parser.add_argument("--model",     default="google/gemma-3-12b-it")
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Rows per batch. Each row generates two model calls (instruction + response). "
             "Reduce if OOM.",
    )
    parser.add_argument(
        "--max-src-length", type=int, default=2048,
        help="Max prompt tokens; longer source text is truncated.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    field_pairs = [
        ("instruction", f"instruction_{args.lang_code}"),
        ("response",    f"response_{args.lang_code}"),
    ]

    # ── device ──────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device found.", flush=True)
        sys.exit(1)
    device = torch.device("cuda")
    print(f"Target language: {args.tgt_lang}  ({args.lang_code})", flush=True)
    print(f"Device: {device}  dtype: bfloat16", flush=True)

    # ── data ─────────────────────────────────────────────────────────────────
    print(f"Loading {args.input} …", flush=True)
    rows = load_jsonl(Path(args.input))
    print(f"Total rows: {len(rows):,}", flush=True)

    n_done = count_done(output_path)
    if n_done:
        print(f"Resuming from row {n_done:,} ({n_done}/{len(rows)} already done)", flush=True)
    todo = rows[n_done:]

    if not todo:
        print("Nothing left to translate.")
        sys.exit(0)

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"Loading {args.model} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"  # required for batched decoder-only generation

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    # transformers auto-compiles the forward pass for static/hybrid caches (Gemma 3);
    # torch.compile needs a C compiler and isn't supported on Python 3.14+.
    model.generation_config.disable_compile = True

    # ── translate ─────────────────────────────────────────────────────────────
    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    write_mode = "a" if n_done else "w"

    with open(output_path, write_mode, encoding="utf-8") as out:
        for i in tqdm(range(0, len(todo), args.batch_size), total=n_batches,
                      desc="Translating", unit="batch"):
            batch_rows = todo[i : i + args.batch_size]

            # Translate each field in a separate model call (instructions and
            # responses have very different lengths; keeping them separate
            # avoids extreme padding waste in the response pass).
            for src_col, tgt_col in field_pairs:
                texts = [row[src_col] for row in batch_rows]
                translations = translate_batch(
                    texts, args.tgt_lang, tokenizer, model, device,
                    args.max_src_length,
                    MAX_NEW_TOKENS[src_col],
                )
                for row, translation in zip(batch_rows, translations):
                    row[tgt_col] = translation

            # Only write rows once both fields are done — keeps resume safe.
            for row in batch_rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    total = n_done + len(todo)
    print(f"\nDone. {total:,} rows written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
