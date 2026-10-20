"""
Translate a Magpie English JSONL file into a target language using a Gemma instruction-tuned model.

Translates the 'instruction' and 'response' fields; output JSONL preserves all original columns
and adds 'instruction_{lang_code}' and 'response_{lang_code}'.

Runs on vLLM (bf16, greedy), with resume (safe to Ctrl+C and restart).

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

"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams

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


def truncate(text: str, tokenizer, max_tokens: int) -> str:
    """Cut overly long source text to max_tokens so prompts fit in max_model_len."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text if len(ids) <= max_tokens else tokenizer.decode(ids[:max_tokens])


def generate(llm, prompts: list[str], max_new_tokens: int, stop_token_ids: list[int]) -> list[str]:
    params = SamplingParams(temperature=0, max_tokens=max_new_tokens, stop_token_ids=stop_token_ids)
    outputs = llm.generate(prompts, params, use_tqdm=False)
    return [o.outputs[0].text.strip() for o in outputs]


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
        "--batch-size", type=int, default=500,
        help="Rows per vLLM call; output is written (and resume checkpointed) after each batch.",
    )
    parser.add_argument(
        "--max-src-length", type=int, default=2048,
        help="Max tokens of source text per field; longer source text is truncated.",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()

    output_path = Path(args.output)
    field_pairs = [
        ("instruction", f"instruction_{args.lang_code}"),
        ("response",    f"response_{args.lang_code}"),
    ]

    print(f"Target language: {args.tgt_lang}  ({args.lang_code})", flush=True)

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
    # enforce_eager: no torch.compile / CUDA graphs, so no C compiler needed (same as gen_ins/gen_res).
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0},
    )
    tokenizer = llm.get_tokenizer()
    stop_token_ids = [
        t for t in {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")}
        if t is not None and t != tokenizer.unk_token_id
    ]

    # ── translate ─────────────────────────────────────────────────────────────
    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    write_mode = "a" if n_done else "w"

    with open(output_path, write_mode, encoding="utf-8") as out:
        for i in tqdm(range(0, len(todo), args.batch_size), total=n_batches,
                      desc="Translating", unit="batch"):
            batch_rows = todo[i : i + args.batch_size]

            # Translate each field in a separate call (they have different max output lengths).
            for src_col, tgt_col in field_pairs:
                prompts = [
                    make_prompt(truncate(row[src_col], tokenizer, args.max_src_length), args.tgt_lang, tokenizer)
                    for row in batch_rows
                ]
                translations = generate(llm, prompts, MAX_NEW_TOKENS[src_col], stop_token_ids)
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
