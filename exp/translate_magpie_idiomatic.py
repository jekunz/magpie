"""
Translate a Magpie English JSONL file into a target language using a Gemma instruction-tuned model,
producing idiomatic, culturally-adapted translations rather than literal ones.

Like translate_magpie.py, this translates the 'instruction' and 'response' fields and preserves all
original columns, adding 'instruction_{lang_code}' and 'response_{lang_code}'. The difference is the
prompting: the model is instructed to write as a native speaker would, and to adapt culture-bound
details in the instruction (settings, holidays, place names, currencies, units, etc.) to natural
target-language equivalents where that doesn't change the underlying task being asked for. The
response is then translated with the already-localized instruction as context, so it stays consistent
with whatever adaptation happened there (e.g. if the instruction's setting changed, the response's
matching details change the same way).

Runs on vLLM (bf16, greedy), with resume (safe to Ctrl+C and restart).

Usage — English → Swedish:
    python translate_magpie_idiomatic.py \\
        --input magpie_eng_for_swe_2000.jsonl \\
        --output magpie_eng_for_swe_sv_2000_idiomatic.jsonl \\
        --tgt-lang Swedish --lang-code sv

Usage — English → Norwegian Bokmål:
    python translate_magpie_idiomatic.py \\
        --input magpie_eng_for_nob_2000.jsonl \\
        --output magpie_eng_for_nob_nb_2000_idiomatic.jsonl \\
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


def make_instruction_prompt(text: str, tgt_lang: str, tokenizer) -> str:
    messages = [{"role": "user", "content": (
        f"Translate the following English text into natural, idiomatic {tgt_lang} — the way a native "
        f"speaker of {tgt_lang} would actually write it, not a literal word-for-word translation. "
        "Preserve all formatting, markdown, code blocks, and lists exactly.\n\n"
        "If the text refers to a setting, custom, holiday, place name, currency, or unit of measurement "
        f"that is specifically tied to English-speaking or American culture, adapt it to a natural "
        f"equivalent from {tgt_lang}-speaking culture, as long as doing so keeps the underlying task or "
        "question being asked exactly the same. Leave proper names of real people, organizations, "
        "products, and technical terms unchanged unless there is a well-established localized form. "
        "If nothing in the text is culturally specific, just translate it naturally.\n\n"
        "Respond with only the translation, nothing else.\n\n"
        + text
    )}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def make_response_prompt(text: str, tgt_lang: str, localized_instruction: str, tokenizer) -> str:
    messages = [{"role": "user", "content": (
        f"Translate the following English text into natural, idiomatic {tgt_lang} — the way a native "
        f"speaker of {tgt_lang} would actually write it, not a literal word-for-word translation. "
        "Preserve all formatting, markdown, code blocks, and lists exactly.\n\n"
        "This text is the response to the instruction below, which has already been translated and "
        f"adapted to {tgt_lang}-speaking culture. Keep the response consistent with that localized "
        "instruction: if its setting, examples, names, currency, or units were adapted, adapt the "
        "response's matching details in the same way. Otherwise, just translate faithfully.\n\n"
        f"Localized instruction ({tgt_lang}):\n{localized_instruction}\n\n"
        "Text to translate:\n" + text
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
    instruction_col = f"instruction_{args.lang_code}"
    response_col = f"response_{args.lang_code}"

    print(f"Target language: {args.tgt_lang}  ({args.lang_code})  [idiomatic/localized]", flush=True)

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

            # Instruction first: this also localizes any culture-bound setting, which the
            # response pass below then translates *with*, to keep the two consistent.
            instruction_prompts = [
                make_instruction_prompt(truncate(row["instruction"], tokenizer, args.max_src_length),
                                        args.tgt_lang, tokenizer)
                for row in batch_rows
            ]
            localized_instructions = generate(
                llm, instruction_prompts, MAX_NEW_TOKENS["instruction"], stop_token_ids,
            )
            for row, translation in zip(batch_rows, localized_instructions):
                row[instruction_col] = translation

            response_prompts = [
                make_response_prompt(truncate(row["response"], tokenizer, args.max_src_length),
                                     args.tgt_lang, row[instruction_col], tokenizer)
                for row in batch_rows
            ]
            translated_responses = generate(
                llm, response_prompts, MAX_NEW_TOKENS["response"], stop_token_ids,
            )
            for row, translation in zip(batch_rows, translated_responses):
                row[response_col] = translation

            # Only write rows once both fields are done — keeps resume safe.
            for row in batch_rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    total = n_done + len(todo)
    print(f"\nDone. {total:,} rows written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
