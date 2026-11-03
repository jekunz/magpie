"""
Extract named entities from the native, translated and localized 20k sets (and the English source) with
Gemma, for the entity-diversity analysis in analysis/compare_datasets.ipynb (section 10).

For every sample the model lists the named entities in instruction + response, each with its canonical
English name (so that inflected and translated mentions of the same entity match across languages) and the
country it is most associated with. Uses the same samples as the topic section of the notebook: TOPIC_N
English source samples, the same ids in every translated/localized set, and an independent seeded sample of
each native set.

Writes one file per set, <out_dir>/<condition>_<lang>.jsonl, with id, the parsed entities and the raw model
output. Runs on vLLM (bf16, greedy) and resumes per set (safe to Ctrl+C and restart).

Usage:
    python extract_entities.py                                  # all sets
    python extract_entities.py --conditions native --langs is fo lb
    python extract_entities.py --dry_run                        # print sample counts and one prompt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

NATIVE_REPO = "jekunz/magpie-filtered-20k"
CONDITIONS = {
    "native": NATIVE_REPO,
    "translated": "jekunz/magpie-filtered-20k-translated-from-en",
    "localized": "jekunz/magpie-filtered-20k-translated-from-en-with-localization",
}
LANGS = ["de", "fr", "es", "it", "pt", "ru", "pl", "nl", "sv", "da", "nb", "cs", "ro", "uk", "hu", "fi", "bg", "ca",
         "hr", "sk", "sr", "sl", "et", "be", "mk", "gl", "is", "fo", "lb",
         "tr", "el", "lt", "bs", "lv", "sq", "ga", "mt", "cnr"]
SEED = 42          # same seed and sample size as the topic section of the notebook
TOPIC_N = 2000

ENTITY_TYPES = {"PER", "LOC", "ORG", "PRODUCT", "WORK", "EVENT", "OTHER"}

PROMPT = """Below is a conversation between a user and an AI assistant. List every named entity mentioned in it: specific people, places (countries, cities, regions, landmarks), organizations, companies, brands and products, works (books, films, songs, laws), events and holidays. Do not list generic concepts, common nouns, units, programming languages, software libraries or other technical terms.

Write one line per entity, in the format:
TYPE | canonical English name | country
- TYPE is one of PER, LOC, ORG, PRODUCT, WORK, EVENT, OTHER.
- The canonical English name is the entity's name as in English Wikipedia, in its base form (not inflected), even if the conversation is in another language.
- country is the English name of the country the entity is most associated with, or - if it is international or unclear.

List each entity only once. If there are no named entities, write NONE. Output only the list.

USER:
{instruction}

ASSISTANT:
{response}"""


def get_args():
    parser = argparse.ArgumentParser(description="Named-entity extraction for the entity-diversity analysis.")
    parser.add_argument("--conditions", nargs="+", default=["english", *CONDITIONS],
                        choices=["english", *CONDITIONS])
    parser.add_argument("--langs", nargs="+", default=LANGS)
    parser.add_argument("--out_dir", type=str, default="../data/entities")
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--max_src_tokens", type=int, default=3000, help="Response is cut to this many tokens")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--max_model_len", type=int, default=6144)
    parser.add_argument("--chunk_size", type=int, default=500, help="Samples per generate call (resume granularity)")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load(repo, config):
    try:
        return load_dataset(repo, config, split="train").to_pandas()
    except Exception as e:
        print(f"  missing {repo} [{config}]: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return None


def sample_sets(args):
    '''{(condition, lang): DataFrame of id/instruction/response}, sampled exactly as in the notebook's topic section.'''
    en_full = load(NATIVE_REPO, "en")
    en_sample = en_full.iloc[np.random.default_rng(SEED).choice(len(en_full), TOPIC_N, replace=False)]
    en_ids = set(en_sample["id"])
    sets = {}
    if "english" in args.conditions:
        sets[("english", "en")] = en_sample
    for code in args.langs:
        for cond, repo in CONDITIONS.items():
            if cond not in args.conditions:
                continue
            df = load(repo, code)
            if df is None:
                continue
            if cond == "native":
                sub = df.iloc[np.random.default_rng(SEED).choice(len(df), min(TOPIC_N, len(df)), replace=False)]
            else:
                sub = df[df["id"].isin(en_ids)]
            sets[(cond, code)] = sub
    return {k: v[["id", "instruction", "response"]].reset_index(drop=True) for k, v in sets.items()}


def parse(text):
    '''Model output -> list of {type, name, country}; malformed lines are skipped, duplicates dropped.'''
    entities, seen = [], set()
    for line in text.splitlines():
        parts = [p.strip() for p in line.strip().lstrip("-*• ").split("|")]
        if len(parts) != 3 or parts[0].upper() not in ENTITY_TYPES or not parts[1]:
            continue
        name = re.sub(r"\s+", " ", parts[1]).strip(" .\"'*")
        country = None if parts[2].strip(" .") in {"", "-", "–", "none", "None", "N/A"} else parts[2].strip(" .")
        if name.casefold() not in seen:
            seen.add(name.casefold())
            entities.append({"type": parts[0].upper(), "name": name, "country": country})
    return entities


def count_done(path):
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def main():
    args = get_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sets = sample_sets(args)
    todo = {}
    for (cond, code), df in sets.items():
        done = count_done(out_dir / f"{cond}_{code}.jsonl")
        rest = df[~df["id"].isin(done)]
        print(f"{cond:>10} {code:<3} {len(df):>5} sampled, {len(done):>5} done, {len(rest):>5} to do", flush=True)
        if len(rest):
            todo[(cond, code)] = rest
    if args.dry_run:
        if sets:
            df = next(iter(sets.values()))
            print("\n--- example prompt ---\n" + PROMPT.format(instruction=df["instruction"][0], response=df["response"][0]))
        return
    if not todo:
        print("Nothing left to do.")
        return

    from vllm import LLM, SamplingParams
    # enforce_eager: no torch.compile / CUDA graphs, so no C compiler needed (same as translate_magpie.py).
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.9, enforce_eager=True, limit_mm_per_prompt={"image": 0})
    tokenizer = llm.get_tokenizer()
    stop_token_ids = [t for t in {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")}
                      if t is not None and t != tokenizer.unk_token_id]
    params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, stop_token_ids=stop_token_ids)

    def truncate(text, max_tokens):
        ids = tokenizer.encode(text or "", add_special_tokens=False)
        return text if len(ids) <= max_tokens else tokenizer.decode(ids[:max_tokens])

    def make_prompt(row):
        content = PROMPT.format(instruction=truncate(row.instruction, 512),
                                response=truncate(row.response, args.max_src_tokens))
        return tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)

    for (cond, code), df in todo.items():
        path = out_dir / f"{cond}_{code}.jsonl"
        with open(path, "a", encoding="utf-8") as out:
            for s in range(0, len(df), args.chunk_size):
                chunk = df.iloc[s:s + args.chunk_size]
                outputs = llm.generate([make_prompt(r) for r in chunk.itertuples()], params, use_tqdm=False)
                for row, o in zip(chunk.itertuples(), outputs):
                    raw = o.outputs[0].text
                    out.write(json.dumps({"id": row.id, "entities": parse(raw), "raw": raw,
                                          "truncated": o.outputs[0].finish_reason == "length"},
                                         ensure_ascii=False) + "\n")
                out.flush()
                print(f"{cond} {code}: {min(s + args.chunk_size, len(df))}/{len(df)}", flush=True)


if __name__ == "__main__":
    main()
