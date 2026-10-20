"""
Sample one shared English source file from Magpie-Pro-1M for translation into all target languages.

Unlike prepare_magpie_corpora.ipynb in annotation_sommarjobb_2026 (which matched each language's English sample to the
task/difficulty distribution of a native Nordic Magpie dataset), every language translates the
same English rows, so the translations are parallel across languages.

Output schema matches the earlier magpie_eng_for_*.jsonl files:
    instruction, response, task_category, difficulty, language ("eng")

Usage (from scripts/, like the other exp/ scripts):
    python ../exp/prepare_eng_source.py --n 10000     # → ../data/magpie_eng_10000.jsonl
"""

import argparse

import numpy as np
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1")
    parser.add_argument("--output", default=None, help="Defaults to ../data/magpie_eng_{n}.jsonl")
    args = parser.parse_args()

    output = args.output or f"../data/magpie_eng_{args.n}.jsonl"

    ds = load_dataset(args.dataset, split="train")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=args.n, replace=False)
    df = ds.select(idx.tolist()).to_pandas()

    # Same column handling as prepare_magpie_corpora.ipynb (Magpie-Pro uses input/output).
    instruction_col = "instruction" if "instruction" in df.columns else "input"
    response_col = "response" if "response" in df.columns else "output"

    out = df[[instruction_col, response_col, "task_category", "difficulty"]].rename(
        columns={instruction_col: "instruction", response_col: "response"}
    )
    out["task_category"] = out["task_category"].str.lower().str.strip()
    out["difficulty"] = out["difficulty"].str.lower().str.strip()
    out["language"] = "eng"

    out.to_json(output, orient="records", lines=True, force_ascii=False)
    print(f"Saved {len(out):,} rows → {output}")
    print(out["task_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
