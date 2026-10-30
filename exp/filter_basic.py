import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

# Conservative cleanup of the basic Magpie runs: drop broken (missing / extremely short)
# samples, then deduplicate instructions within each run by exact match and word n-gram
# Jaccard similarity. Writes <run>_ins_res_filtered.json next to each input, plus a
# <run>_ins_res_removed.jsonl listing every dropped sample with its reason for inspection.
# Usage: python filter_basic.py [--data_dir ../data] [--pattern '*_basic_*'] [--ngram 5] [--threshold 0.5]

def get_args():
    parser = argparse.ArgumentParser(description="Filter and deduplicate basic Magpie runs.")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--pattern", type=str, default="*_basic_*", help="Glob for run directories inside data_dir")
    parser.add_argument("--min_ins_chars", type=int, default=10, help="Drop instructions shorter than this (stripped)")
    parser.add_argument("--min_res_chars", type=int, default=20, help="Drop responses shorter than this (stripped)")
    parser.add_argument("--ngram", type=int, default=5, help="Word n-gram size for near-duplicate detection")
    parser.add_argument("--threshold", type=float, default=0.5, help="Jaccard similarity on instruction n-grams at or above which a later sample counts as a duplicate")
    parser.add_argument("--dry_run", action="store_true", help="Only print statistics, write nothing")
    return parser.parse_args()

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

def shingles(text, n):
    '''
    Lowercased word n-grams of `text`, ignoring punctuation. Texts shorter than n words
    become a single shingle, so for them the Jaccard test reduces to exact matching.
    '''
    tokens = TOKEN_RE.findall(text.lower())
    if len(tokens) < n:
        return {tuple(tokens)}
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

def broken_reason(sample, min_ins_chars, min_res_chars):
    instruction = (sample.get("instruction") or "").strip()
    response = (sample.get("response") or "").strip()
    if not instruction:
        return "empty_instruction"
    if not response:
        return "empty_response"
    if len(instruction) < min_ins_chars:
        return "short_instruction"
    if len(response) < min_res_chars:
        return "short_response"
    return None

def filter_run(dataset, args):
    '''
    Returns (kept, removed). The first occurrence of a near-duplicate cluster is kept.
    Candidate duplicates are found through an inverted index from n-gram to kept samples,
    so only samples sharing at least one n-gram are ever compared.
    '''
    kept, removed = [], []
    kept_shingles = []
    index = defaultdict(list)
    for sample in dataset:
        reason = broken_reason(sample, args.min_ins_chars, args.min_res_chars)
        if reason:
            removed.append({**sample, "filter_reason": reason})
            continue

        sh = shingles(sample["instruction"], args.ngram)
        overlap = Counter(j for s in sh for j in index[s])
        best_j, best_sim = None, 0.0
        for j, inter in overlap.items():
            sim = inter / (len(sh) + len(kept_shingles[j]) - inter)
            if sim > best_sim:
                best_j, best_sim = j, sim
        if best_j is not None and best_sim >= args.threshold:
            removed.append({**sample, "filter_reason": "exact_duplicate" if best_sim == 1.0 else "near_duplicate",
                            "duplicate_of": kept[best_j]["id"], "jaccard": round(best_sim, 3)})
            continue

        for s in sh:
            index[s].append(len(kept))
        kept_shingles.append(sh)
        kept.append(sample)
    return kept, removed

def main():
    args = get_args()
    run_dirs = sorted(d for d in glob.glob(os.path.join(args.data_dir, args.pattern)) if os.path.isdir(d))
    if not run_dirs:
        raise SystemExit(f"No run directories matching {args.pattern} in {args.data_dir}")

    reasons = ["empty_instruction", "empty_response", "short_instruction", "short_response", "exact_duplicate", "near_duplicate"]
    header = f"{'run':<45} {'total':>7} {'kept':>7} " + " ".join(f"{r:>17}" for r in reasons)
    print(header)
    totals = Counter()
    for run_dir in run_dirs:
        files = glob.glob(os.path.join(run_dir, "*_ins_res.json"))
        if len(files) != 1:
            print(f"{os.path.basename(run_dir):<45} skipped: expected one *_ins_res.json, found {len(files)}")
            continue
        input_file = files[0]
        with open(input_file, encoding="utf-8") as f:
            dataset = json.load(f)
        kept, removed = filter_run(dataset, args)

        counts = Counter(r["filter_reason"] for r in removed)
        counts["total"], counts["kept"] = len(dataset), len(kept)
        totals.update(counts)
        print(f"{os.path.basename(run_dir):<45} {len(dataset):>7} {len(kept):>7} " + " ".join(f"{counts[r]:>17}" for r in reasons))

        if not args.dry_run:
            stem = input_file[:-len(".json")]
            with open(f"{stem}_filtered.json", "w", encoding="utf-8") as f:
                json.dump(kept, f, indent=2, ensure_ascii=False)
            with open(f"{stem}_removed.jsonl", "w", encoding="utf-8") as f:
                for sample in removed:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"{'TOTAL':<45} {totals['total']:>7} {totals['kept']:>7} " + " ".join(f"{totals[r]:>17}" for r in reasons))

if __name__ == "__main__":
    main()
