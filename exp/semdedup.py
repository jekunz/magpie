import argparse
import glob
import json
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Semantic deduplication of the basic Magpie runs down to a fixed size per language (input: output of
# filter_lid.py). Instructions are embedded with a multilingual sentence model; samples are then visited
# in a seeded random order and kept only if no already-kept instruction has cosine similarity >= eps.
# eps is binary-searched per run as the strictest value that still keeps at least --target samples, and
# the small surplus is dropped at random. Runs with fewer than --target samples are kept whole.
# Writes <run>_ins_res_filtered_lid_semdedup.json and <run>_ins_res_filtered_lid_semdedup_removed.jsonl
# (with the most similar kept instruction for inspection), and caches embeddings as <run>_..._emb.npy.
# Usage: python semdedup.py [--data_dir ../data] [--target 20000] [--sentence_model BAAI/bge-m3]

def get_args():
    parser = argparse.ArgumentParser(description="Semantic deduplication to a fixed number of samples per run.")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--pattern", type=str, default="*_basic_*", help="Glob for run directories inside data_dir")
    parser.add_argument("--input_suffix", type=str, default="_ins_res_filtered_lid.json")
    parser.add_argument("--target", type=int, default=20000, help="Number of samples to keep per run")
    parser.add_argument("--sentence_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--encoding_batch_size", type=int, default=256)
    parser.add_argument("--k", type=int, default=100, help="Nearest neighbours precomputed per sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", help="Only print statistics, write nothing but the embedding cache")
    return parser.parse_args()

def embed(model, texts, args, cache_file):
    if os.path.exists(cache_file):
        embeddings = np.load(cache_file)
        if len(embeddings) == len(texts):
            return embeddings
    embeddings = model.encode(texts, batch_size=args.encoding_batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    np.save(cache_file, embeddings)
    return embeddings

def top_k_neighbours(emb, k, chunk=4096):
    '''
    Cosine similarities and indices of each sample's k nearest neighbours (itself excluded),
    sorted by decreasing similarity. `emb` is the normalised embedding matrix on the GPU.
    '''
    k = min(k, len(emb) - 1)
    sims, idxs = [], []
    for start in range(0, len(emb), chunk):
        block = emb[start:start + chunk] @ emb.T
        block[torch.arange(len(block)), torch.arange(start, start + len(block))] = -1.0
        s, i = block.topk(k, dim=1)
        sims.append(s.float().cpu().numpy())
        idxs.append(i.cpu().numpy())
    return np.concatenate(sims), np.concatenate(idxs)

def greedy(eps, order, emb, nn_sims, nn_idxs):
    '''
    Keep each sample (in `order`) unless an already-kept sample has similarity >= eps.
    Uses the precomputed neighbours, and falls back to an exact check against all kept
    samples on the GPU when even the k-th neighbour is >= eps (the neighbour list may be
    incomplete). Returns the boolean keep mask.
    '''
    kept = np.zeros(len(order), dtype=bool)
    kept_emb = torch.empty_like(emb)
    n_kept = 0
    for i in order:
        close = nn_idxs[i][nn_sims[i] >= eps]
        if kept[close].any():
            continue
        if nn_sims[i][-1] >= eps and n_kept and (kept_emb[:n_kept] @ emb[i]).max().item() >= eps:
            continue
        kept[i] = True
        kept_emb[n_kept] = emb[i]
        n_kept += 1
    return kept

def select(target, order, emb, nn_sims, nn_idxs, rng, iterations=16):
    '''
    Binary-search the smallest eps whose greedy pass keeps >= target samples, then drop the
    surplus at random. Returns (keep mask, eps).
    '''
    lo, hi = 0.0, 1.0001
    best = greedy(hi, order, emb, nn_sims, nn_idxs)
    best_eps = hi
    for _ in range(iterations):
        mid = (lo + hi) / 2
        kept = greedy(mid, order, emb, nn_sims, nn_idxs)
        if kept.sum() >= target:
            best, best_eps, hi = kept, mid, mid
        else:
            lo = mid
    surplus = int(best.sum()) - target
    if surplus > 0:
        best[rng.choice(np.flatnonzero(best), surplus, replace=False)] = False
    return best, best_eps

def main():
    args = get_args()
    run_dirs = sorted(d for d in glob.glob(os.path.join(args.data_dir, args.pattern)) if os.path.isdir(d))
    if not run_dirs:
        raise SystemExit(f"No run directories matching {args.pattern} in {args.data_dir}")

    model = SentenceTransformer(args.sentence_model, device=f"cuda:{args.device}")
    model.half()
    model.max_seq_length = args.max_seq_length
    model_tag = args.sentence_model.split("/")[-1]

    print(f"{'run':<45} {'total':>7} {'kept':>7} {'eps':>6}")
    for run_dir in run_dirs:
        files = glob.glob(os.path.join(run_dir, f"*{args.input_suffix}"))
        if len(files) != 1:
            print(f"{os.path.basename(run_dir):<45} skipped: expected one *{args.input_suffix}, found {len(files)}")
            continue
        input_file = files[0]
        stem = input_file[:-len(".json")]
        with open(input_file, encoding="utf-8") as f:
            dataset = json.load(f)
        rng = np.random.default_rng(args.seed)

        if len(dataset) <= args.target:
            keep, eps = np.ones(len(dataset), dtype=bool), None
        else:
            embeddings = embed(model, [s["instruction"] for s in dataset], args, f"{stem}_{model_tag}_emb.npy")
            emb = torch.from_numpy(embeddings).to(device=f"cuda:{args.device}", dtype=torch.float16)
            nn_sims, nn_idxs = top_k_neighbours(emb, args.k)
            order = rng.permutation(len(dataset))
            keep, eps = select(args.target, order, emb, nn_sims, nn_idxs, rng)

        eps_text = "-" if eps is None else f"{eps:.3f}"
        print(f"{os.path.basename(run_dir):<45} {len(dataset):>7} {int(keep.sum()):>7} {eps_text:>6}")
        if args.dry_run:
            continue

        kept_idx, removed_idx = np.flatnonzero(keep), np.flatnonzero(~keep)
        removed = []
        for start in range(0, len(removed_idx), 4096):
            chunk = removed_idx[start:start + 4096]
            sims, nearest = (emb[chunk] @ emb[kept_idx].T).max(dim=1)
            for i, sim, j in zip(chunk, sims.float().cpu().numpy(), kept_idx[nearest.cpu().numpy()]):
                removed.append({**dataset[i], "filter_reason": "semantic_duplicate", "duplicate_of": dataset[j]["id"],
                                "duplicate_instruction": dataset[j]["instruction"], "similarity": round(float(sim), 3)})
        with open(f"{stem}_semdedup.json", "w", encoding="utf-8") as f:
            json.dump([dataset[i] for i in kept_idx], f, indent=2, ensure_ascii=False)
        with open(f"{stem}_semdedup_removed.jsonl", "w", encoding="utf-8") as f:
            for sample in removed:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
