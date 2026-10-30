import argparse
import glob
import json
import os
import re
from collections import Counter

import fasttext

# Language-identification filter for the deduplicated basic Magpie runs (output of filter_basic.py).
# Runs GlotLID on instruction and response separately and drops a sample when either is confidently
# identified as any language other than the target, including close neighbours (e.g. Bokmål in a
# Danish run) and the wrong script (srp_Latn in the Serbian Cyrillic run).
# Writes <run>_ins_res_filtered_lid.json (kept samples, with a "lid" field) and
# <run>_ins_res_filtered_lid_removed.jsonl next to each input.
# Usage: python filter_lid.py [--data_dir ../data] [--threshold 0.9] [--lid_model path/to/model.bin]

def get_args():
    parser = argparse.ArgumentParser(description="Language-identification filter for basic Magpie runs.")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--pattern", type=str, default="*_basic_*", help="Glob for run directories inside data_dir")
    parser.add_argument("--lid_model", type=str, default=None, help="Path to GlotLID model.bin; downloaded from cis-lmu/glotlid if omitted")
    parser.add_argument("--threshold", type=float, default=0.9, help="Minimum probability for a wrong-language prediction to remove a sample")
    parser.add_argument("--min_chars", type=int, default=20, help="Texts shorter than this after cleaning are not judged")
    parser.add_argument("--check", type=str, default="both", choices=["both", "instruction", "response"], help="Which fields can cause removal")
    parser.add_argument("--dry_run", action="store_true", help="Only print statistics, write nothing")
    return parser.parse_args()

# Target language (as in gen_input_configs.language) -> GlotLID label.
TARGET_LABELS = {
    "Belarusian": "bel_Cyrl", "Bulgarian": "bul_Cyrl", "Catalan": "cat_Latn", "Croatian": "hrv_Latn",
    "Czech": "ces_Latn", "Danish": "dan_Latn", "Dutch": "nld_Latn", "English": "eng_Latn",
    "Estonian": "ekk_Latn", "Faroese": "fao_Latn", "Finnish": "fin_Latn", "French": "fra_Latn",
    "Galician": "glg_Latn", "German": "deu_Latn", "Hungarian": "hun_Latn", "Icelandic": "isl_Latn",
    "Italian": "ita_Latn", "Luxembourgish": "ltz_Latn", "Macedonian": "mkd_Cyrl",
    "Norwegian Bokmål": "nob_Latn", "Polish": "pol_Latn", "Portuguese": "por_Latn",
    "Romanian": "ron_Latn", "Russian": "rus_Cyrl", "Serbian (Cyrillic script)": "srp_Cyrl",
    "Slovak": "slk_Latn", "Slovenian": "slv_Latn", "Spanish": "spa_Latn", "Swedish": "swe_Latn",
    "Ukrainian": "ukr_Cyrl",
}

CODE_BLOCK_RE = re.compile(r"```.*?(```|$)", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MATH_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]*\$|\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
MARKUP_RE = re.compile(r"[#*_>|\-]{2,}|^\s*[#>*\-]\s*", re.MULTILINE)

def clean(text):
    '''
    Strip content that carries no language signal (code, URLs, math, markdown) and flatten to
    one line, as fastText predicts on a single line.
    '''
    text = CODE_BLOCK_RE.sub(" ", text)
    text = INLINE_CODE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = MATH_RE.sub(" ", text)
    text = MARKUP_RE.sub(" ", text)
    return " ".join(text.split())

def predict(model, text):
    # model.f.predict instead of model.predict: the latter breaks under numpy 2 (copy=False).
    prob, label = model.f.predict(text, 1, 0.0, "strict")[0]
    return label.replace("__label__", ""), min(float(prob), 1.0)

def judge(model, text, target, args):
    '''
    Returns (label, prob, verdict) with verdict one of: target, uncertain, too_short, wrong.
    '''
    text = clean(text or "")
    if len(text) < args.min_chars:
        return None, None, "too_short"
    label, prob = predict(model, text)
    if label == target:
        return label, prob, "target"
    # und_* / zxx_*: unknown language or no linguistic content (e.g. leftover symbols).
    if prob < args.threshold or label.startswith(("und_", "zxx_")):
        return label, prob, "uncertain"
    return label, prob, "wrong"

def removal_reason(lid, args):
    fields = ["instruction", "response"] if args.check == "both" else [args.check]
    for field in fields:
        if lid[field]["verdict"] == "wrong":
            return f"wrong_language_{field}"
    return None

def load_model(path):
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
    return fasttext.load_model(path)

def main():
    args = get_args()
    run_dirs = sorted(d for d in glob.glob(os.path.join(args.data_dir, args.pattern)) if os.path.isdir(d))
    if not run_dirs:
        raise SystemExit(f"No run directories matching {args.pattern} in {args.data_dir}")
    model = load_model(args.lid_model)

    verdicts = ["target", "uncertain", "too_short", "wrong"]
    print(f"{'run':<45} {'field':<11} {'total':>6} {'kept':>6} " + " ".join(f"{v:>10}" for v in verdicts) + "  most frequent non-target labels")
    for run_dir in run_dirs:
        files = glob.glob(os.path.join(run_dir, "*_ins_res_filtered.json"))
        if len(files) != 1:
            print(f"{os.path.basename(run_dir):<45} skipped: expected one *_ins_res_filtered.json, found {len(files)}")
            continue
        input_file = files[0]
        with open(input_file, encoding="utf-8") as f:
            dataset = json.load(f)
        if not dataset:
            continue
        language = dataset[0]["gen_input_configs"]["language"]
        if language not in TARGET_LABELS:
            print(f"{os.path.basename(run_dir):<45} skipped: no GlotLID label for language {language!r}")
            continue
        target = TARGET_LABELS[language]

        kept, removed = [], []
        stats = {field: Counter() for field in ["instruction", "response"]}
        others = {field: Counter() for field in ["instruction", "response"]}
        for sample in dataset:
            lid = {}
            for field in ["instruction", "response"]:
                label, prob, verdict = judge(model, sample[field], target, args)
                lid[field] = {"label": label, "prob": None if prob is None else round(prob, 3), "verdict": verdict}
                stats[field][verdict] += 1
                if verdict == "wrong":
                    others[field][label] += 1
            reason = removal_reason(lid, args)
            if reason:
                removed.append({**sample, "lid": lid, "filter_reason": reason})
            else:
                kept.append({**sample, "lid": lid})

        name = f"{os.path.basename(run_dir)} ({target})"
        for field in ["instruction", "response"]:
            top = ", ".join(f"{l}:{c}" for l, c in others[field].most_common(4))
            print(f"{name:<45} {field:<11} {len(dataset):>6} {len(kept):>6} " + " ".join(f"{stats[field][v]:>10}" for v in verdicts) + f"  {top}")
            name = ""

        if not args.dry_run:
            stem = input_file[:-len(".json")]
            with open(f"{stem}_lid.json", "w", encoding="utf-8") as f:
                json.dump(kept, f, indent=2, ensure_ascii=False)
            with open(f"{stem}_lid_removed.jsonl", "w", encoding="utf-8") as f:
                for sample in removed:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
