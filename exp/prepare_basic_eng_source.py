"""
Turn the filtered + semantically deduplicated English basic Magpie run into the JSONL source file that
translate_all.sh / translate_magpie.py translate into every target language.

Input is the *_ins_res_filtered_lid_semdedup.json written by semdedup.py in the English basic run directory.
Output keeps only what translation needs plus provenance: id, instruction, response, language ("eng"), source_run.

Usage (from scripts/, like the other exp/ scripts):
    python ../exp/prepare_basic_eng_source.py     # → ../data/magpie_basic_eng_20000.jsonl
    INPUT=../data/magpie_basic_eng_20000.jsonl bash translate_all.sh plain
"""

import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--pattern", default="*_English_basic_*", help="Glob for the English run directory inside data_dir")
    parser.add_argument("--output", default=None, help="Defaults to ../data/magpie_basic_eng_{n}.jsonl")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.data_dir, args.pattern, "*_ins_res_filtered_lid_semdedup.json"))
    if len(files) != 1:
        raise SystemExit(f"Expected one semdedup file under {args.data_dir}/{args.pattern}, found {len(files)}: {files}")
    with open(files[0], encoding="utf-8") as f:
        dataset = json.load(f)

    source_run = os.path.basename(os.path.dirname(files[0]))
    output = args.output or os.path.join(args.data_dir, f"magpie_basic_eng_{len(dataset)}.jsonl")
    with open(output, "w", encoding="utf-8") as f:
        for sample in dataset:
            row = {"id": sample["id"], "instruction": sample["instruction"], "response": sample["response"],
                   "language": "eng", "source_run": source_run}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset):,} rows from {files[0]} → {output}")


if __name__ == "__main__":
    main()
