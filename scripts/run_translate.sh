#!/bin/bash
#SBATCH -A berzelius-2026-268
#SBATCH -J gemma27-translate
#SBATCH -C "fat"
#SBATCH --gpus 1
#SBATCH -t 3-00:00:00

# Generic English → target-language translation job; submitted per language/variant by translate_all.sh.
# Usage: sbatch run_translate.sh <plain|idiomatic> <tgt_lang> <lang_code> [input] [model]
# Safe to resubmit if the job crashes or times out: the Python scripts resume from the
# last complete row already written to the output file.

variant=$1
tgt_lang=$2
lang_code=$3
input=${4:-../data/magpie_eng_10000.jsonl}
model=${5:-google/gemma-3-27b-it}

# ── environment (same as run_magpie.sh) ─────────────────────────────────────
module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/dl4nlp/users/x_jenku/magpie/magpie

export HF_HOME="/proj/dl4nlp/users/x_jenku/.huggingface_cache"

# No working C compiler on the nodes: if anything still hits torch.compile,
# fall back to eager instead of crashing.
export TORCHDYNAMO_SUPPRESS_ERRORS=1

# sbatch copies this script into a spool directory and runs it from there,
# so hardcode the real path (same as run_magpie.sh).
cd /proj/dl4nlp/users/x_jenku/magpie/scripts

# ── eng → $lang_code ─────────────────────────────────────────────────────────
case "$variant" in
    plain)     script=../exp/translate_magpie.py;    suffix="" ;;
    idiomatic) script=../exp/translate_magpie_idiomatic.py; suffix="_idiomatic" ;;
    *) echo "Unknown variant: $variant (expected plain or idiomatic)"; exit 1 ;;
esac

output="${input%.jsonl}_${lang_code}${suffix}.jsonl"
echo "[translate] $variant  eng → $tgt_lang ($lang_code)  $input → $output  ($model)"

python "$script" \
    --input     "$input" \
    --output    "$output" \
    --tgt-lang  "$tgt_lang" \
    --lang-code "$lang_code" \
    --model     "$model"
