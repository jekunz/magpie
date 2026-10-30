#!/bin/bash
#SBATCH -A berzelius-2026-268
#SBATCH -J semdedup
#SBATCH --gpus 1
#SBATCH -t 0-12:00:00

# Semantic deduplication of all basic runs down to a fixed size per language
# (expects the *_ins_res_filtered_lid.json files from filter_basic.py + filter_lid.py).
# Usage: sbatch run_semdedup.sh [target] [sentence_model]
# Safe to resubmit: instruction embeddings are cached per run.

target=${1:-20000}
sentence_model=${2:-BAAI/bge-m3}

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/dl4nlp/users/x_jenku/magpie/magpie

export HF_HOME="/proj/dl4nlp/users/x_jenku/.huggingface_cache"

# sbatch copies this script into a spool directory and runs it from there,
# so hardcode the real path (same as run_magpie.sh).
cd /proj/dl4nlp/users/x_jenku/magpie/exp

python semdedup.py \
    --target         "$target" \
    --sentence_model "$sentence_model"
