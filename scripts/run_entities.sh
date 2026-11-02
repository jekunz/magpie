#!/bin/bash
#SBATCH -A berzelius-2026-268
#SBATCH -J entities
#SBATCH -C "fat"
#SBATCH --gpus 1
#SBATCH -t 3-00:00:00

# Named-entity extraction for the entity-diversity analysis (writes ../data/entities/<condition>_<lang>.jsonl).
# Usage: sbatch run_entities.sh [extract_entities.py arguments]
#   sbatch run_entities.sh --conditions english native
#   sbatch run_entities.sh --conditions translated
# Safe to resubmit: finished samples are skipped.

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/dl4nlp/users/x_jenku/magpie/magpie

export HF_HOME="/proj/dl4nlp/users/x_jenku/.huggingface_cache"
export TORCHDYNAMO_SUPPRESS_ERRORS=1

# sbatch copies this script into a spool directory and runs it from there,
# so hardcode the real path (same as run_magpie.sh).
cd /proj/dl4nlp/users/x_jenku/magpie/exp

python extract_entities.py "$@"
