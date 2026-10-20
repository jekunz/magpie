#!/bin/bash
# Basic (no persona) Magpie data for Ukrainian.
# Usage: bash basic/ukrainian.sh [total_prompts] [model_path]
total_prompts=${1:-2000}
model_path=${2:-"google/gemma-3-27b-it"}
sbatch --job-name=magpie_basic_uk \
    /proj/dl4nlp/users/x_jenku/magpie/scripts/run_magpie_basic.sh \
    "$model_path" "$total_prompts" "Ukrainian"
