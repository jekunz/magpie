#!/bin/bash
#SBATCH -A berzelius-2026-268
#SBATCH -C "fat"
#SBATCH -t 3-00:00:00
#SBATCH --job-name=magpie_basic
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

# Basic (persona-free, idiom-free) language-conditioned Magpie run.
# Usage: sbatch run_magpie_basic.sh [model_path] [total_prompts] [language] [n]
# Per-language launchers live in scripts/basic/.

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/dl4nlp/users/x_jenku/magpie/magpie

export HF_HOME="/proj/dl4nlp/users/x_jenku/.huggingface_cache"

model_path=${1:-"google/gemma-3-27b-it"}
total_prompts=${2:-2000}
language=${3:-"Swedish"}
# Prompts per generation round; one checkpoint is written per round.
n=${4:-1000}
device="0"
tensor_parallel=1
gpu_memory_utilization=0.9

if [ "$n" -gt "$total_prompts" ]; then
    n=$total_prompts
fi

# Some vLLM custom ops (e.g. Gemma's RMSNorm) fall back to an internal
# torch.compile path regardless of enforce_eager. suppress_errors makes
# torch._dynamo fall back to eager per-graph on a compile failure instead
# of crashing, so we don't need a working C compiler at all.
export TORCHDYNAMO_SUPPRESS_ERRORS=1

# sbatch copies this script into a spool directory and runs it from there,
# so $(dirname "$0") points at the spool dir, not the repo's scripts/ dir.
cd /proj/dl4nlp/users/x_jenku/magpie/scripts

timestamp=$(date +%s)
lang_tag=$(echo "$language" | tr -cs '[:alnum:]' '_' | sed 's/_$//')
job_name="${model_path##*/}_${lang_tag}_basic_${timestamp}"
job_path="../data/${job_name}"
mkdir -p "$job_path"

echo "[run_magpie_basic] Model: $model_path"
echo "[run_magpie_basic] Total prompts: $total_prompts"
echo "[run_magpie_basic] Language: $language"
echo "[run_magpie_basic] n per round: $n"
echo "[run_magpie_basic] Job name: $job_name"

echo "[run_magpie_basic] Generating instructions..."
CUDA_VISIBLE_DEVICES=$device python ../exp/gen_ins.py \
    --device $device \
    --model_path "$model_path" \
    --total_prompts $total_prompts \
    --n $n \
    --top_p 1 \
    --temperature 1.15 \
    --tensor_parallel_size $tensor_parallel \
    --gpu_memory_utilization $gpu_memory_utilization \
    --enforce_eager \
    --checkpoint_every 1 \
    --language "$language" \
    --job_name "$job_name" \
    --timestamp $timestamp

ins_file="${job_path}/Magpie_${model_path##*/}_${total_prompts}_${timestamp}_ins.json"

echo "[run_magpie_basic] Generating responses..."
CUDA_VISIBLE_DEVICES=$device python ../exp/gen_res.py \
    --device $device \
    --model_path "$model_path" \
    --batch_size $n \
    --top_p 1 \
    --temperature 0.7 \
    --repetition_penalty 1 \
    --tensor_parallel_size $tensor_parallel \
    --gpu_memory_utilization $gpu_memory_utilization \
    --enforce_eager \
    --checkpoint_every 1 \
    --input_file "$ins_file" \
    --offline

echo "[run_magpie_basic] Done. Output: ${ins_file%.json}_res.json"
