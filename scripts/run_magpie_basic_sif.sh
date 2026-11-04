#!/bin/bash
#SBATCH -A berzelius-2026-268
#SBATCH -C "fat"
#SBATCH -t 3-00:00:00
#SBATCH --job-name=magpie_basic_sif
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

# Same as run_magpie_basic.sh (persona-free, idiom-free, language-conditioned Magpie run), but runs in an Apptainer
# image with a newer vLLM, for models the magpie conda env cannot load. The model needs an entry in
# configs/model_configs.json.
# Usage: sbatch run_magpie_basic_sif.sh [model_path] [total_prompts] [language] [n]
# Default image: official vLLM 0.12.0 (needed for Ministral 3), created once with
#   apptainer pull /proj/dl4nlp/users/x_jenku/vllm-0.12.0.sif docker://vllm/vllm-openai:v0.12.0
# Another image: SIF=/proj/dl4nlp/users/x_jenku/euroeval/euroeval.sif sbatch run_magpie_basic_sif.sh ...
SIF=${SIF:-/proj/dl4nlp/users/x_jenku/vllm-0.12.0.sif}
# --cleanenv / PYTHONNOUSERSITE: keep host conda and ~/.local packages out of the container.
run() {
    apptainer exec --nv --cleanenv -B /proj \
        --env HF_HOME=/proj/dl4nlp/users/x_jenku/.huggingface_cache,PYTHONNOUSERSITE=1,TORCHDYNAMO_SUPPRESS_ERRORS=1,CUDA_VISIBLE_DEVICES=$device \
        --pwd /proj/dl4nlp/users/x_jenku/magpie/scripts "$SIF" "$@"
}

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

# sbatch copies this script into a spool directory and runs it from there,
# so $(dirname "$0") points at the spool dir, not the repo's scripts/ dir.
cd /proj/dl4nlp/users/x_jenku/magpie/scripts

timestamp=$(date +%s)
lang_tag=$(echo "$language" | tr -cs '[:alnum:]' '_' | sed 's/_$//')
job_name="${model_path##*/}_${lang_tag}_basic_${timestamp}"
job_path="../data/${job_name}"
mkdir -p "$job_path"

echo "[run_magpie_basic_sif] Model: $model_path"
echo "[run_magpie_basic_sif] Total prompts: $total_prompts"
echo "[run_magpie_basic_sif] Language: $language"
echo "[run_magpie_basic_sif] n per round: $n"
echo "[run_magpie_basic_sif] Job name: $job_name"

echo "[run_magpie_basic_sif] Generating instructions..."
run python3 ../exp/gen_ins.py \
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

echo "[run_magpie_basic_sif] Generating responses..."
run python3 ../exp/gen_res.py \
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

echo "[run_magpie_basic_sif] Done. Output: ${ins_file%.json}_res.json"
