#!/bin/bash
#SBATCH -A berzelius-2026-79
#SBATCH -C "fat"
#SBATCH -t 1-00:00:00
#SBATCH --job-name=magpie
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/dl4nlp/users/x_jenku/magpie/magpie

export HF_HOME="/proj/dl4nlp/users/x_jenku/.huggingface_cache"

model_path=${1:-"google/gemma-3-27b-it"}
total_prompts=${2:-2000}
language=${3:-"Swedish"}
persona=${4:-"all"}
device="0"
tensor_parallel=1
gpu_memory_utilization=0.9

# Some vLLM custom ops (e.g. Gemma's RMSNorm) fall back to an internal
# torch.compile path regardless of enforce_eager. suppress_errors makes
# torch._dynamo fall back to eager per-graph on a compile failure instead
# of crashing, so we don't need a working C compiler at all.
export TORCHDYNAMO_SUPPRESS_ERRORS=1

cd "$(dirname "$0")"

timestamp=$(date +%s)
job_name="${model_path##*/}_${language}_${timestamp}"
job_path="../data/${job_name}"
mkdir -p "$job_path"

# When cycling through every persona (--persona all), gen_ins.py picks one
# persona per generation round, and repeat = ceil(total_prompts / n). So n
# needs to be small enough that repeat covers all personas at least once.
if [ "$persona" = "all" ]; then
    num_personas=$(python3 -c "import json; print(len(json.load(open('../configs/personas.json'))))")
    n=$(( total_prompts / num_personas ))
    if [ "$n" -lt 1 ]; then
        n=1
    fi
else
    n=$total_prompts
fi

echo "[run_magpie] Model: $model_path"
echo "[run_magpie] Total prompts: $total_prompts"
echo "[run_magpie] Language: $language"
echo "[run_magpie] Persona: $persona"
echo "[run_magpie] n per round: $n"
echo "[run_magpie] Job name: $job_name"

echo "[run_magpie] Generating instructions..."
CUDA_VISIBLE_DEVICES=$device python ../exp/gen_ins.py \
    --device $device \
    --model_path "$model_path" \
    --total_prompts $total_prompts \
    --n $n \
    --top_p 1 \
    --temperature 1 \
    --tensor_parallel_size $tensor_parallel \
    --gpu_memory_utilization $gpu_memory_utilization \
    --enforce_eager \
    --checkpoint_every 1 \
    --language "$language" \
    --persona "$persona" \
    --job_name "$job_name" \
    --timestamp $timestamp

ins_file="${job_path}/Magpie_${model_path##*/}_${total_prompts}_${timestamp}_ins.json"

echo "[run_magpie] Generating responses..."
CUDA_VISIBLE_DEVICES=$device python ../exp/gen_res.py \
    --device $device \
    --model_path "$model_path" \
    --batch_size $n \
    --top_p 1 \
    --temperature 0 \
    --repetition_penalty 1 \
    --tensor_parallel_size $tensor_parallel \
    --gpu_memory_utilization $gpu_memory_utilization \
    --enforce_eager \
    --checkpoint_every 1 \
    --input_file "$ins_file" \
    --offline

echo "[run_magpie] Done. Output: ${ins_file%.json}_res.json"
