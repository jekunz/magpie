#!/bin/bash
#SBATCH -A berzelius-2026-268
#SBATCH -C "fat"
#SBATCH -t 3-00:00:00
#SBATCH --job-name=magpie-sif
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

# Same as run_magpie.sh, but runs in an Apptainer image with a newer vLLM, for models the magpie conda env cannot
# load. Usage as for run_magpie.sh; the model needs an entry in configs/model_configs.json. Default image: official
# vLLM 0.12.0 (needed for Ministral 3), created once with
#   apptainer pull /proj/dl4nlp/users/x_jenku/vllm-0.12.0.sif docker://vllm/vllm-openai:v0.12.0
# Another image: SIF=/proj/dl4nlp/users/x_jenku/euroeval/euroeval.sif sbatch run_magpie_sif.sh ...
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
persona=${4:-"all"}
device="0"
tensor_parallel=1
gpu_memory_utilization=0.9

# Auto-select the personas file to match --language, unless explicitly overridden.
if [ -n "$5" ]; then
    personas_file="$5"
else
    case "$language" in
        Swedish) personas_file="../configs/personas.json" ;;
        Danish)  personas_file="../configs/personas_dk.json" ;;
        German)  personas_file="../configs/personas_de.json" ;;
        *)       personas_file="../configs/personas.json" ;;
    esac
fi

# Optional: require a specific idiom (from idioms_definitions/) to be used per round.
# We only have a Danish idiom/definition dataset right now, so it's the only auto-default;
# pass a 6th arg explicitly to use a different file, or "none" to disable even for Danish.
if [ -n "$6" ]; then
    if [ "$6" = "none" ]; then
        idioms_file=""
    else
        idioms_file="$6"
    fi
elif [ "$language" = "Danish" ]; then
    idioms_file="../idioms_definitions/talemaader_csv/talemaader_leverance_1.csv"
else
    idioms_file=""
fi

# Some vLLM custom ops (e.g. Gemma's RMSNorm) fall back to an internal
# torch.compile path regardless of enforce_eager. suppress_errors makes
# torch._dynamo fall back to eager per-graph on a compile failure instead
# of crashing, so we don't need a working C compiler at all.
# sbatch copies this script into a spool directory and runs it from there,
# so $(dirname "$0") points at the spool dir, not the repo's scripts/ dir.
# Hardcode the real path instead (this script is already account/env-specific).
cd /proj/dl4nlp/users/x_jenku/magpie/scripts

timestamp=$(date +%s)
job_name="${model_path##*/}_${language}_${timestamp}"
job_path="../data/${job_name}"
mkdir -p "$job_path"

# When cycling through every persona (--persona all), gen_ins.py picks one
# persona per generation round, and repeat = ceil(total_prompts / n). So n
# needs to be small enough that repeat covers all personas at least once.
if [ "$persona" = "all" ]; then
    num_personas=$(run python3 -c "import json; print(len(json.load(open('$personas_file'))))")
    n=$(( total_prompts / num_personas ))
    if [ "$n" -lt 1 ]; then
        n=1
    fi
else
    n=$total_prompts
fi

echo "[run_magpie_sif] Model: $model_path"
echo "[run_magpie_sif] Total prompts: $total_prompts"
echo "[run_magpie_sif] Language: $language"
echo "[run_magpie_sif] Persona: $persona"
echo "[run_magpie_sif] Personas file: $personas_file"
echo "[run_magpie_sif] Idioms file: ${idioms_file:-none}"
echo "[run_magpie_sif] n per round: $n"
echo "[run_magpie_sif] Job name: $job_name"

idiom_args=()
if [ -n "$idioms_file" ]; then
    idiom_args=(--idioms_file "$idioms_file")
fi

echo "[run_magpie_sif] Generating instructions..."
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
    --persona "$persona" \
    --personas_file "$personas_file" \
    "${idiom_args[@]}" \
    --job_name "$job_name" \
    --timestamp $timestamp

ins_file="${job_path}/Magpie_${model_path##*/}_${total_prompts}_${timestamp}_ins.json"

echo "[run_magpie_sif] Generating responses..."
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

echo "[run_magpie_sif] Done. Output: ${ins_file%.json}_res.json"
