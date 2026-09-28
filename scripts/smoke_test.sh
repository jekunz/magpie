model_path=${1:-"google/gemma-3-27b-it"}
total_prompts=${2:-100}
language=${3:-"Swedish"}
persona=${4:-"se_ake_dalarna"}
device="0"
tensor_parallel=1
gpu_memory_utilization=0.9

timestamp=$(date +%s)
job_name="${model_path##*/}_smoketest_${timestamp}"
job_path="../data/${job_name}"
mkdir -p "$job_path"

echo "[smoke_test] Model: $model_path"
echo "[smoke_test] Total prompts: $total_prompts"
echo "[smoke_test] Language: $language"
echo "[smoke_test] Persona: $persona"

echo "[smoke_test] Generating instructions..."
CUDA_VISIBLE_DEVICES=$device python ../exp/gen_ins.py \
    --device $device \
    --model_path "$model_path" \
    --total_prompts $total_prompts \
    --n $total_prompts \
    --top_p 1 \
    --temperature 1 \
    --tensor_parallel_size $tensor_parallel \
    --gpu_memory_utilization $gpu_memory_utilization \
    --enforce_eager \
    --language "$language" \
    --persona "$persona" \
    --job_name "$job_name" \
    --timestamp $timestamp

ins_file="${job_path}/Magpie_${model_path##*/}_${total_prompts}_${timestamp}_ins.json"

echo "[smoke_test] Generating responses..."
CUDA_VISIBLE_DEVICES=$device python ../exp/gen_res.py \
    --device $device \
    --model_path "$model_path" \
    --batch_size $total_prompts \
    --top_p 1 \
    --temperature 0 \
    --repetition_penalty 1 \
    --tensor_parallel_size $tensor_parallel \
    --gpu_memory_utilization $gpu_memory_utilization \
    --enforce_eager \
    --input_file "$ins_file" \
    --offline

echo "[smoke_test] Done. Output: ${ins_file%.json}_res.json"
