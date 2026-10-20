#!/bin/bash
# Submit English → target-language Magpie translation jobs.
# Usage: bash translate_all.sh [plain|idiomatic|both] [lang_code ...]
#   bash translate_all.sh                 # both variants, all languages
#   bash translate_all.sh idiomatic de fo # idiomatic only, German + Faroese
# Env overrides: INPUT (default ../data/magpie_eng_10000.jsonl, relative to scripts/), MODEL (default google/gemma-3-27b-it)

dir="$(cd "$(dirname "$0")" && pwd)"
variant=${1:-both}
shift

lang_name() {
    case "$1" in
        de) echo German ;;    da) echo Danish ;;      sv) echo Swedish ;;   is) echo Icelandic ;; fo) echo Faroese ;;
        es) echo Spanish ;;   pt) echo Portuguese ;;  it) echo Italian ;;   fr) echo French ;;    ro) echo Romanian ;;
        ru) echo Russian ;;   uk) echo Ukrainian ;;   bg) echo Bulgarian ;; sl) echo Slovenian ;; pl) echo Polish ;;
    esac
}
ORDER=(de da sv is fo es pt it fr ro ru uk bg sl pl)

codes=("$@")
[ ${#codes[@]} -eq 0 ] && codes=("${ORDER[@]}")

case "$variant" in
    both) variants=(plain idiomatic) ;;
    plain|idiomatic) variants=("$variant") ;;
    *) echo "Unknown variant: $variant (expected plain, idiomatic or both)"; exit 1 ;;
esac

input=${INPUT:-../data/magpie_eng_10000.jsonl}
model=${MODEL:-google/gemma-3-27b-it}
if [ ! -f "$dir/$input" ]; then
    echo "Input $dir/$input not found — run 'python ../exp/prepare_eng_source.py' from scripts/ first."
    exit 1
fi

for code in "${codes[@]}"; do
    lang=$(lang_name "$code")
    if [ -z "$lang" ]; then
        echo "Unknown language code: $code"; exit 1
    fi
    for v in "${variants[@]}"; do
        tag=$([ "$v" = idiomatic ] && echo "-idio")
        sbatch --chdir="$dir" -J "gemma27-${code}${tag}" \
            "$dir/run_translate.sh" "$v" "$lang" "$code" "$input" "$model"
    done
done
