#!/bin/bash
# Submit basic (no persona) Magpie jobs for every language in this directory.
# Usage: bash basic/all.sh [total_prompts] [model_path]
dir="$(cd "$(dirname "$0")" && pwd)"
for f in german danish swedish icelandic faroese spanish portuguese italian french romanian russian ukrainian bulgarian slovenian polish luxembourgish norwegian dutch catalan galician belarusian croatian czech slovak serbian macedonian finnish estonian hungarian; do
    bash "$dir/$f.sh" "$@"
done
