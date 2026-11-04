"""
Debug the Magpie pre-query template of a model: generate a few instructions without any stop strings and print
exactly what comes out (raw text and first token ids), for two placements of the language task:
  system: task in the system turn, then an open user turn (the config's system_prompt_prefix/suffix)
  user:   task at the start of the open user turn, followed by a blank line (as for Gemma, which has no system role)

Usage (from exp/): python3 debug_prequery.py Qwen/Qwen3.8-27B Swedish
"""

import json
import sys

from vllm import LLM, SamplingParams

import str_utils
from prompt_utils import make_prompt_encoder


def main():
    model_path, language = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "Swedish"
    config = json.load(open("../configs/model_configs.json", encoding="utf-8"))[model_path]
    task = str_utils.build_persona_language_system_prompt(language=language)
    user_prefix = config["pre_query_template"]
    templates = {
        "system": config["system_prompt_prefix"] + task + config["system_prompt_suffix"],
        "user": user_prefix + task + "\n\n",
    }

    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9, enforce_eager=True,
              limit_mm_per_prompt={"image": 0}, **config.get("vllm_kwargs", {}))
    encode = make_prompt_encoder(model_path, config)
    params = SamplingParams(n=6, temperature=1.15, top_p=1.0, max_tokens=120, seed=0,
                            stop_token_ids=config["stop_token_ids"], skip_special_tokens=False)
    for name, template in templates.items():
        print(f"\n========== {name}: {template!r}")
        for o in llm.generate(encode(template), params)[0].outputs:
            print(f"--- finish={o.finish_reason} stop={o.stop_reason!r} first ids={list(o.token_ids[:6])}")
            print(repr(o.text[:300]))


if __name__ == "__main__":   # vLLM starts its engine in a spawned process
    main()
