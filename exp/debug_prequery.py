"""
Debug the Magpie pre-query template of a model: generate a few instructions without any stop strings and print
exactly what comes out (raw text and first token ids), for up to three prompt variants:
  system: the config's system_prompt_prefix + language task + system_prompt_suffix (what gen_ins.py uses)
  user:   language task at the start of the open user turn, followed by a blank line (Gemma 3 style)
  closed: language task as a complete user message, rendered with the model's chat template; the model's answer is
          the instruction (prompted generation rather than Magpie's open user turn)

Usage (from exp/):
  python3 debug_prequery.py Qwen/Qwen3.8-27B --language Swedish
  python3 debug_prequery.py google/gemma-4-31B-it --top_p 0.95 --top_k 64 --variants system closed
--min_tokens N suppresses the stop tokens for the first N generated tokens (forces a non-empty turn).
"""

import argparse
import json

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import str_utils
from prompt_utils import make_prompt_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--language", default="Swedish")
    parser.add_argument("--variants", nargs="+", default=["system", "user", "closed"],
                        choices=["system", "user", "closed"])
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.15)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_tokens", type=int, default=0)
    args = parser.parse_args()

    config = json.load(open("../configs/model_configs.json", encoding="utf-8"))[args.model_path]
    task = str_utils.build_persona_language_system_prompt(language=args.language)
    templates = {
        "system": config["system_prompt_prefix"] + task + config["system_prompt_suffix"],
        "user": config["pre_query_template"] + task + "\n\n",
    }
    if "closed" in args.variants:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        templates["closed"] = tokenizer.apply_chat_template([{"role": "user", "content": task}], tokenize=False,
                                                            add_generation_prompt=True)

    llm = LLM(model=args.model_path, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.9,
              enforce_eager=True, limit_mm_per_prompt={"image": 0}, **config.get("vllm_kwargs", {}))
    encode = make_prompt_encoder(args.model_path, config)
    params = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                            max_tokens=120, min_tokens=args.min_tokens, seed=0,
                            stop_token_ids=config["stop_token_ids"], skip_special_tokens=False)
    print(f"sampling: temperature={args.temperature} top_p={args.top_p} top_k={args.top_k} min_tokens={args.min_tokens}")
    for name in args.variants:
        template = templates[name]
        print(f"\n========== {name}: {template!r}")
        for o in llm.generate(encode(template), params)[0].outputs:
            print(f"--- finish={o.finish_reason} stop={o.stop_reason!r} first ids={list(o.token_ids[:6])}")
            print(repr(o.text[:300]))


if __name__ == "__main__":   # vLLM starts its engine in a spawned process
    main()
