"""HW4.a part 2 and HW4.5 Part D: vLLM against a plain HF pipeline.

python serve.py hf          HF pipeline, batch 1 (baseline)
python serve.py vllm TAG    vLLM at batch 1, 8, 32; TAG is hw4 or partd
"""
import sys, time
import numpy as np
from common import *

engine, tag = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "hw4")
NEW = 16 if SMOKE else 128
QUESTIONS = ["Explain why the sky is blue to a ten year old.", "Give three tips for a first time marathon runner.",
             "Write a short poem about debugging code at midnight.", "What is the difference between a GPU and a CPU?",
             "Summarize the plot of a heist movie you invent.", "How do vaccines train the immune system?",
             "Suggest a weekend plan in San Jose on a student budget.", "Why does BERT use a [CLS] token?"]

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(LLM)
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
           for q in QUESTIONS]


def bench(run, b, n_req):
    """run(list_of_prompts) returns generated token counts. Latency of a request is its batch call time."""
    run(prompts[:b])   # warmup
    lat, toks, t0 = [], 0, time.time()
    for i in range(0, n_req, b):
        batch = [prompts[(i + j) % len(prompts)] for j in range(b)]
        t = time.time()
        toks += sum(run(batch))
        lat += [time.time() - t] * b
    return dict(batch=b, requests=n_req, tokens_per_s=toks / (time.time() - t0),
                p50_ms=1000 * np.percentile(lat, 50), p95_ms=1000 * np.percentile(lat, 95))


def main():
    if engine == "hf":
        import torch
        from transformers import pipeline
        pipe = pipeline("text-generation", model=LLM, dtype=torch.float16, device=DEV)

        def run(batch):
            out = pipe(batch[0], max_new_tokens=NEW, min_new_tokens=NEW, do_sample=False, return_full_text=False)
            return [len(tok(out[0]["generated_text"], add_special_tokens=False).input_ids)]
        rows = [bench(run, 1, 4 if SMOKE else 16)]
        sample = pipe(prompts[0], max_new_tokens=NEW, do_sample=False, return_full_text=False)[0]["generated_text"]
    else:
        from vllm import LLM as VLLM, SamplingParams
        llm = VLLM(model=LLM, dtype="half", max_model_len=2048, gpu_memory_utilization=0.8, seed=SEED)
        sp = SamplingParams(temperature=0, max_tokens=NEW, ignore_eos=True)

        def run(batch):
            return [len(o.outputs[0].token_ids) for o in llm.generate(batch, sp, use_tqdm=False)]
        rows = [bench(run, b, max(16, 4 * b)) for b in (1, 8, 32)]
        sample = llm.generate(prompts[:1], SamplingParams(temperature=0, max_tokens=NEW))[0].outputs[0].text

    for r in rows:
        print(r)
    save(f"serve_{engine}_{tag}", dict(model=LLM, dtype="float16", new_tokens=NEW, prompt=QUESTIONS[0],
                                       sample_output=sample, rows=rows))


if __name__ == "__main__":   # vLLM spawns a child process that re-imports this file
    main()
