"""02 실제 커널 측정: torchao weight-only INT8 / INT4(group 128) vs FP16·BF16 (계획: PLAN.md).

측정: 프롬프트 처리 지연, 생성 tokens/s(128토큰 greedy), 최대 GPU 메모리, tegrastats VDD_IN 평균.
torchao INT4 커널은 BF16 모델을 요구하므로 BF16 기준도 함께 잰다. 커널이 동작하지 않으면 에러를 그대로 기록한다.
사용: python bench_torchao.py --variant fp16 --out llm_results  (구성마다 별도 프로세스로 실행)
"""
import argparse, gc, json, re, statistics, subprocess, time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default=str(Path.home() / "llm/qwen15"))
p.add_argument("--data", default=str(Path.home() / "llm"))
p.add_argument("--out", default="llm_results")
p.add_argument("--reps", type=int, default=5)
p.add_argument("--new-tokens", type=int, default=128)
p.add_argument("--variant", required=True, choices=["fp16", "bf16", "int8wo_fp16", "int4wo_g128_bf16", "int8wo_bf16"],
               help="구성 하나만 실행(메모리 부족 방지). int8wo_bf16은 사후 확인용")
a = p.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)

tok = AutoTokenizer.from_pretrained(a.model)
row = pd.read_parquet(f"{a.data}/arc_test.parquet").iloc[0]
opts = "\n".join(f"{l}. {t}" for l, t in zip(row["choices"]["label"], row["choices"]["text"]))
msg = [{"role": "user", "content": f"{row['question']}\n{opts}\nExplain your reasoning step by step, then give the letter."}]
ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").cuda()


def torchao_cfg(kind):
    import torchao.quantization as tq
    if kind == "int8wo":
        return tq.Int8WeightOnlyConfig() if hasattr(tq, "Int8WeightOnlyConfig") else tq.int8_weight_only()
    return tq.Int4WeightOnlyConfig(group_size=128) if hasattr(tq, "Int4WeightOnlyConfig") else tq.int4_weight_only(group_size=128)


class Power:
    """tegrastats로 VDD_IN(mW)을 200ms마다 읽는다. 사용할 수 없으면 None."""

    def __enter__(self):
        self.log = out / "_tegra.log"
        try:
            self.p = subprocess.Popen(["tegrastats", "--interval", "200", "--logfile", str(self.log)])
        except Exception:
            self.p = None
        return self

    def __exit__(self, *e):
        if self.p:
            self.p.terminate(); self.p.wait()

    def mean_mw(self):
        if not self.p or not self.log.exists():
            return None
        v = [int(m.group(1)) for m in re.finditer(r"VDD_IN (\d+)mW", self.log.read_text())]
        self.log.unlink()
        return statistics.mean(v) if v else None


@torch.no_grad()
def bench(name, dtype, kind=None):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    r = {"variant": name, "dtype": str(dtype)}
    try:
        m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).cuda().eval()
        if kind:
            from torchao.quantization import quantize_
            quantize_(m, torchao_cfg(kind))
        r["weights_mem_gb"] = torch.cuda.memory_allocated() / 1e9
        gen = dict(max_new_tokens=a.new_tokens, min_new_tokens=a.new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        for _ in range(2):  # 워밍업
            m(ids); m.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        pre, tot = [], []
        with Power() as pw:
            for _ in range(a.reps):
                torch.cuda.synchronize(); t = time.perf_counter(); m(ids); torch.cuda.synchronize(); pre.append(time.perf_counter() - t)
                t = time.perf_counter(); o = m.generate(ids, **gen); torch.cuda.synchronize(); tot.append(time.perf_counter() - t)
        r["power_mw_mean"] = pw.mean_mw()
        pf, tt = statistics.median(pre), statistics.median(tot)
        r.update(prompt_tokens=int(ids.shape[1]), prefill_ms=pf * 1000, total_s=tt,
                 decode_tok_s=a.new_tokens / max(tt - pf, 1e-9), peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                 sample=tok.decode(o[0, ids.shape[1]:ids.shape[1] + 40], skip_special_tokens=True))
        del m
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"[:500]
    print(json.dumps(r, ensure_ascii=False), flush=True)
    return r


VARIANTS = {"fp16": (torch.float16, None), "bf16": (torch.bfloat16, None),
            "int8wo_fp16": (torch.float16, "int8wo"), "int4wo_g128_bf16": (torch.bfloat16, "int4wo"),
            "int8wo_bf16": (torch.bfloat16, "int8wo")}
dtype, kind = VARIANTS[a.variant]
r = bench(("torchao_" if kind else "") + a.variant, dtype, kind)
r["posthoc"] = a.variant == "int8wo_bf16"
with open(out / "bench_torchao.jsonl", "a", encoding="utf-8") as fh:  # 구성마다 바로 저장
    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print("BENCH_DONE")
