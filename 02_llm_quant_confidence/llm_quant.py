"""02: PyTorch로 LLM 가중치 양자화(RTN W8 채널별 / W4 그룹 128)를 직접 구현해
ARC-Challenge에서 정확도·확신 오답을 재고, 층별 민감도로 혼합 정밀도를 정한다 (계획: PLAN.md).

단계
  smoke : validation 20문항, FP16 — 채점(글자 확률 질량)과 속도 점검
  sens  : validation 전체 — FP16, 전체 W8/W4, 층 하나만 W4, 모듈 종류 하나만 W4 → 층 순위 저장
  final : test 전체 — FP16, W8, W4, Mixed-k2, Mixed-k4 (층 순위는 sens 결과에서만 가져옴)
사용: python llm_quant.py --stage smoke|sens|final --out results
"""
import argparse, glob, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--stage", choices=["smoke", "sens", "final"], required=True)
p.add_argument("--model", default=str(Path.home() / "llm/qwen15"))
p.add_argument("--data", default=str(Path.home() / "llm"))
p.add_argument("--out", default="results")
p.add_argument("--dataset", choices=["arc", "mmlu"], default="arc")
p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16", help="양자화하지 않은 층의 자료형")
a = p.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)
LET, THR = ["A", "B", "C", "D"], 0.8
MMLU_N = 2000  # MMLU test에서 seed 0으로 뽑는 문항 수 (실행 전 고정, PLAN.md 확장 절)
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(a.model)
DT = torch.float16 if a.dtype == "fp16" else torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=DT).cuda().eval()
LIN = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.startswith("model.layers.")}
N_LAYERS = model.config.num_hidden_layers
KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 원본 가중치는 메모리에 복사하지 않고 safetensors에서 필요할 때만 다시 읽는다 (Jetson은 CPU·GPU 메모리 공유)
SRC = {}
for f in glob.glob(f"{a.model}/*.safetensors"):
    h = safe_open(f, framework="pt")
    for k in h.keys():
        SRC[k] = h
STATE = dict.fromkeys(LIN, "fp16")


@torch.no_grad()
def rtn(w, mode):
    w = w.float()
    if mode == "w8":  # 출력 채널별 대칭 INT8
        s = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
        return ((w / s).round().clamp(-127, 127) * s)
    if mode == "w4":  # 입력 방향 128개 그룹별 대칭 INT4 [-8, 7]
        o, i = w.shape
        g = w.reshape(o, i // 128, 128)
        s = g.abs().amax(2, keepdim=True).clamp(min=1e-8) / 7
        return ((g / s).round().clamp(-8, 7) * s).reshape(o, i)
    raise ValueError(mode)


@torch.no_grad()
def apply(cfg):
    """cfg: 이름 → 'fp16'|'w8'|'w4' (없으면 fp16). 바뀐 층만 원본에서 다시 읽어 적용."""
    for n, m in LIN.items():
        want = cfg.get(n, "fp16")
        if STATE[n] == want:
            continue
        w0 = SRC[n + ".weight"].get_tensor(n + ".weight").cuda()
        m.weight.data.copy_(w0 if want == "fp16" else rtn(w0, want).to(DT))
        STATE[n] = want


def letter_ids():
    ids = {}
    for L in LET:
        ids[L] = sorted({tok.encode(x, add_special_tokens=False)[0] for x in (L, " " + L)})
    return ids


LID = letter_ids()


def load(split, limit=0):
    if a.dataset == "arc":
        df = pd.read_parquet(f"{a.data}/arc_{split}.parquet")
        df = df[df["choices"].apply(lambda c: list(c["label"]) == LET)].reset_index(drop=True)
    else:  # MMLU(보기 4개 목록 + 정답 번호)를 ARC 형식으로 맞춘다. test만 사용
        m = pd.read_parquet(f"{a.data}/mmlu_test.parquet")
        df = pd.DataFrame({"id": [f"{s}_{i}" for i, s in enumerate(m["subject"])], "question": m["question"],
                           "choices": [{"label": LET, "text": list(c)} for c in m["choices"]],
                           "answerKey": [LET[int(x)] for x in m["answer"]]})
        df = df.sample(n=MMLU_N, random_state=0).reset_index(drop=True)
    return df.head(limit) if limit else df


def enc(row):
    opts = "\n".join(f"{l}. {t}" for l, t in zip(row["choices"]["label"], row["choices"]["text"]))
    msg = [{"role": "user", "content": f"{row['question']}\n{opts}\nAnswer with only the letter of the correct option."}]
    return tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").cuda()


@torch.no_grad()
def evaluate(df, cache):
    rows = []
    for i, r in df.iterrows():
        pr = torch.softmax(model(cache[i]).logits[0, -1].float(), -1)
        lp = np.array([pr[LID[L]].sum().item() for L in LET])
        mass = float(lp.sum()); lp = lp / max(mass, 1e-12)
        pred = LET[int(lp.argmax())]
        rows.append({"id": r["id"], "answer": r["answerKey"], "probs": lp.tolist(), "letter_mass": mass,
                     "pred": pred, "conf": float(lp.max()), "correct": pred == r["answerKey"]})
    return rows


def summ(rows, ref=None):
    c = np.array([x["correct"] for x in rows]); f = np.array([x["conf"] for x in rows])
    s = {"acc": float(c.mean()), "confident_wrong": float(((~c) & (f >= THR)).mean()),
         "mean_letter_mass": float(np.mean([x["letter_mass"] for x in rows]))}
    if ref is not None:
        s["agree_fp16"] = float(np.mean([x["pred"] == y["pred"] for x, y in zip(rows, ref)]))
    return s


def all_mode(mode):
    return {n: mode for n in LIN}


def save(rows, name):
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")


if a.stage == "smoke":
    df = load("validation" if a.dataset == "arc" else "test", 20); cache = [enc(r) for _, r in df.iterrows()]
    t = time.time(); rows = evaluate(df, cache); dt = (time.time() - t) / len(df)
    print("smoke", summ(rows), f"{dt * 1000:.0f} ms/question", flush=True)
    apply(all_mode("w4")); print("smoke w4", summ(evaluate(df, cache), rows), flush=True); apply({})
    print("DONE")

elif a.stage == "sens":
    df = load("validation"); cache = [enc(r) for _, r in df.iterrows()]
    print("validation questions", len(df), flush=True)
    res = {}
    apply({}); ref = evaluate(df, cache); res["fp16"] = summ(ref); save(ref, "val_fp16")
    for mode in ("w8", "w4"):
        apply(all_mode(mode)); res[f"all_{mode}"] = summ(evaluate(df, cache), ref); print(mode, res[f"all_{mode}"], flush=True)
    for li in range(N_LAYERS):
        apply({n: "w4" for n in LIN if n.startswith(f"model.layers.{li}.")})
        res[f"layer{li}_w4"] = summ(evaluate(df, cache), ref); print(f"layer{li}", res[f"layer{li}_w4"], flush=True)
    for k in KINDS:
        apply({n: "w4" for n in LIN if n.endswith(k)})
        res[f"kind_{k}_w4"] = summ(evaluate(df, cache), ref); print(k, res[f"kind_{k}_w4"], flush=True)
    apply({})
    base_cw = res["fp16"]["confident_wrong"]
    rank = sorted(range(N_LAYERS), key=lambda li: (-(res[f"layer{li}_w4"]["confident_wrong"] - base_cw),
                                                   res[f"layer{li}_w4"]["agree_fp16"]))
    res["layer_rank"] = rank
    json.dump(res, open(out / "sens_validation.json", "w", encoding="utf-8"), indent=2)
    print("layer_rank", rank[:8]); print("DONE")

else:
    sens = out / "sens_validation.json"  # 없으면(MMLU) 혼합 정밀도 없이 FP16·W8·W4만
    rank = json.load(open(sens, encoding="utf-8"))["layer_rank"] if sens.exists() else []
    df = load("test"); cache = [enc(r) for _, r in df.iterrows()]
    print("test questions", len(df), "rank top4", rank[:4], flush=True)
    cfgs = {"fp16": {}, "w8": all_mode("w8"), "w4": all_mode("w4")}
    for k in ((2, 4) if rank else ()):
        keep = {f"model.layers.{li}." for li in rank[:k]}
        cfgs[f"mixed_k{k}"] = {n: ("w8" if any(n.startswith(p_) for p_ in keep) else "w4") for n in LIN}
    ref, summary = None, {}
    for name, cfg in cfgs.items():
        apply(cfg); t = time.time(); rows = evaluate(df, cache)
        ref = ref or rows
        summary[name] = summ(rows, ref); save(rows, f"pred_{name}")
        print(name, summary[name], f"({time.time() - t:.0f}s)", flush=True)
    apply({})
    json.dump(summary, open(out / "test_quick_summary.json", "w", encoding="utf-8"), indent=2)
    print("DONE")
