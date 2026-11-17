"""ARC-Challenge 객관식에서 답 글자(A~D)의 다음 토큰 확률로 정답·확신도를 잰다 (계획: PLAN.md).

모델마다 llama-server를 띄우고, 세 모델에 같은 프롬프트를 보낸다. 생성 문장은 파싱하지 않는다.
사용: python eval_mc.py --models f16 q8_0 q4_k_m --out results
"""
import argparse, json, subprocess, time, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--models", nargs="+", default=["f16", "q8_0", "q4_k_m"])
p.add_argument("--work", default=str(Path.home() / "llm"))
p.add_argument("--out", default="results")
p.add_argument("--limit", type=int, default=0, help="0이면 전체 (점검용으로만 사용)")
a = p.parse_args()
W, out = Path(a.work), Path(a.out); out.mkdir(exist_ok=True)
LETTERS = ["A", "B", "C", "D"]
PORT = 8089

df = pd.read_parquet(W / "arc_challenge_test.parquet")
df = df[df["choices"].apply(lambda c: list(c["label"]) == LETTERS)].reset_index(drop=True)
if a.limit:
    df = df.head(a.limit)
print("questions", len(df), flush=True)


def prompt(row):
    opts = "\n".join(f"{l}. {t}" for l, t in zip(row["choices"]["label"], row["choices"]["text"]))
    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{row['question']}\n{opts}\nAnswer with only the letter of the correct option.<|im_end|>\n"
            "<|im_start|>assistant\n")


def post(payload):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def letter_probs(resp):
    """다음 토큰 후보 중 A~D에 해당하는 확률을 모아 4개로 정규화한다."""
    cp = resp["completion_probabilities"][0]
    cands = cp.get("top_logprobs") or cp.get("probs")
    acc = dict.fromkeys(LETTERS, 0.0)
    for c in cands:
        tok = c.get("token", c.get("tok_str", "")).strip()
        pr = np.exp(c["logprob"]) if "logprob" in c else c["prob"]
        if tok in acc:
            acc[tok] += pr
    s = sum(acc.values())
    return [acc[l] / s if s > 0 else 0.25 for l in LETTERS], s


for m in a.models:
    gguf = W / f"qwen15-{m}.gguf"
    srv = subprocess.Popen([str(W / "llama.cpp/build/bin/llama-server"), "-m", str(gguf), "-ngl", "99",
                            "-c", "2048", "--port", str(PORT), "-np", "1"],
                           stdout=subprocess.DEVNULL, stderr=open(out / f"server_{m}.log", "w"))
    try:
        for _ in range(120):
            try:
                if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        rows, t0 = [], time.time()
        for _, r in df.iterrows():
            resp = post({"prompt": prompt(r), "n_predict": 1, "n_probs": 100, "temperature": 0,
                         "cache_prompt": False})
            pr, mass = letter_probs(resp)
            pred = LETTERS[int(np.argmax(pr))]
            rows.append({"id": r["id"], "answer": r["answerKey"], "probs": pr, "letter_mass": mass,
                         "pred": pred, "conf": max(pr), "correct": pred == r["answerKey"]})
        with open(out / f"pred_{m}.jsonl", "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
        acc = np.mean([x["correct"] for x in rows])
        cw = np.mean([(not x["correct"]) and x["conf"] >= 0.8 for x in rows])
        print(f"{m} acc {acc:.3f} confident_wrong {cw:.3f} n {len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        srv.terminate(); srv.wait()
print("EVAL_DONE")
