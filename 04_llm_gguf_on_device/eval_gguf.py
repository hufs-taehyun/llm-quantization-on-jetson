"""04: GGUF 파일을 llama-server로 띄워 ARC-Challenge 다음 토큰 확률로 채점한다 (계획: PLAN.md).

프롬프트는 02와 같은 문장을 HF 채팅 템플릿으로 만든 문자열 그대로 /completion에 보낸다.
출력 형식은 02의 pred_*.jsonl과 같아서 02/analyze.py를 그대로 쓴다 (F16→fp16, Q8_0→w8, Q4_K_M→w4).
사용: python eval_gguf.py --gguf x.gguf --model ~/llm/qwen15 --name fp16 --out results_qwen [--split validation --limit 20]
"""
import argparse, json, subprocess, time, urllib.request
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--gguf", required=True)
p.add_argument("--model", required=True, help="HF 모델 폴더 (채팅 템플릿용)")
p.add_argument("--name", required=True, choices=["fp16", "w8", "w4"])
p.add_argument("--out", required=True)
p.add_argument("--split", default="test")
p.add_argument("--limit", type=int, default=0)
p.add_argument("--data", default=str(Path.home() / "llm"))
p.add_argument("--server", default=str(Path.home() / "llama.cpp/build/bin/llama-server"))
p.add_argument("--port", type=int, default=8089)
a = p.parse_args()
LET, NPROBS = ["A", "B", "C", "D"], 100
URL = f"http://127.0.0.1:{a.port}"


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def letter_probs(resp):
    """응답의 첫 토큰 상위 확률에서 A~D(공백 붙은 것 포함)를 글자별로 합친다. 서버 판에 따라 필드 이름이 다르다."""
    first = resp["completion_probabilities"][0]
    cand = first.get("top_logprobs") or first.get("top_probs") or first.get("probs")
    lp = dict.fromkeys(LET, 0.0)
    for c in cand:
        s = c.get("token", c.get("tok_str", "")).strip(" ")
        pr = c["prob"] if "prob" in c else 2.718281828459045 ** c["logprob"]
        if s in lp:
            lp[s] += pr
    return [lp[L] for L in LET]


def score(lp, answer, qid):
    mass = sum(lp); n = [x / max(mass, 1e-12) for x in lp]
    pred = LET[n.index(max(n))]
    return {"id": qid, "answer": answer, "probs": n, "letter_mass": mass, "pred": pred, "conf": max(n), "correct": pred == answer}


def _selfcheck():
    r = {"completion_probabilities": [{"top_logprobs": [{"token": " B", "logprob": -0.5108}, {"token": "B", "logprob": -2.3026},
                                                         {"token": "A", "logprob": -1.6094}, {"token": "x", "logprob": -3.0}]}]}
    lp = letter_probs(r); s = score(lp, "B", "q")
    assert abs(lp[1] - 0.7) < 1e-3 and abs(lp[0] - 0.2) < 1e-3 and s["pred"] == "B" and abs(s["conf"] - 0.7 / 0.9) < 1e-3
    r2 = {"completion_probabilities": [{"probs": [{"tok_str": "C", "prob": 0.9}]}]}
    assert score(letter_probs(r2), "A", "q")["correct"] is False


if __name__ == "__main__":
    _selfcheck()
    tok = AutoTokenizer.from_pretrained(a.model)
    df = pd.read_parquet(f"{a.data}/arc_{a.split}.parquet")
    df = df[df["choices"].apply(lambda c: list(c["label"]) == LET)].reset_index(drop=True)
    df = df.head(a.limit) if a.limit else df
    out = Path(a.out); out.mkdir(exist_ok=True)
    # --cache-ram 0: 서버는 지난 프롬프트 상태를 기본 8GB까지 RAM에 쌓는다. Jetson 전체 메모리(7.6GB)보다 커서
    # SmolLM2 F16에서 기기가 멈췄다(README 실행 기록). 슬롯도 1개로 고정.
    srv = subprocess.Popen([a.server, "-m", a.gguf, "-ngl", "99", "-c", "2048", "-np", "1", "--cache-ram", "0",
                            "--port", str(a.port), "--no-webui"],
                           stdout=open(out / f"server_{a.name}.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(120):  # 모델 로딩 대기
            try:
                if json.load(urllib.request.urlopen(URL + "/health", timeout=2)).get("status") == "ok":
                    break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("server not ready")
        rows, t = [], time.time()
        avail = lambda: int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 1024
        for i, r in df.iterrows():
            if i % 100 == 0:
                print(a.name, "q", i, "mem_avail_mb", avail(), flush=True)
                if avail() < 600:  # 기기가 멈추기 전에 스스로 끝낸다
                    raise SystemExit(f"LOW_MEMORY at question {i}")
            opts = "\n".join(f"{l}. {x}" for l, x in zip(r["choices"]["label"], r["choices"]["text"]))
            msg = [{"role": "user", "content": f"{r['question']}\n{opts}\nAnswer with only the letter of the correct option."}]
            prompt = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
            resp = post("/completion", {"prompt": prompt, "n_predict": 1, "n_probs": NPROBS, "temperature": 0, "cache_prompt": False})
            rows.append(score(letter_probs(resp), r["answerKey"], r["id"]))
        fname = f"pred_{a.name}.jsonl" if a.split == "test" and not a.limit else f"smoke_{a.name}.jsonl"
        with open(out / fname, "w", encoding="utf-8") as fh:
            for x in rows:
                fh.write(json.dumps(x) + "\n")
        n = len(rows)
        print(a.name, {"n": n, "acc": sum(x["correct"] for x in rows) / n,
                       "confident_wrong": sum((not x["correct"]) and x["conf"] >= 0.8 for x in rows) / n,
                       "mean_letter_mass": sum(x["letter_mass"] for x in rows) / n}, f"({time.time() - t:.0f}s)", flush=True)
    finally:
        srv.terminate(); srv.wait()
