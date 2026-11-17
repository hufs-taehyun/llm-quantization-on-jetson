"""llm_quant.py --stage final 결과(test)를 계획(PLAN.md)에 정한 지표·통계로 정리한다.

- 구성별 정확도, 확신 오답률(conf>=0.8 & 오답), ECE(10구간)
- 짝 비교: 정확도·확신 오답률 차이, 문항 단위 부트스트랩 10,000회(seed 0) 95% 구간, 판정 일치율, 맞→틀/틀→맞
- 주된 비교 3개(확신 오답률: W8−FP16, W4−FP16, Mixed-k4−W4)에 Holm 보정(부트스트랩 양측 p)
사용: python analyze.py --dir results
"""
import argparse, json
from pathlib import Path

import numpy as np

THR, B, SEED = 0.8, 10_000, 0


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return {r["id"]: r for r in rows}


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def metrics(conf, correct):
    return {"acc": float(correct.mean()),
            "confident_wrong": float(((~correct) & (conf >= THR)).mean()),
            "ece": ece(conf, correct)}


def boot_diff(x, y, rng):
    """짝지은 문항 지표 차이(y-x)의 부트스트랩 분포. x, y는 문항별 0/1 배열."""
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    return (y[idx].mean(1) - x[idx].mean(1))


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * pvals[i]); adj[i] = min(1.0, run)
    return adj


CONFIGS = ["fp16", "w8", "w4", "mixed_k2", "mixed_k4"]
# (비교 이름, 기준, 대상, 주된 비교 여부) — 실행 전 고정
COMPARISONS = [("w8-fp16", "fp16", "w8", True), ("w4-fp16", "fp16", "w4", True),
               ("mixed_k4-w4", "w4", "mixed_k4", True), ("mixed_k2-w4", "w4", "mixed_k2", False),
               ("mixed_k4-fp16", "fp16", "mixed_k4", False)]


def main(d):
    d = Path(d)
    P = {c: load(d / f"pred_{c}.jsonl") for c in CONFIGS if (d / f"pred_{c}.jsonl").exists()}
    ids = sorted(P["fp16"])
    arr = lambda c, k: np.array([P[c][i][k] for i in ids])
    rng = np.random.default_rng(SEED)
    res = {"n_questions": len(ids), "configs": {c: metrics(arr(c, "conf"), arr(c, "correct").astype(bool)) for c in P},
           "comparisons": {}}
    primary = {}
    for name, base, tgt, is_primary in COMPARISONS:
        if base not in P or tgt not in P:
            continue
        bc, bk = arr(base, "conf"), arr(base, "correct").astype(bool)
        c, k = arr(tgt, "conf"), arr(tgt, "correct").astype(bool)
        out = {"agree": float(np.mean(arr(tgt, "pred") == arr(base, "pred"))),
               "right_to_wrong": int(np.sum(bk & ~k)), "wrong_to_right": int(np.sum(~bk & k))}
        for mname, x, y in (("acc_diff", bk.astype(float), k.astype(float)),
                            ("confident_wrong_diff", ((~bk) & (bc >= THR)).astype(float), ((~k) & (c >= THR)).astype(float))):
            dist = boot_diff(x, y, rng)
            out[mname] = {"est": float(y.mean() - x.mean()),
                          "ci95": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))]}
            if mname == "confident_wrong_diff" and is_primary:
                primary[name] = float(min(1.0, 2 * min((dist <= 0).mean(), (dist >= 0).mean())))
        out["primary"] = is_primary
        res["comparisons"][name] = out
    if primary:
        names = list(primary); adj = holm(np.array([primary[n] for n in names]))
        for n, pa in zip(names, adj):
            cw = res["comparisons"][n]["confident_wrong_diff"]
            cw["p_boot"], cw["p_holm"] = primary[n], float(pa)
    json.dump(res, open(d / "summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


def _selfcheck():
    c = np.array([0.9, 0.9, 0.6, 0.95]); k = np.array([True, False, True, False])
    m = metrics(c, k)
    assert abs(m["acc"] - 0.5) < 1e-9 and abs(m["confident_wrong"] - 0.5) < 1e-9
    assert np.allclose(holm(np.array([0.01, 0.04])), [0.02, 0.04])
    assert ece(np.array([0.5, 0.5]), np.array([True, False])) < 1e-9


if __name__ == "__main__":
    _selfcheck()
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results")
    main(ap.parse_args().dir)
