"""eval_mc.py 결과를 계획(PLAN.md)에 정한 지표·통계로 정리한다.

- 정확도, 확신 오답률(conf>=0.8 & 오답), ECE(10구간), F16 대비 판정 일치율과 바뀐 방향
- 문항 단위 부트스트랩 10,000회(seed 0) 95% 구간, F16과의 짝 차이
- 주된 비교 2개(확신 오답률 Q8_0-F16, Q4_K_M-F16)에 Holm 보정(부트스트랩 양측 p)
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


def main(d):
    d = Path(d)
    base = load(d / "pred_f16.jsonl")
    ids = sorted(base)
    rng = np.random.default_rng(SEED)
    res, primary = {}, {}
    arr = lambda rows, k: np.array([rows[i][k] for i in ids])
    bc, bk = arr(base, "conf"), arr(base, "correct").astype(bool)
    res["f16"] = metrics(bc, bk)
    for m in ("q8_0", "q4_k_m"):
        f = d / f"pred_{m}.jsonl"
        if not f.exists():
            continue
        r = load(f)
        c, k = arr(r, "conf"), arr(r, "correct").astype(bool)
        out = metrics(c, k)
        out["agree_f16"] = float(np.mean(arr(r, "pred") == arr(base, "pred")))
        out["right_to_wrong"] = int(np.sum(bk & ~k)); out["wrong_to_right"] = int(np.sum(~bk & k))
        for name, x, y in (("acc_diff", bk.astype(float), k.astype(float)),
                           ("confident_wrong_diff", ((~bk) & (bc >= THR)).astype(float), ((~k) & (c >= THR)).astype(float))):
            dist = boot_diff(x, y, rng)
            out[name] = {"est": float(y.mean() - x.mean()), "ci95": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))]}
            if name == "confident_wrong_diff":
                primary[m] = float(min(1.0, 2 * min((dist <= 0).mean(), (dist >= 0).mean())))
        res[m] = out
    if primary:
        names = list(primary); adj = holm(np.array([primary[n] for n in names]))
        for n, pa in zip(names, adj):
            res[n]["confident_wrong_diff"]["p_boot"] = primary[n]; res[n]["confident_wrong_diff"]["p_holm"] = float(pa)
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
