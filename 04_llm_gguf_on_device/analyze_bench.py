"""run_bench.sh 출력(llama-bench json + tegrastats)을 표로 정리한다 (계획: PLAN.md 측정 1).

- 속도: llama-bench의 프롬프트 처리·생성 tok/s (5회 평균·표준편차·범위)
- 메모리: tegrastats RAM 최댓값 − 실행 전 5초 평균 (CPU·GPU 공유 메모리 전체 기준)
- 전력: VDD_IN 평균. 계획대로 실행 전체 구간 평균을 쓴다(프롬프트·생성 구간을 나눌 수 없어 근사).
  참고로 GPU 사용률 50% 이상인 표본만의 평균도 함께 낸다(사후, 모델 로딩 시간을 뺀 값).
- 토큰당 에너지(근사) = 전력 평균 ÷ 생성 tok/s
사용: python analyze_bench.py --dir bench
"""
import argparse, json, re
from pathlib import Path

import numpy as np

RX = re.compile(r"(\d\d:\d\d:\d\d) RAM (\d+)/.*?GR3D_FREQ (\d+)%.*?tj@([\d.]+)C.*?VDD_IN (\d+)mW")


def sec(t):
    h, m, s = map(int, t.split(":")); return h * 3600 + m * 60 + s


def tegra(path):
    return [(sec(t), int(ram), int(gpu), float(tj), int(mw)) for t, ram, gpu, tj, mw in RX.findall(open(path).read())]


def one(d, name):
    meta = dict(l.split(" ", 1) for l in open(d / f"{name}.meta").read().splitlines() if l.startswith(("bench_", "start")))
    t0, t1 = sec(meta["bench_start"].split()[0]), sec(meta["bench_end"].split()[0])
    oc0, oc1 = int(meta["start"].split("oc3 ")[1]), int(meta["bench_end"].split("oc3 ")[1])
    rows = tegra(d / f"{name}.tegra")
    pre = [r for r in rows if r[0] < t0]; run = [r for r in rows if t0 <= r[0] <= t1]
    act = [r for r in run if r[2] >= 50]
    out = {"seconds": t1 - t0, "ram_base_mb": float(np.mean([r[1] for r in pre])), "ram_peak_mb": max(r[1] for r in run),
           "power_idle_w": float(np.mean([r[4] for r in pre])) / 1000, "power_run_w": float(np.mean([r[4] for r in run])) / 1000,
           "power_gpu_active_w": float(np.mean([r[4] for r in act])) / 1000 if act else None,
           "temp_start_c": run[0][3], "temp_end_c": run[-1][3], "oc3_events": oc1 - oc0}
    out["ram_used_mb"] = out["ram_peak_mb"] - out["ram_base_mb"]
    for r in json.load(open(d / f"{name}.json")):
        k = "prompt" if r["n_prompt"] else "gen"
        out[k] = {"mean": r["avg_ts"], "std": r["stddev_ts"], "min": min(r["samples_ts"]), "max": max(r["samples_ts"])}
    out["j_per_token_run"] = out["power_run_w"] / out["gen"]["mean"]
    if act:
        out["j_per_token_gpu_active"] = out["power_gpu_active_w"] / out["gen"]["mean"]
    return out


def _selfcheck():
    line = "11-19-2026 04:11:10 RAM 632/7620MB (lfb 3x2MB) SWAP 247/3810MB (cached 1MB) CPU [0%@729] GR3D_FREQ 7% cpu@44.6C tj@45.437C soc1@45C VDD_IN 3456mW/3456mW VDD_SOC 1083mW/1083mW"
    assert RX.findall(line) == [("04:11:10", "632", "7", "45.437", "3456")] and sec("01:00:01") == 3601


if __name__ == "__main__":
    _selfcheck()
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="bench"); d = Path(ap.parse_args().dir)
    res = {f"{m}-{q}": one(d, f"{m}-{q}") for m in ("qwen15", "smol17") for q in ("f16", "Q8_0", "Q4_K_M")}
    json.dump(res, open(d / "bench_summary.json", "w"), indent=2)
    for n, r in res.items():
        print(f"{n:15s} gen {r['gen']['mean']:5.1f} [{r['gen']['min']:.1f},{r['gen']['max']:.1f}] prompt {r['prompt']['mean']:6.0f} "
              f"[{r['prompt']['min']:.0f},{r['prompt']['max']:.0f}] ram +{r['ram_used_mb']:.0f}MB W run {r['power_run_w']:.1f} "
              f"active {r['power_gpu_active_w'] or 0:.1f} idle {r['power_idle_w']:.1f} J/tok {r['j_per_token_run']:.2f}/"
              f"{r.get('j_per_token_gpu_active', 0):.2f} temp {r['temp_start_c']:.0f}->{r['temp_end_c']:.0f} oc {r['oc3_events']} {r['seconds']}s")
