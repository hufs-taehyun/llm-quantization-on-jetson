#!/bin/bash
# 측정 1: llama-bench(-p 512 -n 128 -r 5, 모든 층 GPU) + tegrastats(1초 간격). 구성 사이 60초 쉼.
# 출력: bench/<모델>-<양자화>.json(llama-bench), .tegra(전력·메모리·온도 원자료), .meta(과전류 횟수·전원 모드)
set -uo pipefail
B=~/llama.cpp/build/bin; G=~/llm/gguf; O=${1:-bench}; mkdir -p $O
oc() { cat /sys/class/hwmon/hwmon*/oc3_event_cnt 2>/dev/null | head -1; }
for m in qwen15 smol17; do
  for q in f16 Q8_0 Q4_K_M; do
    sleep 60
    T=$O/$m-$q
    { echo "start $(date +%T) oc3 $(oc)"; nvpmodel -q | head -1; free -m | sed -n 2p; } > $T.meta
    tegrastats --interval 1000 > $T.tegra & TP=$!
    sleep 5  # 실행 전 대기 구간(기준 메모리·전력)
    echo "bench_start $(date +%T)" >> $T.meta
    $B/llama-bench -m $G/$m-$q.gguf -ngl 99 -p 512 -n 128 -r 5 -o json > $T.json 2> $T.err
    echo "bench_end $(date +%T) exit $? oc3 $(oc)" >> $T.meta
    kill $TP; wait $TP 2>/dev/null
    echo "$m $q done $(date +%T)"
  done
done
echo BENCH_DONE
