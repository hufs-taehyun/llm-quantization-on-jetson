#!/bin/bash
# 빌드가 끝나기를 기다렸다가 변환 → 측정 1(속도·전력) → 점검(20문항) → 측정 2(정확도)를 사람 없이 이어서 한다.
# 점검에서 PyTorch FP16 판정과 18/20 미만으로 같으면 그 모델의 정확도 측정은 하지 않고 SMOKE_FAIL을 남긴다(PLAN.md).
set -uo pipefail
cd "$(dirname "$0")"; . ~/pt-venv/bin/activate
G=~/llm/gguf
while ! grep -q BUILD_DONE ~/build_llama2.log; do
  pgrep -f "[c]make --build" >/dev/null || { echo "BUILD_FAILED $(date +%T)"; exit 1; }
  sleep 30
done
echo "build ok $(date +%T)"
bash make_gguf.sh || { echo "GGUF_FAILED $(date +%T)"; exit 1; }
bash run_bench.sh bench
declare -A REF=([qwen15]=~/pt_lab/llm_results/val_fp16.jsonl [smol17]=~/pt_lab/results_smol_arc/val_fp16.jsonl)
for m in qwen15 smol17; do
  python eval_gguf.py --gguf $G/$m-f16.gguf --model ~/llm/$m --name fp16 --out res_$m --split validation --limit 20 \
    || { echo "SMOKE_ERROR $m"; continue; }
  AG=$(python - "$m" "${REF[$m]}" <<'EOF'
import json, sys
a = [json.loads(l) for l in open(f"res_{sys.argv[1]}/smoke_fp16.jsonl")]
b = {r["id"]: r for r in map(json.loads, open(sys.argv[2]))}
print(sum(x["pred"] == b[x["id"]]["pred"] for x in a))
EOF
)
  echo "smoke $m agree_with_pytorch $AG/20"
  [ "$AG" -ge 18 ] || { echo "SMOKE_FAIL $m"; continue; }
  for pair in fp16:f16 w8:Q8_0 w4:Q4_K_M; do
    python eval_gguf.py --gguf $G/$m-${pair#*:}.gguf --model ~/llm/$m --name ${pair%%:*} --out res_$m || echo "EVAL_ERROR $m $pair"
  done
done
echo "ALL_DONE $(date +%T)"
