#!/bin/bash
# Jetson Orin Nano(JetPack 6.2.1)에서 llama.cpp 빌드, 모델 변환·양자화, 평가 데이터 준비.
# sudo 없이 사용자 영역에서만 동작한다. 사용: bash setup_llm.sh
set -euo pipefail
W=~/llm; mkdir -p $W; cd $W
export PATH=/usr/local/cuda-12.6/bin:$PATH
. ~/pt-venv/bin/activate

# 1) llama.cpp 빌드 (Orin = sm_87). 커밋을 기록해 재현 가능하게 한다.
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
git -C llama.cpp rev-parse HEAD > llama_cpp_commit.txt
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DLLAMA_CURL=OFF
cmake --build llama.cpp/build -j4 --target llama-server llama-bench llama-quantize

# 2) 변환에 필요한 패키지만 설치 (requirements 파일은 CPU용 torch를 끌어오므로 쓰지 않음)
pip install -q "transformers<5" sentencepiece safetensors huggingface_hub pandas pyarrow requests
pip install -q --no-deps -e llama.cpp/gguf-py

# 3) 원본 모델 → GGUF F16 → Q8_0 / Q4_K_M (모두 같은 원본에서 직접 변환)
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-1.5B-Instruct', local_dir='qwen15')"
python llama.cpp/convert_hf_to_gguf.py qwen15 --outtype f16 --outfile qwen15-f16.gguf
for q in Q8_0 Q4_K_M; do llama.cpp/build/bin/llama-quantize qwen15-f16.gguf qwen15-${q,,}.gguf $q; done

# 4) ARC-Challenge test (공개 데이터셋)
python - <<'EOF'
import pandas as pd
url = "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet"
pd.read_parquet(url).to_parquet("arc_challenge_test.parquet")
EOF
ls -la *.gguf arc_challenge_test.parquet; cat llama_cpp_commit.txt
echo SETUP_DONE
