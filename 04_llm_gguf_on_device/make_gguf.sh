#!/bin/bash
# 02에서 쓴 HF 가중치(~/llm/qwen15, ~/llm/smol17)를 GGUF F16으로 바꾸고 Q8_0, Q4_K_M을 만든다.
set -euo pipefail
. ~/pt-venv/bin/activate
L=~/llama.cpp; B=$L/build/bin; G=~/llm/gguf; mkdir -p $G
# 사전 설치(인터넷 필요, 한 번만): pip install --no-deps $L/gguf-py && pip install sentencepiece
for m in qwen15 smol17; do
  [ -f $G/$m-f16.gguf ] || python $L/convert_hf_to_gguf.py ~/llm/$m --outtype f16 --outfile $G/$m-f16.gguf
  for q in Q8_0 Q4_K_M; do
    [ -f $G/$m-$q.gguf ] || $B/llama-quantize $G/$m-f16.gguf $G/$m-$q.gguf $q
  done
done
(cd $L && echo "llama.cpp $(git rev-parse HEAD)")
ls -la $G; sha256sum $G/*.gguf
echo GGUF_DONE
