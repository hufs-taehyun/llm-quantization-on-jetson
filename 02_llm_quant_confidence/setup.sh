#!/bin/bash
# Jetson(JetPack 6.2.1)의 ~/pt-venv(torch 2.8.0 Jetson 빌드)에 LLM 실험 준비. sudo 없음.
# Jetson용 torch를 덮어쓰지 않도록 torchao는 Jetson 서버에서 --no-deps로, 나머지는 PyPI에서 설치한다.
set -euo pipefail
. ~/pt-venv/bin/activate
mkdir -p ~/llm && cd ~/llm
pip install -q "transformers>=4.45,<5" accelerate safetensors huggingface_hub pandas pyarrow
pip install -q --no-deps --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torchao==0.13.0
python -c "import torch, transformers, torchao; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'torchao', torchao.__version__)"
python - <<'EOF'
from huggingface_hub import snapshot_download
import pandas as pd
snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_dir="qwen15")
base = "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/"
for split in ("validation", "test"):
    pd.read_parquet(base + f"{split}-00000-of-00001.parquet").to_parquet(f"arc_{split}.parquet")
EOF
du -sh qwen15; ls -la arc_*.parquet
echo SETUP_DONE
