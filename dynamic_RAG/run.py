# -*- coding: utf-8 -*-
"""
run_openai_hipporag.py  –  用 *OpenAI GPT‑4o‑mini* 接入 MultiTQmini 数据集
==========================================================================
- 读取 `reproduce/dataset/MultiTQmini/test.txt` 四元组 → 文本 `docs`
- 读取同目录下 `test.json` → `queries / answers`
- 用 `HippoRAG` + OpenAI LLM & embedding 完成索引 + QA

**环境准备**
```bash
export OPENAI_API_KEY="sk‑xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# 可选：指定组织, 代理等 OPENAI_ORG / HTTPS_PROXY
pip install hipporag openai
```

运行：
```bash
python run_openai_hipporag.py \
  --data_dir reproduce/dataset/MultiTQmini \
  --save_dir outputs/openai_multiTQ \
  --llm_model gpt-4o-mini \
  --embed_model text-embedding-3-small
```
"""
# %load_ext autoreload
# %autoreload 2
import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

from src.hipporag import HippoRAG  # 确保 PYTHONPATH 指向 HippoRAG 根目录
# import logging
# logging.disable(logging.INFO)
###############################################################################
# 工具函数
###############################################################################

def triples_to_docs(txt_path: Path) -> List[str]:
    """把四元组 <s, r, o, date> 转成简单英文句子，供向量化索引，并全部转为小写。"""
    docs = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            s, r, o, date = line.rstrip("\n").split("\t")
            # 全部转小写
            s, r, o, date = s.lower(), r.lower(), o.lower(), date.lower()
            docs.append(f"{date} {s} {r.replace('_', ' ')} {o}.")
    return docs




###############################################################################
# 主函数
###############################################################################


parser = argparse.ArgumentParser("HippoRAG + LLAMA on MultiTQmini")
parser.add_argument("--data_dir", default="reproduce/dataset/MultiTQ")
parser.add_argument("--save_dir", default="outputs/LLAMA_multiTQ")
parser.add_argument("--llm_model", default="gpt-4o-mini") #  meta-llama/Llama-3.1-8B-Instruct
parser.add_argument("--embed_model", default="text-embedding-3-small")
args, _ = parser.parse_known_args()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
data_dir = Path(args.data_dir)
txt_path, json_path, temporal_path = data_dir / "train.txt", data_dir / "train.json", data_dir / "tkbc_processed_data/temporal_model_new.pickle"

# 1. 转换 docs
docs = triples_to_docs(txt_path)
logging.info("Loaded %d facts as docs", len(docs))

# 2. 启动 HippoRAG (默认 provider=openai)
# hipporag = HippoRAG(
#         save_dir="outputs/llama2_rag",
#         llm_model_name="gpt-4o-mini",   # 与 vLLM model id 完全一致
#         embedding_model_name="nvidia/NV-Embed-v2",   # 已在白名单中
#         llm_base_url="http://localhost:6578/v1",
#         dynamic=True,
#         temporal_dir = str(temporal_path)
#     )

save_dir = 'outputs/openai'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
llm_model_name = 'gpt-4o-mini'  # Any OpenAI model name

# Startup a HippoRAG instance
hipporag = HippoRAG(save_dir=save_dir,
                    llm_model_name=llm_model_name,
                    embedding_model_name="nvidia/NV-Embed-v2",
                   dynamic=True, temporal_dir = str(temporal_path))

# 3. 索引
hipporag.index(docs, txt_path)
# 4. 读取问题 & 答案
txt_path, json_path, temporal_path = data_dir / "train.txt", data_dir / "train.json", data_dir / "tkbc_processed_data/temporal_model_new.pickle"
def load_qas(json_path: Path) -> Tuple[List[str], List[List[str]]]:
    data = json.loads(json_path.read_text("utf-8"))
    queries  = [item["question"] for item in data]
    answers  = [item["answers"]  for item in data]
    qlabels = [item["qlabel"]  for item in data]
    return queries[:100], answers[:100], qlabels[:100]

queries, answers, qlabels = load_qas(json_path)
logging.info("Running RAG‑QA on %d queries via OpenAI", len(queries))
print(hipporag.rag_qa(queries=queries,gold_docs=None,gold_answers=answers, qlabels = qlabels))
