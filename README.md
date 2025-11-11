# RAGverse
一个以 Retrieval-Augmented Generation（RAG）为核心的系统化项目集合

### 📘 项目简介
🔍 一个围绕 **Retrieval-Augmented Generation（RAG）** 技术展开的系统化实验与工程实现集合，涵盖从基础实现到前沿改进的多层级实验。
> 本仓库为作者个人在RAG领域的工程能力与研究探索，为后续学术研究与工业部署提供可复现、可扩展的解决方案。


### 🏗️ 项目结构

```
RAGverse/
│
├── temporal_RAG/             # 
├── adaptive-rag/             # 动态检索与生成（query routing、信息瓶颈控制、adaptive top-k）
├── causal-rag/               # 因果与反事实RAG（Counterfactual RAG、Causal Graph Integration）
├── eval-bench/               # 评估与指标模块（Faithfulness、Factuality、Hallucination率）
├── visualizer/               # 检索可视化与路径追踪工具
├── docs/                     # 论文笔记与理论总结
└── utils/                    # 通用工具函数与配置文件
```

---

### 🔬 核心特色

* **模块化设计**：每个子项目均可独立运行或组合构建，支持多框架（Haystack、LlamaIndex、LangChain）。
* **系统性探索**：从工程实现到研究创新，逐步覆盖RAG关键问题（检索质量、生成对齐、信息冗余控制）。
* **可扩展性强**：所有模块均提供配置化入口与API接口，便于快速复现实验或进行微调。
* **对齐前沿论文**：包含如 *Counterfactual RAG (2024)*、*Self-RAG (Meta)*、*Atlas (DeepMind)* 等代表性工作复现与延展。

---

### 📂 子项目规划（持续更新中）

| 模块名            | 内容简介                                                     |   状态   |
| :------------- | :------------------------------------------------------- | :----: |
| `basic-rag`    | 从零实现RAG Pipeline（Encoder、Retriever、Generator、Evaluation） |  ✅ 已完成 |
| `causal-rag`   | 基于反事实推理的检索增强生成（参考CF-RAG）                                 | 🔧 进行中 |
| `adaptive-rag` | 基于信息熵与不确定性的动态top-k检索策略                                   | 🧩 规划中 |
| `eval-bench`   | 通用RAG评估框架，支持多维指标                                         | 🧩 规划中 |
| `visualizer`   | 检索路径可视化与知识流向分析                                           | 🧩 规划中 |

---

### ⚙️ 技术栈

* **语言**：Python 3.10+
* **核心框架**：PyTorch / Transformers / Haystack / LangChain
* **检索后端**：FAISS / Milvus / Elasticsearch
* **生成模型**：Llama / Mistral / OpenAI / Qwen
* **可视化**：Streamlit / Plotly / D3.js

---

### 🚀 运行方式

```bash
git clone https://github.com/<yourname>/rag-lab.git
cd rag-lab/basic-rag
pip install -r requirements.txt
python run_pipeline.py
```

---

### 📚 研究参考

* Zhu et a. (2025). *Right Answer at the Right Time — Temporal Retrieval-Augmented Generation via Graph Summarization*
* Min et al. (2024). *Counterfactual RAG: Disentangling Causal from Correlational Knowledge.*
* Izacard et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Evaluate.*
* Borgeaud et al. (2022). *Improving Language Models by Retrieving from Trillions of Tokens.*

---

### ✨ 作者寄语

本仓库既是个人对RAG体系的系统实践，也是一份研究者视角下的“认知实验室”。
希望每一个实验分支都能为理解“如何让模型真正理解知识”提供一块拼图。
