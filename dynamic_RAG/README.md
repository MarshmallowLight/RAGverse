## Code for the submission "Right Answer at the Right Time — Temporal Retrieval-Augmented Generation via Graph Summarization"


## Requirements
We build this code based on the HippoRAG project: https://github.com/OSU-NLP-Group/HippoRAG.git
```
pip install requirement.txt
```

## Datasets
The CronQuestion dataset is public on:
https://github.com/apoorvumang/CronKGQA.git

The Forecast dataset is public on:
https://github.com/ZifengDing/ForecastTKGQA.git

The MultiTQ dataset is public on: https://huggingface.co/datasets/chenziyang/MultiTQ

## Build the rule graph
Cython needs to be compiled before running, run this command:
```
python build_temporal_model.py
```

## Run STAR-RAG for question answering

```
python run.py
```
