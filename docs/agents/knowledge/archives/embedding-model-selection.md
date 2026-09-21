# Embedding Model Selection (2026-09-21)

## Question
- 为何不用 bge-m3 或 jina-embeddings-v3？

## Answer
- bge-m3：硬性排除——不在 fastembed 支持列表（实测 0.7.4 共 30 个模型）；改用需拖入 PyTorch；其 dense+sparse+ColBERT 三路中 sparse 腿已由 SQLite FTS5/BM25 覆盖
- jina-embeddings-v3：在支持列表但权衡排除——2.29GB vs 0.64GB（骨干 XLM-R-large ~570M vs bert-base ~160M）；许可证 CC-BY-NC-4.0 禁商用 vs v2 的 Apache-2.0（HF API 实查）；Matryoshka/task-LoRA 对定长 768 维召回无用；sidecar 冷启动设计偏好小模型
- 结论：v2-base-zh 为中英混合专项训练、Apache-2.0、0.64GB，四维全胜于本项目场景

## Sources
- [embedding-model](../modules/embedding-model.md)
- `./README.md`

## See also
- [[embedding-model]]
