# Embedding Model

> Last updated: 2026-09-28

## Overview
- ctx-semantic 侧车的嵌入层：llama-cpp-python 0.3.35 加载 Qwen3-Embedding-0.6B GGUF Q8_0（dim 1024，pooling last，query 侧 Instruct 前缀），CUDA 后端 + nvidia-smi 实证的设备回退
- Key files: `./pyproject.toml`, `./ctx_semantic/embedder.py`, `./run.sh`, `./scripts/uv-sync.sh`, `~/ctx-semantic/models/Qwen3-Embedding-0.6B-Q8_0.gguf`
- Dependencies: `llama-cpp-python==0.3.35`（cu124 预编译轮 + 本机重建 libggml-cpu，无 torch）、`nvidia-cuda-runtime-cu12`、`nvidia-cublas-cu12`
- See also: [[infra]]

## Decisions

### 选用 qwen/qwen3-embedding-0.6b-gguf-q8 + llama-cpp-python（2026-09-28 验证归档）
- **Source:** T1 spike（GGUF 加载/CUDA/池化实证）+ harness 同库同 CASES 重生基线 A/B；全文见 [embedding-model-selection-v2-qwen3](../archives/embedding-model-selection-v2-qwen3.md)
- **Chosen:** qwen/qwen3-embedding-0.6b-gguf-q8（GGUF Q8_0，dim 1024，pooling last，query 侧 Instruct 前缀）via llama-cpp-python 0.3.35 CUDA
- **Alternatives:** 留守 fastembed——实测 0.7.4 共 28 模型与上游 0.8.1 官方文档站均无任何 Qwen，等待收录路线不通
- **Reason:**
    - llama-cpp-python 为 C 绑定，不破无-torch 轻量 sidecar 架构
    - 中文+代码检索双强：jina hybrid 8/12 vs qwen3 10/12，rescues 2→4，三门禁双 PASS（recall-report.md 对比节）
    - 与 zvec-grep 同路线，GGUF 资产可复用
- **Tradeoff:** 自维护 GGUF 分发与 CUDA runtime——cu124 预编译轮 libggml-cpu 本机 SIGILL → scripts/uv-sync.sh 混合安装（轮子 GPU 库 + 本机重建 CPU 核）；1024 维向量体积 +33%；升级路径：fastembed 收录 Qwen 后可回归（实测 0.7.4/0.8.1 均无）
- **Supersedes:** 2026-09-21 jina-embeddings-v2-base-zh 条目

### 选用 jina-embeddings-v2-base-zh（2026-09-21 验证归档）
> **Superseded:** 已被上方 2026-09-28 qwen3-embedding-0.6b 条目取代，保留决策血缘
- **Source:** commit 8466cc4 (feat(embed): jina-v2-base-zh embedder with device auto-fallback) + 本次 session 实测
- **Chosen:** jinaai/jina-embeddings-v2-base-zh，dim 768，0.64GB，Apache-2.0
- **Alternatives:** BAAI/bge-m3；jinaai/jina-embeddings-v3
- **Reason:**
    - bge-m3 不在 fastembed 支持列表（实测 0.7.4 共 30 个模型，BAAI 系仅 bge-base/small en/zh）；要用必须换 FlagEmbedding/sentence-transformers，拖入 PyTorch，破坏无 torch 轻量 sidecar 架构
    - v3 在支持列表（dim 1024, 2.29GB）但被权衡排除：体积 3.5 倍（XLM-R-large ~570M vs bert-base ~160M）、许可证 CC-BY-NC-4.0 禁商用（HF API 实查，v2 为 Apache-2.0）、Matryoshka/task-LoRA 对定长 768 维召回无用
    - v2-base-zh 本身为中英混合训练，正中语料；sidecar 每次 run.sh 冷启动，小模型启动快
- **Tradeoff:** v2 序列较老；升级路径是等 fastembed 收录 Apache 系新中英模型，而非 v3（许可证卡死）
- **Supersedes:** 无（初始选型）

## Strategies

### 验证 fastembed 支持列表与发行名陷阱（2026-09-21；fastembed 时代历史策略，现栈 llama-cpp-python 下不适用，留存参考）
- **Source:** session 实测 + `./pyproject.toml` 注释
- **Problem:** 确认某模型（如 bge-m3）是否可用；版本元数据查询报错
- **Approach:**
    - 用 `TextEmbedding.list_supported_models()` 实测，不凭记忆
    - `importlib.metadata('fastembed')` 抛 PackageNotFoundError 是陷阱：发行名是 `fastembed-gpu`（`fastembed[gpu]` extra 上游 0.8.0 已移除，fastembed-gpu 为官方 GPU 发行版，onnxruntime-gpu 后端）
    - uv 无法对传递依赖启用 extras，CUDA 库须直接声明 `onnxruntime-gpu[cuda,cudnn]<1.24`
- **When to reuse:** 升级/更换嵌入模型、排查 fastembed 元数据或 GPU 后端问题时

## Open Questions
<!-- Gaps that future work should address -->
- README 引用 `docs/evidence` 但该目录未入库——已消除（2026-09-28 README 改引 `docs/agents/knowledge`）
- v2 序列老化：已由 2026-09-28 迁移 qwen3-embedding-0.6b 解决（见 Decisions 首条；后续候选见该条升级路径）
- 2026-09-23 评估 WeMM-Embedding：多模态 2B VLM，四维仅许可证过关，排除——见 [wemm-embedding-evaluation](../archives/wemm-embedding-evaluation.md)
