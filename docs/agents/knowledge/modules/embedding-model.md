# Embedding Model

> Last updated: 2026-09-21

## Overview
- ctx-semantic 侧车的嵌入层：fastembed 加载 `jinaai/jina-embeddings-v2-base-zh`（dim 768，中英混合），onnxruntime-gpu 后端，设备自动回退
- Key files: `./pyproject.toml`, `./ctx_semantic/`, `./run.sh`, `./models/`
- Dependencies: `fastembed-gpu`（0.7.4，GPU 发行版，import 名仍是 `fastembed`）、`onnxruntime-gpu[cuda,cudnn]<1.24`
- See also: [[infra]]

## Decisions

### 选用 jina-embeddings-v2-base-zh（2026-09-21 验证归档）
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

### 验证 fastembed 支持列表与发行名陷阱（2026-09-21）
- **Source:** session 实测 + `./pyproject.toml` 注释
- **Problem:** 确认某模型（如 bge-m3）是否可用；版本元数据查询报错
- **Approach:**
    - 用 `TextEmbedding.list_supported_models()` 实测，不凭记忆
    - `importlib.metadata('fastembed')` 抛 PackageNotFoundError 是陷阱：发行名是 `fastembed-gpu`（`fastembed[gpu]` extra 上游 0.8.0 已移除，fastembed-gpu 为官方 GPU 发行版，onnxruntime-gpu 后端）
    - uv 无法对传递依赖启用 extras，CUDA 库须直接声明 `onnxruntime-gpu[cuda,cudnn]<1.24`
- **When to reuse:** 升级/更换嵌入模型、排查 fastembed 元数据或 GPU 后端问题时

## Open Questions
<!-- Gaps that future work should address -->
- README 引用 `docs/evidence` 但该目录未入库（find 无结果）——证据文档缺失，是否补齐？
- v2 序列老化：中文召回质量若不足，候选替代模型是什么（须同时满足 Apache 系 + fastembed 收录）？
