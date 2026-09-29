# Upgrade Paths

> Last updated: 2026-09-29

## Overview
- ctx-semantic 侧车三向升级调研（2026-09-29）：向量索引（Q1）、GPU 依赖轮廓（Q2）、脱离 context-mode 独立化（Q3）
- Key files: `./pyproject.toml`, `./scripts/uv-sync.sh`, `./run.sh`, `./ctx_semantic/embedder.py`, `./ctx_semantic/vectors.py`
- Dependencies: `llama-cpp-python==0.3.35` 双轮廓（cpu 默认 / cuda opt-in）、nvidia wheels 仅 cuda 轮廓
- See also: [[embedding-model]]

## Decisions

### 向量索引不换 sqlite-vec，保留暴力 cosine + 量化触发线（2026-09-29 调研归档）
- **Source:** 三向调研 draft Q1（本仓实测 + librarian 一手源 bg_1672b4bc）；触发线落点 `ctx_semantic/vectors.py` ponytail 注释
- **Chosen:** 维持自有 embeddings 表 + numpy 暴力 cosine；触发线 = chunks>~50k 或向量腿 P99>50ms → 直接评估 ANN（usearch/hnswlib，或 sqlite-vec rescore 转正后），跳过同算法换皮
- **依据:**
    - 语料实测 89 向量 / 0.4MB（本仓 store），e2e 稳态单查 21ms — 向量腿占比极小，瓶颈在 embed_query 模型推理
    - sqlite-vec stable（v0.1.9）同为暴力全扫（作者明示 brute-force only）且 pre-v1，ANN 仅 alpha 未转正
    - doobidoo/mcp-memory-service 的 sqlite-vec 栈运维伤疤（锁争用/segfault/完整性监控）交叉印证

### GPU 双安装轮廓 + ctypes RTLD_GLOBAL 预载（2026-09-29 实施归档）
- **Source:** 三向调研 draft Q2 + spike-cpu-wheel.out（Spike B）+ commits 20df083/771936e/54d59f5
- **Chosen:** CPU 为默认轮廓（abetlen cpu 轮 24MB，免 1.7GB 拉取），cuda 轮廓 opt-in（保留 path+sha256 pin 机制）；embedder 在 `import llama_cpp` 前以 ctypes.CDLL(RTLD_GLOBAL) 预载 nvidia 轮三库，免 LD_LIBRARY_PATH
- **依据:**
    - CUDA 轮 import 硬依赖实测：不预载时 `import llama_cpp` 顶层 dlopen libcudart 即死，n_gpu_layers=0 无机会生效
    - abetlen issue #1460 提供官方预载代码先例
    - uv `[tool.uv] conflicts` + 双 index 按 marker 分流有 torch 官方先例
    - Spike B：cpu 轮本机探针 OK（SIGILL 未复现）

### 独立化定方向 A：保持 sidecar（2026-09-29 调研归档）
- **Source:** 三向调研 draft Q3 + librarian 一手源 bg_0b6b4d08（生态证据）
- **Chosen:** 维持 sidecar 逆向依赖 context-mode；方向 B（+自有摄取工具）留作可逆升级路径，不预建
- **依据:**
    - 生态无直接读他库活 schema 的成功先例（主流全为 own-store 摄取或松散文件格式）
    - 本仓 dbadapter EXPECTED_COLUMNS 主动漂移守卫领先同类（同类靠被动修 drift）
    - MCP roots+sampling 自 2026-07-28 协议版起 deprecated，独立索引产品地基不确定
    - B 升级路径 = 加自有摄取工具，生态 hybrid 收敛方向（mcp-memory-service、Basic Memory 双双趋同）

## Strategies

### 触发线复查节奏（2026-09-29）
- **Source:** Q1 决策的量化触发线
- **Problem:** 语料量随 context-mode 索引增删变化，触发线是否逼近需要低成本监控
- **Approach:**
    - 语料量随 Boulder 报告观察；chunks>10k 时复核一次向量腿延迟与触发线距离
- **When to reuse:** 例行复查或向量检索延迟异常时

## Open Questions
<!-- Gaps that future work should address -->
- sqlite-vec rescore/DiskANN 转正观察（v0.1.10-alpha 何时 stable）
- CPU 轮廓召回质量长期观察（如实记录，不 gate）
