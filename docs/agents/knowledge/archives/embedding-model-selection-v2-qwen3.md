# Embedding Model Selection v2: qwen3-embedding-0.6b（2026-09-28）

## Question
- jina-v2-base-zh 中文+代码召回不足，如何换模型？留守 fastembed 等收录是否可行？

## Background
- 旧基线 DB（53883986ad0936d4.db）被 context-mode 升级清除 → 本次 T1 重采样 CASES 并重生基线，A/B 在同库同 CASES 上进行，杜绝漂移
- fastembed 等待路线实测不通：0.7.4 共 28 模型与上游 0.8.1 官方文档站均无任何 Qwen

## Answer
- Chosen: qwen/qwen3-embedding-0.6b-gguf-q8（GGUF Q8_0，dim 1024，pooling last，query 侧 Instruct 前缀）via llama-cpp-python 0.3.35 CUDA
- A/B（同库同 CASES 重生基线）：jina hybrid 8/12 vs qwen3 10/12，rescues 2→4，三门禁双 PASS；对比口径为模型+截断（512→n_ctx）+query 前缀的打包效果，非纯模型差
- Tradeoff: 自维护 GGUF 分发与 CUDA runtime——cu124 预编译轮 libggml-cpu 本机 SIGILL（Zen4 CI 编译含 AVX-VNNI，本机 i7-11800H Tiger Lake）→ scripts/uv-sync.sh 混合安装（轮子 GPU 库 + 本机重建 CPU 核）；1024 维向量体积 +33%
- 升级路径: fastembed 收录 Qwen 后可回归（实测 0.7.4/0.8.1 均无）

## Sources
- [embedding-model](../modules/embedding-model.md)
- ~/.omo/evidence/ctx-semantic/：spike-qwen3-llama-cpp.out、recall-report.md、recall-report-jina-baseline-20260928.md、task-5-uv.out

## See also
- [[embedding-model]]
- [[embedding-model-selection]]
- [[wemm-embedding-evaluation]]
