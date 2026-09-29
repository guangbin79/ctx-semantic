# Knowledge Log

## [2026-09-21] ingest | embedding-model
- 记录嵌入模型选型决策（v2-base-zh；bge-m3 不在 fastembed 列表、v3 因体积/CC-BY-NC 许可证排除）与 fastembed-gpu 发行名/支持列表验证策略；fastembed-gpu 为官方 GPU 发行版（onnxruntime-gpu 后端，fastembed[gpu] extra 上游 0.8.0 已移除）
- Types: Decision, Strategy, Module Info

## [2026-09-21] query | Archived: embedding-model-selection
- 归档「为何不用 bge-m3 / jina-v3」的对比结论（含 HF API 许可证实查）
- Types: Archive

## [2026-09-23] query | Archived: wemm-embedding-evaluation
- 评估 Tencent WeMM-Embedding 替换 jina-v2-base-zh：多模态 2B VLM，四维仅许可证（Apache-2.0）过关，不换
- Types: Archive

## [2026-09-28] ingest | embedding-model
- 嵌入模型迁移 jina-v2-base-zh → qwen/qwen3-embedding-0.6b-gguf-q8（llama-cpp-python CUDA）：同库同 CASES 重生基线 A/B hybrid 8/12→10/12、rescues 2→4、三门禁双 PASS；fastembed 0.7.4/0.8.1 实测无 Qwen，留守路线不通
- Types: Decision, Archive

## [2026-09-29] ingest | embedding-model
- 迁移后评审待办落实：CASES 重采样对齐活语料（75 chunks，第三次 purge 后）恢复召回门禁可运行；uv-sync.sh 加固（WHEEL/VER 派生自 pyproject 钉、glob 守卫、原子 rename 换库、sdist sha256 三钉、分片长度校验）；embedder cache_dir 参数恢复生效；nvidia 轮钉 12.9.*；旧 qwen3 A/B 基线存档 recall-report-qwen3-baseline-20260928.md
- Types: Decision, Module Info

## [2026-09-29] ingest | embedding-model
- 全仓深审（5 路全 PASS）后修复落实：README 探针 2 补 LD 前缀（import llama_cpp 即需 CUDA 库）；prune 兼清退役模型键行（单模型存储设计）；warmup --all/--db 逐库容错（FAIL 行+非零退出，不再因一个漂移库中止）；run.sh 改脚本相对 CTX_DIR + paste 拼接（去尾随冒号）；projhash realpath 分歧入文档；_append_filters 显式组合；零范数守卫对齐；binding ChunkRow 改导入；运行态 QA 残渣（120 行 /tmp 语料）已 prune
- Types: Decision, Module Info

## [2026-09-29] ingest | embedding-model
- 可移植性收口：[tool.uv.sources] 轮钉改仓内相对路径（models/wheels/，uv.lock 同步相对化，任意 clone 位置 uv-sync.sh 全链自洽）；新增 docs/INSTALL.md 固定安装参考（前置/资产/修复原理/验证/MCP 注册/排障/升版契约），README Setup 挂链
- Types: Decision, Module Info
