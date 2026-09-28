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
