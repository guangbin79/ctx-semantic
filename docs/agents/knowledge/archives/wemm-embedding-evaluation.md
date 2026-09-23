# WeMM-Embedding Evaluation (2026-09-23)

## Question
- Tencent WeMM-Embedding（github.com/Tencent/WeMM-Embedding）值得替换本项目的 jina-v2-base-zh 吗？

## Answer
- 定性：多模态（图/视频/文本）嵌入族，Qwen3.5-2B 解码器骨干 + `<embedding>` token 取向量，须 `trust_remote_code=True`——非 BERT 式文本检索模型
- 最小变体 2B / 5.44GB bf16 / dim 2048（MRL 64-2048），无 <1B 型号；现模型 161M / 0.64GB / 768
- 基准只报 MMEB-v2/v3（多模态）：2B MMEB-v2 AVG 77.9；无 C-MTEB/MTEB，无与 jina/bge 的官方对比；MMEB-v3 纯文本 NDCG@5=45.3（2B），不起眼
- 部署：transformers==5.2.0 钉死、官方示例全 `.cuda()`、无 ONNX、无量化、无 CPU/延迟数据；2B VLM 短查询 CPU 交互延迟不可行（推断，无官方数据）
- 许可证 Apache-2.0（代码+权重均确认）——四维中唯一通过项
- fastembed 0.7.4 实测 30 模型无 WeMM；换入须拖 PyTorch，重蹈 bge-m3 否决路径且更重（2B vs 0.6B、钉版本、remote code）
- 语料是纯文本 session 知识，多模态能力零收益
- 结论：四维标准（Apache / fastembed 收录 / 体积延迟 / 中英质量）仅过 1/4，不换
- 升级路径不变：等 fastembed 收录 Apache 系新中英文本模型（候选参考 Qwen3-Embedding-0.6B、bge-m3——后者须先破无-torch 约束）
- 成熟度：2026-08-25 发布（评估时约 1 个月），1.7k stars，微信线上部署（tech report 自述）

## Sources
- [embedding-model](../modules/embedding-model.md)
- https://github.com/Tencent/WeMM-Embedding
- https://huggingface.co/tencent/WeMM-Embedding-2B
- arXiv:2608.24053

## See also
- [[embedding-model]]
