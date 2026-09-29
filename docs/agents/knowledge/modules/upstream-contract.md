# Upstream Contract (context-mode)

> Last updated: 2026-09-29

## Overview
- ctx-semantic 逆向依赖 context-mode 的两个契约：projhash（sha256(yr(path))[:16] → `content/<hash>.db`）与 content DB 三表结构（EXPECTED_COLUMNS：sources / chunks / chunks_trigram）
- 上游 7 个月 223 发版、无公开契约承诺——drift 防护是 sidecar 生存需求
- Key files: `./ctx_semantic/projhash.py`, `./ctx_semantic/dbadapter.py`, `./ctx_semantic/probe.py`, `./ctx_semantic/server.py`
- See also: [[embedding-model]]

## Decisions

### drift 分级门禁：硬 SchemaDrift fail-loud / 软 STALE 仅警告（2026-09-29）
- **Source:** 会话三轮设计确认（探针→提示分级→人的发现渠道）+ TDD 落地（15 新测试先红后绿，168 全绿）
- **Chosen:** 确定性漂移（缺表/列集合不等）抛 SchemaDrift 并拒绝启动（`server.preflight()` 退出 1——死亡注册比逐调用报错更显眼）；mtime 启发式（`STALE_AFTER_DAYS=14d` + 同目录 sibling DB 新旧）只警告不杀
- **Alternatives:** STALE 也硬闸；context-mode 版本号跟踪
- **Reason:** 错误答案比功能缺失危害大；schema 本身即契约，版本号不提供额外信息；启发式必可误报（闲置项目），误杀 server 代价大于收益
- **Tradeoff:** STALE 期间照常返回结果，靠结果前缀 ⚠️ 行 + probe 提示兜底

## Strategies

### drift 修正流程
- SchemaDrift 消息自带修复套件：列差异 + sqlite_master dump + Fix 步骤 + 中文转告行；照 dump 更新 `EXPECTED_COLUMNS` 后 `uv run pytest -m integration` 复验
- `python -m ctx_semantic.probe` 为升级后健康检查（退出码 0 PASS/SKIP · 1 DRIFT · 2 STALE；不 import embedder，亚秒级）；全局 AGENTS.md 挂 ctx_upgrade → probe 路由

### 人的发现渠道（按时刻排序）
- 升级当下：agent 调 ctx_upgrade 后按 AGENTS.md 路由跑 probe
- 下次开会话：MCP 注册失败（preflight 硬门）
- 用到检索时：工具红色错误（消息内嵌转告指令）/ 结果前缀 ⚠️
- 事后追查：server stderr 时间戳日志（stdout 保持 MCP 协议纯净）
