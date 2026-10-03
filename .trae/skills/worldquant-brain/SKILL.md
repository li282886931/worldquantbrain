---
name: worldquant-brain
description: Interact with the WorldQuant BRAIN API — login, fetch data fields, submit simulations, poll progress, and check submissions. Use when the user asks to run alphas, query BRAIN datasets, debug 401/429/timeout issues, or extend alpha_machine.py / alpha_explorer.py. Do not use for offline pandas analysis unrelated to the BRAIN API.
---

# WorldQuant BRAIN 交互

所有与 BRAIN API 的交互统一走项目根目录的 `machine_lib.py`，不要在业务脚本里重写 HTTP 逻辑。

## 凭证

- 凭证从项目根目录 `.env` 读取：`WQB_USERNAME` / `WQB_PASSWORD`，加载函数是 `config.load_wqb_credentials()`。
- 不要把凭证写进代码或日志；调试时只打印 `username[:3] + "***"`。
- 登录用 `machine_lib.login()`，返回带 `session.auth` 的 `requests.Session`，POST `/authentication` 成功状态码是 201。

## 网络可靠性（已踩过的坑，勿回退）

BRAIN API 限流严格、偶发代理/SSL 抖动。machine_lib 已有两个辅助函数，新增请求必须复用：

- `_request_with_retry(session, method, url, ...)` — 处理 429（读 `Retry-After`）、5xx、`ReadTimeout`、`ConnectionError`，指数退避。登录、拉字段、提交模拟都用它。
- `_poll_with_retry_after(session, url, ...)` — 轮询直到无 `Retry-After` 头，上限 `MAX_POLL_ATTEMPTS = 300`，网络异常自动重试。用于模拟进度轮询、alpha 详情、提交检查。
- `DEFAULT_TIMEOUT = 60`；同批提交间隔 `sleep(0.3)`，翻页间隔 `sleep(0.5)`。
- 禁止 `while True` 裸轮询；禁止直接 `session.get/post` 不加重试。
- 后台运行时输出会缓冲：用 `python -u` 或 `PYTHONUNBUFFERED=1`。

## 核心 API 速查

| 目的 | 函数 |
|---|---|
| 登录 | `login()` |
| 拉数据字段（自动翻页） | `get_datafields(s, dataset_id=, region=, universe=, delay=)` |
| 清洗字段 | `process_datafields(df)` |
| 一阶表达式工厂 | `first_order_factory(fields, ts_ops)` |
| 二阶分组 | `get_group_second_order_factory(exprs, group_ops, region)` |
| 三阶 trade_when | `trade_when_factory(op, expr, region)` |
| 分批 | `load_task_pool_single(alpha_list, limit)` |
| 单模拟（带断点回调） | `single_simulate(pool, neut, region, universe, start, on_batch_done=None)` |
| 多模拟 | `multi_simulate(...)` |
| 按指标查 alpha | `get_alphas(start, end, sharpe, fitness, region, limit, usage)` |
| 按字段剪枝 | `prune(tracker, prefix, keep)` |
| 提交前检查 | `check_submission(stone_bag, gold_bag, start)` |
| alpha 详情 | `locate_alpha(s, alpha_id)` |

## 上层入口

- 完整流水线（8 阶段、断点续跑）：`python alpha_machine.py`，断点文件 `alpha_machine_checkpoint.json`，`--fresh` 重跑。断点续跑模式见 `_simulate_or_raise` 的 `checkpoint` + `progress_key` 参数。
- 字段×表达式×settings 探索：`python alpha_explorer.py`，进度文件 `alpha_progress_fine.pkl`。

## 测试

改动 machine_lib 后必须跑：`py -m pytest test_machine_lib.py test_alpha_machine.py test_config.py -q`。测试用 `RecordingSession`/`FakeMachineLib` mock HTTP，不要在测试里发真实请求。
