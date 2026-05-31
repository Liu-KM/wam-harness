# Runner 行为契约

runner 是核心循环：它驱动 backend 在 replan 点推理、缓存动作块、按步消费动作、
发射 trace 事件。它**只用契约词汇**，不含任何上游仓库分支（见 `AGENTS.md`）。

本文件用伪代码 + 边界表固定 runner 的可观察行为，作为 Codex 实现任务卡 006 的契约。
术语（run/episode/step/replan/inference_call/action_chunk）见 `docs/contract.md`。

## 状态

runner 在一个 episode 内维护：

- `pending_actions`: 当前动作块里尚未消费的动作队列。
- `steps_since_replan`: 自上次推理以来已执行的步数。
- `replan_id`: 单调递增的推理计数。

## 循环伪代码

```text
run(config):
  backend.load()                          # → backend_load 事件 + runtime_info
  observer.reset_peak()
  emit run_start(runtime_info, experiment_fields, measurement_fields)

  for episode in workload.episodes():     # → episode_start
    backend.reset(seed=config.seed)       # → reset 事件；决定性对齐
    workload.reset(episode)
    pending_actions = []
    steps_since_replan = 0
    backend.warmup(sample_request)        # → backend_warmup（样本打 warmup=true）

    for step in range(episode.steps):
      if need_replan(pending_actions, steps_since_replan, config.replan_steps):
        obs = workload.observe()
        request = build_request(obs, config)
        emit replan_start(replan_id)
        emit inference_start(replan_id, workload_fields)
        with observer.stage_timer("model"), observer.cuda_timer():
          result = backend.infer(request)         # 内含 preprocess→model→postprocess
        persist_artifact(result.action_chunk, key=(episode,step,replan_id))
        emit inference_end(replan_id, timing, action_chunk_path, shape)
        emit memory_sample(observer.sample_memory())
        pending_actions = list(result.action_chunk)   # 取整块
        steps_since_replan = 0
        replan_id += 1

      action = pending_actions.pop_front()
      from_stale = (steps_since_replan > 0)
      with observer.stage_timer("env_step"):
        workload.step(action)             # 开环下为重放/记录
      emit step(step_id, env_step_ms, from_stale_chunk=from_stale)
      steps_since_replan += 1

    emit episode_end(episode_summary)

  emit run_end(run_summary)
```

## need_replan 判定

```text
need_replan(pending, steps_since_replan, replan_steps):
  if len(pending) == 0:            return True   # 块已耗尽，必须重规划
  if steps_since_replan >= replan_steps: return True  # 到达重规划间隔
  return False
```

## 边界情况表（Codex 必须按此实现并测试）

| 情形 | 条件 | 期望行为 |
|---|---|---|
| 块比间隔长 | `action_horizon > replan_steps` | 执行 `replan_steps` 步后重规划，块尾部未消费的动作被丢弃；丢弃数记入 `chunk_utilization` |
| 块比间隔短 | `action_horizon < replan_steps` | 块耗尽即触发重规划，`replan_steps` 实际不会被先达到；记录此为有效路径 |
| 块恰好等长 | `action_horizon == replan_steps` | 每块刚好消费完即重规划，无丢弃，`chunk_utilization == 1.0` |
| 块长为 1 | `action_horizon == 1` | 每步都重规划，退化为逐步推理 |
| 空块 | backend 返回 0 长度 action_chunk | 发 `error` 事件（stage=infer, recoverable=false），终止该 episode，不静默挂起 |
| reset 语义 | episode 切换 | `backend.reset(seed)` 必须清空 pending_actions 与 cache；跨 episode 状态不得泄露 |
| stale 标记 | `steps_since_replan > 0` 时消费的动作 | `step` 事件 `from_stale_chunk=true`，支撑 chunk 调度分析 |
| warmup 隔离 | warmup 阶段样本 | 打 `warmup=true`，不进 steady-state 统计（见 measurement.md） |
| CPU-only | 无 CUDA | memory_sample 字段名不变，CUDA 字段记 0/null；cuda_timer no-op |
| 决定性 | baseline 与 variant | 同 seed → 同观测顺序 → 同输入；否则数值 gate 判 not_comparable |

## episode / run 摘要

- `episode_end` 应含：`model_calls`、`total_model_ms`、`chunk_utilization`、
  `stale_action_steps`、`steps`（支撑 action_chunk_scheduling 实验，见 systems_ideas.md）。
- `run_end` 应含：跨 episode 聚合 + 按 measurement 协议的 p50/p95/mean/std/n。

## 不变量

- runner 不 import 具体 backend，只依赖 `docs/interfaces.md` 的 Protocol。
- trace 事件 append-only；错误是事件而非仅终端日志（见 contract.md）。
- 动作块以 artifact 形式落盘并按 `(episode_id, step_id, replan_id)` 对齐，
  不嵌入 trace（见 trace_schema.md）。
