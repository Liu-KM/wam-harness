# S2 存在性检查：<论文短标题>

> 阶段：S2 存在性检查 · 主导：Claude · 见 [../../PIPELINE.md](../../PIPELINE.md)
> 回答先决问题：**S1 那个 pressure，在 WAM 推理负载里到底存不存在？** 用数据说话。
> 契约：`docs/contract.md` 的 Characterization Experiment Contract。

## 复述 pressure（来自 S1）

<一句话重述要检验的 LLM 侧 pressure。>

## WAM 侧候选对应物

<在 WAM 推理里，哪个环节可能承受同样的 pressure。
 例：replan 重复推理的 launch 开销 / 视觉历史的重复编码 / 图像-动作模型的显存峰值 ……>

## 可观测信号

<用什么测量能判定它存在。映射到 trace 字段（见 `docs/trace_schema.md`），
 例：model_ms 中 CPU launch 占比、stage_share[preprocess/model/postprocess/env_step]、
 cuda_peak_allocated_mb、cache 命中潜力 ……>

## 画像实验设计（characterization，无 variant）

- **target**：backend + workload（早期可用 fake / open-loop）
- **existence_check**：<被回答的具体问题>
- **metrics**：<决定 pressure 是否存在的测量集>
- **seed / warmup_iters / repetitions / runs**：<计测协议，见 `docs/measurement.md`>

> 该画像实验若需跑 harness，在 S4 用最小复现执行；这里先把设计写死。

## 取证数据

<跑完画像后填：实测分布、stage 占比、内存峰值等。>

## 结论

- **判定**：`exists` / `not_applicable` / `needs_reproduction`
- **依据**：<为什么。若 not_applicable，说明 pressure 为何在 WAM 下不构成瓶颈——这是有价值的负结果。>

## Gate

- [ ] User 认可结论。
- `exists` → 进入 S3；`not_applicable` → 归档（记入 PROGRESS 归档区），流水线终止；
  `needs_reproduction` → 谨慎进入 S3/S4 做最小复现取证。
