# WAM Harness 架构分析

> 只读审查 · 产品原型阶段 · 重点：可扩展性 · 验证基线：`172 tests passed`, `ruff` clean
>
> 结论已对照当前代码核实（截至 commit `148e0aa`）。下文是修正后的版本，不是原始草稿。

## TL;DR

方向是对的：要做"WAM 的 Ollama"+ "PEFT 式加速开关"，当前已有 backend / processor /
registry / trace / native loader+adapter 的正确雏形。真正的不足**不是"抽象不够多"**，
而是几个核心边界还没升格成一等对象：**runtime 装配 + 调用信封**、**optimization adapter**、
**model catalog**。不建议推倒重来。

## 目标定位

价值定位不是 "benchmark harness"，而是**模型库 + 部署入口 + 加速 profile 管理器**。
用户只面对 model id 和少量命令，内部才处理上游 repo / checkpoint / processor / runtime /
trace / 优化开关。

```bash
wam list / info / doctor / prepare
wam run  fastwam-libero --input obs.json
wam run  fastwam-libero --input obs.json --opt vla-cache
wam serve fastwam-libero --opt action-chunk-scheduling
```

## 已经做对的部分

- **核心命令贴近 Ollama UX**：`list/info/doctor/prepare/run/eval/serve`。
- **Backend/Processor 边界已建立**：runner 不含 FastWAM/Cosmos/DreamZero 分支。
- **Native loader/adapter 方向正确**：`NativeRuntimeLoader` 与 `NativeModelAdapter`
  把重依赖加载和调用边界分开。
- **依赖倒置已完成（地基，勿回退）**：`core` 不再 import `backends`；native 映射通过
  `RuntimeResolver` 链在组装根（`defaults.py`）注册。
- **契约已强类型（地基，勿回退）**：`Backend/Processor` 协议用 `InferenceRequest/
  InferenceResult`，`PreflightReport` 已通用化；`BackendSession` 收敛了 load/warmup/
  reset/preflight/infer 的生命周期 trace。

## 真正的不足

| 优先级 | 不足 | 为什么重要 | 代码信号 |
|---|---|---|---|
| **P1** | 缺统一的**调用骨架**（装配 + 信封合一） | run/serve/native-smoke/eval 各自做 manifest 解析、resolve、backend/processor 创建、profile 合并，以及 `run_start/error/run_end` 信封。加模型/加优化时会复制。`BackendSession` 已收编生命周期，但**装配前半段和信封事件仍未收编**。 | `Runner.run`、`ServeApp.__init__`、`NativeSmokeRunner.run`、`EvalRunner.run` 都重复 `resolve_runtime → build_optimization_profiles → create_backend` |
| **P1** | **Processor 所有权是假的** | runner 另起一个 `registry.create_processor` 塞给 session，仅用于 `runtime_contract` 的 modality 上报、**从不参与 infer、也从不被 bind**；真正干活的是 native backend `__init__` 自建、`load()` 里 `bind_runtime` 的那一个。公共边界对象是个未绑定的空壳孪生。 | `BackendSession.processor` vs `FastWAMBackend.processor`（`fastwam.py:299,329` 只 bind 后者） |
| **P1** | **优化 profile 只是参数，不是 adapter** | 目标若是"PEFT 式加速开关"，优化必须能 声明/校验/应用/诊断/trace。今天只有"声明+上报"。 | `_profile_state` 状态机封顶在 `requested`，**没有 `applied`**；`build_optimization_profiles` 只合并参数、无 apply 阶段 |
| **P2** | Manifest 太宽松 | `dict[str, Any]` 快，但跨边界字段缺结构会让扩展靠字符串约定和局部校验。 | `Manifest.backend/processor/workload/optimizations` 均为裸 dict |
| **P2** | `NativeBackendBase` 职责偏多 | 同时管 readiness / assets / upstream repo / env / sys.path / profile helper / runtime info / infer spine。新后端越来越依赖隐式父类约定。 | 单文件 ~820 行，方法跨度大 |
| **P2** | **Model catalog 与 component registry 耦合** | Ollama 式产品要可插拔的 model library（本地目录 / 远程 / 用户 Wamfile）；registry 应只注册组件，不该焊死内置 manifest 来源。 | `Registry.load_manifest()` 直连 `load_builtin_manifest`（`registry.py:84`） |
| **P3** | Serve transport 还不是可替换层 | 现 `http.server` 够原型；未来 remote endpoint / resident server / FastAPI / gRPC 需要 transport adapter。 | `ServeApp` 同时管 runtime lifecycle 和 HTTP |

### P2「spine 漂移」的现行实例

`BackendSession.load_backend(split_end_event=...)` 这个旗标只有 smoke 传 `True`，导致同名
事件 `backend_load` 在 run/serve 上带 `timing`、在 smoke 上不带（timing 跑到了
`backend_load_end`）。这是 trace schema 已经开始不一致的活证据，统一掉即可删除该旗标。

## 优化抽象应如何升格

把 optimization profile 从"名字+参数"升级为"可应用的 runtime adapter"。最小抽象：

```text
OptimizationProfileSpec
  name; scope: request|replan|episode|run|server
  target: processor|backend|model_adapter|workload|deployment
  params; supported_backends; conflicts
  prepare_requirements; doctor_checks; trace_fields

OptimizationAdapter
  validate(runtime); prepare(runtime); apply(runtime); trace_status(runtime)
```

- **用户层仍简单**：`wam run ... --opt vla-cache`，无需知道它作用在哪一层。
- **内部必须可诊断**：trace 记录 supported / applied / fallback / conflict。

> **关键修正——`apply` 不能是"外部包一层"，必须是"与 backend 协商挂载点"。**
> 目标优化（CUDA Graph、torch.compile、dit_cache、VLA-cache）大多侵入模型 forward /
> KV-cache，无法从外部干净套用。正确形状：backend 声明它支持哪些 hook
> （`supports: {cuda_graph, kv_cache_reuse, ...}`），adapter 去**请求**这些 hook，
> 不匹配则 `fallback` 并 trace。否则会造出"只能开关 backend 本就暴露之物"的漏抽象——
> 恰好漏在产品价值所在处。

## 建议演进顺序

1. **一个 `Invocation` 骨架**：装配 bundle → 开 trace → `run_start` →（`BackendSession`
   生命周期）→ `run_end`/`error`/`close`。**装配 + 信封合成一条缝，不要拆成
   `RuntimeAssembler` + `InvocationContext` 两个重叠抽象**；eval 共用前半段装配，执行语义
   保持独立。
2. **统一 processor 所有权**：实例由装配层统一创建并 `bind`，backend 可依赖但不再自建第二份。
3. **`OptimizationProfileRegistry` 升格为一等组件**：先支持 validate/apply/trace 与
   backend hook 协商，**先不做第三方插件安装**。
4. **结构化 Manifest 的跨边界子对象**：只结构化稳定边界，repo-specific config 仍留 dict。
5. **拆 `NativeBackendBase`**：用组合 helper 取代继续扩大父类。

> **排序提醒**：报告默认"先装配、后优化"，理由是 adapter 需要 runtime 对象。但依赖没那么硬
> ——可直接拿现有 backend 跑一条 opt-adapter 薄竖切（让某个优化真正 `apply` 并 trace 出
> `applied/fallback`）。装配去重 de-risk 的是**代码整洁**（用户无感），opt-adapter 竖切
> de-risk 的是**产品赌注**。这一步想先验证哪一个，由你拍板。

## 现在不要过度设计

- 不做第三方插件市场——先把内置 opt-adapter 生命周期跑通。
- 不全量 Pydantic 化 Manifest——只结构化稳定边界。
- 不急换 HTTP server——先把 `ServeApp` 的 runtime lifecycle 抽出来。
- 不把 eval 和 run 混成一个概念——可共享装配，语义分离。

## 关键代码位置

- `core/registry.py` — 组件注册 + manifest 加载（catalog 耦合点）
- `core/backend_session.py` — 已收编的生命周期；`split_end_event` 旗标
- `core/runner.py` / `serve.py` / `backends/native_support/smoke.py` — 三处装配/信封重复
- `core/eval_runner.py` — 第四处装配（执行语义独立）
- `core/types.py` — 公共契约 dataclass；Manifest 裸 dict
- `backends/native.py` — `NativeBackendBase`（职责偏多）
- `backends/fastwam.py` — native loader/adapter 范式；processor 双实例
- `backends/native_support/optimizations.py` — 优化状态上报（封顶 `requested`，无 `applied`）
