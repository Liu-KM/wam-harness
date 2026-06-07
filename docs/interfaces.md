# 上层接口契约（Interfaces）

本文件用 Python `Protocol` 的形式确定 EazyWAM 的核心抽象边界。实现可以在
Phase A 中细化这些草图，但必须保持产品边界：model id → model entry → backend /
processor → action chunk。

设计原则（见 `AGENTS.md`）：核心 runner 只用契约词汇推理，不得出现任何上游仓库
（FastWAM/DreamZero/Cosmos/LingBot/Motus/Qi）的分支判断。backend-native 的张量
布局、归一化、cache 机制、传输协议一律不得穿过这些接口。

> 这些是契约草图，不是 `src/` 实现。类型最终落在 `src/eazywam/core/`，
> 由 Phase A deployment spine 实现。

## 数据类型（契约对象）

```python
from typing import Protocol, Optional, Any, runtime_checkable

# 全部为不可变数据载体（实现用 dataclass(frozen=True) 或等价物）。
# 字段对齐 docs/contract.md 的 Inference Contract。

class Observation(Protocol):
    images: dict[str, Any]          # 命名视图: primary / wrist / exterior_0
    prompt: str                     # 任务指令
    state: Optional[dict[str, Any]] # proprio / joint_position / ...
    history: Optional[Any]
    session: Optional["SessionInfo"]
    metadata: Optional[dict[str, Any]]

class InferenceRequest(Protocol):
    observation: Observation
    action_horizon: int
    replan_steps: int
    num_inference_steps: Optional[int]
    return_future: bool
    return_value: bool
    reset: bool
    cache_control: Optional[dict[str, Any]]
    runtime_options: Optional[dict[str, Any]]

class InferenceResult(Protocol):
    action_chunk: Any               # [T, D]，但核心不假设具体张量库
    action_chunk_shape: tuple[int, ...]
    future_frames: Optional[Any]
    value: Optional[Any]
    warnings: list[str]
    backend_metadata: dict[str, Any]
    timing: Optional[dict[str, float]]
    memory: Optional[dict[str, float]]

class ModelEntry(Protocol):
    id: str
    backend: dict[str, Any]
    processor: dict[str, Any]
    assets: dict[str, Any]
    defaults: dict[str, Any]
    optimizations: dict[str, Any]

class OptimizationProfile(Protocol):
    name: str
    deployment_class: str
    parameters: dict[str, Any]
```

## 生命周期接口

### Backend

backend 暴露契约定义的推理生命周期。它**不**向核心泄露 backend-native 名称、
张量布局、归一化或 cache 机制。

```python
@runtime_checkable
class Backend(Protocol):
    def load(self) -> None: ...
    def warmup(self) -> None: ...
    def reset(self) -> None: ...
    def infer(self, request: InferenceRequest) -> InferenceResult: ...
    # preprocess/postprocess 由 backend 内部经 Processor 完成，不在核心循环显式调用。
    def runtime_info(self) -> "RuntimeInfo": ...
    def close(self) -> None: ...
```

约定：
- `load` 加载权重、processor 绑定、server/client 连接等重资源。
- `runtime_info` 返回 model_entry_id/name/backend/source_repo/mode/device/dtype/
  optimization_profiles，核心由此获知运行元数据，而非用户手写 capabilities。
- `reset` 清理 episode/session/action-buffer/cache 边界；需要 seed 的后续 backend 可通过
  request/runtime_options 或 backend config 传递，但不要把 backend-native 状态泄露给 core。
- `infer` 内部完成 preprocess→model→postprocess，并在 `InferenceResult.timing`
  里填充各阶段毫秒。
- `close` 释放 backend 自己创建的 server、socket、worker、CUDA graph pool 或临时状态；
  runner/smoke/serve shutdown 路径都应调用。

### Processor

processor 在 harness 观测/结果与 backend-native 输入/输出间转换。它声明模态限制。

```python
@runtime_checkable
class Processor(Protocol):
    def to_model_inputs(self, observation: Observation) -> Any: ...
    def to_harness_result(self, raw_output: Any) -> InferenceResult: ...
    def modality_limits(self) -> dict[str, Any]: ...
    def smoke_observation(self) -> Observation: ...
```

### Registry

registry 把 model/backend/processor/workload 名映射到实现，核心据此选择实现，**不得**对仓库名做分支。

```python
@runtime_checkable
class Registry(Protocol):
    def register_backend(self, name: str, factory: Any) -> None: ...
    def register_processor(self, name: str, factory: Any) -> None: ...
    def register_workload(self, name: str, factory: Any) -> None: ...
    def create_backend(self, model_entry: ModelEntry, profiles: list[Any]) -> Backend: ...
    def create_processor(self, model_entry: ModelEntry) -> Processor: ...
```

### Observer（时间/内存）

observer 采集计时与内存，CPU-only 下干净降级。

```python
@runtime_checkable
class Observer(Protocol):
    def stage_timer(self, stage: str) -> "ContextManager[None]": ...  # perf_counter
    def cuda_timer(self) -> "ContextManager[None]": ...               # CUDA events，CPU 下 no-op
    def sample_memory(self) -> dict[str, Optional[float]]: ...        # 字段名恒定，CPU 下 0/None
    def reset_peak(self) -> None: ...                                 # run 边界归零
```

### TraceWriter

JSONL trace 写入。append-only，错误不得静默吞掉（见 contract.md trace 阶段）。

```python
@runtime_checkable
class TraceWriter(Protocol):
    def emit(self, event: "TraceEvent") -> None: ...   # 一行一事件，含 schema_version
    def close(self) -> None: ...
```

### Comparator（compare 步骤）

读取两个 run 的 trace + 输出 artifact，产出 `comparison_result` 事件。

```python
@runtime_checkable
class Comparator(Protocol):
    def compare(
        self, baseline_trace: str, variant_trace: str
    ) -> "ComparisonResult": ...   # faster/slower/same/invalid/not_comparable
```

## 不变量（呼应 contract.md）

- 核心 runner 只依赖以上 Protocol，不 import 任何具体 backend。
- backend-native 键名与张量布局不得出现在 Protocol 签名里。
- 以上签名是 Codex 的实现契约；改动须申报，由 Claude 裁决。
