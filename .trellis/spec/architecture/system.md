# LLM-first 系统架构

## 数据流

```text
S --LiftLLM--> (J,C,π) --ReplayLLM--> R_f
                                      │
                        ┌─────────────┴─────────────┐
                        ▼                           ▼
                Gate1 observer closure       Gate2 execution
```

Lift 和 Replay 是开放世界主执行者。固定核心只实现：

```text
ValidateReferences
EnforceIsolation
CommitArtifact
Build / Run / Observe / OptionalDump
Compare
IssueReceipt
```

## 三个研究命题

- `H_replay`：`J+π+C` 足以构造 source realization，由 Gate1回答；
- `H_sem`：源码与 J-only 语义约等价，由 Gate2 的 `Source↔E_J` 回答；
- `H_partition`：π 不足以独立替代 J，由 masked-J 审计。

三者不得互相替代。特别地，masked-J 不是最终语义 soundness 前提；Gate2 对任意 π
仍然具有 Source–J-only 声度。

## 失败循环

```text
failure evidence
  -> source-aware Lift
  -> fresh J',C',π'
  -> fresh source-blind Replay
```

旧候选只保留在审计记录中，不进入新 Replay 上下文。

## 在线成本上限

- 一次 Lift LLM；
- 一次 normal Replay LLM；
- normal Gate1 闭合后最多一次 masked-J Replay LLM；
- Gate2 不增加 LLM 调用。

更大规模的 mutation/swap/dropout、多模型、多 seed 和 C family 都是离线研究实验。
