# Lift 使用匿名 Source / TTIR 的方法

> 状态：方法决议，TTIR 工具待实现；不修改 SCRC 数学与 `(J,C,π)` 边界。

## 方法

匿名 Source 已可在冻结的匿名 Harness binding 下真实编译。下一步让同一个持久 Lift agent：

```text
匿名 Source + 匿名 Harness view
        → 调用现有 frontend Compile/Observe
        → 查看匿名、规范化 TTIR
        → LLM 自己生成完整 (J,C,π)
        → CandidateCheck
        → 外部正式 TypeCheck → fresh blind Replay
```

Source 用于判断逻辑语义，TTIR 用于观察 def-use、control、loop、address、materialization 与 sharing。固定工具只编译和展示，不从 TTIR 生成、补全或建议任何 J/C/π 字段。

新 frontend 仍只需已有的：

```text
Compile_f + Run_f + Observe_f
```

不新增每 frontend 的 CFG/DFG extractor、Source-to-J/π parser 或 RuleCard。Replay 永远看不到 Source、TTIR、Harness、C values 或 Gate diff，仍只看 `J + π + MaskC(C) + U_f`。

## CandidateCheck

CandidateCheck 只是 **Lift-local TypeCheck preflight**：对 LLM 尚未正式提交的完整草稿运行严格 parse、Decode/Elaborate、匿名 Harness 一致性检查和正式 TypeCheck 的同一 source-free 核心，并把机械错误返回同一个 Lift agent。

它不读取或比较 Source，不判断 J/π 是否正确，不生成 patch，不修改 artifact，也不进入证书。PASS 后 Runner 仍必须对 exact submitted bytes 重新执行正式 TypeCheck。

因此认证公式始终只有：

```text
SCRC_CERT = TypeCheck ∧ Gate1 ∧ Gate2
```

## 当前状态

- 已有：匿名 Source、真实编译绑定、Compile/Observe、CandidateCheck、Source-blind Replay。
- 待做：向 Lift workspace 开放只读 Source-side TTIR 查询，并补齐匿名化、权限和回归测试。
