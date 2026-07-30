# Lift 使用匿名 Source / TTIR 的方法

> 状态：已实现；不修改 SCRC 数学与 `(J,C,π)` 边界。

## 方法

匿名 Source 在冻结的匿名 Harness binding 下编译一次，raw TTIR与现有Gate1 compiler fingerprint进入该branch的只读Lift workspace。
同一个持久Codex Lift agent像普通coding agent一样直接读文件、局部搜索并编辑：

```text
匿名 Source + 匿名 Harness view
        → 调用现有 frontend Compile/Observe
        → 先看紧凑Source/Harness/compiler brief，再按需查询CFG、SSA-DFG、effect、time、program-address或TTIR局部
        → LLM 自己生成完整 (J,C,π)
        → CandidateCheck
        → 可选 public SemanticProbe(Source, RefEval(J))
        → 外部正式 TypeCheck → fresh blind Replay
```

Source 用于判断逻辑语义，compiler view用于观察def-use、control、loop、address、materialization与sharing。
brief和查询只是现有Source/Harness/observer事实的有界投影；它们不映射IntentOp、不生成字段、不补全J/C/π。

新 frontend 仍只需已有的：

```text
Compile_f + Run_f + Observe_f
```

不新增每 frontend 的 CFG/DFG extractor、Source-to-J/π parser 或 RuleCard。Replay 永远看不到 Source、TTIR、Harness、C values 或 Gate diff，仍只看 `J + π + MaskC(C) + U_f`。

## CandidateCheck

CandidateCheck 只是 **Lift-local TypeCheck preflight**：对 LLM 尚未正式提交的完整草稿运行严格 parse、Decode/Elaborate、匿名 Harness 一致性检查和正式 TypeCheck 的同一 source-free 核心，并把机械错误返回同一个 Lift agent。

它不读取或比较 Source，不判断 J/π 是否正确，不生成 patch，不修改 artifact，也不进入证书。PASS 后 Runner 仍必须对 exact submitted bytes 重新执行正式 TypeCheck。

`SemanticProbe`与CandidateCheck严格分离：宿主对每份通过CandidateCheck+SourceReconcile的精确草稿自动在一个固定public seed上执行Source与J-only RefEval，
若有分歧就把逐输出差异交还同一个source-aware Lift owner。它不读π、不运行Replay、不自动修J，也不进入认证；
hidden Gate2仍只在Gate1 winner确定后独立生成。因此它只是coding-agent的“运行—观察—再编辑”工具。

因此认证公式始终只有：

```text
SCRC_CERT = TypeCheck ∧ Gate1 ∧ Gate2
```

## 当前状态

Codex Lift工作区只给匿名Source、Harness、queryable compiler observation、可查询closed Schema/IntentOp参考、feedback文件与
`candidate_check / semantic_probe`；首prompt不嵌入大对象。Replay只消费同一source-blind packet，但通过
`packet_query`按需读取，并以`submit_replay`一次完成ingress+compile诊断；own-TTIR inspect仍只看自己的Replay。正式验收只看真实pilot的首稿时间、工具数、
累计输入、Gate1 frontier和最终闭合，不恢复独立测试套件。
