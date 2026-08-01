# IntentIR：J-only Replay 与多前端 J/C Structural Gate1 最终实施

> 日期：2026-08-01  
> 性质：当前实现与实验的唯一活动增补。  
> 上位依据：`IntentIR_J-only_Replay与JC结构Gate1_2026-07-31.md` 与
> `IntentIR_四核Lift瓶颈与pi归一化重构诊断.md`。  
> 取代：π-fed Replay、π约束 Replay Compiler、Composer/route/binding 主路径。

## 1. 最终边界

```text
Source_f + Harness
        │
        ▼
source-aware 作者（本实验由主代理直接承担，不调用 Lift agent）
        │
        ├── J       计算语义、logical domain/access/effect/control/output
        ├── C       匿名 implementation-selection 参数与 J attachment
        └── π       certification-only realization witness
                │
                ▼
TypeCheck(J,C,π)

fresh source-blind Replay_f(J, MaskC(C); U_f) -> R
                         # 不读取 π

Φ_f,p(P,c0) = Canon_f,p(Observe_f,p(Compile_f(P,c0)))
Φ^π_f,p(P,c0) = N^π_f,p(Φ_f,p(P,c0))

Gate1 = defined(Source) ∧ defined(Replay) ∧ Φ^π(Source)=Φ^π(Replay)
Gate2 = Source–Replay ∧ Source–RefEval(J) ∧ Replay–RefEval(J)
```

`J+C` 是论文提升、跨前端复用与 production backend 消费的对象。π 不进入 Replay、RefEval 或
production lowering，也不出现在 Gate1 的最终 J/C structural closure 结论中。

## 2. Replay 的绝对隔离

Replay request、只读文件、query、prompt、continuation、cache identity 与 compile callback 只能依赖：

- exact J；
- `MaskC(C)` 的匿名 slot/type/role/attachment，不含 valuation；
- 固定 public frontend context `U_f`；
- Replay 自己的源码、编译诊断与 own-IR。

它们不得依赖：

- π 或任何 π-derived ledger/view；
- Source、Harness valuation、Source TIR/PTX/TTIR；
- Gate1 observation/diff/frontier；
- 旧 Replay、sibling Replay 或历史 winner；
- kernel identity、target recipe 或 per-kernel template。

`S_J=M(J)` 与只读 cross-reference `V(S_J,MaskC(C))` 可以保留，但必须是 J-only/MaskC-only 的
确定性重索引。它们不能生成 target skeleton、program axis、loop/lane/guard 或 target API。

### 2.1 MaskC 是参数化绑定，不是待猜答案

隐藏 C valuation 不等于要求 Replay 猜一个常数。compiler-kind slot 是参数化 Replay source 与固定
Harness compile/launch 之间的接口：

- TileLang worker 在加载 Replay module 前，以 exact `c*` 名称注入本轮 compile valuation；Replay 只
  直接引用该 host global，不得定义、shadow 或猜值；
- CUDA 的 launch slot 只由外部 Harness 设置 grid/block，`c*` 不得成为 kernel runtime parameter 或
  kernel-body value；
- source-blind construction compile 可以注入类型正确的 representative valuation，但没有 Gate 权限；
- formal Gate1/Gate2 对同一份 exact Replay source 重新绑定 accepted Harness valuation `c₀`。

因此 `MaskC(C)` 隐藏的是 valuation，不是 binding convention。`U_f` 必须公开该前端如何引用或外部消费
匿名 slot，否则 agent 会把“未知值”误解为需要自己作者化的常数，既破坏隔离，也产生无意义的搜索。
固定 ingress 必须拒绝 Replay 对 compiler slot 的重绑定。

## 3. π 与 normalizer

fresh artifact 作者化只记录当前 profile 已有固定单臂 consumer 的 compact π。第一纵切采用：

```json
{
  "schema": "intentir.pi.normalization-witness.v1",
  "program_axes": []
}
```

当前 `jc-normalizer.v2` 是 identity/fail-closed 基线。即使 `program_axes` 非空，也先保持 hard；在
observer 无法把 physical coordinate 唯一回指 J domain 时，不做 `x/y/z` 标签折叠。以后新增
program/iteration/control/materialization relation，必须先有 profile-scoped、kernel-independent、单臂
recognizer与 hard-preservation audit，再进入 wire。不能先让 Lift 填字段、以后再解释。

## 4. 每前端 Gate1 profile

### 4.1 TileLang

主 observation 位于公开 TileLang/TIR compiler boundary：

- hard：匿名 ABI、alpha-normalized TIR structure、TIR topology、TIR operation multiset；
- canonical：schedule iterator spelling、单 basic block 的 call-free single-use let、声明式
  `T.reads/T.writes` region-set 顺序；
- diagnostic：PTX value/memory/control/coordinate/raw topology；
- hard safety envelope：PTX schema、ABI、target。

TIR normalization 只能消除 proof-irrelevant spelling；buffer shape、index、operator、load/store、loop bound、
control nesting 与 effect 必须保留。未知或歧义结构返回 `observer_unsupported` 或保持 hard。

### 4.2 CUDA

CUDA 没有可复用的高层 TIR，第一纵切选择 PTX semantic-effects observer：

- hard：匿名 ABI、semantic value operations、semantic memory effects、address-coordinate folds、
  collectives、control flow、coordinate uses 与 target；
- diagnostic：raw instruction topology、raw opcode multiset、raw memory strings、cache hint、虚拟寄存器与
  label spelling。

不能把 raw PTX 文本等同语义，也不能为闭合临时忽略 value op、predicate、memory effect 或 control。
若 PTX atom无法唯一分类为 J/C-owned 或 representation-only，fail closed。

### 4.3 共享与不共享

TileLang/CUDA共享 J/C/π wire、TypeCheck、Replay visibility、Gate1数学与Gate2。它们不共享
`Φ`、`H^(J,C)` 或 `N^π` 的实现；一个 profile 的 PASS 不能转移到另一个 profile。

## 5. 实验执行：主代理直接作者化，而非调用 Lift agent

为隔离 Replay/Gate 的核心变量并避免 Lift agent 的多轮延迟，正式实验由主代理直接读取当前 Source 与编译观察，重新作者化 `(J,C,π)`：

```text
anonymous Source + Harness/compiler view
  -> 主代理直接作者化完整 candidate.json
  -> Decode + CandidateCheck + SourceReconcile + TypeCheck
  -> fresh J-only Replay
```

主代理亲自作者化 J/C/π，不调用 Lift agent，不从历史 artifact复制为正式结果，也不把 Gate diff改写成
答案。历史证据仅可用来理解 wire 结构；每一项 operator、predicate、domain、access/effect、C ownership
与 compact π 都必须重新对照当前 Source/编译观察。固定工具只做 schema/reference、CandidateCheck、
SourceReconcile 与 TypeCheck 的机械拒绝。

## 6. 第一对实验：add2d

选择：

- `tilelang:add2d`；
- `cuda:tilelang_add2d`。

原因：两者表达同一逐元素加法，具有真实 dataset Source、Harness、编译器和 Gate2；语义足够简单，适合
首先检验协议边界，而不是把复杂算法能力与 observer能力混在一起。

必须得到的证据：

1. 两份 source-aware 主代理作者化记录；
2. 两份 current compact `(J,C,π)` 与 TypeCheck PASS；
3. Replay context receipt机械证明 π 不可见；
4. TileLang TIR profile 与 CUDA PTX profile各自 Gate1；
5. 各自三边 Gate2；
6. 两个 accepted J/C 的 canonical comparison：报告共同语义与真实前端差异，不强迫字节一致。

首个 `tilelang:add2d` baseline（`run-001`）已经给出协议反例：手工 J/C/π TypeCheck PASS，Replay
输入机械证明无 π，fresh Codex 也生成并编译了两读、一加、一写的正确数值程序；但它不知道 TileLang
既有的 host-global C binding convention，先后猜测并重绑 `c0`，最终选择 `128`，加入 carrier 与两个
J-absent guard。identity TIR Gate1 因而正确地拒绝，而不是为闭合临时放松 observer。修复面只限于
公开 `U_f`/ingress：明确 `c0` 由 host 注入并禁止重绑，同时要求直接 lower J-owned logical domain；
不修改 J，不向 Replay 暴露 `C=16`、Source、π 或 Gate diff，也不添加 kernel-specific matcher。

fresh `run-002` 验证了这个归因：同一手工 artifact 与 identity observer 不变，Codex 在只看
`J+MaskC(C)+U_f` 时生成14行参数化 TileLang，直接引用 host-injected `c0`，Gate1 的 TIR structure/
topology/operation multiset全部 hard PASS；随后有限隐藏输入上的 Source–Replay、Source–J、Replay–J
三边均为0误差。Replay episode 从 baseline 的555.32 s降到158.57 s，construction tool calls从18降到3。
这里的提升来自补全通用匿名编译槽 ABI，不来自 π、Source recipe、Gate反馈或放松 observer。

add2d 通过后，第二对使用 `tilelang:gather2d` / `cuda:tilelang_gather2d`。它们共享
`M=64`、block `[128,1,1]`、grid `[256,1,1]`，但包含 data-dependent row/column 间接寻址，用来检验
两种 observer 对合法 address realization 的误拒率。

## 7. 论文只证明四件事

1. **J-only sufficiency**：Replay没有π/Source/Gate仍能闭合，说明计算实现由J与匿名C接口驱动。
2. **跨前端可复用因子化**：同一数学 kernel 的TileLang/CUDA作者化结果具有可对应的J/C核心，而工程差异限制在
   frontend profile与Replay target code。
3. **Gate1的提前结构价值**：正确Replay通过；wrong op/index/mask/effect mutation在Gate2前被对应profile
   拒绝。Gate1不替代Gate2。
4. **保守性**：报告合法 realization false reject与`observer_unsupported`；π identity基线不伪造收益。

最小表格字段：artifact作者化wall time、candidate bytes、人工修订次数、TypeCheck、Replay wall time/segments/
compile-fix cycles、Gate1 verdict/primary category、Gate2三边、π bytes、Replay π-visible=false、observer
profile、persistent kernel-specific LoC（目标0）。

## 8. 停止条件

- Replay输入仍出现π：立即停止，不跑实验；
- profile依赖peer diff或kernel identity：立即停止；
- add2d Gate1失败：先区分J/C错误、Replay不忠实、observer unsupported，不增加provider预算碰运气；
- Gate1通过而Gate2失败：按三边定位，不盲目relift；
- 只有add2d两前端都完成fresh Gate1+Gate2，才进入gather2d与mutation实验。

最终原则：

> **source-aware作者直接给出J/C/π；Replay只从J与匿名C接口编译；π只供固定单臂Gate1 normalizer；每前端在自己的公开compiler boundary保守观察，最终由三边Gate2锚定行为。**
