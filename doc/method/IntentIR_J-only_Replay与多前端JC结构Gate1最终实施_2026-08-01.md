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
source-aware Codex Lift
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

## 3. π 与 normalizer

fresh Lift 只作者化当前 profile 已有固定单臂 consumer 的 compact π。第一纵切采用：

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

## 5. 自动 Lift，而非手写 artifact

正式实验只运行 source-aware Codex Lift：

```text
anonymous Source + Harness/compiler view
  -> Codex owner作者化完整 candidate.json
  -> Decode + CandidateCheck + SourceReconcile + TypeCheck
  -> fresh J-only Replay
```

主代理不手写 J/C/π，不从历史 artifact复制，不把 Gate diff改写成答案。固定工具只维护 schema/reference
机械合法性并拒绝；所有 operator、predicate、domain、access/effect、C ownership 与 π relation 选择由
source-aware owner完成。

## 6. 第一对实验：add2d

选择：

- `tilelang:add2d`；
- `cuda:tilelang_add2d`。

原因：两者表达同一逐元素加法，具有真实 dataset Source、Harness、编译器和 Gate2；语义足够简单，适合
首先检验协议边界，而不是把复杂算法能力与 observer能力混在一起。

必须得到的证据：

1. 两个 source-aware fresh Lift trajectory；
2. 两份 current compact `(J,C,π)` 与 TypeCheck PASS；
3. Replay context receipt机械证明 π 不可见；
4. TileLang TIR profile 与 CUDA PTX profile各自 Gate1；
5. 各自三边 Gate2；
6. 两个 accepted J/C 的 canonical comparison：报告共同语义与真实前端差异，不强迫字节一致。

add2d 通过后，第二对使用 `tilelang:gather2d` / `cuda:tilelang_gather2d`。它们共享
`M=64`、block `[128,1,1]`、grid `[256,1,1]`，但包含 data-dependent row/column 间接寻址，用来检验
两种 observer 对合法 address realization 的误拒率。

## 7. 论文只证明四件事

1. **J-only sufficiency**：Replay没有π/Source/Gate仍能闭合，说明计算实现由J与匿名C接口驱动。
2. **跨前端可复用因子化**：同一数学 kernel 的TileLang/CUDA Lift得到可对应的J/C核心，而工程差异限制在
   frontend profile与Replay target code。
3. **Gate1的提前结构价值**：正确Replay通过；wrong op/index/mask/effect mutation在Gate2前被对应profile
   拒绝。Gate1不替代Gate2。
4. **保守性**：报告合法 realization false reject与`observer_unsupported`；π identity基线不伪造收益。

最小表格字段：Lift wall time、candidate bytes、CandidateCheck次数、TypeCheck、Replay wall time/segments/
compile-fix cycles、Gate1 verdict/primary category、Gate2三边、π bytes、Replay π-visible=false、observer
profile、persistent kernel-specific LoC（目标0）。

## 8. 停止条件

- Replay输入仍出现π：立即停止，不跑实验；
- profile依赖peer diff或kernel identity：立即停止；
- add2d Gate1失败：先区分J/C错误、Replay不忠实、observer unsupported，不增加provider预算碰运气；
- Gate1通过而Gate2失败：按三边定位，不盲目relift；
- 只有add2d两前端都完成fresh Gate1+Gate2，才进入gather2d与mutation实验。

最终原则：

> **Lift自动作者化J/C/π；Replay只从J与匿名C接口编译；π只供固定单臂Gate1 normalizer；每前端在自己的公开compiler boundary保守观察，最终由三边Gate2锚定行为。**

