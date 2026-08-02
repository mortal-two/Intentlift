# Task07 Stage 2：现有 pi 图足够作为第一版 residual contract

## 决策

本阶段不扩 `RealizationWitness` JSON shape，不增加 address expression、cast op、mask placement、
TTIR skeleton 或自由文本 recipe。先把现有 `region + instance + route + order + sharing` 的
全局含义定义完整，并由一次 fresh canary 检验。

v3 只证明 Lift 输出没有 `p3`，没有证明 schema 禁止 `p3`。当前 parser 已允许 instance
引用 J/C/interface handle，并允许多个 instance 引用同一 handle。因此 row-stride 与 flat
至少具有以下可区分图不变量：

```text
row-stride:
  p* event chain -> shared fanout -> access-base event -> final access event

flat:
  final access event only; no p* dependency, base chain or shared fanout
```

## 唯一解释

- instance：一个 compiler-observable realization event；不是新 semantic value。
- semantic_ref：event 最近的 J/C/interface anchor；只引用，不重述其内容。
- route：event 间匿名 def-use，`slots` 是 target-local opaque input ordinal。
- order：数据依赖不能表达的必要 source partial order。
- sharing：一个实际 materialization producer 被指定 consumers 共同复用。
- region/placement：event 的 source realization scope。

同一 semantic_ref 可出现多次，只表示多个 source event。它不能被用作任意 bit string：图的
解释仅限上述合同，不能私下约定“出现三次就是乘法/出现四次就是 cast”。Replay 根据完整
J/C/interface 和公共前端知识决定具体写法，Gate1 观察最终编译拓扑是否闭合。

## Source-reference completeness

Lift prompt 现在要求：源码中实际参与 address、transfer、synchronization、placement 或
sharing realization 的每个匿名 interface/C handle 都必须在 pi 出现，并 route 到受影响的
event。固定代码不从源码抽取这张图，也不按名字猜 `stride/BLOCK`；完整性由 source-aware
Lift 提议、source-blind Replay 构造、strict Gate1 终检。

Replay prompt 同时要求：不得压平多级 chain、丢弃 interface/C dependency、复制 shared
producer 或发明不存在的 event。所有 operator/predicate/fill/constant/index relation/effect
仍只能来自 J。

## LayerNorm 边界

以下二者不是靠 pi 修复的同一 J：

```text
select(mask, centered, zero) -> mul -> reduce
mul(centered, centered) -> select(mask, square, zero) -> reduce
```

operator、predicate、fill 和 value edges 由 J 唯一拥有，故 J 必须不同。固定上述 J 后，
“selected value 只物化一次并共享给 mul 两个 operand”与“两个 realization event”才是 pi
可表达的 materialization/share 差异。

## 离线证据

`python/tests/test_gate1_direct_pi.py` 已覆盖：

1. 同一 `p*` 的多 event anchor；
2. runtime row-base graph 与 flat graph 不同；
3. `p*`-anchored instance 到 access-anchored instance 的 routes 和 sharing 原样进入
   source-blind Replay packet；raw `p*` 本身不是 graph endpoint；
4. Replay system contract 明确禁止 chain collapse/dependency drop；
5. LayerNorm pre/post mask 产生不同 J，pi 不含 `select` 等语义 atom；
6. 既有 closed schema 继续拒绝 operator/predicate/value/API/text。

## 升级门槛

只有在 fresh artifact 已正确包含 non-C parameter dependency、多级 chain 和 sharing，而 normal
Replay 仍稳定地压平 address topology 时，才允许提出 schema 扩展。第一个候选上限仅为两个
source-neutral access mechanism：program-uniform `address_base` 与 lane-varying
`address_lane`；仍不得记录公式、operator、width、signedness、dtype 或 frontend API。任何
扩展都必须新建 protocol epoch，不能修改历史 v1/v2/v3/v4 evidence。
