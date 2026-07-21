# Task08 Stage 0：π Graph Node Namespace 审计

## 冻结反例

权威输入是 protocol-v4 的未修改 Lift response：

```text
artifacts/real_llm_batch_28_v4/recovery/liger_swiglu_forward/
  calls/01_lift/response.txt
  receipt.json
```

- response SHA：`289aa3b37fb04c198941f31f76e31a4d05b118049dcf149e4593e8bde8e7db0a`
- receipt SHA：`98a3efbc9b9027e2480e68347e053ea62ce7014b6f34db95b707fd39364573d2`
- 17 个 declared instances：`i0..i16`
- 30 条 routes：9 条 source 非 `i*`，所有 target 都是 declared `i*`
- 16 条 order edges：所有 before/after 都是 declared `i*`
- 4 条 sharing relations：其中 1 条 source 是 raw `p3`，所有 consumers 都是 declared `i*`

9 个非法 route source 按原始顺序是：

```text
c1, p0, p3, p1, p3, p2, p3, c0, p4
```

因此 v4 不只有首先触发 parser 的 route foreign-key 错误；它还有一条
`{"source":"p3","consumers":["i1","i2","i3"]}` sharing namespace 错误。parser 在第一条
invalid route fail closed，尚未到 sharing validation。这一补充不改变 v4 receipt 的失败分类。

## 歧义来自哪里

机器合同原本只有一种合法结构：

- `instances[*].semantic_ref` 接受声明过的 J/C/interface handle；
- `instances[*].id` 声明 π 内部 graph node，命名为 `i*`；
- route/order/sharing 的端点 schema pattern 都是 `^i[0-9]+$`；
- 本地 parser 进一步检查这些 `i*` 是否真的存在。

但自然语言投影不完整：

1. Lift prompt 要求每个 Interface/C handle “occur in pi and route to the affected instances”，没有
   明说 raw handle 只能通过一个 anchored instance 进入 graph；
2. Task07 研究文档使用了 `p* -> access routes` 简写；
3. Responses text format 为 `strict=false`。JSON Schema 是强提示，不是 provider-side 完整性保证，
   本地 parser 才是权威接受边界；
4. `semantic_ref` 名称虽沿用历史，但它实际上是 J/C/interface 的中性 anchor，并不授权 raw ref
   成为 graph endpoint。

LLM 因而把“raw handle 参与 realization”直接实现为 `p3 -> i1`，而不是声明一个
`semantic_ref=p3` 的 `i*` node 再连接到 `i1`。

## 唯一 Graph Grammar

令：

```text
R = {r0, ..., r(m-1)}                    declared region IDs
I = {i0, ..., i(n-1)}                    declared realization-event IDs
H = declared anonymous J/C/interface handles
```

唯一合法解释为：

```text
region.parent        ∈ R ∪ {null}
instance.id          ∈ I                 and declares exactly one graph node
instance.semantic_ref∈ H                 and only anchors that node
instance.region      ∈ R
route.source/target  ∈ I
order.before/after   ∈ I
sharing.source       ∈ I
sharing.consumers    ⊆ I
```

`semantic_ref` 和 graph endpoint 是两个不同角色，不是两个可互换的 node namespace。edge 不会
隐式声明 instance。若一个 runtime parameter 参与 access realization，最小通用写法是：

```json
{
  "instances": [
    {"id":"i0","semantic_ref":"p0","region":"r0","mechanism":"realize"},
    {"id":"i1","semantic_ref":"a0","region":"r0","mechanism":"transfer_direct"}
  ],
  "routes": [{"source":"i0","target":"i1","slots":[0]}]
}
```

不是：

```json
{"source":"p0","target":"i1","slots":[0]}
```

同一 H anchor 可以对应多个 `i*`，因为多个 compiler-visible stages 可能锚定同一个 declared
handle；这不允许 parser 按 anchor 猜选其中一个 instance。`slots` 只是在 target instance 内的
opaque input ordinal，不是 J operand schema，也不能编码 operator。

## 字段决策

Task08 **保留现有 JSON shape 和字段名**，不把 `semantic_ref` 改成 `anchor_ref`，也不把
`source/target` 改成第二套别名。理由不是兼容优先于正确性，而是：

1. schema/parser 已经只接受一种 graph endpoint namespace，错误位于合同投影而非接受域；
2. rename 不增加类型能力，只会迫使历史 artifact schema 分叉或同时接受两种 spelling；
3. 同时接受新旧字段反而违反“一个真值”，版本分派则增加与本问题无关的固定控制面；
4. 一个由 schema description、Lift prompt、Replay prompt 共用的短 grammar 可以直接消除歧义。

这不是 response patch：模型仍必须一次生成合法对象，本地代码仍只验证/拒绝。

## 阶段 1 实现边界

- 在 π 权威模块定义一个共享 grammar renderer，投影到 Responses schema、Lift instructions 和
  Replay instructions；不得复制三份文字；
- schema 对 instance、route、order、sharing 各字段增加 role-specific description；
- parser 对 raw `p*/c*/n*` endpoint 与 dangling `i*` 给出带数组下标/字段名的确定性错误；
- 不自动创建 instance，不按 `semantic_ref` 匹配，不重新排序或补 edge；
- 不新增 mechanism、address formula、cast、predicate、dtype、shape 或 operator；
- region-parent/order 的既有 cycle 检查保持，是否扩 route topology 不是本任务范围。

## 阶段 1 测试边界

至少覆盖：

- raw `p0/c0/n0` 分别出现在 route source/target、order before/after、sharing source/consumer；
- 合法形状但未声明的 `i99` endpoint；
- repeated `semantic_ref` 的两个不同 `i*` 都可被 edge 引用；
- duplicate/declaration-order instance ID、region/order cycle、slot uniqueness；
- `parse -> to_json -> parse` 以及 Replay packet 原样保留 graph；
- v4 response 离线仍在第一条 `pi.routes[0].source='c1'` 精确拒绝。

这些测试证明合同闭合，不证明合法 π 已足够令 strict Gate1 闭合。后者只能由新 epoch canary
给出证据。
