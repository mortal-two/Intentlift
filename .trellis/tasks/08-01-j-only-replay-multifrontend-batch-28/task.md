# Task 28：J-only多前端闭合扩展到TileLang/CUDA各10个

> 状态：`P0_FROZEN`（2026-08-01）
>
> 上游正式证据：Task27 add2d/gather2d两对，共TileLang 2、CUDA 2。

## 0. 目标与计数

当前满足`TypeCheck PASS + Gate1 replay_closed + Gate2 gate2_pass`的current J-only正式证据只有：

- TileLang：`add2d`、`gather2d`，2个；
- CUDA：`tilelang_add2d`、`tilelang_gather2d`，2个。

旧π-fed、旧observer或旧Replay证据不计入current accepted。用户要求新增7组并使两边各10个；按机械
基数，新增7组只能达到9。因此本Task冻结“7组主批次 + 1组补足批次”，新增8个same-origin pair，最终
TileLang 10、CUDA 10。

## 1. 冻结cohort

七组主批次：

1. `relu2d`：unary max/relu；
2. `floor2d`：unary floor；
3. `exp2d`：unary exp；
4. `where2d`：comparison + select；
5. `add_bias2d`：rank-1 broadcast + add；
6. `clamp2d`：broadcast lower/upper bound + max/min；
7. `copy2d_divmod`：identity data movement，Source带pid div/mod realization。

补足批次：

8. `row_sum`：row-wise f32 sum reduction。

每个stem分别运行`tilelang:<stem>`与`cuda:tilelang_<stem>`。若某case在current Source/descriptor/TypeCheck
阶段确定性unsupported，保留失败并只用预注册fallback：`transpose2d`，其次`row_max`；不能按Gate结果
挑更容易成功的case。

## 2. 不变量

1. 主代理直接读取current Source/descriptor/compiler view并手工作者化`(J,C,π)`；不调用Lift agent；
2. 历史artifact只可核对wire语法，不作为正式输入；每个语义、ABI、predicate、effect均重新核对；
3. Replay payload根恰为`j,c,u_f`，π、Source、Harness valuation、Gate diff、prior Replay不可见；
4. 每个case使用fresh source-blind Codex episode，不能复用对话、源码或sibling candidate；
5. TileLang hard=TIR semantic structure、PTX diagnostic；CUDA hard=semantic PTX、raw PTX diagnostic；
6. 失败只允许沉淀kernel-independent J消费/U_f/observer机制；禁止case id、kernel名、Source path matcher；
7. Gate1闭合后才运行三边Gate2；失败case不借用paired frontend的PASS；
8. 不恢复`python/tests`，不以mock/smoke替代真实closure workflow；
9. 所有计数只认current run中的TypeCheck/Gate1/Gate2机器证据。

## 3. 批量执行

- [ ] 手工构造16份artifact并全部Decode/TypeCheck；
- [ ] runner接入新CASE_BUILDERS/receipts；
- [ ] 每个case原子分配独立run目录；
- [ ] provider episode可有界并行，单case仍保持fresh session；
- [ ] GPU/Gate2阶段限制并发，避免资源竞争污染结果；
- [ ] 先跑7组主批次，再跑row_sum补足批次；
- [ ] 每个失败保存exact candidate、compiler observation与hard category；
- [ ] 最终生成10+10总表、按pair的J共同核心/真实delta与论文claim boundary。

## 4. 最小记录字段

每个case记录：artifact/J/C/π canonical bytes、TypeCheck、Replay model/provider calls/time/source bytes、
π visibility、Gate1 profile/hard/diagnostic、三边Gate2与最大误差。每个pair额外记录共同数学关系、full-J
差异原因、generic production LoC、kernel-specific production LoC。

## 5. 完成条件

- current accepted TileLang=10、CUDA=10；
- 新增正式run的`lift_agent_calls=0`且`pi_visible_to_replay=false`；
- 8个pair各自有两边独立TypeCheck/Gate1/Gate2证据，或诚实记录预注册fallback/无法达成原因；
- 不把旧流程结果、compile-only或Gate1-only计入accepted；
- 论文文档明确这10+10只支持measured cohort，不外推任意frontend/kernel；
- 提交后工作区clean。
