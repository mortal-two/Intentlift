# Task 24：J 主导的 π_min 与三前端 Episode-Local Composer Replay

> 历史状态：`ARCHIVED_P5_9_CERTIFIED_1_NO_CLOSURE`（2026-07-31；后继Task 25亦已归档，当前为Task 26）。
>
> 本文和run metadata中的原`.trellis/tasks/07-31-j-driven-replay-composer-24/...`路径保留执行时身份；
> 当前文件整体位于本归档目录，历史artifact内容不因移动而重写。
>
> 定位：Task 15/Task 11 的 Replay 构造机制升级，以及 Task 23 多前端实证的真实 closure 执行线。
> 本任务不接管 production backend；backend 仍在 SCRC 后只消费 accepted `J+C`。
>
> 执行模型：真实 Replay episode 默认使用 Codex。简单纵切从 `high` 开始；只有 exact artifact 下重复
> 暴露模型构造能力不足时才升 `xhigh`，不能用 effort 升级掩盖 schema、consumer 或工具错误。

## 0. 要证明什么

本任务只证明四件事：

1. `artifact.v12` 是首版 `π_min` 兼容基线；真实反例需要的 `artifact.v13` 也只能增加
   J/C-dependent、target-independent 的窄 realization delta，不能复制
   op/dataflow/type/shape/Source/API/target recipe；
2. source-blind Codex可以在fresh episode内借助纯语法Composer更快地构造、编译和修正Replay；
3. 同一共享Composer core可以被Triton、TileLang和CUDA薄profile复用，而无需每前端J lowerer或
   kernel模板；
4. 最终仍由exact TypeCheck、各前端Gate1和三边Gate2认证，Composer不改变数学或信任边界。

Task 24时期的历史设计增补（已由Task 25权威设计取代）：

- `doc/archive/2026-07-31-pre-agent-internal-route-composer/IntentIR_J主导_pi_min与Episode_Local_Composer_Replay最终设计_2026-07-31.md`

当前Replay作者化权威设计：

- `doc/IntentIR_共享J语义物化与pi约束Replay编译Agent最终设计_2026-07-31.md`

上位数学与对象边界继续由原四份SCRC设计和第五份backend设计决定。

## 1. 固定公式

```text
Lift_H(Source) -> (J,C,π_min)
Replay_f = FreshAgent_f(J,π_min,MaskC(C);U_f)

SCRC_CERTIFIED
  iff TypeCheck(J,C,π_min)
  and Gate1_f(Source(C),Replay(C))
  and Gate2(Source(C),Replay(C),RefEval(J))
```

Composer只存在于`FreshAgent_f`内部；最终提交物仍是完整`replay.source`。

## 2. 当前基线

### 2.1 已有能力

- artifact/decode/typecheck：`v13 / v18 / v31`（v12 继续作为严格 legacy wire 回放）；
- π已有closed regions/instances/routes/order/sharing/program_axes/iterations/physical time/terminal control/
  route extent transform；
- Codex Replay每个`generate`已创建fresh workspace，并有exact patch、ReplayCheck、compile、own-IR、
  checkpoint/rollback与exact submit；
- Triton以TTIR为hard observer；
- TileLang以结构TIR为hard observer、PTX只作diagnostic；
- CUDA以声明的normalized PTX semantic topology为observer，并有native Gate2 runner；
- 三边Gate2已绑定exact J。

### 2.2 当前缺口

- Replay仍主要维护一次性大文本，缺少稳定node id、局部结构编辑和source map；
- normal/masked臂可复用同一writer接口，J-necessity隔离不够强；
- “composer”在旧文档中被一概禁止，未区分semantic lowerer与syntax-only editing mechanism；
- 三前端有compile/observe，但没有同一Composer core上的真实provider closure证据；
- Task 15轨迹显示大量时间/token耗在重复构造与修整源码，不是check/compile本身。

### 2.3 当前实现检查点（尚无closure claim）

- `python/intentir/verify/gate1/replay_composer.py`已实现共享stdlib-only core：closed
  `line/block/raw_block`、stable ID、局部insert/replace/delete/move、Python/C-like render、source map、
  bounded state/audit以及plan/source/map一致的checkpoint/rollback；
- Codex Replay backend现升为`openai.codex-cli.intentir-replay.v36`，Composer protocol为v2；每个
  fresh workspace安装自己的通用builder/plan/editor；agent仍可直接作者化完整source，两条路径都进入
  原有exact ingress/compile/submit；
- Composer失败或未使用不改变Replay接受集合；host checkpoint/diagnostic map只作非权威辅助；
- `off/available/preferred`只控制episode作者体验并写入proof-irrelevant evidence；`preferred`优先使用
  `work/build_replay.py`，仍保留raw escape。telemetry区分offer、成功/失败action与最终source路径；
- Triton/TileLang/CUDA共享同一core，profile只选择Python-like或C-like排版；
- normal与Mask-J由不同writer对象发起，正式workload runtime会为每个formal round另外创建masked
  writer，证据按normal candidate与masked物理角色分开保存；
- 每份Codex trajectory附带`pi-min.ownership-receipt.v1`，它只重述已经通过的strict artifact decode、
  dependent TypeCheck与closed Replay packet事实，不是第二个validator或新artifact；
- 已完成不调用provider的三前端render、全部八种局部edit、source-map、checkpoint/rollback、copied CLI、
  stdlib-only import与writer identity smoke；独立只读审计未发现J/π读取、target recipe、kernel template或
  Gate bypass；
- 已形成TileLang/CUDA五类代表case的真实横向矩阵：`add2d/row_sum/gather2d/rowmask_where2d`
  双前端及TileLang `transpose2d`共9项达到`SCRC_CERTIFIED`；CUDA `transpose2d`在历史PTX v5
  profile下为真实`NO_CLOSURE`。这些证据证明现有Composer不妨碍闭合，并有一组同artifact A/B
  显示单样本成本下降；它们不证明当前syntax-only Composer已解决复杂J/π消费成本。

### 2.4 P5真实运行日志

### Pilot A：修复前的外部临时运行

`tilelang:add2d / /tmp/intentir-task24-composer/tilelang_add2d/run-001`（Codex Lift + Codex
Replay，`gpt-5.6-sol high`，Mask-J配置开启）：

- real Lift产生`artifact.v12`，normal Codex Replay在一个CLI turn内生成完整TileLang source，并通过
  `replay-check.v15`与source-blind construction compile；
- 本例agent选择direct-source raw escape，Composer telemetry为`installed=true, used=false`；这是真实
  行为，不能重标为Composer成功；
- 进入共享Gate1 non-closure feedback时，`SCRCGateRunner`错误读取
  `TileLangCompileResult.ttir`，以`AttributeError`终止；masked臂和Gate2均未运行，故无closure claim；
- 根因不是Lift信息丢失或Replay无法消费J，而是共享feedback的Triton-specific result contract；CUDA
  也有同类潜伏错误；
- 修复采用`observer_profile.compiler_stage`驱动的通用compiler-counterexample合同，并把共享
  `Gate1ArmEvidence`的compile/fingerprint/comparison类型改为frontend-neutral Protocol；没有给TileLang
  伪造TTIR或增加case特判。

### Pilot B：修复后的正式fresh rerun

`tilelang:add2d / .trellis/tasks/07-31-j-driven-replay-composer-24/runs/tilelang_add2d/run-001`
（Codex Lift + Codex Replay，`gpt-5.6-sol high`，Mask-J开启）：

- 最终状态为`SCRC_CERTIFIED`，accepted attempt为1，formal Gate1 round为1；
- artifact strict decode与dependent TypeCheck通过；Lift在第一次CandidateCheck暴露scope/handle问题后自行
  修正，第二次通过，Source--J semantic probe精确通过；
- normal Replay以TileLang **TIR** hard observer闭合，Source/Replay TIR fingerprint一致；PTX没有替代
  TIR成为hard closure；
- normal和masked Replay由不同writer、workspace、provider thread和fresh session生成；masked臂TIR
  mismatch，因此`J_NECESSARY=proved`；
- Gate2仅在normal Gate1通过后运行，`Source--Replay`、`Source--J`、`Replay--J`三边均通过，唯一输出的
  `max_abs_error`均为0；Source、Replay和RefEval(J)各有一次真实执行；
- normal Replay为1个CLI turn、5个provider segments、7个tool actions、81.632秒；masked Replay为1个
  CLI turn、6个provider segments、10个tool actions、141.783秒；Lift为1个CLI turn、7个provider
  segments、48个tool actions、287.174秒；
- normal与masked都安装了Composer，但agent对该简单case选择direct-source，`used=false`。这证明raw
  escape和Composer不改变接受集合，但**不能**证明Composer加速；后续必须在结构化case做fresh A/B。

### Pilot C：首次CUDA纵切暴露C-slot宏污染

`cuda:tilelang_add2d / .trellis/tasks/07-31-j-driven-replay-composer-24/runs/cuda_tilelang_add2d/run-001`
（Codex Lift，`gpt-5.6-sol high`）：

- Lift产生的artifact通过strict decode和dependent TypeCheck；在任何normal/masked Replay调用之前，
  Gate1 Source compile以`SCRCInfrastructureError` fail closed，因此本轮不评价Replay或Composer；
- 原因是`CudaCompiler`把匿名C-slot valuation `c0=16`无条件转成Translation Unit全局
  `-Dc0=16`。预处理器由此把CUTLASS/CuTe头文件中的合法参数名`c0`也替换为数字`16`；
- C中的`c0`是artifact handle，`role=launch_block_x`才是固定adapter语义。CUDA已经通过`grid/block`
  消费launch metadata，并通过独立白名单`compiler_options`消费真实NVCC选项；匿名handle不属于Source
  标识符或预处理器API；
- 通用修复是保留`CudaCompileCase.schedule`用于C绑定、cache key和PTX observer，但禁止把它物化为
  `-D`。没有修改dataset、Source、J/π、observer或case-specific规则；
- 修复后对同一Dataset Source、同一Harness case直接重编译成功：NVCC 12.8、`sm_120`、PTX 2140
  bytes、空诊断。

### Pilot D：修复后的CUDA fresh certification

`cuda:tilelang_add2d / .trellis/tasks/07-31-j-driven-replay-composer-24/runs/cuda_tilelang_add2d/run-002`
（Codex Lift + Codex Replay，`gpt-5.6-sol high`，Mask-J开启）：

- 没有复用Pilot C artifact或会话；fresh run最终为`SCRC_CERTIFIED`，accepted attempt 1、formal
  Gate1 round 1；Lift 1次逻辑调用、8个physical provider segments；
- CUDA hard observer为`cuda_ptx_semantic_effects@cuda-ptx-semantic-effects-v3`。normal臂的interface、
  target、coordinate uses、memory effects、value operations、control flow和collectives全部hard pass；
  cache/load形式等raw PTX topology差异仍只保留为diagnostic；
- masked臂由独立writer和独立fresh session生成，`coordinate_uses` hard mismatch，故
  `J_NECESSARY=proved`；
- 三边Gate2均通过且`max_abs_error=0`，Source、Replay和RefEval(J)各真实执行一次；
- normal Replay为1个CLI turn、4个provider segments、4个tool actions、70.304秒；masked Replay为
  1个CLI turn、6个provider segments、6个tool actions、130.303秒；
- CUDA C-like Composer在两臂均安装但`used=false`。因此当前证明的是共享Composer机制不妨碍两种
  原生前端closure，尚未证明它带来效率收益；下一阶段必须转向row_sum/gather等结构化case。

### Pilot E：TileLang reduction纵切与Composer负结果

`tilelang:row_sum / .trellis/tasks/07-31-j-driven-replay-composer-24/runs/tilelang_row_sum/run-001`
（Codex Lift + Codex Replay，`gpt-5.6-sol high`，Mask-J开启）：

- 最终为`SCRC_CERTIFIED`，accepted attempt 3；3个formal Gate1 rounds、3次Lift episode、3次
  normal Replay episode，accepted round另有1次独立masked Replay；
- 三轮J和C保持相同，前两轮的PTX/数学语义相容，但TileLang hard TIR structure/topology/operations
  不闭合：Source使用`T.copy + T.region`物化一行，Replay使用显式逐元素parallel loop；
- Gate1 feedback只使Lift调整π中的materialization ownership/refinement。第三轮fresh Replay根据当前
  exact J+π改用`T.copy`，Source/Replay TIR digest一致，normal Gate1 hard pass；
- Gate2三边均通过：Source--Replay最大误差0，Source--J与Replay--J最大误差均
  `2.384185791015625e-07`，在声明的f32 policy内；
- masked臂与normal臂物理隔离；它能从π拓扑猜测存在load/materialize/reduce/store阶段并产出可编译
  reduction，但缺少exact J后在TIR结构、坐标使用、PTX collective和ABI上均不闭合，故
  `J_NECESSARY=proved`。这证明J对exact语义必要，不声称π零统计信息；
- 三轮normal Replay分别使用7、6、6个provider/model segment lower bounds，episode elapsed分别
  241.689、168.446、123.484秒；accepted masked为14个segments、414.276秒。这里的segments不是
  raw HTTP调用，`call_budget.replay_physical_calls=4`只计3次normal+1次masked Replay执行；
- 三轮normal与masked的Composer都`installed=true, used=false`。各episode仍反复整文件写
  `output/replay.source`，第一轮5次submit失败后通过、第二轮3次失败后通过、第三轮2次失败后通过；
  因此这是Composer当前“可选但无效用”的真实负结果，不能作为提速证据；
- 下一步必须以accepted exact artifact做fresh `composer off / preferred` A/B。若preferred分支仍不
  使用或成本不降，则继续收缩/重构Composer，而不是扩大kernel模板。

### Pilot F：同artifact Composer v2 fresh A/B

父artifact固定为Pilot E accepted attempt-03，digest
`2c9fe2c20b37700aaa97fd4141c786d17db3bed06f5626ba1c6e9a97caddb0a9`。两臂均使用
`gpt-5.6-sol high`，只运行authoritative TypeCheck与fresh normal Gate1；不调用Lift、masked-J或Gate2，
不改变parent verdict：

| 指标 | off / recheck-001 | preferred / recheck-002 | preferred相对变化 |
|---|---:|---:|---:|
| Gate1 | replay_closed | replay_closed | 相同 |
| Composer最终路径 | raw_escape | composer | 实际采用 |
| elapsed | 141.212 s | 111.523 s | -21.02% |
| input tokens | 174,443 | 142,642 | -18.23% |
| cached input tokens | 143,872 | 115,968 | -19.40% |
| model segments lower bound | 6 | 5 | -16.67% |
| tool actions | 9 | 7 | -22.22% |
| output tokens | 3,063 | 3,379 | +10.32% |
| reasoning tokens | 1,905 | 2,097 | +10.08% |
| submit_replay | 1 fail + 1 pass | first pass | 少1次失败修复 |

- 两臂最终normal hard Gate1均通过，Source/Replay TIR digest均为
  `094a01244932fb98c87651c3f572c3a012e6f8e827c0a64bcb5016c7685e55b1`；
- preferred臂为Composer v2：`compose=1`、10个stable nodes（2 block + 8 line）、source map 10 nodes、
  plan/source/map revision一致、无failed composer action；
- 该单对样本证明“agent真实使用Composer且本次减少构造/修复成本”，不证明总体平均提速或统计显著；
  output/reasoning tokens反而略升，后续至少增加CUDA reduction或Triton难例复核。

### Pilot G：CUDA reduction反例推动`artifact.v13`

`cuda:tilelang_row_sum / .trellis/tasks/07-31-j-driven-replay-composer-24/runs/cuda_tilelang_row_sum/run-001`
（Codex Lift + Codex Replay，`gpt-5.6-sol high`，Composer preferred）：

- 三个formal rounds均完成真实fresh Replay，Composer每轮都实际生成最终source，但最终为
  `NO_CLOSURE`；该记录是有效失败证据，不重标为Replay或Composer成功；
- 三轮J与C的逻辑事实稳定：J拥有`[k0,16] -> [k0]`的row-sum语义，C拥有匿名
  `launch_block_x=128`外部binding。Source的128个物理线程通过`threadIdx.x & 15`循环覆盖16个逻辑
  lanes，而三轮Replay都直接使用`threadIdx.x`，造成错误row/OOB并让不同线程竞争同一归约结果；
- v12 π只能表达lane-domain realization/routes，不能表达“物理participant循环投影到exact J domain”。
  Lift在重复realize、identity materialization和barrier之间振荡，证明缺的是closed relation，不是继续
  增加instance数量；
- 旧PTX observer只统计`%tid.x` coordinate use，且memory-effect投影有意忽略地址operand，因此Source与
  Replay看起来都使用同一个coordinate；最终hard failure来自偶然的`add.f32`数量差，无法给Lift准确归因；
- 不采用`x+0`消除或进一步放松PTX：浮点`x+0`对signed zero、NaN等并非全局数学恒等。机制修复为
  `artifact.v13`的closed `participant_mappings{domain_instance,axis,extent_slot,projection=cyclic}`，以及
  PTX v4沿memory-address reaching definition观察`and/rem` coordinate fold；两者都不含kernel模板；
- 本地严格检查已经证明：v12 round-trip不变；v13缺字段或加入target recipe均fail closed；删除所引用
  J domain后dependent TypeCheck失败；normal与Mask-J packet中的relation均不含logical period、公式、
  Source/API或C value；旧错误Source/Replay现在在`address_coordinate_folds` hard category稳定分离；
- 当前只声称机制已实现并通过本地回归。必须从Source重新fresh Lift/Replay/Gate1/Gate2后，才能声称
  CUDA row_sum闭合；不得手改旧artifact追认run-001。

### Pilot H：v13 fresh run定位feedback、observer与identity三个通用缺口

`cuda:tilelang_row_sum / .trellis/tasks/07-31-j-driven-replay-composer-24/runs/cuda_tilelang_row_sum/run-002`
（Codex Lift + Codex Replay，`gpt-5.6-sol high`，Composer preferred）：

- fresh run完成3个TypeCheck-valid formal rounds，最终仍为`NO_CLOSURE`；masked-J与Gate2均未运行，
  因而这是一份真实失败证据，不是closure；
- 三轮均为`artifact.v13`，J稳定拥有`f32 [x0,16] -> [x0]`的sum与identity 0，C稳定拥有匿名
  `launch_block_x=128`；Lift从第一轮起就作者化并保持
  `{domain_instance:i3,axis:0,extent_slot:c0,projection:cyclic}`，证明v13 relation已被Lift消费；
- 三轮normal Replay都实际走Composer最终路径。第一轮Replay正确写出`threadIdx.x % 16`并生成4步
  shuffle，证明J、π和Composer数据流本身已经工作；
- PTX v4却把Replay的NVCC等价形式`(tid << 2) & 60`记为`bitmask:60`，把Source的
  `(tid & 15) << 2`记为`cyclic:16`。这不是realization差异，而是byte scaling后的同一低位投影；
  PTX v5现只在纯power-of-two scale、整除mask/positive remainder、精确非窄化整数conversion且不跨
  predicated/control boundary时作等价归一，其他路径保守保留原类别；
- v5重放第一轮exact PTX后，interface、address folds、memory effects、collectives、control、coordinate
  uses和target全部hard pass，唯一hard差异是Source `add.f32=5`、Replay `add.f32=4`。额外一次add来自
  Source显式`identity + incoming carrier`阶段，不能把浮点`x+0`全局NormalizeAway；
- relift frontier正确列出`value_operations`与`address_coordinate_folds`，但旧Codex Lift workspace只
  挂载5个Triton legacy category文件，`feedback-category value_operations`因故意不存在的bulk
  `feedback.json`失败。Lift随后从raw PTX误加`signed_widen`，后两轮丢失cyclic spelling，第三轮还
  引入未授权barrier并使collective category回归；
- 通用修复不是kernel patch：Codex Lift v31/knowledge v90以固定文件集合中的稳定exact category index
  支持任意合法Gate1 label（含`ptx.*`），每轮原子覆盖、无旧类别残留；`typecheck.v31`把已有
  `identity_materialization`的一个窄拓扑解释闭合为“slot-0 incoming realize -> J reduce synchronize”的
  显式pre-collective identity combine。π仍不含reducer、identity value、dtype、Source/API或target
  recipe；TileLang原有shaped representation-stage语义保持不变；
- 本地检查已覆盖稳定persistent feedback刷新、动态类别精确查询/路径拒绝、真实run-002第一轮PTX v5
  等价、identity正负TypeCheck、normal/Mask-J无语义泄漏以及旧TileLang artifact回放。上述修复必须以
  run-003 fresh provider episode验证后才可声称CUDA row_sum闭合。

### Pilot I：CUDA row_sum在J主导Composer流程中正式闭合

`cuda:tilelang_row_sum / .trellis/tasks/07-31-j-driven-replay-composer-24/runs/cuda_tilelang_row_sum/run-003`
（Codex Lift v31 + Codex Replay v36，`gpt-5.6-sol high`，Composer preferred，PTX v5）：

- fresh run最终为`SCRC_CERTIFIED`，accepted attempt 2；2个formal Gate1 rounds、2次normal Replay，
  accepted round另有1次独立masked Replay与1个Gate2 case；两轮TypeCheck均为v31 PASS；
- Lift从第一轮起就同时作者化closed cyclic participant mapping与
  `transfer_direct -> identity_materialization(realize) -> J-reduce synchronize`拓扑。π只含opaque
  handle、slot和refinement kind；reducer=sum、identity=0.0、dtype=f32、logical extent=16仍只在J，C=128
  未进入Replay packet；
- 第一轮fresh Replay已经消费identity，`value_operations` hard pass，但遗漏cyclic projection，唯一hard
  category为`address_coordinate_folds`。第二轮同一persistent Lift owner通过新的stable category index
  成功执行`feedback-category address_coordinate_folds`，得到Source有
  `ld.global.f32:%tid.x:cyclic:16`而Replay为空的exact evidence；没有再出现`feedback.json`缺失；
- 两轮artifact的计算语义相同，第二轮只对anonymous J index variable做alpha-equivalent重命名；闭合来自
  fresh Replay真正消费当前participant relation，而不是Lift把modulo或CUDA recipe写进π。accepted
  Replay自行写出`threadIdx.x % 16`、`0.0f + loaded`、4步xor shuffle与store；
- accepted normal Replay实际使用Composer v2：`selected_path=composer`、1次compose、12个stable nodes、
  source map与rendered source一致、无failed action；normal Gate1的schema/interface/value operations/
  memory effects/address folds/collectives/control/coordinate/target九类全部hard pass；
- masked臂使用不同writer、workspace、thread与generate call；它虽能从π拓扑构造可编译程序，但在
  value operations、memory effects、address folds和collectives上hard mismatch，故
  `J_NECESSARY=proved`；这证明exact J必要，不声称π零统计信息；
- Gate2三边均真实执行并通过：Source--Replay最大误差0，Source--J与Replay--J最大误差均
  `2.384185791015625e-07`，小于`atol=1e-6, rtol=1e-5`；Source、Replay、RefEval(J)各执行一次；
- accepted normal Replay为6个provider segments、8个tool actions、146.613秒；masked为7个segments、
  13个tool actions、196.276秒。整轮相对run-002从3轮无闭合收敛为2轮认证，但这里只把改进归因到
  “新机制与fresh agent共同作用”，不把单样本写成统计显著提速。

### Pilot J：CUDA gather前端闭合，暴露J reference consumer边界

`cuda:tilelang_gather2d / .trellis/tasks/07-31-j-driven-replay-composer-24/runs/cuda_tilelang_gather2d/run-001`
（Codex Lift v31 + Codex Replay v36，`gpt-5.6-sol high`，Composer preferred）：

- accepted attempt 1，`artifact.v13 + typecheck.v31` PASS；normal Replay与masked Replay都实际使用
  Composer，normal 19 nodes、masked 6 nodes；
- normal CUDA PTX v5的全部hard categories通过，Source--Replay Gate2也以最大误差0通过；masked臂在
  value operations与control flow上hard fail，因此`J_NECESSARY=proved`；
- 主状态仍是`GATE2_FAILED / reference_unsupported`，不能声称SCRC closure。唯一阻塞是
  `RefEval(J)`返回`NotImplementedError: functional boundary supports only logical-domain predicates`；
  Source和Replay native执行都成功；
- J拥有两级typed gather、index bounds、value-dependent合法性predicate、zero-fill select与精确effects；
  π只有14个realization instances、routes/order/sharing，不复制gather/predicate/index公式。故这不是
  Replay信息缺失，也不应通过给CUDA Composer加入gather模板解决；
- 当前RefEval其实已有通用ContractIR/structured gather执行器，阻塞来自fallback capability guard过严：
  它要求每个boundary access都带predicate/fill，错误拒绝了domain-total的row/column index直接load。

### Pilot K：TileLang gather闭合失败暴露长TIR反馈transport缺口

`tilelang:gather2d / .trellis/tasks/07-31-j-driven-replay-composer-24/runs/tilelang_gather2d/run-001`
（Codex Lift v31 + Codex Replay v36，`gpt-5.6-sol high`，Composer preferred）：

- 三个formal rounds均TypeCheck PASS且Composer实际使用，最终`NO_CLOSURE`；J/C三轮稳定，只有π的
  transfer/order topology变化；
- 前两轮Replay把data load显式保存为typed scalar，Source直接把load写入store，因而
  `tir_structure/tir_topology/tir_operations` hard fail；第三轮fresh Replay已inline该load，使operations与
  topology通过，但把row/column index load顺序反转，最终只剩`tir_structure` hard fail；
- frontier正确列出`tir_structure/tir_topology/tir_operations/ptx.raw_instruction_topology`，但旧Codex
  continuation的本地`feedback-category`全部得到“no observer-diff categories”；Lift只能看空examples和
  aggregate component变化，随后无依据地删order/transfer关系；
- 该问题不是TileLang应改用PTX。第一版修复保持结构TIR为hard observer：Codex Lift v32/knowledge v91
  挂载已经删除previous artifact与compiler blob的标准closed `feedback.json`作为canonical query source，
  stable category index只作冗余；任意超长TIR/PTX string由通用`feedback-slice`按exact JSON path与
  UTF-8 byte offset分页，array/object按item offset递归有界分页并携带whole/chunk receipt；frontier与
  分页工具都不解释语义或建议repair；
- 同名Dataset的两个前端并不要求J字节一致：两者共享两级gather核心；CUDA Source另有越界mask/
  zero-fill与runtime M，因此J多出predicate/select/value-dimension facts，π也按各自Source realization
  不同。这个差异属于真实跨前端语义/实现边界，不是失败或泄漏。

### Pilot L前的通用机制修复

- `j_scalar_cpu_mlir_v25`只放宽既有indexed-A capability guard：domain-total direct access允许无
  predicate/fill；非direct load必须由J `index_bounds`锚定；任何predicated load仍必须有J-owned fill。
  执行仍走既有typed structured reference，不生成index/predicate/gather语义；
- exact CUDA gather artifact的本地J-only执行已与`inp[row,col]`逐元素完全一致；删除typed index-bounds的
  artifact在canonical schema即fail closed，reduction等其他graph不进入该fallback；
- exact TileLang run-001 feedback已通过两轮persistent workspace刷新、四个动态category读取和长字符串
  exact slice回归；合成多字节/恶意path/超大nested container还验证了byte-exact重组、shell quoting与
  递归有界输出。旧的fixed dataflow repair hint已经删除。上述修改必须fresh rerun两个gather case后
  才能更新历史verdict。

### Pilot M：CUDA gather经v25获得完整SCRC认证

`cuda:tilelang_gather2d / runs/cuda_tilelang_gather2d/run-002`（fresh Codex Lift v32、Replay v36、
Composer preferred、high）：

- accepted attempt 2，TypeCheck v31 PASS；normal与Mask-J Replay是不同thread的fresh session，均选择
  Composer，分别21/11 nodes；normal CUDA PTX Gate1全部hard pass；
- Mask-J在`value_operations/control_flow/memory_effects`上hard fail，`J_NECESSARY=proved`；
- Gate2的Source--Replay、Source--J、Replay--J三条边均exact pass，最大误差全部0，主状态
  `SCRC_CERTIFIED`；`j_scalar_cpu_mlir_v25`因此把旧run-001的唯一`reference_unsupported`闭合；
- π receipt仍只有13 instances、13 routes、3 order、1 program axis与2 regions；无iteration/carrier/
  participant mapping/sharing/refinement，也无gather、predicate、index公式或target recipe。该变化来自
  通用J reference consumer能力，不是CUDA Composer模板。

### Pilot N：TileLang run-002证明v32真实root适配仍不完整

`tilelang:gather2d / runs/tilelang_gather2d/run-002`（fresh Codex Lift v32、Replay v36、Composer
preferred、high）仍为`NO_CLOSURE`：

- 三轮TypeCheck PASS、Replay compile PASS且均选择Composer（9/9/8 nodes）；第三轮已令
  `tir_operations/tir_topology`通过，只剩`tir_structure` hard fail；J/C保持不变，只有π order/instance
  topology变化，public Source--J probe每轮exact pass；
- frontier已有四个exact category label，但agent执行相对脚本而未传roots；copied tool的默认root仍写死
  `/input`，而Codex真实mount是`/workspace/input`，于是查询不存在root并误报“no categories”。这解释了
  synthetic显式-root回归通过而real continuation失败；
- Codex Lift v33/knowledge v92与MiniSWE knowledge v70改为从copied tool自身位置解析paired roots；Codex
  frontier/continuation query显式携带`--input-root input --output-root output`。只改transport identity，
  不改TIR observer、J/π、Gate或Replay；必须fresh run-003验证后才能更新TileLang verdict。

### Pilot O：TileLang gather首轮完整认证

`tilelang:gather2d / runs/tilelang_gather2d/run-003`（fresh Codex Lift v33/knowledge v92、Replay
v36、Composer preferred、high）：

- accepted attempt 1，TypeCheck v31 PASS；normal与Mask-J为独立fresh session，均选择Composer，分别
  8/13 nodes；normal Replay只需1个physical provider call；
- hard observer保持TileLang TIR：`tir_structure/tir_topology/tir_operations`与interface/schema/target全部
  通过；`ptx.raw_instruction_topology`仍有diagnostic差异但不参与hard verdict；
- Mask-J在三项TIR hard categories上全部失败，`J_NECESSARY=proved`；Gate2的Source--Replay、
  Source--J、Replay--J三条边均exact pass，最大误差全部0，主状态`SCRC_CERTIFIED`；
- π ownership仍为J-only semantics与J/C-dependent realization；13 instances、14 routes、1 sharing、
  1 program axis、2 regions，未出现operator/index/predicate/target recipe副本；
- 本run没有formal relift，因此它证明v33/v92首轮可闭合，不单独证明category-query root fix的真实
  continuation；该fix由copied `/workspace/input/tools`布局回归与独立审计证明，仍需未来自然触发的
  relift evidence补强，不把未执行路径写成provider认证。

### Pilot P：rowmask双前端横向认证与真实root continuation

`cuda:tilelang_rowmask_where2d / runs/cuda_tilelang_rowmask_where2d/run-001`与
`tilelang:rowmask_where2d / runs/tilelang_rowmask_where2d/run-001`均使用fresh Codex Lift/Replay、
Composer preferred、Mask-J与`high` effort：

- 两项均在accepted attempt 2达到`SCRC_CERTIFIED`；两轮artifact均通过strict decode与dependent
  TypeCheck，accepted normal/masked臂使用不同writer/workspace/session并实际选择Composer；
- CUDA normal的PTX semantic hard categories全部通过，masked臂在observable value/memory上hard
  mismatch；TileLang normal以结构TIR闭合，masked臂同样hard mismatch，因此两项均为
  `J_NECESSARY=proved`；
- 两项Gate2的Source--Replay、Source--J、Replay--J三边均真实执行并通过，最大误差全部为0；
- TileLang attempt 2发生真实formal relift，Lift trajectory中的`feedback-category tir_structure`、
  `tir_topology`、`tir_operations`与PTX diagnostic查询均携带显式`--input-root input --output-root
  output`并返回exact payload。这是Codex Lift v33 root修复第一次由真实provider continuation验证；
- π仍只含opaque instance/route/order/sharing/program-axis/region关系；没有rowmask predicate、select、
  target API或kernel recipe副本。

### Pilot Q：transpose横向结果揭示CUDA observer表示商边界

`tilelang:transpose2d / runs/tilelang_transpose2d/run-001`与
`cuda:tilelang_transpose2d / runs/cuda_tilelang_transpose2d/run-001`：

- TileLang在accepted attempt 2达到`SCRC_CERTIFIED`。attempt 1 Replay发明了Source不存在的逐scalar
  participant loop，导致TIR structure/topology/operations hard fail；Lift通过v33 exact rooted category
  query获得差异，fresh attempt 2删除该额外stage后结构TIR闭合；masked-J hard fail，Gate2三边最大误差
  全部为0；
- CUDA完成3个TypeCheck-valid formal rounds后仍为`NO_CLOSURE`，未运行masked-J或Gate2，不能重标为
  closure。三轮J稳定表达`[x0,16] -> [16,x0]` transpose，Source--J public probe每轮exact pass；
- 最终v5 hard差异集中在`value_operations/memory_effects`：Source wrapper使用恒真predicated
  `ld.global.v4.u32`及zero-init，Replay使用unpredicated `ld.global.v4.f32`，两侧均执行4次32-bit
  vector load与16次32-bit store。该记录只支持“需要审查observer表示商”，不在没有fresh Gate1/Gate2
  的情况下宣称二者已经认证等价。

### Pilot R：CUDA PTX v6机制检查点，尚无fresh closure claim

- `cuda-ptx-semantic-effects-v6`只把非atomic纯`ld/st/suld/sust`的terminal
  `b/s/u/f{8,16,32,64,128}`归一为相同bit width，并只消去由straight-line closed propagation证明
  恒真的memory predicate；proven-false effect被删除，unknown predicate保持hard polarity；
- vector/tuple load的所有destination进入def-use；unknown predicated write保留旧/新definition alternatives。
  算术opcode类型、atomic/reduction、address space、vector width、control、coordinate、collective、ABI与
  target继续为hard evidence，raw instruction/opcode/memory始终保留diagnostic；
- 两路独立只读审计未发现会把unknown predicate、算术类型或atomic差异误认证的阻塞项。对Pilot Q
  exact历史PTX的本地重新观察使九项semantic hard view一致而raw diagnostic仍不同；这只是当前实现
  的机制证据，绝不重判v5历史artifact；
- 在新的权威Replay设计重构完成前不启动CUDA transpose fresh v6 provider rerun。Task 25若恢复该项，
  必须从fresh episode重新生成exact Replay并完整运行Gate1、Mask-J与三边Gate2。

### Task 24冻结与Task 25迁移裁决

Task 24保留上述真实成功、失败和成本证据，但不再继续给现有line/block Composer追加局部功能。真实
轨迹说明主要剩余成本是agent反复遍历扁平J/π、按physical occurrence维护route/local/region并重写完整
源码，而不是缺少更多宿主semantic template。后继Task 25以“fresh agent自建episode-local composer、
固定宿主只做syntax/route linking”为唯一新执行线；Task 24所有run目录保持历史verdict，不迁移、不
重判，也不作为新设计已经提速的证据。

## 3. 交付物

1. 权威设计增补、spec与任务索引更新；
2. 无J/C/π/Source依赖的共享syntax Composer core；
3. stable node id、insert/replace/delete/move、render/source-map、checkpoint/rollback、raw escape；
4. Triton、TileLang、CUDA三个薄profile与公共文档；
5. Composer接入fresh Codex Replay workspace/tool bridge；
6. normal/masked独立writer/workspace/thread/composer；
7. π静态ownership/leakage audit复用同一TypeCheck事实，不维护第二套宽松validator；
8. add2d、row_sum、gather2d/masked、rowmask与transpose等真实多前端纵切evidence；
9. 至少一个sharing/physical-loop与一个effect/control纵切（前端能力允许时）；
10. 对Task 15 Triton难例或已闭合代表项的fresh A/B，证明Composer确实降低构造成本；
11. 干净提交与可追溯真实运行记录。

## 4. 实施阶段

### P0：设计与协议对齐

- 建立设计文档和Task 24；
- 更新文档/spec：禁止deterministic semantic composer，允许episode-local syntax-only composer；
- 明确v12为兼容基线；只有真实CUDA reduction反例支持的closed participant projection进入v13；
- 列出normal/masked writer生命周期与三前端compile/observe边界；
- provider调用为0。

### P1：共享Composer core

- 建立最小structured-text tree：document/line/block/raw_block；
- stable id与parent/children invariant；
- local insert/replace/delete/move；
- deterministic render与node-to-line source map；
- in-memory/workspace revision、checkpoint与rollback；
- exact bytes原子写入现有`output/replay.source`；
- Composer package本身不import artifact schema、J model、π model、Source/Gate或frontend compiler。

### P2：Replay runtime接入

- 每次`generate`创建episode-local Composer service/state；
- Codex可直接写完整source，也可用Composer；两条路径都走同一exact submit；
- compile错误映射回rendered line/node id；
- 复用现有ReplayCheck/compile/own-IR/commit/rollback；
- episode结束销毁Composer state，不跨formal round复用；
- telemetry只作proof-irrelevant成本记录。

### P3：三前端profile

- Triton：Python block/indent、安全import/decorator/function surface，TTIR hard；
- TileLang：Python/TIR surface，TIR hard、PTX diagnostic；
- CUDA：C-like braces/helper/kernel surface，PTX observer与native Gate2；
- profile不包含J op→API映射、π→skeleton展开、schedule/reduction/attention模板。

### P4：Mask-J隔离

- normal与masked各自实例化fresh writer、workspace、provider conversation与Composer；
- 两臂不共享draft、own-IR、compiler error、checkpoint或tool trajectory；
- `J_NECESSARY`仍独立于主认证；
- report验证物理隔离与normal/masked closure gap。

### P5：真实pilot

顺序：

1. 三前端add2d；
2. 三前端row_sum；
3. 三前端gather2d或masked access；
4. sharing/materialization或physical loop；
5. effect/control；
6. TileLang/CUDA选定组横向扩展；
7. Triton Task15难例/已闭合代表项A/B。

每次系统修改后必须fresh rerun。不能在同一实验中一边看单个failure一边临时增加template或放松
observer。

### P6：收尾

- 汇总first-compilable、compile/fix cycles、provider turns/tokens、closure@round、raw escape与
  normal/masked gap；
- 报告shared core与三profile LoC，区分接入代码、dataset修复与kernel-specific代码；
- 更新Task 11/15/23事实，不重写历史verdict；
- 提交并保持worktree clean。

## 5. 真实验收标准

### 5.1 结构与安全

- Composer没有artifact/J/π语义依赖；
- 没有kernel/family模板、RuleCard或target algorithm helper；
- profile只含syntax/safety/compile wiring；
- raw escape使未知语法不要求扩建宿主grammar；
- exact final source仍经过现有ingress/compile/Gate；
- normal/masked状态物理隔离；
- production backend入口和结果不含π/Composer。

### 5.2 真实运行

对每个声称closed的case，必须有：

```text
real provider episode
exact artifact + TypeCheck PASS
exact Replay source
frontend compile/observer evidence
Gate1 PASS
Source--Replay PASS
Source--J PASS
Replay--J PASS
```

compile-only、dataset membership、provider写出源码、ReplayCheck PASS或PTX存在均不是SCRC closure。

### 5.3 效率

至少对同一artifact做direct-source与Composer两种fresh episode比较。Composer只有在以下至少一项有
实质改善且没有显著损害另一项时才保留：

- time-to-first-compilable；
- compile/fix cycles；
- provider turns/input tokens；
- closure@1；
- 跨前端共享LoC。

若Composer成为额外操作层而不改善结果，继续收缩到最小structured-text core或删除。

## 6. 禁止退化

- 不实现`J/π -> target source` deterministic composer；
- 不添加`emit_attention/emit_softmax/emit_reduction`等答案模板；
- 不让Lift知道Composer调用规则或target API映射；
- 不把Composer plan/state/source map变成第四artifact；
- 不把Source、TTIR/TIR/PTX、Gate diff、C value或normal draft给masked/normal Replay；
- 不按kernel name、dataset path或failure选择profile内容；
- 不把TileLang PTX设为hard closure替代TIR；
- 不让CUDA relaxed address/cache规则放松observable value/effect/control/ABI；
- 不恢复`python/tests`或以mock代替真实pilot；
- 不用hash、digest、commitment或重复冻结门作为工作进度；
- 不虚构、补写或重标实验结果。

## 7. 完成条件

1. 设计/spec/代码对Composer边界没有矛盾；
2. 共享Composer core及三薄profile落地，raw escape可用；
3. normal/masked独立fresh episode已由真实evidence验证；
4. add2d、row_sum、gather2d/masked在可运行的Triton/TileLang/CUDA组达到TypeCheck+Gate1+三边Gate2；
5. 至少一个更复杂realization纵切闭合，或以完整evidence明确归因具体系统缺口；
6. 对Triton Replay有同artifact A/B效率证据；
7. 没有semantic template/lowerer、第四artifact或backend π消费；
8. Task 11/15/23状态按事实更新，工作区clean。

最终原则：

> **给agent更好的手，而不是在宿主里替agent写答案；用J决定语义、π_min约束realization、Composer稳定编辑，最终仍让Gate1/Gate2裁决。**
