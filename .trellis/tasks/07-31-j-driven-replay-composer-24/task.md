# Task 24：J 主导的 π_min 与三前端 Episode-Local Composer Replay

> 状态：`P5_TILELANG_ADD2D_CERTIFIED_CUDA_INFRA_FIXED_RERUN_PENDING`（2026-07-31）。
>
> 定位：Task 15/Task 11 的 Replay 构造机制升级，以及 Task 23 多前端实证的真实 closure 执行线。
> 本任务不接管 production backend；backend 仍在 SCRC 后只消费 accepted `J+C`。
>
> 执行模型：真实 Replay episode 默认使用 Codex。简单纵切从 `high` 开始；只有 exact artifact 下重复
> 暴露模型构造能力不足时才升 `xhigh`，不能用 effort 升级掩盖 schema、consumer 或工具错误。

## 0. 要证明什么

本任务只证明四件事：

1. `artifact.v12` 的 π 在现有 closed schema/TypeCheck 下可以作为首版 `π_min`：它依赖 J/C、没有
   op/dataflow/type/shape/Source/API/target recipe副本；
2. source-blind Codex可以在fresh episode内借助纯语法Composer更快地构造、编译和修正Replay；
3. 同一共享Composer core可以被Triton、TileLang和CUDA薄profile复用，而无需每前端J lowerer或
   kernel模板；
4. 最终仍由exact TypeCheck、各前端Gate1和三边Gate2认证，Composer不改变数学或信任边界。

权威设计增补：

- `doc/IntentIR_J主导_pi_min与Episode_Local_Composer_Replay最终设计_2026-07-31.md`

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

- artifact/decode/typecheck：`v12 / v17 / v29`；
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
- Codex Replay backend已升为`openai.codex-cli.intentir-replay.v33`，每个fresh workspace安装自己的
  Composer；agent仍可直接作者化完整source，两条路径都进入原有exact ingress/compile/submit；
- Composer失败或未使用不改变Replay接受集合；host checkpoint/diagnostic map只作非权威辅助；
- Triton/TileLang/CUDA共享同一core，profile只选择Python-like或C-like排版；
- normal与Mask-J由不同writer对象发起，正式workload runtime会为每个formal round另外创建masked
  writer，证据按normal candidate与masked物理角色分开保存；
- 每份Codex trajectory附带`pi-min.ownership-receipt.v1`，它只重述已经通过的strict artifact decode、
  dependent TypeCheck与closed Replay packet事实，不是第二个validator或新artifact；
- 已完成不调用provider的三前端render、全部八种局部edit、source-map、checkpoint/rollback、copied CLI、
  stdlib-only import与writer identity smoke；独立只读审计未发现J/π读取、target recipe、kernel template或
  Gate bypass；
- 真实TileLang `add2d` provider纵切已达到`SCRC_CERTIFIED`；CUDA纵切和Composer/direct-source效率
  比较尚未完成，因此当前只对该单个TileLang case声称closure，不外推三前端或Composer提速。

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
  bytes、空诊断。正式provider fresh rerun待执行。

## 3. 交付物

1. 权威设计增补、spec与任务索引更新；
2. 无J/C/π/Source依赖的共享syntax Composer core；
3. stable node id、insert/replace/delete/move、render/source-map、checkpoint/rollback、raw escape；
4. Triton、TileLang、CUDA三个薄profile与公共文档；
5. Composer接入fresh Codex Replay workspace/tool bridge；
6. normal/masked独立writer/workspace/thread/composer；
7. π静态ownership/leakage audit复用同一TypeCheck事实，不维护第二套宽松validator；
8. add2d、row_sum、gather2d/masked等真实三前端纵切evidence；
9. 至少一个sharing/physical-loop与一个effect/control纵切（前端能力允许时）；
10. 对Task 15 Triton难例或已闭合代表项的fresh A/B，证明Composer确实降低构造成本；
11. 干净提交与可追溯真实运行记录。

## 4. 实施阶段

### P0：设计与协议对齐

- 建立设计文档和Task 24；
- 更新文档/spec：禁止deterministic semantic composer，允许episode-local syntax-only composer；
- 明确v12为`π_min-v1`兼容基线，不在无证据时升schema；
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
