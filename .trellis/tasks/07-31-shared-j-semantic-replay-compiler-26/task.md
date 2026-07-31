# Task 26：共享J语义物化与π约束Replay编译Agent

> 状态：`P1_SEMANTIC_MATERIALIZER_CORE_REAL_ARTIFACT_PASS`（2026-07-31）。
>
> 权威设计：
> `doc/IntentIR_共享J语义物化与pi约束Replay编译Agent最终设计_2026-07-31.md`。
>
> Task 24/25及两版Composer设计已归档。Task 26不接管production backend；backend继续只消费
> accepted `J+C`，Replay独有π约束不进入production lowering。

## 0. 任务目标

把Replay从“Agent重新解释复杂J并借固定Composer组织源码”改成：

```text
exact J
  -> deterministic parameterized SemanticMaterializer
  -> read-only S_J

fresh Replay Compiler Agent(J,S_J,π,MaskC,U_f)
  -> 编写、编译、查看并重构完整target source
  -> exact submit
  -> Gate1 + Mask-J + Gate2
```

本任务不建立holes、bindings、route assembler或target skeleton。Agent仍拥有完整编译职责，但不再浪费
能力重建J已经确定的typed semantic program。

## 1. 固定不变量

1. accepted canonical J是唯一语义权威；`S_J=M(J)`是只读内部派生视图，不是第四artifact；
2. SemanticMaterializer只读J，不读π、C valuation、Source、Gate、frontend或历史Replay；
3. `S_J`参数化保留J anchors，不复制production concrete M0/C/search state；
4. Replay agent同时理解`S_J+π`，输出一份完整source，不做per-node填空；
5. π只约束Source realization，不能携带算式、operator、predicate、API或recipe；
6. 宿主只提供query、普通编辑、compile/own-IR/checkpoint与exact submit，不拼源码；
7. normal/masked分别fresh物化`S_J`，绝不把normal skeleton给masked臂；
8. TypeCheck、Gate1和三边Gate2公式不变；
9. production与Replay只共享J语义materialization，不共享π、target recipe或C search；
10. 不用mock/test文件替代真实kernel closure。

## 2. 当前事实基线

- Task 24真实矩阵保留9项`SCRC_CERTIFIED`与1项CUDA transpose `NO_CLOSURE`；
- syntax Composer证明了普通编辑工具可用，也在单个row_sum A/B观察到成本下降，但复杂case仍要求agent
  从扁平J/π重新建立semantic program；
- Task 25只建立文档和route/binding计划，尚未开始P1代码迁移，现已归档；
- 当前Replay接口仍让writer返回完整任意source，尚无`S_J`生成或只读mount；
- 当前production已有canonical J elaboration、target-neutral J function materialization和Intent MLIR emitter，
  但production `materialize_j`还混有concrete extents/constexpr等不应进入Replay Core的职责；
- 现有`replay_composer.py`是syntax-only，不读J/π；新主线稳定后应删除其host plan/scaffold主路径，普通
  coding工具继续保留。

## 3. 目标代码目录

```text
python/intentir/replay_compiler/
  __init__.py
  semantic_program.py
  semantic_materializer.py
  semantic_query.py
  packet.py
```

职责：

- `semantic_program.py`：只读、参数化、带exact J anchors的`S_J`；
- `semantic_materializer.py`：复用/迁入canonical J与Intent MLIR逻辑，仅`J -> S_J`；
- `semantic_query.py`：declaration-order、bounded、exact回指查询；
- `packet.py`：normal/masked各自的J/S_J/π/MaskC/U_f只读mount描述。

原有Gate1 orchestration、Codex provider bridge、frontend compiler/observer和Gate2 runner保持各自所有权。

## 4. 实施阶段

### P0：权威、归档与干净入口（本提交）

- 归档Task 24/25完整目录与历史证据；
- 归档Task 24/25两版Composer设计；
- 两份外部AI反馈进入source-notes，不作为规范；
- 建立唯一active正式设计、Task 26与索引；
- 不修改runtime代码、不调用provider，提交后工作区clean。

### P1：抽取参数化SemanticMaterializer Core

- 对比本分支与`/home/kingdom/phdworks/TianchenIR-backend-certified-j-search`的canonical J/Intent MLIR
  materialization实现；
- 选择性迁入已经验证的J elaboration、type/shape、boundary、control/effect和result-relation逻辑；
- 明确排除concrete valuation、constexpr specialization、C attachment、production loop choice与search；
- 建立`S_J`只读数据模型和确定性文本表示，每个entity回指exact J handle；
- 未支持J form明确`semantic_materializer_unsupported`，不fallback到Agent猜语义。

P1不改Replay agent、不删旧Composer；先在真实Task 24 accepted artifact的formal workflow中生成`S_J`
evidence，确认J-only、参数化和当前RefEval/Gate2语义一致。

P1完成证据（2026-07-31）：

- 新增`intentir.replay_compiler`的不可变`SemanticProgram`、严格J-only materializer、bounded query与
  read-only mount receipt；Core API只接收exact source-isolated J mapping；
- normal/masked入口具有方向性拒绝：normal J不能进入masked物化，MaskJ view不能进入normal物化；
- 使用Task 24真实`SCRC_CERTIFIED`的`cuda:tilelang_row_sum` run-003/attempt-02 exact artifact运行当前
  `TypeCheck(J,C,pi)`，得到`intentir.scrc.typecheck.v31=pass`；
- fresh normal `S_J`保留symbolic `x0`、typed signature、reduce dataflow、boundary access/effect/domain/
  reduction/result relation，并明确排除π、C valuation、Source、Gate、frontend、target API与历史Replay；
- 证据位于`runs/p1_tilelang_row_sum/run-001/`。本阶段未修改Replay writer、Gate或旧Composer路径。

### P2：normal/masked materialization与packet接入

- normal从exact J fresh生成`S_J`；
- masked从MaskJ(J)独立物化`S_J_mask`，不能复用normal materialization/cache；
- packet只给文件路径、大小和短边界，Agent按需查询；
- exact J仍可只读回查，`S_J`不是替代authority；
- π query只做无损region/instance/route/control邻接，不生成target structure。

### P3：Replay Compiler Agent主路径

- Codex按普通coding-agent方式读取J/S_J/π并维护完整source；
- 保留exact patch、checkpoint/rollback、compile、own-IR与atomic submit；
- 删除强制Composer plan、per-node binding、route renderer和semantic scaffold；
- Agent可在本episode自建helper/composer，workspace销毁即消失；
- normal/masked writer、conversation、workspace、compiler history完全隔离。

### P4：同artifact真实A/B

固定model、effort、wall-time、artifact、C masking、U_f、compiler/Gate profile，比较：

```text
baseline: exact J + π -> complete-source Replay
new:      exact J + read-only S_J + π -> complete-source Replay
```

首组：

1. TileLang row_sum；
2. CUDA row_sum；
3. TileLang/CUDA gather或rowmask；
4. TileLang transpose；
5. 一个Task 15高occurrence/loop/carrier/terminal CFG难例。

每臂都fresh，不复用source/conversation；两臂最终都必须按各自exact Replay运行真实Gate1。只有新路径
normal闭合后才运行其masked与Gate2，不改变baseline历史verdict。

### P5：横向扩展与旧Composer退役

- 若P4在多个case降低semantic reconstruction失败或主要成本，扩展CUDA transpose和原22项代表组；
- 共同失败只提升J materializer覆盖、query或Agent编译环境，不增加kernel模板；
- 新主路径完成真实三前端closure后删除host `replay_composer.py`安装/plan/scaffold入口；
- 保留普通文件编辑/compile/checkpoint工具，不保留两套Replay construction栈。

## 5. 真实验收

每个closed claim必须有：

```text
exact artifact + current TypeCheck
fresh normal S_J + read-only receipt
fresh complete-source Replay Compiler trajectory
frontend compiler observation + Gate1
fresh MaskJ materialization/episode
Source--Replay + Source--J + Replay--J Gate2
```

`S_J receipt`只作proof-irrelevant运行审计，不成为artifact、第四输入、TypeCheck对象或certificate字段。

成本报告至少包含：

- J/S_J/π bytes与agent实际读取bytes；
- time-to-first-compilable、compile/fix cycles；
- Replay elapsed、provider segments、input/output/reasoning tokens；
- closure@1/2/3；
- semantic reconstruction与π realization失败分类；
- complete-source rewrite次数；
- normal/masked gap；
- SemanticMaterializer/shared core LoC、frontend基础设施LoC、persistent kernel-specific LoC。

`persistent kernel-specific semantic LoC`必须保持0。若`S_J`只增加输入和token、没有降低错误或成本，
就收缩表示或终止该路径，不能因为已经实现而保留。

## 6. 执行纪律

- 默认Codex `high`；只有materializer、packet、query、compiler与Gate确认正确，仍重复暴露模型能力不足时
  才升`xhigh`；
- 每次系统修改都fresh运行，不patch旧Replay、不重判历史artifact；
- Replay看不到Source或Gate diff，diff只回同一source-aware Lift owner；
- π schema只在真实case证明缺少不可恢复realization事实时升级；
- 不同步修改Gate observer、RefEval、π schema和SemanticMaterializer，保持因果可归因；
- 不恢复`python/tests`，不以mock/smoke/protocol test替代真实closure；
- 不以hash、digest、commitment或文件冻结冒充进度。

## 7. P1第一步

只读比较当前分支与backend-certified-j-search的J materialization，实现一份“可直接迁入、必须参数化、
必须排除”的符号清单。随后只建立`S_J` Core，不同时改Replay生成路径。第一个代码提交必须由真实
accepted J的formal workflow产出`S_J` evidence后结束。

最终原则：

> **固定Core只把J物化为只读semantic program；fresh Replay agent在π约束下承担完整编译职责并提交整份源码；宿主不填洞、不拼程序，最终仍由Gate1/Gate2认证。**
