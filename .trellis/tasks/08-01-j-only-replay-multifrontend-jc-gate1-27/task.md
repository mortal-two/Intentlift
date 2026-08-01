# Task 27：J-only Replay 与 CUDA/TileLang J/C Structural Gate1

> 状态：`P4_GATHER_PAIR_CLOSED`（2026-08-01）
> 权威实施文档：
> `doc/method/IntentIR_J-only_Replay与多前端JC结构Gate1最终实施_2026-08-01.md`。  
> 前序 Task 26 已归档；其 π-fed Replay 代码与运行只作反例证据。

## 0. 唯一目标

```text
source-aware 主代理直接作者化 Source_f -> (J,C,π)
TypeCheck(J,C,π)
fresh Replay_f(J,MaskC(C);U_f) -> R       # π不可见
Gate1_f,p^π(Source,R,c0)                  # J/C structural
Gate2(Source,R,RefEval(J))                # 三边
```

完成一对真实 TileLang/CUDA add2d 的人工可审计作者化与 current-version Gate1/Gate2；通过后再运行 gather2d。正式运行不调用 Lift agent。

## 1. 不变量

1. 主代理直接读取 Source/编译观察并作者化 artifact；不调用慢速 Lift agent，也不把历史 artifact 当作正式输入；
2. Replay request/workspace/query/cache/continuation不含π、Source、Harness valuation、Gate或历史Replay；
3. `S_J`只由J确定，compiler index最多并置MaskC attachment；
4. π只进入固定单臂normalizer，candidate不能选择allowed diff或擦除字段；
5. TileLang hard=TIR structure，PTX diagnostic；CUDA hard=PTX semantic effects，raw PTX diagnostic；
6. current `jc-normalizer.v2`保持identity/fail-closed，不声称π消融收益；
7. 每个frontend独立作者化前端事实并执行 fresh Replay/Gate，不转移另一frontend的PASS；
8. Gate1闭合后才进入三边Gate2；
9. 不恢复`python/tests`，不用mock/smoke替代真实workflow；
10. persistent kernel-specific semantic/normalization LoC必须为0。

## 2. P0：归档与同步

- [x] 提交旧π-fed实验checkpoint `bf553e75`，工作区clean；
- [x] Task26与旧权威Replay设计归档；
- [x] 建立Task27与唯一活动实施文档；
- [x] 选择性同步`frontend-backend-op-opt`的J-only Replay、compact π与`jc-normalizer.v2`；
- [x] 保留本分支较新的TileLang TIR v5、CUDA PTX semantic-effects v6，禁止observer降级。

## 3. P1：删除π→Replay通道

- [x] Replay payload从`{j,c,pi,u_f}`收缩为`{j,c,u_f}`；
- [x] 删除π critical_facts、obligation ledger、prompt与query；
- [x] normal/masked/fresh writer与cache key审计均证明π不可见；
- [x] 保留J-only semantic program与bounded own-IR review；
- [x] compact π fresh wire只保留当前有consumer的closed relation；旧v13 graph仅历史读取。

## 4. P2：多前端Gate1协议

- [x] raw observer与Canon完全π-blind；
- [x] TileLang/CUDA各自接入profile-scoped unary identity normalizer receipt；
- [x] identity normalizer保持所有unknown/physical atom hard；
- [x] Gate evidence记录profile、normalizer version、definedness与J/C hard coverage；
- [x] TileLang继续TIR hard/PTX diagnostic；CUDA继续semantic PTX hard/raw diagnostic；
- [x] 不实现kernel-specific matcher、pairwise exception或allowed diff。

## 5. P3：主代理直接作者化 add2d pair

- [x] `tilelang:add2d`：主代理作者化→TypeCheck→fresh J-only Replay→Gate1→Gate2（run-002 全闭合）；
- [x] `cuda:tilelang_add2d`：同一完整流程（run-002 全闭合）；
- [x] 保存Source、Harness、candidate、作者化记录、Replay trajectory、compiler observation、Gate1、Gate2；
- [x] canonical比较两份accepted J/C：共同数学结构与真实frontend差异分别报告；
- [x] 确认两份Replay receipt均无π输入。

### 5.1 run-001 反例与机制修正

- 手工 artifact TypeCheck PASS；`lift_agent_calls=0`；Replay workspace root 恰为 `j,c,u_f`，π key 为0；
- fresh `gpt-5.6-sol/xhigh` 一次 provider call，约 574.84 s，最终 source 可编译；
- agent 把 host-owned `c0` 重绑为128，生成 carrier 与两个额外 guard；TileLang TIR hard Gate1 拒绝；
- 现有 worker 本来会在 module exec 前注入 `case.schedule[c0]`，无需新增替换器；
- 修正 `U_f`：公开 host-global binding convention；修正 ingress：禁止定义/shadow compiler slot；
- 不改变 artifact、observer、normalizer、C valuation visibility 或 Gate acceptance。

### 5.2 run-002 TileLang 真实闭合

- fresh `gpt-5.6-sol/xhigh`，1次 provider call，Replay episode 158.57 s；
- 直接生成14行/323 bytes TileLang，正确引用 host-injected `c0`，无猜值、carrier或额外 guard；
- construction tool calls 从 run-001 的18次降为3次，live checkpoint 从15个降为2个；
- TypeCheck PASS；TileLang TIR structure/topology/operation multiset全部 hard PASS；
- Gate2 的 Source–Replay、Source–J、Replay–J 三边全部 PASS，f32 max abs error均为0；
- receipt：`replay-input.v4`、`replay-check.v17`、`pi_visible_to_replay=false`；
- 首版实验 runner 在 Gate2 已落盘后误读 suite `.status` 而抛异常；run-002 summary 对 wall time 明确标成
  filesystem-mtime approximation，runner 已改用 `.passed/.status_counts` 并增量保存 timing，未重跑或改写 Gate。

### 5.3 run-001 CUDA 反例

- fresh `gpt-5.6-sol/xhigh`，1次 provider call，Replay episode 91.18 s，首次 construction/submit均PASS；
- agent 正确恢复两读、一加、一写，但把 J logical domain 错当成 runtime access predicate：加入
  `if (blockIdx.x < p3)` 与 `threadIdx.x += blockDim.x` grid-stride loop；
- PTX 因而增加 `ld.param p3`、`blockDim.x`、比较、逻辑谓词与分支；Gate1 在 value operations、memory
  effects、control flow、coordinate uses四个hard类别拒绝；
- 该失败不归因于 J 缺失，也不通过放松 semantic PTX observer解决；下一轮只澄清通用 ownership：
  `boundary.domains` 是迭代集合，只有 J access predicate/control 才授权 runtime guard。

### 5.4 run-002 CUDA 真实闭合与 pair summary

- fresh `gpt-5.6-sol/xhigh`，1次 provider call，Replay episode 96.28 s；
- 首个385-byte checkpoint即 construction PASS，2次工具调用；
- semantic PTX 的 ABI/value/memory/address/control/coordinate/target全部hard PASS；3项raw PTX差异仅诊断；
- Gate2三边全部PASS，f32 max abs error均为0；
- add2d pair机器指标见`pair_metrics.json`，论文解释与claim boundary见`report.md`。

## 6. P4：gather2d与Gate1价值

- [x] add2d双闭合后运行TileLang/CUDA gather2d（TileLang run-003、CUDA run-002均全闭合）；
- [x] 报告间接寻址下两个observer的hard/diagnostic差异与合法realization false reject；
- [ ] 对add2d做最小wrong-op、wrong-index、wrong-effect mutation sensitivity；
- [ ] mutation只改受控Replay副本并重新compile/observe，不进入正式certificate；
- [ ] Gate1拒绝后不运行Gate2，统计提前拒绝成本。

### 6.1 gather2d 手工作者化 checkpoint

- 主代理直接读取当前注册 Source/Harness：`M=64`、logical length 256、column extent 64、`c0=128`；
- TileLang J：两级 typed gather + 两个 index_bounds，无 runtime guard；
- CUDA J：相同间接寻址核心，额外保留 Source 的 row/col bounds、zero fill与select；
- 两份 current v13 artifact decode + TypeCheck一次PASS；canonical bytes分别2696/4381；
- host侧非正式可闭合性预检：参数化 target source在synthetic与formal compile均PASS；TileLang TIR hard全PASS，
  CUDA semantic PTX hard全PASS；该预检不进入正式certificate；
- `lift_agent_calls=0`，历史artifact只用于核对wire语法，不作为正式输入。

### 6.2 TileLang gather run-001：合法 realization 被 identity profile拒绝

- fresh `gpt-5.6-sol/xhigh`，1次provider call，Replay恢复了正确间接读取且construction PASS；
- Replay选择`T.Kernel(1,threads=c0)+T.Parallel(256)`，Source使用`T.Kernel(256,threads=128)`的program id；
- 数学访问一致，但TIR多出一个`control.for`，structure/topology/operation multiset hard拒绝；
- 该结果不归因于J缺失，也不放松observer；当前identity profile需要一个固定、source-neutral的TileLang
  canonical mapping：首个output-covering J domain直接成为`T.Kernel` program grid，不能改成singleton grid
  下的parallel loop。该规则进入公开`U_f`，不读取π、Source、C value或Gate diff。

### 6.3 TileLang gather run-002：program mapping修复后暴露pure-intermediate物化

- fresh Replay已采用`T.Kernel(256,threads=c0)`，证明canonical output-domain mapping生效；
- 但它把两级逻辑gather逐级lower为`T.alloc_local(64)`、serial copy、再按column读取；
- J的中间值无storage、无boundary effect且只有一个consumer，Source TIR直接融合为`inp[row,col]`；
- identity TIR Gate因多出allocation/loop/load-store结构拒绝；这不是per-kernel问题，而是通用编译职责：
  pure single-use J intermediate应沿boundary access composition融合，只有J storage/effect或observable sharing
  才授权target materialization。下一fresh run只升级这条J-derived编译契约。

### 6.4 TileLang gather run-003：J-only正式闭合

- fresh `gpt-5.6-sol/xhigh`，1次provider call；Replay episode 161.91 s，Gate1总计181.23 s；
- Replay直接生成12行/297 bytes TileLang：`T.Kernel(256,threads=c0)`与融合后的
  `p3[v2] = p0[p1[v2], p2[v2]]`，没有中间allocation、serial loop或额外target recipe；
- Replay输入receipt为`replay-input.v4`/`replay-check.v17`，可见根仅`j,c,u_f`，
  `pi_visible_to_replay=false`，且π、Source、Harness valuation、Gate observation、prior Replay均列为排除依赖；
- TileLang TIR的interface、structure、topology、operation multiset全部hard PASS；PTX value/control/raw差异
  只作为diagnostic，不参与TileLang认证；
- Gate2的Source–Replay、Source–J、Replay–J三边均为exact PASS，f32 max abs error全部为0；
- 全程`lift_agent_calls=0`。两次失败分别沉淀为output-domain canonical mapping与pure-intermediate
  fusion两条通用J消费契约，没有增加kernel名称、Source matcher、历史winner或Gate-diff规则。

### 6.5 CUDA gather run-001：未授权thread coordinate被hard Gate拒绝

- fresh `gpt-5.6-sol/xhigh`，1次provider call，Replay episode 81.39 s；candidate一次construction PASS；
- Replay正确恢复ABI、四个row/col bounds、zero fill、两级间接寻址与output store；semantic PTX的
  interface、value operations、memory effects、control flow、target与address folds均hard PASS；
- 但它将唯一output axis写成`blockIdx.x * blockDim.x + threadIdx.x`，Source/J canonical ownership只以
  `blockIdx.x`枚举该域；真实launch下前者会额外枚举128倍坐标，并非合法schedule差异；
- `coordinate_uses`作为唯一hard失败正确提前拒绝，Gate2未运行；raw PTX topology/opcode/cache差异只诊断；
- 归因不是J丢失数学/effect，也不能放松observer。通用缺口是CUDA U_f尚未明确：没有独立J-owned
  lane/participant axis时，`launch_block_*`仅是外部launch配置，`threadIdx`/`blockDim`不得进入J地址；
  下一fresh run只增加该source-neutral ownership契约。

### 6.6 CUDA gather run-002：J-only正式闭合

- fresh `gpt-5.6-sol/xhigh`，1次provider call；Replay episode 120.78 s，Gate1总计124.18 s；
- 18行/426 bytes CUDA直接使用`blockIdx.x`，保留四个J-owned bounds、zero fill、两级间接寻址与store；
- semantic PTX的interface、value operations、memory effects、address folds、control flow、coordinate uses、
  target与schema全部hard PASS；3个raw PTX字段差异仅diagnostic；
- Gate2的Source–Replay、Source–J、Replay–J三边均为exact PASS，f32 max abs error全部为0；
- receipt为Replay backend v72，visible root仅`j,c,u_f`，`pi_visible_to_replay=false`，全程
  `lift_agent_calls=0`；Gate1在run-001提前拒绝了真实错误坐标，run-002没有通过放松observer获得PASS。
- gather pair机器指标见`gather_pair_metrics.json`，论文解释与claim boundary见`gather_report.md`。

## 7. 最小实验输出

每个fresh正式case记录：

- J/C/π作者化wall time、candidate bytes、人工修订次数、TypeCheck；
- J/C/π bytes与π relation count；
- Replay wall time、segments、tokens、compile/fix cycles；
- `pi_visible_to_replay=false`的机械receipt；
- frontend/profile/normalizer；
- Gate1 closure与primary hard category；
- Gate2 Source–Replay、Source–J、Replay–J；
- shared core LoC、frontend profile LoC、kernel-specific LoC。

论文主表只放add2d与gather2d两对；row_sum/rowmask/transpose与旧Task26数据只作背景或附录，不扩张
本任务时间预算。

## 8. 完成条件

- 两个add2d case均由主代理根据当前Source重新作者化，不调用Lift agent、不使用历史artifact作为正式输入；
- 两个Replay输入中π物理缺席；
- 两个profile各自Gate1闭合；
- 两个case各自三边Gate2 PASS；
- 若gather2d无法在预算内闭合，准确报告J/Replay/observer capability边界，不伪造数据；
- 提交后工作区clean。
