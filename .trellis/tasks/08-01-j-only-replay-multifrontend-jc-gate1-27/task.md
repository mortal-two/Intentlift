# Task 27：J-only Replay 与 CUDA/TileLang J/C Structural Gate1

> 状态：`P3_ADD2D_REAL_EXECUTION`（2026-08-01）
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

- [ ] `tilelang:add2d`：主代理作者化→TypeCheck→fresh J-only Replay→Gate1→Gate2（run-001 已完成到 Gate1，因 Replay 重绑/猜测隐藏 `c0` 被 identity TIR 拒绝；通用 ABI 契约修正后 fresh 重跑）；
- [ ] `cuda:tilelang_add2d`：同一完整流程；
- [ ] 保存Source、Harness、candidate、作者化记录、Replay trajectory、compiler observation、Gate1、Gate2；
- [ ] canonical比较两份accepted J/C：共同数学结构与真实frontend差异分别报告；
- [ ] 确认两份Replay receipt均无π输入。

### 5.1 run-001 反例与机制修正

- 手工 artifact TypeCheck PASS；`lift_agent_calls=0`；Replay workspace root 恰为 `j,c,u_f`，π key 为0；
- fresh `gpt-5.6-sol/xhigh` 一次 provider call，约 574.84 s，最终 source 可编译；
- agent 把 host-owned `c0` 重绑为128，生成 carrier 与两个额外 guard；TileLang TIR hard Gate1 拒绝；
- 现有 worker 本来会在 module exec 前注入 `case.schedule[c0]`，无需新增替换器；
- 修正 `U_f`：公开 host-global binding convention；修正 ingress：禁止定义/shadow compiler slot；
- 不改变 artifact、observer、normalizer、C valuation visibility 或 Gate acceptance。

## 6. P4：gather2d与Gate1价值

- [ ] add2d双闭合后运行TileLang/CUDA gather2d；
- [ ] 报告间接寻址下两个observer的hard/diagnostic差异与合法realization false reject；
- [ ] 对add2d做最小wrong-op、wrong-index、wrong-effect mutation sensitivity；
- [ ] mutation只改受控Replay副本并重新compile/observe，不进入正式certificate；
- [ ] Gate1拒绝后不运行Gate2，统计提前拒绝成本。

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
