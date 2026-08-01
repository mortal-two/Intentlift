# Task 27：J-only Replay 与 CUDA/TileLang J/C Structural Gate1

> 状态：`P0_DESIGN_AND_CLEAN_CHECKPOINT`（2026-08-01）  
> 权威实施文档：
> `doc/method/IntentIR_J-only_Replay与多前端JC结构Gate1最终实施_2026-08-01.md`。  
> 前序 Task 26 已归档；其 π-fed Replay 代码与运行只作反例证据。

## 0. 唯一目标

```text
Codex Lift(Source_f) -> (J,C,π)
TypeCheck(J,C,π)
fresh Replay_f(J,MaskC(C);U_f) -> R       # π不可见
Gate1_f,p^π(Source,R,c0)                  # J/C structural
Gate2(Source,R,RefEval(J))                # 三边
```

完成一对真实 TileLang/CUDA add2d 的自动 Lift 与 current-version Gate1/Gate2；通过后再运行 gather2d。

## 1. 不变量

1. 主代理不手写或复制 artifact；正式 candidate由source-aware Codex owner生成；
2. Replay request/workspace/query/cache/continuation不含π、Source、Harness valuation、Gate或历史Replay；
3. `S_J`只由J确定，compiler index最多并置MaskC attachment；
4. π只进入固定单臂normalizer，candidate不能选择allowed diff或擦除字段；
5. TileLang hard=TIR structure，PTX diagnostic；CUDA hard=PTX semantic effects，raw PTX diagnostic；
6. current `jc-normalizer.v2`保持identity/fail-closed，不声称π消融收益；
7. 每个frontend独立fresh Lift/Replay/Gate，不转移另一frontend的PASS；
8. Gate1闭合后才进入三边Gate2；
9. 不恢复`python/tests`，不用mock/smoke替代真实workflow；
10. persistent kernel-specific semantic/normalization LoC必须为0。

## 2. P0：归档与同步

- [x] 提交旧π-fed实验checkpoint `bf553e75`，工作区clean；
- [x] Task26与旧权威Replay设计归档；
- [x] 建立Task27与唯一活动实施文档；
- [ ] 选择性同步`frontend-backend-op-opt`的J-only Replay、compact π与`jc-normalizer.v2`；
- [ ] 保留本分支较新的TileLang TIR v5、CUDA PTX semantic-effects v6，禁止observer降级。

## 3. P1：删除π→Replay通道

- [ ] Replay payload从`{j,c,pi,u_f}`收缩为`{j,c,u_f}`；
- [ ] 删除π critical_facts、obligation ledger、prompt与query；
- [ ] normal/masked/fresh writer与cache key审计均证明π不可见；
- [ ] 保留J-only semantic program与bounded own-IR review；
- [ ] compact π fresh wire只保留当前有consumer的closed relation；旧v13 graph仅历史读取。

## 4. P2：多前端Gate1协议

- [ ] raw observer与Canon完全π-blind；
- [ ] TileLang/CUDA各自接入profile-scoped unary normalizer；
- [ ] identity normalizer保持所有unknown/physical atom hard；
- [ ] Gate evidence记录profile、normalizer version、definedness与J/C hard coverage；
- [ ] TileLang继续TIR hard/PTX diagnostic；CUDA继续semantic PTX hard/raw diagnostic；
- [ ] 不实现kernel-specific matcher、pairwise exception或allowed diff。

## 5. P3：自动Lift add2d pair

- [ ] `tilelang:add2d`：Codex Lift→TypeCheck→fresh J-only Replay→Gate1→Gate2；
- [ ] `cuda:tilelang_add2d`：同一完整流程；
- [ ] 保存Source、Harness、candidate、Lift/Replay trajectory、compiler observation、Gate1、Gate2；
- [ ] canonical比较两份accepted J/C：共同数学结构与真实frontend差异分别报告；
- [ ] 确认两份Replay receipt均无π输入。

## 6. P4：gather2d与Gate1价值

- [ ] add2d双闭合后运行TileLang/CUDA gather2d；
- [ ] 报告间接寻址下两个observer的hard/diagnostic差异与合法realization false reject；
- [ ] 对add2d做最小wrong-op、wrong-index、wrong-effect mutation sensitivity；
- [ ] mutation只改受控Replay副本并重新compile/observe，不进入正式certificate；
- [ ] Gate1拒绝后不运行Gate2，统计提前拒绝成本。

## 7. 最小实验输出

每个fresh正式case记录：

- Lift wall time、candidate bytes、checker calls、TypeCheck；
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

- 两个add2d case均由fresh Codex自动Lift，不使用历史artifact；
- 两个Replay输入中π物理缺席；
- 两个profile各自Gate1闭合；
- 两个case各自三边Gate2 PASS；
- 若gather2d无法在预算内闭合，准确报告J/Replay/observer capability边界，不伪造数据；
- 提交后工作区clean。

