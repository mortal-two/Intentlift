# Task 33 P1 checkpoint：line-level semantic-SLoC ownership

> 冻结日期：2026-08-02  
> 协议：`intentir.experiment.frontend-semantic-sloc-ledger.v1`  
> 输入状态：`1ce53f68391c7da57c112926c46dc331f649c9e1`

## 已完成的边界

P1 不是把仓库总行数包装成“前端成本”，而是把每个被审计的物理行唯一归入 11 个 ownership
bucket。论文核心只取：

1. `frontend_semantic`：Source ABI/effect/shape 到逻辑接口的所有权；
2. `observer_semantic`：CUDA/PTX 与 TileLang/TIR 的观察语义；
3. `per_kernel_persistent`：历史 B0 中为 kernel 保留的 J 语义构造代码。

编译/执行 adapter、公开 prompt context、共享协议、sidecar、receipt、tests、runner、docs 和生成物均
分列，不能进入核心 semantic SLoC。

## 冻结结果

| 视图 | raw | nonblank/noncomment | audited semantic core |
|---|---:|---:|---:|
| 初次联合 onboarding `d30753e7^..d30753e7` | +3,619 / -5 | 3,149 | 589 |
| 当前 production frontend footprint | 12,139 present | 10,694 | 2,311 |
| final-five `b4f1b6da..df95382c` | +56,765 / -124 | 1,707 line-audited | 265 |
| Task 28 四个作者化模块 | 2,223 present | 2,009 | 734 |

final-five 的 56,765 新增行中有 55,050 行是 `runs/` 下 immutable Gate/trajectory artifacts；它们
在 `window_raw_totals.csv` 保留总量，但不重复展开进逐行代码账本，也永不进入 semantic core。

当前 2,311 semantic-core SLoC 由 359 frontend-semantic 与 1,952 observer-semantic 组成。它是最终
能力 footprint，不是“第一次加入一个前端”的边际成本；后者使用冻结的 d307 窗口并继续按
TileLang/CUDA/shared 分类。

## B0 冻结 12 对的作者化曲线

仅计 J semantic constructor，冻结顺序的累计值为 531 SLoC；单 kernel 最小 8、中位数 40.5、
平均 44.25、最大 114。额外 203 SLoC 的 frontend ABI/signature ownership 在逐行 ledger 中单列。
共享 graph/J/C/pi helper、receipt 和 runner 也均可见，但不重复摊到每个 kernel。

这个 531 不是 IntentIR 最终论文优势结论，而是 B0 的可核验输入曲线。P2 的 B1 converter 必须用
相同 12 对、相同 J schema/TypeCheck/Gate 2/backend 生成自己的 persistent semantic-SLoC 曲线，
两者才能形成 matched comparison。

## 可复算文件

- `audit_semantic_loc.py`：冻结 revisions、路径集合、symbol ownership 与计数实现；
- `semantic_loc_audit/semantic_loc_ledger.csv`：19,825 个唯一 line events；
- `semantic_loc_audit/frontend_onboarding_summary.csv`：按 window/frontend/bucket/capability 汇总；
- `semantic_loc_audit/kernel_authoring_summary.csv`：冻结 12 对的 B0 constructor 曲线；
- `semantic_loc_audit/window_raw_totals.csv`：整 commit raw 与 line-audited scope 对账；
- `semantic_loc_audit/manifest.json`：全部派生文件的 SHA256 receipt。

连续执行两次审计器后所有文件 SHA256 完全一致；逐行主键
`(window,path,change_kind,line_number)` 亦已检查唯一。

## 对 P2 的约束

- B1 只能读取 Source、公开 frontend observation 与共享 schema；
- 禁止读取 Task 28 manual artifacts、Replay、Gate diff 或 sibling baseline；
- converter rule 只能按 source op / typed relation 泛化，禁止 kernel id/path/family matcher；
- 每个冻结 pair checkpoint 记录新增 persistent semantic SLoC、rule count、closure 与失败；
- P1 分类和冻结顺序不得因 B1 的结果而回改。
