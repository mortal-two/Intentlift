# run-008 基础设施审计

Source projection、adapter、request 与实际 `c0` 已在 provider 调用前保存。第一次 Lift HTTP 请求
在 ordinary `high` 的 600 秒客户端窗口内没有返回有效 response，随后 provider client 按既有策略
自动开始重试。为避免同一不可用请求最多重复占用约 40 分钟，第二次网络等待在整个进程运行超过
12 分钟后由操作者中断。

本 run 没有 Lift response、candidate、TypeCheck、Replay、Gate1 或 Gate2 证据，因此只能记为
`INFRASTRUCTURE_FAILED`。它不支持任何关于系统闭合能力或 candidate 正确性的结论，也不计入正式
Gate1 round。中断没有改写 Source、规则、Q 或任何 candidate；事实上本 run 尚未产生 candidate。
