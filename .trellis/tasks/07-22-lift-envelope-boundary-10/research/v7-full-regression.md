# Task10 Protocol-v7 Full Regression

在提交 protocol-v7 manifest/wire/lineage 后，于 clean worktree 运行：

```text
PYTHONPATH=python pytest -q
```

结果：

```text
867 passed
4 skipped
60 warnings
0 failed
372.78 seconds
online LLM calls = 0
```

四个 skip 是既有环境/能力条件；60 条 warning 由 Triton deprecated autotuner 参数与
FlagGems SQLAlchemy automap class replacement 组成，没有新增 Gate1/Lift warning。

该回归同时证明：

- parser 收紧 `scopes/storage` wire required 没有破坏现有 canonical fixtures；
- v5 canonical response 与 v6 frozen rejection 保持可审计；
- v1--v6 historical manifests/evidence 没有被 current v7 builder 重新解释；
- root/J/Boundary/access mutation 在 schema/parser 双侧按合同关闭；
- v7 manifest 与 current schema/prompt/client identity 完全一致；
- v7 recovery 在 provider/ledger 前精确绑定 v6 envelope rejection；
- normal 未闭合不运行 masked-J 的 runner 约束保持不变。

因此可以进入 0-LLM formal preflight；在 preflight 28/28 ready 且提交冻结之前，不允许执行
protocol-v7 canary。
