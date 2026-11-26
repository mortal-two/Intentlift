# Intentlift

成长性的 CUDA → IntentDSL 源码编译器。正常编译不调用模型；缺少能力或实际结果不一致时，具备固定先验知识的 API Agent 修改功能组件、重新编译并调用独立 Bench。通过后的通用能力长期保留。

## 项目关系

- Intentlift 独立实现并运行编译器。src/intentlift/core 是固定管理和权限机制，src/intentlift/extensions 是允许扩充的组件，knowledge 是受保护的 Agent 先验知识。
- CompilerInfra-lift 是独立的规则和结构参考项目，无代码依赖或运行调用。
- lift-bench 独立编译、执行原始 CUDA 与输出 DSL，管理合法输入条件和精确比较。编译器 Agent 永远不能修改其内容。
- /home/wwq/compilerinfra、/home/wwq/Intent 只读，不参与新系统的构建或运行。

## 环境与运行

当前专用环境是 /home/wwq/.venvs/lift-bench，使用 cuTile 1.6、PyTorch CUDA、CUDA Driver bindings、OpenAI SDK 和 Clang。CUDA 源码使用系统 nvcc 编译，DSL 由 Bench 的独立作者语言解析器转换成 cuTile。依赖原本只读项目的 SDK、动态库或 Python 包均不需要。

两个项目已在上述环境中 editable 安装，可以直接使用该环境的 intentlift 和 lift-bench 命令。也可以从源目录运行：

```bash
export PYTHONPATH=/home/wwq/Intentlift/src
export LIFT_PYTHON=/home/wwq/.venvs/lift-bench/bin/python

"$LIFT_PYTHON" -m intentlift.cli inspect

"$LIFT_PYTHON" -m intentlift.cli run \
  --dataset /home/wwq/Dataset/TianchenIR-MultiFrontend \
  --case vector_add \
  --tilelang-root /home/wwq/.venvs/tilelang018/lib/python3.10/site-packages/tilelang
```

API Agent 从 OPENAI_API_KEY、OPENAI_BASE_URL 读取连接配置；使用 --model 或 OPENAI_MODEL 显式选择服务支持的模型。没有固定模型回退，也不使用 CLI Agent 回退。凭据只进入 API 客户端，不进入编译 worker 或 Bench。

```bash
"$LIFT_PYTHON" -m intentlift.cli run \
  --dataset /home/wwq/Dataset/TianchenIR-MultiFrontend \
  --case vector_mul \
  --tilelang-root /home/wwq/.venvs/tilelang018/lib/python3.10/site-packages/tilelang \
  --model "$OPENAI_MODEL"
```

--no-agent 可在没有 API 配置时报告能力缺失，不能绕过 Bench。--bench、--bench-python、--clang、--cuda-path、--nvcc 均可显式指定。--run-root 只选择临时工作目录的父目录，运行结束仍会清理。

默认在系统临时目录完成编译、候选和 Bench 验证，成功或失败结束后删除中间文件。验证通过的 DSL 直接放在返回 JSON 的 dsl 字段；需要持久成品时用 --output 显式指定仓库外的输出位置。不保留历史运行目录。

## 扩充与边界

Registry 提供具名表示及 verifier、具有输入/输出表示的阶段、功能规则和可检查的 pipeline。阶段连接必须吻合，最终输出必须是 IntentDSL 源码。组件注册位置由 inspect 查询。Agent 可增加表示、阶段和连接，不能改固定机制。

API 工具没有任意 shell。候选修改只能写扩展文件；编译 worker 由 bubblewrap 使用只读根文件系统、独立 PID/网络命名空间和临时目录执行，不能写 Bench 或正式框架。候选验证通过且完成通用性、已有功能保护说明后，宿主才接受到正式扩展目录。没有 bubblewrap 时失败关闭。

每次编译完成必须验证当前真实 Kernel。默认不运行历史 Kernel；Agent 必须先做源码影响分析。无固定修复次数限制；环境/API/Bench 故障、越界、明确 DSL 表达障碍或已确认无进展时停止。

## 当前能力边界

初始目标是 rank-one float32、一个完整 block 的 threadIdx.x 成员映射、直接指针访问和一个总输出赋值。功能按操作和适用条件注册，不按 Kernel 身份分支。其他索引、控制流、类型等需要后续扩充；不宣称任意 CUDA 已支持。

Bench 当前使用固定 shape 的多组真实数值条件，精确数值相等，NaN 位置一致，无穷符号一致，正负零按数值相等。不是逐位相等，也没有浮点容差。性能不参与通过判定。

当前状态、必要修改结论和实际测试结果统一维护在 [进度记录](PROGRESS.md)。中间文件使用后清理，记录不依赖历史运行文件。
