# Intentlift

成长性的 CUDA → IntentDSL 源码编译器。正常编译不调用模型；缺少能力或实际结果不一致时，具备固定先验知识的 Agent 修改功能组件、重新编译并调用独立 Bench。默认使用 API，也可显式选择 CLI、MiniSWE 或 MCP。通过后的通用能力长期保留。

## 项目关系

- Intentlift 独立实现并运行编译器。src/intentlift/core 是固定编译机制，src/intentlift/extensions 是允许扩充的组件，src/intentlift/agents 集中全部 Agent 调用与权限宿主，knowledge 是受保护的先验知识。
- CompilerInfra-lift 是独立的规则和结构参考项目，无代码依赖或运行调用。
- lift-bench 独立编译、执行原始 CUDA 与输出 DSL，管理合法输入条件和精确比较。编译器 Agent 永远不能修改其内容。
- /home/wwq/compilerinfra、/home/wwq/Intent 只读，不参与新系统的构建或运行。

## 环境与运行

当前专用环境是 /home/wwq/.venvs/lift-bench，使用 cuTile 1.6、PyTorch CUDA、CUDA Driver bindings、OpenAI SDK 和 Clang。原始 CUDA 源码使用系统 nvcc 编译。Bench 独立解析 IntentDSL，支持 cuTile 和有序标量 CUDA 两种 GPU 执行路径。

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

默认 API Agent 从 OPENAI_API_KEY、OPENAI_BASE_URL 读取连接配置；用 --model 或 OPENAI_MODEL 选择模型。通过 --agent-backend 显式选择其他入口，不因失败自动切换接口、模型或调用方式。凭据不进入编译 worker 或 Bench。

```bash
"$LIFT_PYTHON" -m intentlift.cli run \
  --dataset /home/wwq/Dataset/TianchenIR-MultiFrontend \
  --case vector_mul \
  --tilelang-root /home/wwq/.venvs/tilelang018/lib/python3.10/site-packages/tilelang \
  --model "$OPENAI_MODEL"
```

--no-agent 可在没有 API 配置时报告能力缺失，不能绕过 Bench。--bench、--bench-python、--clang、--cuda-path、--nvcc 均可显式指定。--run-root 只选择临时工作目录的父目录，运行结束仍会清理。

默认在系统临时目录完成编译、候选和 Bench 验证，成功或失败结束后删除中间文件。验证通过的 DSL 直接放在返回 JSON 的 dsl 字段；需要持久成品时用 --output 显式指定仓库外的输出位置。不保留历史运行目录。

## Agent 调用入口

全部实现集中在 src/intentlift/agents/，host.py 是唯一修改和验证宿主，protocol.py 管理统一模型轮次，broker.py 和 mcp_bridge.py 提供受控工具连接。core 中不保留重复 Agent 实现。

| --agent-backend | 实现文件 | 模型与认证 |
| --- | --- | --- |
| api / responses（默认） | responses.py | OPENAI_MODEL、OPENAI_API_KEY、可选 OPENAI_BASE_URL |
| chat-completions | chat_completions.py | 同上，显式使用 Chat Completions 接口 |
| codex | codex.py | --model 或 OPENAI_MODEL；OpenAI API 配置或已有 Codex 认证 |
| miniswe | miniswe.py | 同 OpenAI 配置；使用实际 MiniSWE 控制器和受限工具环境 |
| mcp | mcp.py、mcp_bridge.py | 模型由外部 Agent 管理，服务自身不需要模型密钥 |
| claude | claude.py | --model / ANTHROPIC_MODEL；ANTHROPIC_API_KEY、ANTHROPIC_AUTH_TOKEN、CLAUDE_CODE_OAUTH_TOKEN 或已有 OAuth 认证 |
| gemini | gemini.py | --model / GEMINI_MODEL；GEMINI_API_KEY、Google/Vertex 配置或已有 Gemini OAuth 认证 |

Claude Code 可使用 ANTHROPIC_BASE_URL 配置其兼容网关；Gemini 支持 GEMINI_API_KEY（也接受 GOOGLE_API_KEY），或 GOOGLE_GENAI_USE_VERTEXAI=true 配合 Google Cloud 项目、区域和 ADC 配置。已有 Gemini OAuth 认证也可复用。ADC 与 OAuth 的必要认证文件只复制到私人临时 HOME，结束后清理。两者不会继承 OpenAI 的模型名和密钥；未显式指定模型时使用各自 CLI 的默认设置。接入不绑定开发者个人账号。

```bash
python -m pip install -e '.[all-agents]'
intentlift agents
intentlift run --dataset /path/to/Dataset/TianchenIR-MultiFrontend \
  --case vector_sub --tilelang-root /path/to/site-packages/tilelang \
  --agent-backend codex
```

api、chat-completions、miniswe、claude、gemini 均可替换上例的后端名。选择后端不意味着强制调用；只有编译失败或结果不一致才介入。

Codex CLI 通过其官方发行方式安装。Claude Code 和 Gemini CLI 可独立安装，也可使用以下可发现目录：

```bash
npm install --prefix "$HOME/.local/share/intentlift/agent-clis" \
  @anthropic-ai/claude-code @google/gemini-cli
```

LIFT_CODEX_BIN、LIFT_CLAUDE_BIN、LIFT_GEMINI_BIN 可以覆盖对应可执行文件位置。intentlift agents 报告依赖和认证是否配置，不显示密钥，也不将配置存在视为真实模型验证通过。

CLI 使用独立临时 HOME 和只读文件系统，原项目、Bench 和参考项目不可写。只允许通过 intentlift MCP 工具提交修改；Claude 关闭内置工具，Gemini 使用只允许本 MCP 的策略，Codex 关闭 shell 和多 Agent 工具。所有调用使用同一份知识、权限、无进展判断及真实验证。验证终态后宿主结束 CLI，不需要额外模型请求来确认通过；CLI 的底层 API 次数不作推测。

## 外部 Agent 的 MCP 接口

独立入口 intentlift-agent-mcp 使用标准 MCP stdio，可以由支持 MCP 的客户端启动：

```json
{
  "mcpServers": {
    "intentlift": {
      "command": "/path/to/venv/bin/intentlift-agent-mcp",
      "args": ["--dataset", "/path/to/Dataset/TianchenIR-MultiFrontend",
               "--case", "vector_sub", "--tilelang-root", "/path/to/site-packages/tilelang"]
    }
  }
}
```

提供 task_context、inspect、read_file、propose_changes、compile、validate、stop 七个工具。客户端先读取 task_context 获取同一先验知识与实际输入。正常可编译任务不允许额外修改，验证不一致或能力缺失后才进入修复；任何修改最终仍由独立 Bench 判断。

同一 CLI 也支持 run --agent-backend mcp：需要修复时切换为 stdio 工具服务，最终摘要写 stderr，stdout 只传 MCP 消息。独立 MCP 入口更适合客户端的固定配置。连接结束后临时会话清理，不留下认证副本、轨迹或运行目录。

## 扩充与边界

Registry 提供具名表示及 verifier、具有输入/输出表示的阶段、功能规则和可检查的 pipeline。阶段连接必须吻合，最终输出必须是 IntentDSL 源码。组件注册位置由 inspect 查询。Agent 可增加表示、阶段和连接，不能改固定机制。

API 工具没有任意 shell。候选修改只能写扩展文件；编译 worker 由 bubblewrap 使用只读根文件系统、独立 PID/网络命名空间和临时目录执行，不能写 Bench 或正式框架。候选验证通过且完成通用性、已有功能保护说明后，宿主才接受到正式扩展目录。没有 bubblewrap 时失败关闭。

每次编译完成必须验证当前真实 Kernel。默认不运行历史 Kernel；Agent 必须先做源码影响分析。无固定修复次数限制；环境/API/Bench 故障、越界、明确 DSL 表达障碍或已确认无进展时停止。

## 当前能力边界

当前编译器支持静态一维向量和二维矩阵的 float32 逐元素计算。单个 CUDA block 完整覆盖一个输出视图。编译时根据 Clang AST 检查每个线程的存取索引，支持同坐标读写和二维只读转置。表达式支持 f32 常量、加减乘除和基于 f32 比较的逐元素选择。

混合精度路径覆盖 f16 矩阵乘法和带共享内存暂存的 GEMV。编译器检查访存和循环范围，生成显式转为 f32 的乘法、有序 f32 累加和 f32 输出。支持行内有序求和及 maximum-number 归约。

整数路径支持 f32 向零截断到 i32，以及把行内多轮最大值选择转成 f32 值和 i32 索引的配对输出。配对输出保留源码的并列取索引及候选屏蔽规则。

单块单 tile 的 FlashAttention 路径检查共享暂存、异步拷贝及同步顺序，恢复四维 Q/K/V 和输出视图。计算保留 f32 点积与归一化、权重先转为 f16 的舍入点及 f16 输出。已实测的形状为 `(1, 1, 64, 64)`。多块或多 tile 注意力、因果遮罩、动态形状、通用 f16 算术、BF16 和 FP64 仍待扩充。

Bench 按 Dataset 的固定形状和类型独立分配两侧缓冲区，比较全部输出。浮点沿用五组基础条件，含 f16 输入时追加有限次正规数条件。配对选择追加并列值条件。数值精确比较，浮点的 NaN 位置与无穷符号必须一致，正负零按数值相等。

当前状态、必要修改结论和实际测试结果统一维护在 [进度记录](PROGRESS.md)。中间文件使用后清理，记录不依赖历史运行文件。
