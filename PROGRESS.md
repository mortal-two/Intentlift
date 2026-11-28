# 项目进度与验证记录

本文件是 Intentlift、CompilerInfra-lift 和 lift-bench 唯一持续维护的进度记录。记录当前合同、必要修改、验证条件与结果、未解决事项。临时运行报告、候选、数值、生成程序和构建产物用后删除，不归档，不以临时路径作为永久证据。

## 当前合同

- CUDA 是唯一前端，IntentDSL 源码是编译输出。Intentlift 独立运行编译器，不依赖或调用 CompilerInfra-lift；后者是独立可运行的结构与固定机制参考实现。
- 正常编译先尝试，能力缺失或结果不一致后才调用具有受保护先验知识的 Agent，API 是默认入口。规则按功能和类型组织，不按 Kernel 信息编写个例分支。
- Agent 只通过固定扩展接口修改编译组件，不能修改框架、权限、知识库、CompilerInfra-lift 或 Bench 的任何内容。
- Bench 管理合法输入、实际执行、独立高精度参考和固定通过阈值。原始 CUDA 和 DSL 分别对同一参考判定，两侧均达标才通过。浮点默认使用容差，整数和索引等明确要求精确的结果逐元素比较。性能不作为通过条件。
- 默认只验证当前 Kernel；保护已有功能依靠源码分析，后续实际发现问题再修复。无固定修复次数上限，按成功、故障、越界、表达障碍及无进展结束或停止。
- /home/wwq/Intent 与 /home/wwq/compilerinfra 只读。必要构建、测试和真实验证直接进行，无需重复询问。

## 已完成的建设与验证结论

| 范围 | 条件与过程 | 结论 |
| --- | --- | --- |
| Intentlift 初始编译 | 真实加法，f32，固定长度 128，五组数值条件 | 5/5 精确一致 |
| API Agent 成长 | 真实乘法先缺少能力；7 次 API 交互，仅扩充通用算术登记并保留加法 | 重新编译后 5/5 精确一致，修改接受为正式能力 |
| 成长能力复用 | 后续真实乘法直接编译，未调用 Agent | 5/5 精确一致 |
| CompilerInfra-lift 独立参考产品 | 独立 Clang 解析与 DSL 生成，真实加法 | 5/5 精确一致 |
| 独立维护流程 | 真实乘法能力缺失、派生候选、通用扩充、检查点恢复、Bench、接受修改 | 全流程完成，5/5 精确一致 |
| 独立 wheel 安装 | 专用环境仅有框架、pip 和 setuptools；安装入口执行真实乘法 | 5/5 精确一致，无旧项目运行依赖 |
| 输入绑定保护 | 使用不同真实输入恢复检查点；以不同 Bench 输入验证编译结果 | 分别拒绝错误恢复和错误验证绑定 |

五组数值条件：两个固定种子正态输入、一个均匀输入、零与符号、IEEE 边界值。这里记录的是实际运行结论；原始中间文件按用户要求清理，不保留下载链接或历史报告路径。

## 2026-11-26：临时产物与进度记录调整

- 原分散运行产物不再保留，进度文档统一到本文件，旧 final-task 文档移除。
- Intentlift 完整运行采用临时目录并自动清理；验证后的 DSL 返回调用者，仅显式输出成品可保留。
- Bench 独立运行默认自动清理；跨步骤所需证据仅在调用者管理的临时会话中存在。
- CompilerInfra-lift 提供 run 完整流程，临时编译与验证结束后清理；底层构建、候选和检查点只在活动维护会话内传递。
- 已删除 lift-runs 和 compilerinfra-lift-runs 全部历史目录，无备份或归档；旧 final-task 文档不存在。
- 调整后真实乘法验证：Intentlift 正常流程 5/5 精确一致、未调用 Agent，结束后临时目录为空；Bench 独立运行 5/5 精确一致并自动清理；CompilerInfra-lift 独立安装入口的 run 流程 5/5 精确一致并自动清理。
- 实际临时候选编译通过：候选位于系统临时目录时，在只读 worker 内仍可解析真实 CUDA 并生成 DSL；没有使用模拟组件。
- 失败退出检查：真实减法输入报告未支持能力后清理完成；真实乘法编译与加法 Bench 输入不匹配时被拒绝，临时编译和验证目录全部清理。
- 独立安装已更新，构建 wheel、分发源树及验证父目录均在临时会话结束后删除。代码和所需运行环境保留。
- 语法、差异与安装一致性检查通过；上述机械检查不增加数值验证计数。

## 2026-11-26：统一 Agent 目录与多入口接入

- 所有 Agent 宿主、调用适配、配置发现、本地工具连接和 MCP 入口集中到 src/intentlift/agents/，删除原 core/agent.py，不保留重复实现。
- 已实现 Responses API（api 默认别名）、Chat Completions、Codex CLI、MiniSWE、外部 MCP、Claude Code、Gemini CLI。命令行显式选择后端，不自动切换；OpenAI 默认模型保持 gpt-6-sol。Claude/Gemini 分别读取自身模型和认证配置。
- 各入口共用七个受控工具、先验知识、候选修改、通用性说明、源码保护、重新编译、Bench 验证和无进展规则。MCP 并发请求在宿主串行执行，候选修改与验证不能交错。
- CLI 在只读隔离进程和私人临时 HOME 中运行，只通过 MCP 修改扩展；不提供任意写文件或 shell 修改通道。原项目、Bench 与参考项目隐藏。认证副本、配置、CLI 输出和候选在会话结束时清理。
- 使用实际上游 MiniSWE 控制器及受限 Model/Environment 适配，不把 MiniSWE 入口伪装成直接 API 循环；运行后恢复临时环境变量。外部 MCP 使用官方 SDK 的 stdio 协议。
- 已安装实际 Codex、Claude Code、Gemini CLI，及 MCP/MiniSWE Python 依赖。新增 intentlift agents 状态入口与 intentlift-agent-mcp 独立工具服务入口，README 已补充安装、选择、认证和外部连接方式。

### 真实数值与成长流程验证

每个模型入口使用独立的真实编译器扩展副本，从仅支持加法和乘法的状态处理 Dataset 中真实 vector_sub CUDA：先报告减法能力缺失，Agent 分析并生成通用操作登记，再重新编译并由独立 Bench 比较五组真实输入。临时副本经宿主接受修改后清理，没有将减法测试能力加入正式编译器。

| 入口 | 实际模型/API 交互 | 受控工具调用 | 独立 Bench 结果 |
| --- | --- | --- | --- |
| Responses API | 8 次真实 API 请求 | 8 次 | 5/5 f32 精确一致 |
| Chat Completions | 9 次真实 API 请求 | 9 次 | 5/5 f32 精确一致 |
| MiniSWE | 8 次真实 API 请求 | 8 次 | 5/5 f32 精确一致 |
| Codex CLI | 使用实际 Codex 模型调用；底层 API 次数不可观测，不作推测 | 9 次 | 5/5 f32 精确一致 |

- 上述验证未修改正式加法/乘法扩展。Codex 验证成功后由宿主结束进程，其进程终止码与已验证成功分别记录，不将终止码误报为编译失败。
- 外部 MCP：官方客户端完成初始化、列出七工具、读取上下文与组件、重新编译真实 vector_mul 并通过 5/5 精确比较；会话清理通过。
- 越界 MCP：在真实 vector_sub 输入下尝试越界修改被拒绝，随后编译不能解除终止状态；受保护文件未变，会话清理通过。
- 加入宿主串行保护后再次通过实际 MCP 并发只读请求，并重新编译 vector_mul，5/5 精确一致，会话清理通过。
- 正常编译回归：显式选择 Gemini、移除其认证配置后，真实 vector_mul 仍通过 5/5 精确一致，agent_invoked=false，临时产物清理完成。后端选择不会强迫已支持 Kernel 调用模型。

### Claude Code 与 Gemini CLI 的验证范围

- 两者已完成接口实现、官方客户端安装、私人配置加载与隔离环境下到真实 Intentlift MCP 服务的健康连接，均显示 Connected。Claude 的连接诊断只向私人临时配置注册服务，没有写入开发者全局配置。
- 两者实际非交互运行参数均通过原生客户端解析；当前缺少各自认证时明确停在认证阶段，没有模型推理结果。不能把 MCP 连通或参数解析算作真实模型修复通过；也不将开发者个人凭据作为完成公共接口的前置条件。
- Claude 支持 Anthropic API、兼容网关和已有 OAuth；Gemini 支持 Gemini API、Google API key 映射、Vertex/ADC 和已有 OAuth。必要认证副本在临时会话内使用，不写入仓库或进度记录。
- 语法解析、CLI 入口、依赖发现和 git diff --check 通过。这些检查不增加数值正确性验证计数。

## 2026-11-27：逐元素算术、条件选择与二维访存

- Intentlift 的正式扩展加入 f32 减法、除法、常量及逐元素比较选择。选择可以来自 CUDA 条件表达式，或给同一局部变量赋值的完整 if/else。两种形式都要求唯一输出写入。
- 编译器从 Clang AST 计算固定单 block 上每个线程的整数索引。输出必须完整覆盖一维或二维静态视图且无重复写入。读索引支持相同坐标和二维只读转置，输出 DSL 使用每个轴一层 parallel 的逻辑坐标。
- Bench 按固定 ABI 形状生成一维或二维 f32 输入，并独立解析、执行同坐标读写、二维转置和新增表达式。CUDA 和 DSL 都采用分别舍入的乘加运算，原来的五组数值条件和精确比较标准继续使用。

| 真实 Dataset 样例 | 验证范围 | 独立 Bench 结果 |
| --- | --- | --- |
| vector_sub、vector_div | 减法和除法 | 各 5/5 精确一致 |
| vector_scale、vector_neg、vector_axpy、vector_fma | f32 常量、表达式组合、乘法和加法分别舍入 | 各 5/5 精确一致 |
| vector_select、vector_abs | 同一局部变量的完整 if/else 两分支，以及 IEEE 边界值下的比较 | 各 5/5 精确一致 |
| matrix_add、matrix_transpose | 16×16 二维视图，分别覆盖同坐标访问和坐标交换 | 各 5/5 精确一致 |
| vector_add、vector_mul | 共用路径修改后的回归，重新编译生成 DSL | 各 5/5 精确一致 |

- 12 个 Kernel 在 RTX 5090 D 上合计通过 60/60 组验证，均未调用 Agent。用真实 row_sum 确认当前归约路径返回 capability_missing。测试会话及中间产物全部清理。

## 2026-11-27：float16 输入与 float32 计算

- 扩充编译器和独立 Bench，以实际 f16 矩阵作为只读输入。识别 CUTLASS half 的显式转 f32、线程私有 f32 结果及有序循环。按所有输出坐标和循环位置检查行列索引映射，生成保留转换点、累加顺序及 f32 输出的 IntentDSL。
- Bench 按各参数 ABI 分配缓冲区并独立执行 DSL。原有 f32 的五组条件继续使用。含 f16 的合同用对应类型的 IEEE 边界值，并追加一组有限次正规数及临界正规值，直接覆盖低精度小值转 f32。输出继续精确比较。

| 真实 Dataset 样例 | 数值路径 | 独立 Bench 结果 |
| --- | --- | --- |
| matrix_gemm | 16×16，只读 f16×f16，显式转 f32 乘法、有序 f32 累加、f32 输出 | 6/6 精确一致，含 f16 有限次正规数条件 |
| vector_fma、matrix_transpose | 受影响的纯 f32 算术和二维读取路径，重新编译 | 各 5/5 精确一致 |
| softmax | f16 输出的真实例子 | 仍报告 bench_unsupported |

- 所有运行使用 RTX 5090 D，未调用 Agent，临时会话全部清理。

## 2026-11-27：有符号 int32 输出

- 编译器从 Clang AST 检查直接读取 f32 成员并转换为有符号 32 位整数的 CUDA 表达式，就地保留向零截断语义和输出坐标，生成 I.cast(value, I.i32) 与 i32 输出视图。
- Bench 为转换选择有限 f32 输入，范围为 [-2³¹, 2³¹)。六组条件覆盖种子随机数、符号、截断点及 i32 范围两端。两侧使用实际 i32 输出缓冲区，逐成员检查完整写入及精确一致性。

| 真实 Dataset 样例 | 验证范围 | 结果 |
| --- | --- | --- |
| vector_cast | f32 读取、向零截断、有符号 i32 输出 | GPU 6/6 精确一致。检查实际产物，两侧原生 i32 结果均与独立截断参考一致 |
| vector_div、matrix_gemm | 涉及输入条件和输出比较的浮点回归 | f32 除法 5/5，f16/f32 矩阵乘法 6/6 精确一致 |
| topk | f32 值及 i32 索引的多输出 | 当前仍返回 bench_unsupported |

- 运行使用 RTX 5090 D，临时会话全部清理。

## 2026-11-27：FlashAttention、行归约、GEMV 与 Top-K

- Intentlift 根据 Clang AST 检查向量暂存、两次异步等候与同步、线程与输出成员映射、点积和归一化的浮点顺序。由真实 CUDA 恢复四维 Q/K/V 及输出视图，保留 f32 计算、注意力权重的 f16 舍入和最终 f16 输出。当前规则覆盖单块单 tile 调度。
- GEMV 分析共享内存复制、同源重复写入、同步及真实读取来源，生成有序 f32 点积。行内求和和最大值保持原 CUDA 的顺序及初值。配对选择保存 shuffle 分组、并列取最大索引和候选屏蔽规则。
- Bench 增加有序标量 CUDA 执行方式，独立检查 DSL 的类型、索引、循环界限及全部输出。旧 GEMV 和 FlashAttention 快照缺省动态共享内存大小，Bench 使用与原始 CUDA 摘要绑定的启动记录。编译原始 FlashAttention 时显式包含同一 TileLang 安装中的 256 位存储辅助头文件。

| 真实 Dataset 样例 | 已验证的能力 | 独立 Bench 结果 |
| --- | --- | --- |
| flash_attention | `(1, 1, 64, 64)`，单 CTA 单 tile，f16 Q/K/V 与 f16 输出，f32 有序计算及 f16 权重舍入 | 6/6 精确一致，含 f16 次正规数 |
| gemv | 共享暂存和同步，f16 向量、矩阵与 f32 输出 | 6/6 精确一致 |
| row_sum、row_max | 线程归属检查及有序行归约 | 各 5/5 精确一致 |
| topk | 跨线程最大值语义、并列项处理、f32 值和 i32 索引 | 两个输出均 6/6 精确一致 |
| vector_cast、matrix_gemm | 涉及类型与执行路径的回归 | 各 6/6 精确一致 |

- RTX 5090 D 共通过 40/40 组，未调用 Agent，临时目录清理完成。当前 softmax 仍报告能力缺失。

## 2026-11-27：多块前缀和、有序点积与向量数学

- 检查共享内存暂存、同步及扫描调用，枚举可信启动合同中的所有 block，证明每块负责互不重叠的完整扫描线。输出 DSL 保留 warp 分阶段加法和分段进位。真实扫描形状为 `(1, 64, 4)`，启动四个 CUDA block，每块一条独立扫描线，每条由两个完整 warp 分段构成。
- Bench 从 DSL 独立推导整数索引与有序循环界限，检查视图访问安全。扫描条件在原有五组之外增加大小量混合与抵消条件。实际原 CUDA 按快照启动记录分配每块 256 字节动态共享内存。
- 有序点积保留每次 f32 乘法和累加。逐元素定义增加最大值、最小值、exp、exp2 和 log，并验证截断、ReLU 和 Sigmoid 的组合。

| 真实 Dataset 样例 | 独立 Bench 结果 |
| --- | --- |
| chunk_cumsum | 6/6 精确一致，含扫描舍入条件 |
| vector_dot | 5/5 精确一致 |
| vector_min、vector_max、vector_clamp、vector_relu | 各 5/5 精确一致 |
| vector_exp、vector_exp2、vector_log、vector_sigmoid | 各 5/5 精确一致 |
| flash_attention、topk、gemv、vector_cast | 重新编译回归，各 6/6 精确一致 |

- RTX 5090 D 上 10 个新增样例共 51/51 组通过，4 个既有样例共 24/24 组通过。默认入口复验真实扫描未触发 Agent。临时产物均已清理。

## 2026-11-27：配对基线、卷积与倒平方根、并发提交

- 初始 Git 基线为 Intentlift `ba2c80a0` 与 lift-bench `933695f6acf80892db62ed442a791d1bebcb9082`。本节的 Intentlift 代码配对验证 Bench `c881b08a9332d7baafae541ab46076490bfacfe5`。四个旧 CUDA 快照的动态共享内存启动记录随 Bench wheel 安装，声明绑定已校验的原始 CUDA 摘要。
- 用隔离构建的 Bench wheel 安装到临时目录，再通过 Intentlift 正常入口调用。真实卷积在安装版本上通过 6/6 精确比较，实际 CUDA 使用四个 block 和每块 1,536 字节动态共享内存。
- 卷积阶段独立注册，为每个输出及有序归约项证明暂存来源、边缘 padding 和尾部补零。验证范围为 FP16 NHWC 输入与输出、FP32 累加、3×3 滤波器、静态四块调度。Bench 对 DSL 的条件分支独立收紧访问范围。
- 将公共 Clang AST 检查与逐元素数学登记拆成候选目录内的可复用模块。默认 Responses API 从真实 vector_rsqrt 缺失开始，8 次 API 与 8 次工具调用，只修改 _cuda_math.py。Bench 5/5 精确一致后自动保留，后续直接编译亦为 5/5。
- 在临时副本移除卷积阶段的管线连接，保留已实现的阶段。真实卷积触发缺失后，默认 Responses API 以 8 次 API 和工具调用恢复 pipeline.json。候选验证 6/6，接受后的直接复用 6/6。此项验证复杂 Kernel 的阶段发现、接线和复用。
- 正式扩展编译持共享锁，候选发布持独占锁并核对基线和实测摘要，使用目录交换整组接受。两个独立进程从同一临时基线修复真实 vector_rsqrt 和 convolution。两份候选均通过实际 Bench，第一份完整接受，另一份基线过期被拒绝。临时目录已清理。
- FlashAttention、Top-K、扫描、GEMM、整数转换重新编译回归，各 6/6。卷积和倒平方根分别新增 6/6、5/5。连同此前覆盖盘点补测通过的 vector_square 和 vector_cube，Dataset 去重后的实测覆盖为 33/37。
- RMSNorm 原始源码的 Clang 临时对象取地址诊断已通过受限兼容重试，后续仍缺向量化归约与写回捕获。剩余实测缺口为 rms_norm、softmax、vector_sin 和 vector_cos。

## 2026-11-27：FP32 RMSNorm

- 新增归一化捕获阶段，检查每个线程的行内平方和顺序、原始行宽除法和 epsilon，以及 256 位负载和写回中八个 f32 分量的映射。只有所有输出行完整且互不重叠才生成 DSL。
- DSL 使用矩形并行域与静态有序循环表达归约，保留每次乘法和加法、倒平方根和逐元素缩放。Bench 独立支持元组并行坐标、静态循环和有类型的 f32 常量。
- 真实 rms_norm 使用静态 16×32 f32 输入与输出，在 RTX 5090 D 上 5/5 精确一致。默认入口复验未调用 Agent。重新编译回归的 convolution 和 vector_rsqrt 分别为 6/6、5/5。临时产物均已清理。
- Dataset 去重后的实测覆盖为 34/37，剩余 softmax、vector_sin 和 vector_cos。

## 2026-11-27：剩余真实 Kernel 复跑

| Kernel | 当前编译结果 |
| --- | --- |
| `vector_sin` | `cuda.pointwise` 报 `capability_missing`，缺少 `sinf` 的 f32 转换规则 |
| `vector_cos` | `cuda.pointwise` 报 `capability_missing`，缺少 `cosf` 的 f32 转换规则 |
| `softmax` | `cuda.pointwise` 报 `capability_missing`，当前通用入口只接受只读 f16 视图。此例为 4×32 f16 输入与输出，使用跨线程分组归约 |

- 使用现有编译器复跑，禁用 Agent 以记录当前覆盖。三个样例均未生成 DSL，未进入 Bench 数值比较。实测覆盖仍为 34/37，临时产物已清理。

## 2026-11-27：Bench 严格数学与边界输入

- 标量 CUDA 执行器接受显式源数值语义、严格 FP32 除法和数学调用中的严格浮点选项。独立 Bench 用标注了显式选项的 DSL 对真实 vector_div、vector_log 各通过 5/5。显式近似对数返回 `bench_unsupported`。已重新编译复核 vector_div、vector_log、rms_norm，各 5/5 精确一致。
- 二维 FP16 指数归一化的固定条件由 6 组增至 40 组。追加并列与近并列、有限尾部、大幅值及 32 组固定种子的多尺度输入。手写 softmax 候选保留分组归约的加法括号及 FP16 写回，使用自然对数换底。最初 8/8 通过，额外 256 组有 5 组不一致。最终固定矩阵在种子 27 检出 1 个 FP16 输出差异，候选被判定为 `result_mismatch`。
- FP32 正弦和余弦的固定条件由 5 组增至 6 组。新增象限两侧的相邻可表示角度及大幅值有限输入，真实 vector_sin 和 vector_cos 的原始 CUDA 均执行通过。两者仍待可精确比较的 IntentDSL 表达。
- 本轮扩充独立 Bench。Intentlift 正式编译通过覆盖为 34/37。比较全程使用 RTX 5090 D 和原有精确标准，临时产物已清理。
- Bench 本地提交 `1b408d4` 后复跑。Intentlift 在禁用 Agent 时仍对 softmax、vector_sin、vector_cos 报 `capability_missing`。独立 Bench 再次检出 softmax 手写候选在种子 27 的 1 个输出差异，通过 39/40 组。正弦和余弦的原 CUDA 各执行 6/6 组。重新编译的 rms_norm、vector_div、vector_log 各通过 5/5 组精确比较，临时产物已清理。

## 2026-11-27：通过正常入口复测自动修复

- 真实 Kernel 测试统一使用 `intentlift run`。主入口移除了关闭 Agent 的参数。能力缺失、生成错误和结果不一致会自动触发所选 Agent，候选通过 Bench 后才接受。
- 默认 Responses API 真实复跑，vector_sin、vector_cos、softmax 均自动进入 Agent。两个三角函数的候选编译后被独立 Bench 判为 `bench_unsupported`，执行器拒绝候选中的 DSL 调用。softmax 的 Agent 请求读取允许范围之外的文件，宿主以 `boundary_required` 停止。
- 同一入口重新编译 rms_norm，Bench 5/5 精确一致，`agent_invoked=false`。
- 三个样例均无通过验证并接受的候选，已验证覆盖仍为 34/37。Bench 使用本地提交 `1b408d4`，运行产物已清理。

## 2026-11-27：独立高精度参考与误差阈值

- Bench 为 Dataset 中 37 个已校验的 CUDA 快照绑定独立 CPU 参考。参考用 FP64 计算，保留算子显式中间转换并按输出 ABI 舍入。未登记参考的快照报告 `reference_unavailable`。RMSNorm 的有限极值条件取在源码 FP32 平方和范围内，继续保留 NaN、无穷与次正规数条件。
- 原始 CUDA 和生成 DSL 各自与同一参考比较。两侧的所有输出和条件达标才整体通过。FP16 默认 `atol=1e-5`、`rtol=1e-3`，FP32 默认 `atol=1e-6`、`rtol=1e-5`。线性归约另用输入绝对贡献和舍入次数推导绝对误差界。整数、索引及取值重排等使用精确比较。报告包含分侧绝对误差、相对误差、ULP 中位数、P99、最大值和首个超差位置。
- 正常入口重新编译的 vector_add、rms_norm、vector_cast、topk、chunk_cumsum、vector_dot、matrix_gemm、gemv、convolution、vector_log、vector_rsqrt 和 vector_div 均通过，合计 66/66 条件。FlashAttention 通过 5/6。种子 83 下 CUDA 与 DSL 在同一个输出均超差，判为 `source_result_mismatch`。诊断发现 FP32 分数后的一个 FP16 权重舍入到相邻值，对接近零的输出产生约 `2.6e-5` 的绝对误差。
- 同一正常入口触发默认 Agent 复跑 softmax、vector_sin 和 vector_cos。softmax 以 `no_progress` 停止，正弦和余弦在候选执行前返回 `bench_unsupported`，三者均没有新增数值通过结果。37 个参考已完成 229 组条件的 CPU 计算检查，wheel 已检查包含参考计算及摘要绑定。数值与打包临时产物已清理。

## 2026-11-27：Agent 扩充同类型加法

- 用 TileLang 为原 Dataset 向量加法生成 f16、bf16 和有符号 i32 三种源码特化，固定为单块长度 128。外部输入和元数据位于 `/home/wwq/Dataset/TianchenIR-TypeSpecializations`。Bench 独立验证存储类型、浮点边界及整数合法取值。
- 默认 API Agent 自动修改并通过 Bench 后保留三种加法规则。f16、bf16 各 6/6，i32 5/5。最终编译器复跑三者均无需调用 Agent。重新编译原 f32 加法、矩阵乘法和 f32 到 i32 转换分别通过 5/5、6/6、6/6。

## 2026-11-27：Agent 扩充 INT64 加法和 8 位转置

- 对原 Dataset 的向量加法和矩阵转置只特化存储类型，由 TileLang 生成 i64 加法和 i8、u8 转置三个 CUDA 输入，并记录来源、ABI、launch 和源码摘要。三者均从正常入口自动进入默认 Agent。
- i64 加法通过 5/5 组精确比较，覆盖超过 FP64 整数精确范围的输入。Bench 对 i8、u8 转置各使用四组输入，其中一组遍历所有 256 种字节值。首轮候选成功编译，Bench 的输出缓冲区却沿用了转置参考的内存步幅。将缓冲区改为连续行主序后，两次 Agent 重跑均通过 4/4，候选已保留。
- 最终编译器直接复跑 i64 加法、i8 转置、u8 转置，分别通过 5/5、4/4、4/4，均未调用 Agent。原 f32 转置以及 i32、f16、bf16 加法各通过 5/5、5/5、6/6、6/6。运行产物已清理，外部 Dataset 保留六个可重跑输入。

## 2026-11-28：FlashAttention 校准、旧样例复验与 CPU 执行

- 单 tile FlashAttention 由输入推导逐输出误差界。传播 FP32 点积、缩放、指数与归约区间，在权重端保留显式 FP16 舍入，再将分子、分母与最终写回的范围传到输出。种子 83 的已知差异落入该界，原有通用阈值照常适用。CUDA DSL 路径通过 6/6。
- 原 Dataset 尚待按新合同验证的 20 个样例均从正常入口重新编译，合计 100/100 组通过，无需调用 Agent。包含 matrix_add、row_max、row_sum 以及其余 17 个逐元素向量样例。
- Bench 增加原生 CPU 路径，使用 Clang 编译和执行生成 DSL，并实际执行原始 CUDA。两侧使用同一独立参考和阈值。初次系统默认 C++ 编译器不支持 `_Float16`，切换至 Clang 后复验。FP32 转置、FlashAttention、i64 加法、f16 加法和乘法、i8 及 u8 扩位乘法、尾部加法合计 43/43 组通过，无需调用 Agent。

## 2026-11-28：Agent 扩充乘法与双块尾部

- 外部 Dataset 新增五份 TileLang CUDA 输入。默认 Agent 保留 f16、bf16 乘法（各 6/6），i8、u8 扩至 i32 后的乘法（各 5/5），以及长度 130、两个 block 的带尾部 FP32 加法（5/5）。
- 新尾部阶段一度截住现有多块扫描。对真实 chunk_cumsum 重走默认流程，Agent 将共享扫描路由回既有结构检查，通过 6/6 并保留修正。回归重新生成 DSL，尾部加法、i8 和 u8 转置及 bf16 乘法分别通过 5/5、4/4、4/4、6/6。
- Bench 已登记 48 个输入快照。按新合同的 CUDA 执行记录已有 45 个不同样例通过。外部 Dataset 保留可重跑输入，实际执行的临时产物已清理。

## 当前能力与后续事项

- 已验证的逐元素范围为单个完整 block、静态一维或二维 f32 视图及唯一输出。真实向量长度为 128，矩阵尺寸为 16×16。多组条件覆盖数值输入变化。
- 混合精度覆盖 f16 数据的 f32 有序乘加，以及已验证注意力调度中的中途 f16 舍入和 f16 输出。同类型 f16、bf16 逐元素加法已实测，其他低精度运算和 FP64 仍待扩充。
- i32 已覆盖同类型加法、向零截断的 f32 到 i32 转换和配对选择中的整数索引输出。i64 加法、i8 和 u8 转置及扩位乘法已实测。窄整数写回、其他位宽和布尔参数继续按真实输入扩充。
- 多块扫描已覆盖独立扫描线和完整 32 lane 分段，卷积已覆盖所验证的多块静态 padding 和归约尾部。带尾部的扫描、跨块通信、多块注意力、因果遮罩、其他卷积调度、一般访存和动态形状仍待扩展。
- 静态检查、文档、接口和构建成功不计作数值一致性通过。不创建模拟 Kernel 或测试目录替代真实验证。
