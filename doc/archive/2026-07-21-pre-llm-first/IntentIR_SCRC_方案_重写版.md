# IntentIR 验证方案:源条件重放闭包(SCRC)

> 本文档自成一体,不需要先读任何旧版方案。所有概念在第一次出现时都会解释。
> 全文用同一个例子(Triton 版 row-wise softmax)贯穿。

---

## 一、我们要解决什么问题

项目的大目标:拿到别人写好的 GPU Kernel(Triton / TileLang / CUDA 源码),
用 LLM 把它"提升"成一份高层语义描述——IntentIR JSON。这份 JSON 说清楚
Kernel **在算什么**(A 层:计算图、归约轴)和**边界行为是什么**(B 层:
mask、越界处理),但不关心**怎么实现**(C 层:tile 大小、warp 数,单独存)。

问题:**LLM 提升出的 JSON,凭什么相信它是对的?**
LLM 完全可能看着一个 softmax Kernel,漏掉"减 max"那一步,输出一份
不稳定 softmax 的 JSON。我们没有形式化工具能证明"JSON ≡ 源码",
因为两边都没有严格数学语义。

本方案(Gate1)就是在没有形式化证明的前提下,给出一个可操作、
可测量、可辩护的可信度检查;Gate2 再用实际执行做行为兜底。

**核心思路一句话(考试类比):**

> 检验一个学生是否真懂一篇文章,不是问他"你懂了吗",
> 而是把文章收走,让他凭笔记复述,再拿复述和原文对照。

对应到系统:把源码收走,让一个从没见过源码的 Agent 凭 JSON(+ 一份
实现笔记)把 Kernel 重新写出来;重写的 Kernel 和原 Kernel 喂给
**同一个编译器**,比较编译器"看到的东西"是否一致。一致,说明 JSON
真的装下了源程序的语义——我们称之为**重放闭合(replay closure)**。

---

## 二、系统总览:五个角色 + 一个考场

```text
              源 Kernel(固定不动,每一轮都和它比)
                    │
      ┌─────────────┴──────────────┐
      │                            │
      ▼                            ▼
 ① 同前端编译器               ② 提升官(LLM,能看源码)
      │                            │ 产出:J 语义档案
      ▼                            │       π 重放计划
 源指纹 F(S)                       │       C 调度参数
 (纯代码提取)                     │       定位表(附件)
                                   ▼
                     ③ 重写员(LLM,看不到源码)
                        只拿到 J + π + "调度有哪些空位"
                                   │
                                   ▼
                        参数化重放 Kernel Ŝ(空着调度)
                                   │
                     ④ 注入器(程序)把 C 填进去
                                   │
                                   ▼
                        同一个编译器 → 重放指纹 F(Ŝ)
                                   │
                                   ▼
                  ⑤ 裁判(纯代码):F(S) vs F(Ŝ) 逐项硬比对
                        │
              ┌─────────┴──────────┐
           全部通过             有不一致
              │                    │
              ▼                    ▼
        反作弊终检          ⑥ 诊断员(LLM):写差异报告
              │               (错在哪、疑似谁的错、允许改哪)
              ▼                    │
       REPLAY_CLOSED          差异回流 Lift,进入全新一轮
              │
              ▼
 ═════════ Gate2 考场(动态)═════════
   三方执行对答案:源 / 重放 / 生产Lowering
```

角色分工的**总原则:裁决权归代码,理解权归 LLM。**

| 角色 | 是谁 | 能看什么 | 职责 |
|---|---|---|---|
| 提升官(Lift Agent) | LLM | 源码、编译器视图摘要、历史差异报告 | 产出 J、π、C、定位表 |
| 重写员(Replay Agent) | LLM | 只有 J、π、调度空位表、规则卡 | 重写同前端 Kernel |
| 注入器 | 程序 | C 的具体值 | 把调度值填进重放模板 |
| 裁判(Comparator Core) | **纯代码** | 两份语义指纹 | 逐项硬比对,判过/不过;LLM 无权推翻 |

> v8 实现说明：这里的“提升官/Lift Agent”按完整 Lift 过程理解。source-aware LLM
> 负责 J、C、接口和定位 core；同一次 Lift 内的 deterministic finalizer 根据
> J/interface/C/RuleRegistry 生成 π。对 Gate1 而言，Lift 的最终产物仍是完整
> `J + π + C + 接口 + 定位表`，π 没有被移到 Gate1 或独立外部流程。
| 诊断员(Diagnostician) | LLM | 裁判输出 + 定位表 + 源视图 | 只解释和归属,**无裁决权** |

为什么裁判必须是代码:Gate 的裁决必须确定、可重跑、不会看漏。
而"重放的 IR 里少了一个 reduce(max)"这种判断,数数就能做,
代码百分之百可靠,LLM 只是大概率可靠。诊断员判错的代价只是修复
方向走弯路,不会造成错误 lifting 被放行——所以那部分才交给 LLM。

**Gate1 全程不执行任何代码**(只生成、编译、比指纹);
执行发生在 Gate2,且只有过了 Gate1 的才进 Gate2。

---

## 三、提升官产出的四样东西

### 3.1 J:语义档案(就是 IntentIR 的 A+B)

用 softmax 举例:

```text
Layer A(算什么):
  m = reduce_max(X, axis=N)
  e = exp(X - m)
  s = reduce_sum(e, axis=N)
  Y = e / s

Layer B(边界规则):
  合法域:n < N
  max 的越界填充值:-inf
  sum 的越界贡献:0
  只写回合法的 n
```

J 拥有:算子种类、操作数与数据流、归约函数与轴、shape/dtype、
mask/bounds、broadcast、别名与副作用、输出映射。
J 不拥有:tile 大小、warp 数、前端 API 写法、源码文本。

### 3.2 π:重放计划(源程序的"实现笔记")

π 回答一个问题:**这个源程序用了当前前端的哪些机制来实现 J?**
它是给重写员看的施工说明,不是第二份语义。softmax 的 π:

```text
1. 用"程序域绑定"机制,把并行实例绑到 J 的行轴
2. 用"带掩码的 tile 加载"机制,按 J 的合法域物化一行
3. 用"前端原生归约"机制,实现 J 里的 reduce_max 节点
4. 用"逐元素映射"机制,实现 J 里的 sub 和 exp 节点
5. 用"前端原生归约"机制,实现 J 里的 reduce_sum 节点
6. 用"带掩码的存储"机制,按 J 的合法域写回输出
```

**π 的三条铁律(防作弊的第一道结构约束):**

**铁律一:只许引用,不许复述。** π 里所有语义信息必须写成"J 的第几个
节点"这样的引用,不许自己写出内容。

```json
// 错误:把 softmax、行、轴全偷藏进了 π
{ "action": "emit_row_softmax_axis_1" }

// 正确:语义全部指回 J
{ "rule": "realize_reduce",
  "node_ref": "J.layer_a.node[2]",
  "mechanism": "native_reduce" }
```

**铁律二:禁止出现任何前端 API 的具体名称。** 不许写 `tl.max`——
这个字符串本身就说出了"这是 max 归约",遮盖 J 之后重写员照样能猜出
语义。π 里的 `mechanism` 只允许一个语义无关的词汇表:
`native_reduce`、`masked_tile_load`、`masked_tile_store`、
`program_domain_bind`、`elementwise_map`、`native_dot`……
具体用 tl.max 还是 tl.sum,由系统在生成时拿 `(mechanism, J 里的 fn)`
去查规则卡(见 3.5)解析。

**铁律三:禁止源码文本、完整表达式、完整计算图。** 并设长度与
字面量预算,超限直接拒绝。

**标识符同样属于通道预算。** Lift 内部和定位表可以保存便于诊断的名字,但正常
replay handoff 里的 J/B/π 引用必须统一改写为 declaration-order opaque handle
(`n0/t0/d0/a0/v0/p0/c0`)。例如 `a_load`、`reduce_max_0`、`row_domain` 均不得交给
重写员。ABI/C handle 只通过不包含原名的 JSON path 绑定到 interface parameter 或
schedule slot；重写员在目标对象中读取真正需要输出的 ABI 名称。

### 3.3 C:调度参数

```text
BLOCK_N = 128, num_warps = 4, num_stages = 2, ...
```

重写员**只知道"存在哪些调度空位"**(有一个叫 BLOCK_N 的 constexpr、
有一个叫 num_warps 的 launch 参数),**不知道具体值**。它生成一个
留空的参数化 Kernel,再由注入器把 C 的值填进去。这样 C 这个通道
没有机会偷带语义。注意:源和重放必须用**同一份 C** 编译,两边指纹
才可比(constexpr 会改变 IR 里的张量形状)。

### 3.4 定位表(附件,不属于 π)

一张"源码行号 ↔ J 节点"的对照表,提升官顺手产出。例如:
"源码第 12~13 行(tl.max 与减法)对应 J 的 reduce_max 节点和 sub 节点"。

它的唯一用途是帮**诊断员**把差异翻译成人能定位的位置。三条纪律:
永远不给重写员看;裁判不依赖它(裁判只看指纹);诊断员把它当提示
而非真值——因为它是提升官写的,本身可能就错。

> 旧版方案把这张表塞在 π 内部(叫 π.anchor),要求"只给比较器可见",
> 结构绕且有泄漏风险;现在独立成附件,归属清晰。

### 3.5 规则卡(系统资产,不是提升官产出)

每个前端预先注册一个规则库,每条机制一张卡:

```yaml
id: realize_reduce
适用的 J 算子: [reduce]
从 J 读取: [fn, axes, operand]
前端构造映射: { max: tl.max, sum: tl.sum, min: tl.min }   # API 名只住在这里
最小合法示例: ...
常见错误: [broadcast 错误, 掩码恒等元填错]
```

工作方式是"先计划,再检索,再生成":提升官只输出用了哪些机制,
系统按 π 检索出对应的几张卡给重写员——既不用把整个规则库塞进
prompt,也不用每条机制单独调用一次模型。

### 3.6 信息所有权总表

| 信息 | 唯一的家 |
|---|---|
| 算子、数据流、归约轴/函数 | J.A |
| mask、bounds、别名、副作用 | J.B |
| 实现机制(语义无关词汇) | π |
| 前端 API 具体名称 | 规则卡 |
| 源码行 ↔ J 节点对照 | 定位表(只给诊断员) |
| tile/warps/stages 的值 | C |
| 源码文本 | 重写员永远不可见 |

---

## 四、裁判怎么判:语义指纹

### 4.1 什么是语义指纹

不比源码文本(写法风格全是噪音),不比 PTX(优化太深,等价代码也
长得不同),而是比**编译器中间层"看到的东西"的一张固定格式摘要表**:

```text
语义指纹 F = {
  接口:     参数列表、dtype、rank、输入/输出/inout、constexpr、别名声明
  算子多重集: {masked_load:1, reduce_max:1, sub:1, exp:1,
              reduce_sum:1, div:1, masked_store:1}
  归约清单:  [(max, axis=…, 填充=-inf), (sum, axis=…, 填充=0)]
  数据流边:  load→max→sub→exp→sum→div→store
  谓词:     规范化后的 mask 条件(如 offs < N)
  效应:     带掩码读/写、atomic、同步 的计数
  调度视图:  block 形状、warp 布局(只记录,不裁决)
}
```

这张表由**纯代码**从编译器 dump 的 IR 里机械提取(各前端怎么提取
见第五章)。提取器本身用变异测试做单元测试(见第十二章)。

### 4.2 判定标准

| 项 | 标准 | 性质 |
|---|---|---|
| 接口 | 逐项精确相等 | 硬:不过即 FAIL |
| 算子多重集 + 数据流边 + 归约清单(A) | **规范化后精确相等** | 硬 |
| 谓词 + 效应(B) | 效应计数相等;谓词允许有界改写(offs<N ≡ N>offs) | 硬 |
| 调度视图(C) | 只报告差异 | 软:不影响裁决 |
| PTX 规范化后逐字节相同 | 记 bonus 标记 | 意外之喜,非必需 |

A 为什么用精确匹配而不是"差异小于某阈值":比对前先做规范化
(常量折叠、交换律算子操作数排序、公共子表达式合并、抵消的纯
layout 变换消除、dtype 别名归一),规范化已经吸收了合法的写法差异;
剩下的 A 图很小(一个 Kernel 十来个语义算子),精确匹配完全可达,
而且"精确"在论文里比"阈值以内"好辩护得多。

### 4.3 softmax 上裁判的一次工作

```text
源指纹的算子多重集:   {masked_load, reduce_max, sub, exp, reduce_sum, div, masked_store}
重放指纹的算子多重集: {masked_load,                  exp, reduce_sum, div, masked_store}

裁判输出:缺 reduce_max、缺 sub;归约清单少一条(max, 填充=-inf)。
判定:A 项 FAIL。结束。诊断员接手解释(见第六章)。
```

发现"少了一个算子"不需要智能,数数就行——这就是裁决权必须归代码的原因。

---

## 五、每个前端从哪里填指纹表

三个前端共用同一张指纹表和同一个裁判,差别只在**从编译链的哪一层、
用什么方式提取**。新增一个前端 = 写一个薄的提取适配器,不是重写 Gate。

### 5.1 Triton(论文主线,裁决强度最高)

编译链:Triton → **TTIR → TTGIR** → LLVM IR → PTX。提取层:TTIR。

TTIR 是张量算子级 IR,和 J 的 A/B 几乎一一对应:`tt.reduce` 自带
函数和轴,`tt.load/tt.store` 自带 mask 和填充值。文本规整,
约 200 行脚本(或 MLIR Python API 遍历)即可填满指纹表。
获取方式:`TRITON_KERNEL_DUMP=1` 或 `triton.compile()` 后读 asm 字典。
C 视图从 TTGIR 取 layout/warp 信息。

### 5.2 TileLang(第二阶段,粒度稍粗但全是读字段)

编译链:TileLang DSL → **TVM TIR** → 生成 CUDA 源 → nvcc → PTX。提取层:TIR。

两个事实让提取比想象中容易:

1. **TIR 不是文本,是 Python 对象树。** 用 TVM 官方遍历 API
   (`post_order_visit`)直接读节点字段,零 parser。
2. **归约轴是显式标注的。** TIR block 的每个迭代变量自带类型标记,
   归约轴是一个直接可读的字段——别处要模式识别的事,这里是读字段。

填表方式:接口读 PrimFunc 签名;算子多重集收集 Call 节点的
intrinsic;归约读 block 迭代变量类型 + combine 算子 + init 值;
数据流读 block 的 reads/writes 区域;谓词收集 if_then_else 条件;
C 视图记录 shared copy 与线程绑定。

注意:TIR 会被 lowering pass 逐步改写,**dump 点必须固定在早期
stage**(高层结构尚在),并把 pass 位置与 TVM 版本 pin 进缓存 key。
粒度是 loop/buffer 级,提取器要把 block+loop 折叠成算子摘要,
但每一步都是读注解,没有反向工程。

### 5.3 CUDA(降级档:接口硬检查 + 必要条件指纹,语义裁决交给 Gate2)

编译链:CUDA → Clang AST → LLVM IR → PTX。**每一层都是标量/线程级,
不存在张量算子概念。** 从 LLVM IR 的 phi 节点和循环回边还原
"这是 axis=1 的 max 归约"等于写反编译器——明确不做。

纯代码能做的:

```text
① 接口硬检查(有否决权):
   clang -ast-dump=json → 参数类型/顺序/const/__restrict__、模板实参
   源码扫描 → shared memory 声明与大小、__launch_bounds__
   加上 specialization 里的 launch 配置,逐项精确比对

② 指令多重集指纹(必要条件):
   数学指令集合:expf→ex2.approx、fmaxf→max.f32(映射稳定)
   bar.sync 计数;atomic;谓词化 store 的存在性(mask 的 PTX 痕迹)
   ptxas -v → 寄存器数、shared memory 字节数
   用法:源 softmax 指纹含 max+sub+ex2+div,重放缺 max → FAIL
        源 2 个 barrier、重放 0 个 → FAIL
```

诚实的局限:多重集抓得住"缺算子、缺同步、缺 mask",抓不住
数据流接错(a/b 与 b/a 的指令多重集相同)。所以 CUDA 的语义
充分性裁决**移交 Gate2 三方执行**(第九章),反作弊消融也改用
行为版(第八章)。论文中如实声明此 scope。

### 5.4 分级汇总

| 前端 | 提取层 | 提取方式 | 裁决强度 |
|---|---|---|---|
| Triton | TTIR(张量算子级) | 规整文本脚本 / MLIR API | 完整 A/B 结构闭合 |
| TileLang | TVM TIR(loop 级,归约轴显式标注) | 官方 AST visitor 读字段 | A/B 结构闭合,粒度稍粗 |
| CUDA | Clang AST + PTX 统计(标量级) | ast-dump + ptxas | 接口硬检查 + 必要条件;语义靠 Gate2 |

---

## 六、比对失败之后:差异报告

裁判只负责判"过/不过"并列出不一致项;把不一致翻译成"该谁改、
怎么改"是诊断员的工作,产物叫**差异报告**——一份机器可读的清单,
固定四栏:

```json
{
  "不一致项": [
    "算子多重集缺 reduce_max、缺 sub",
    "归约清单缺 (max, 填充=-inf)"
  ],
  "疑似责任方": "重放计划",        // intent | 重放计划 | 接口/调度
  "已确认无误的字段": ["接口", "J 的 exp/sum/div 节点"],
  "允许 Lift 修改的位置": ["J", "π", "定位表"]
}
```

责任方怎么判?诊断员对照三方关系(J 声明了什么、源指纹里有什么、
重放指纹里有什么),按下表归属:

| J 里有吗 | 源指纹里有吗 | 重放指纹里有吗 | 结论 | 修谁 |
|---|---|---|---|---|
| 有 | 有 | 没有 | 当前 J/π 合同未让 writer 完整重放 | 差异回流 Lift，重产 J/π |
| 没有 | 有 | 没有 | 提升时就漏了 | 差异回流 Lift，修 J + 依赖它的 π |
| 有 | 没有 | 有 | J 多编了源里没有的东西 | 差异回流 Lift，修 J/π |
| — | — | 只有调度不同 | 可接受 | 不修；C 仍冻结 |

"已确认无误的字段"和"允许修改的位置"这两栏是给下一轮的护栏:
防止模型把已经对的部分改坏,也防止它大面积重写。

---

## 七、修复循环:只回流 Lift，禁止原地修 replay

当前主线只有一个 source-aware 外循环。无论失败发生在 replay 生成、前端编译、
接口检查还是 TTIR 硬指纹比较，都不得把旧 replay code、编译反馈或 source/replay
差分交回重写员做局部代码修补。原因是这种修补只能提高某个候选代码的通过率，
不能证明 Lift 生成的 J/π 本身已经足以指导一次全新重放。

正确恢复方式是：提取确定性的失败类别/差异报告，与原始 source 一起交回提升官；
提升官重新检查并产出完整 J、π、定位表，interface/C 冻结；随后从静态泄漏扫描、
源编译、全新 source-blind replay、重放编译和指纹比较开始重新执行。旧 replay 只
保留在审计历史中，永远不是下一轮 writer 的输入。

Lift 自己在输出 JSON 不合法时可以在同一次 Lift 调用中修复 artifact；这仍是
source-aware Lift 修复，不是 replay-code repair。provider 对同一请求的 HTTP 重试
也只是传输容错，不属于语义修复。

主流程伪代码:

```python
def run_gate1(spec):
    源指纹 = 编译并提取指纹(spec)                 # 纯代码,缓存
    历史报告, 最优候选 = [], None

    for Lift轮次 in range(最多5轮):
        J, π, C, 定位表 = 提升官(源码, 源指纹摘要, 历史报告)
        静态泄漏扫描(J, π)                        # 每轮必跑,见第八章

        模板 = 重写员(J, π, 调度空位, 检索规则卡(π))  # 全新生成，无旧 replay
        重放源 = 注入器(模板, C)
        结果 = 前端编译(重放源)
        if 生成失败 or 编译失败:
            报告 = 结构化失败类别(生成/编译/接口)
            历史报告.append(报告)
            continue                                # 回到 Lift 起点

        重放指纹 = 提取指纹(结果)
        裁决 = 裁判(源指纹, 重放指纹, J)          # 纯代码

        if 裁决.硬项全过:
            if 终检消融通过(J, π, C):             # 第八章第②档
                return REPLAY_CLOSED
        报告 = 诊断员(裁决, J, 定位表)
        历史报告.append(报告)
        最优候选 = 保留更优(最优候选, 裁决)       # 仅供预算耗尽后诊断/Gate2对照

    return NO_CLOSURE(最优候选)
```

每轮 artifact 都独立裁决，不能把上一轮 replay code 拼进下一轮。最优候选仍按
编译 > 接口 > A > B > 泄漏 > C 的字典序保存，但只用于预算耗尽后的诊断和离线
Gate2 对照，不能越过当前 Gate1 verdict。

接受条件(全部满足才 REPLAY_CLOSED):

```text
重放可编译
接口精确相等
A 指纹规范化后精确相等
B 无关键不一致
每轮静态泄漏扫描通过 且 masked-J/actual-π 终检出现 compile rejection 或硬指纹差异
连续两轮无新增关键不一致
```

---

## 八、反作弊:怎么证明语义真的住在 J 里

为什么要防:这个方案存在三条作弊通道——π 偷带语义(把"softmax"
写进计划,J 成空壳)、C 偷带语义、重写员偷看源码。源隔离和 π 铁律
挡住了大部分,剩下的用消融实验证明。按成本分三档执行:

**第①档:静态扫描,每轮跑(便宜)。**
检查 π 是否只用引用不写内容、字面量是否超预算、是否含源码 n-gram、
是否出现前端 API 名称。纯字符串/schema 检查。

**第②档:遮盖消融,最终接受前必跑一次。**
把 J 的 A/B 内容遮盖掉,只给 π,让重写员重放:

```text
正常:  重写(J, π, C)          → 应当闭合
遮盖:  重写(遮盖(J), π, C)    → 应当崩掉
```

如果没有 J、光凭 π 还能重构出接近源的东西,说明 π 偷藏了语义,
这次提升作废。两个执行细节决定这个测试是否有效:

- **名字始终要遮。** 正常 replay 已先对 J/B/π 做一致的不透明重命名
  (`reduce_max_0 → n3`),让引用只剩 declaration order 与拓扑信息；masked-J
  复用同一原则并进一步遮盖 A/B 内容。不能只在消融时才隐藏名字。
- **裁决必须确定。** masked writer 必须先形成通过源码与 opaque ABI 校验的可审计
  候选。该合法候选在同一前端编译期被拒绝，或编译后
  `op_multiset/dataflow/reduction/predicate_effect` 任一硬分量与 source 不同，均
  表示“无 J 时确定性退化”；全部语义硬分量仍闭合则判泄漏。没有形成 generation、
  ABI 恢复失败或指纹工具失败只能记为 inconclusive，不能把 provider/基础设施失败
  冒充成 π 无泄漏；score/ratio 不进入接受条件。

为什么放在最终而不是只在第一轮:**迭代修复恰恰是泄漏产生的时刻**——
每轮差异报告都在诱导模型把修正塞进 π(改 π 比改 J 更容易过下一轮),
所以必须在收敛后的最终版 (J, π) 上验一次。

**第③档:完整套件,进 CI 和论文实验(贵)。**

在当前 π 仍处于资格验证和规则扩展阶段时，正式 live/held-out 证据默认在最终完整
π 控制组 `REPLAY_CLOSED` 后执行一次完整套件；只有 quick smoke 和纯单元测试允许
显式 `--no-pi-ablations`。消融失败不得触发 relift，也不得修补消融 replay：生成/
编译失败只记为 inconclusive。即使 π 协议成熟，schema/rule/prompt 变化和分层
held-out 样本仍必须持续做消融审计，不能因为“π 已成型”永久取消。

| 消融 | 做法 | 证明什么 |
|---|---|---|
| 只给 J 不给 π | 重写(J, ∅, 任意C) | A/B 应尚可、但实现结构与源的相似度应明显下降 → π 确实携带实现知识 |
| 换别人的 π | 重写(J, 别的Kernel的π, C) | 应劣于本 Kernel 的 π → π 含源实例特定信息,不是空话 |
| 随机删 π 一步 | step dropout | 应造成对应位置的闭合下降 → π 每步有信息量 |
| 换一组合法 C | 重写(J, π, C') | A/B 指纹应不变 → C 没有替代语义 |

**CUDA 特例:** 结构指纹弱,消融改用行为版——遮盖 J 后重放的
Gate2 执行失败率必须显著上升,以执行结果代替结构距离。

---

## 九、Gate2:三方执行对答案

### 9.1 定位

Gate1 全程静态(只编译、比指纹,不跑代码);**只有 REPLAY_CLOSED
的 artifact 进入 Gate2**。重放 Kernel Ŝ 是 Gate1 造出来的文件,
到 Gate2 才第一次被执行——"造"在 Gate1,"跑"在 Gate2。
(唯一例外:论文评估需要对一部分被 Gate1 拒绝的样本也离线跑
Gate2,用于 9.4 的对照指标;那是实验,不是生产流水线。)

### 9.2 三方是哪三方

```text
O_src = 原 Kernel 在 GPU 上执行的输出
O_rep = 重放 Kernel Ŝ 在 GPU 上执行的输出(Gate1 产物,零额外生成成本)
O_low = J 走生产 Lowering(Intent MLIR → Linalg → 标量 → CPU LLVM JIT)执行的输出
        (复用项目现有链;IntentIR 的 Python 解释器可作第四参照)
```

为什么三方而不是两方:两两比较的一致/偏离模式可以**机械地**判定
错误归属:

| 观察(容差内) | 结论 | 动作 |
|---|---|---|
| 三方一致 | 全链正确 | GATE2_PASS |
| 源≈重放,Lowering 偏离 | 生产 Lowering 有 bug | 修 emitter/pass,**不动 J** |
| 源≈Lowering,重放偏离 | J 对,重放/π 错 | 分析 Gate1 为何没抓到 |
| 重放≈Lowering,源偏离 | **J 错但错得自洽** | Gate1 的理论盲区被抓到:回流重提,收进 hard case 集 |

第四行值得强调:J 和 π 一起错得自洽(两个 LLM 犯了同一个理解错误)
是 Gate1 结构上防不住的唯一情形,三方差分是系统性捕获它的唯一手段。
这使 Gate2 不只是工程回归,而是**对 Gate1 假设的实证审计**。

### 9.3 输入怎么生成(三个维度缺一个就有整类错误测不到)

```text
形状:  N ∈ {1, BLOCK−1, BLOCK, BLOCK+1, 非整除值, 远大于BLOCK}
       —— mask 错误只在非整除形状下触发,N 整除时测试通过是假象
数值:  正态/均匀之外,必须含大动态范围与 ±inf/NaN
       —— B 层声明了"填充值=-inf"却不测 inf 输入,等于没验证这条声明;
          漏掉数值稳定化的 softmax 只在大值输入下露馅
种子:  每组形状至少 3 个随机种子
```

比较一律容差制(按 dtype 定 rtol/atol,归约类容差随规模放宽),
**禁止逐位比较**——GPU 树形归约与 CPU 顺序归约的浮点结果位级必然不同。
全部 形状 × 数值分布 × 种子 组合通过才算 GATE2_PASS。

### 9.4 指标必须成对报告

```text
P(Gate2 通过 | Gate1 = REPLAY_CLOSED)   —— 精确率
P(Gate2 通过 | Gate1 = NO_CLOSURE)      —— 对照
```

只报第一个没有意义:如果所有 Kernel 的 Gate2 基线通过率本来就是
95%,Gate1 等于没贡献。两个条件概率拉开差距,才证明 Gate1 有区分度。

---

## 十、这套方案能宣称什么

**能:**

> 在重写员无法访问源码、π 经不透明化消融证明未偷带 A/B 语义、
> C 与 A/B 分离的条件下,J 与源特定重放计划足以重构一个实现,
> 该实现经同一前端编译器提取的语义指纹与源程序精确闭合
> (接口与 A 精确、B 有界等价);差异可经责任归属定向修复而收敛;
> 闭合的 artifact 进一步在源/重放/生产 Lowering 的三方差分执行上
> 与源行为一致。

**不能:** ∀x, S(x)=J(x) 的形式化等价;浮点、atomic、并发、动态
shape 的全域行为等价。对外术语用 replay-closed lifting /
source-conditioned replay validation,不用 formal equivalence proof。

---

## 十一、相关工作与我们的位置

**形式化 verified lifting 线:** STNG(PLDI'16)、Casper(SIGMOD'18,
arXiv:1801.09802)、Dexter(SIGGRAPH Asia'19)、C2TACO(GPCE'23)、
Tenspiler(ECOOP'24, arXiv:2404.18249)、**LLMLift(NeurIPS'24,
arXiv:2406.03003)**。LLMLift 是最近邻(LLM 提升 + LLM 生成形式化
证明),审稿人必问"为何不生成证明";答案引 C2TACO 原文:即使简单
benchmark,SMT 验证在 10×10×10 张量、1 小时超时内也无法完成——
GPU Kernel 的 mask、online softmax、并行归约远超定理证明器实际能力。
我们的取舍:以编译器指纹闭合换 soundness,换取覆盖真实 GPU Kernel。

**round-trip 线:** Meta LLM Compiler(arXiv:2407.02524)用
"IR 编译回汇编逐字节全等"作反汇编正确性下界,exact match 仅 14%——
佐证我们不把 PTX 全等作必要条件、只作 bonus。SCRC 与朴素 round-trip
的三点差异:盲重放 + 信息分区(可测量地证明语义住在 J 里);闭合
判据选在编译器语义可见的中层 IR;结构化差异报告驱动的定向修复。

**执行验证线(领域基线):** KernelBench(arXiv:2502.10517)、
TritonBench(arXiv:2502.14752)、Robust Agentic CUDA Benchmarking
(arXiv:2509.14279)、ProofWright(arXiv:2511.12294)。执行验证是
领域默认做法;SCRC 的增量 = 结构闭合 + 语义归属证明 + 定向修复,
Gate2 三方差分吸收其成熟实践。投稿前需再做一轮系统检索。

---

## 十二、实验设计与指标

**研究问题:**

- RQ1 SCRC 是否优于 one-shot lifting?(对比:裸提升 / 提升+自由重放 /
  提升+π但不隔离 / 完整 SCRC)
- RQ2 π 是否偷带语义?(第八章消融套件,全部在不透明重命名下执行,
  报告 masked hard-closure / hard-mismatch / inconclusive 三类结果；不以比值裁决)
- RQ3 C 后注入是否必要?(重写员直接看 C vs 空位后注入)
- RQ4 结构化差异报告是否优于"请重试"?(自由文本错误 / 原始编译器
  报错 / 差异报告 / 差异报告+责任归属)
- RQ5 规则供给方式的成本效果?(全库进 prompt / 每条规则一次调用 /
  计划-检索-生成 / 三段 checkpoint)
- RQ6 Gate1 闭合是否预测 Gate2 行为?(9.4 的成对条件概率)
- RQ7 确定性提取器是否可靠?(变异集上的检出率;诊断员归属准确率)

**变异集(一鱼两吃):** 对 J 注入错误(错轴、sum↔max、operand
互换、删稳定化、错 broadcast、错输出映射、错 dtype)、对 π 注入
错误(漏 masked store、错归约机制、错域映射、删步骤)、对 C 注入
扰动。它既测试系统的错误定位能力,又充当**指纹提取器的单元测试**:
每个 J 变异必须被裁判确定性检出。

**关键指标:**

| 指标 | 含义 |
|---|---|
| 重放可编译率 | 重写员产出可编译代码的比例 |
| 闭合率 | 达到 REPLAY_CLOSED 的 Kernel 比例 |
| G2 精确率 / G2 对照 | 9.4 的两个条件概率 |
| 平均 Lift/relift 轮数 | 修复成本 |
| 归属准确率 | 差异报告把错误判给正确责任方的比例 |
| 提取器检出率 | 变异集上确定性指纹的检出率 |
| masked-J 终检结果 | actual-π 遮盖消融的 hard closure、hard mismatch 与 inconclusive |
| Agent 成本 | 每 Kernel 的 token 与延迟 |
| held-out 泛化 | 未参与开发的 Kernel 上的闭合率 |

held-out 集要求:未参与 prompt 开发,且来自规则卡编写时未看过的仓库。

**成本预算参考:**

| Kernel 类型 | Lift/relift 调用 | 正常重放调用 | 典型总调用(不含最终消融) |
|---|---:|---:|---:|
| elementwise / copy | 1~2 | 1~2 | 2~4 |
| reduce / matmul | 1~3 | 1~3 | 2~6 |
| fused / attention(三段 checkpoint) | 1~5 | 1~5 | 2~10 |

必缓存:源指纹(key = 源hash+specialization+编译器版本,TileLang 另含
dump stage)、规则检索、重放编译、最优候选。SCRC 是提升/认证阶段的
一次性成本,不进 Kernel 运行的 fast path。

---

## 十三、施工顺序

```text
Phase 0  基础:J schema 收紧;π schema + 机制词汇表 + 规则卡格式;
         定位表格式;artifact 与缓存格式
Phase 1  Triton MVP:TTIR 指纹提取器 + 裁判 + 诊断员;10~15 张规则卡;
         elementwise/reduce/softmax/matmul 共 20~40 个 Kernel;
         relift-only 单外循环跑通
Phase 2  反作弊:静态扫描;不透明重命名 + actual-π masked-J 硬指纹终检;
         变异集(兼提取器单测)
Phase 3  Gate2:输入生成器(边界形状+对抗值+种子);容差比较;
         打通 J→标量→CPU JIT 链;三方差分与成对条件概率评估
Phase 4  TileLang:TIR visitor 提取器;固定 dump stage;
         复用裁判与循环
Phase 5  CUDA(降级档):Clang AST 接口检查 + PTX 指纹;
         行为版消融;语义裁决接入 Gate2
Phase 6  规模化:数百 Kernel 批处理、缓存、失败分诊、
         held-out 评估、论文实验
```

---

## 十四、一句话总结

> SCRC 把源 Kernel 因子化为语义档案 J、实现笔记 π 和调度参数 C;
> 让看不到源码的重写员凭 J+π 重写 Kernel;源与重放经同一编译器
> 提取语义指纹,由纯代码裁决是否闭合,由 LLM 诊断错误归属,
> 把所有正常 replay 失败统一回流 source-aware Lift,重产 J/π 并从头重放直至闭合;
> 再用不透明化的
> 遮盖消融量化证明语义确实住在 J 里而不是 π 里;最后由三方差分
> 执行做行为审计。**裁决链上没有 LLM,理解链上全是 LLM。**
