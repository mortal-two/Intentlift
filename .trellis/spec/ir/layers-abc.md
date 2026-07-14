# Layer A / B / C 三层表示

## 脊柱:从 SPMD 视角升到 whole-tensor 视角

kernel 源码是 **SPMD 单程序视角**(每个 program/thread 在干什么);提取就是把它升回 **whole-tensor 整张量视角**(这个算子对整个 tensor 算什么)。三层由此自动分出:

- 描述"整张量算子算什么" → **Layer A**
- 描述"使切分到 grid 上这件事合法的条件" → **Layer B**
- 描述"整张量算子如何切分到 grid 上" → **Layer C**

## Layer A:算法意图

整张量级的 tensor 计算:算子类、数据类型、符号 shape、轴角色,以及**会改变输出解释的 layout**。

- A 层是**可执行的张量程序**,是后续被源码执行一致性验证的语义对象。
- layout 只有在"改变张量轴如何被解释"时才属于 A(例如 matmul 读 `A[i,k]` 还是 `A[k,i]`);只为局部性的内存布局属于 C。
- A 层用**张量级原子算子**(见 [atomic-ops.md](atomic-ops.md)),必要时用带状态结构化循环(见 [structured-loop.md](structured-loop.md))。

## Layer B:可移植正确性结构

跨目标必须成立的结构:index 表达式、mask、边界、shape/layout 约束、依赖、同步、受控 atomic、**归约的结合性 / 顺序自由度**。

- B 层不是性能提示——后端可以改 schedule,但**必须保持 A 层计算 + 满足 B 层约束**。
- B 层使"把 A 层算子切分到并行硬件上"这件事合法。
- 持久化 schema 必须显式记录每个 access 的 `index_expression`、load mask/fill、
  store guard、effect 与 output mapping；每个 tiled reduction 必须记录 axis 及
  identity 或 masked contribution。禁止把这些信息留在无约束自由字典中。

## Layer C:非绑定调度提示

源实现里的性能知识:tile 大小、线程数、向量宽度、流水深度、循环展开因子、仅为局部性的内存布局。

- C 层是**非绑定**的:后端可拿来当重调度的先验,也可完全忽略。
- 改 C 层影响性能,但**不改变被提升计算的含义**。
- 在 SCRC Gate1 中，重写员只看到 C 槽，不看到绑定值；注入器随后为源与
  重放注入完全相同的 C。C 的语义地位仍是非绑定，但“同 C 编译”是结构
  比较成立的实验控制条件。

## π:受限重放计划，不是第四层 IR

SCRC 的 π 只保存系统规则卡 id 及对 J/B 标识的引用，用于把源实例的实现
机制交给源码隔离重写员。π 不是语义真值，也不是 Layer D：它不得含前端
API、源码表达式、自由 action/code/comment 或完整计算图。任何与计算含义或
边界正确性有关的事实都必须回到 A/B；规则卡是版本化系统资产。

完整约束见 [../verification/scrc-gate1.md](../verification/scrc-gate1.md)。

## 分层不变量

- **A + B 是绑定的**(验证它们);**C 是非绑定的**(后端可重调)。
- 验证检查 A 和 B;后端可以重调 C。

## 因式分解原则(关键)

一个源结构可能**同时**承载多层信息。此时必须**因式分解**成各层的份额,而不是二选一归类。

**示例:softmax**(单层张量代数)
- A:`exp` / `reduce-max` / `reduce-sum` / `div`(几个张量原子)
- B:行长被 block 覆盖、masked load、store guard
- C:block 大小、向量宽度、流水深度

**示例:FlashAttention**(分层的关键案例,必须支持)
- 它的 K 维循环**同时**是两件事:
  - **tiling**(把 K 切块流式处理)→ 属于 **C**
  - **online-softmax 递推**(running max / running sum / 输出累加器跨迭代携带)→ 属于 **A 的 scan**
- 正确动作:把这一个源循环**拆**成 `C 层的 tiling hint` + `A 层的递推语义`。
- 反例:把整个 K 循环简单归类成"schedule"会丢掉递推语义;归类成"semantics"会冻死 tiling、无法重调。

> 见 [structured-loop.md](structured-loop.md) 了解递推在 A 层如何表达。
