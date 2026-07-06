# 张量级原子算子集

## 粒度原则(地基)

原子算子是**张量级结构化原语**(linalg 风格),不是标量、也不是库级大算子。

- **不是标量级**(`add`/`mul` 逐标量):否则 softmax 摊成几十个、FlashAttention 摊成几百个 op,LLM 生成质量崩、token 爆炸、验证困难。
- **不是库级大算子**(`softmax`/`attention` 当单个原子):否则其内部的 mask/bound/归约(B 层约束)被吞进算子里看不见,无法验证、无法重调度。
- **张量级刚好**:softmax = 几个原子;FlashAttention = 有限个原子(两个 contraction + 一个带状态 scan),不爆炸;且直接对应 MLIR 已有构造(linalg/vector)。

## 每个算子的硬要求(吸取旧仓库教训)

旧仓库把 112 个算子扁平堆放、`dtype_rules`/`shape_rules` 全空、还有注册了却不可执行的算子。本项目禁止重蹈:

1. **可执行 reference 语义**:每个算子必须有解释器可执行的参考实现,用于源码执行一致性验证。注册了不可执行的算子不允许存在。
2. **显式 shape / dtype 规则**:每个算子必须显式声明形状与类型推导规则,不留空。
3. **明确的层归属**:算子产出的信息属于 A 还是 B 还是 C,必须明确(见 [layers-abc.md](layers-abc.md))。

## 核心算子集(v1)

> 这是受控开放集:核心集封闭明确,新增算子必须走规范(给出语义 + shape/dtype 规则 + 层归属),不得随手加。

> **FlashAttention 所需子集已端到端验证(2026-06-25)。** `map.unary{exp}` /
> `map.binary`(含 broadcast)/ `reduce`(keepdims)/ `contraction` 与 `scf` 携带
> carry 的递推,都已通过真实 C++ dialect → linalg → gpu → sm_120a cubin 在
> 5090 D 上执行,且与解释器参考实现一致(execute==interpret,见
> [../verification/validation.md](../verification/validation.md))。要求 #1
> "每个算子必须有可执行 reference 语义"对这个子集已闭环验证。

**逐元素族**(Layer A,`parallel` 迭代)
- `map.unary` — 逐元素一元(exp / log / sqrt / abs / neg / recip / sigmoid / tanh / erf / ...)
- `map.binary` — 逐元素二元,带 broadcast(add / sub / mul / div / max / min / pow / cmp / ...)
- `select` — `where(cond, a, b)`,cond 是 tensor(数据依赖的"值",图形状不变)
- `cast` — dtype 转换

**归约族**(Layer A 语义 + Layer B 结合性)
- `reduce` — 沿轴归约,op ∈ {sum, max, min, prod, any, all};带**结合性标注** `associative` / `float_nondeterministic`(见下)
- `arg_reduce`(C6)— 沿轴归约到**极值的整数索引**(argmax / argmin),而非极值本身;结果是整数 tensor(轴折叠或 keepdims)。并列取**最低索引**(与 torch 一致)。lowering:block 归约携带 `(best_val, best_idx)` 对(register-tile 局部最优 + 共享内存 pair-tree),整数输出。见 `convert-intent-arg-reduce-to-gpu`。

**收缩族**(Layer A,`reduction` 轴 + `indexing maps`)
- `contraction` — einsum 风格的通用收缩(matmul / batched-matmul / dot / 带轴标注)

**扫描 / 递推族**(Layer A)
- `scan` — 沿轴的带 carry 递推,fn ∈ {sum, max, min}(cumsum / cummax / cummin / online-softmax 累加器)。max/min(C6)复用与 sum 相同的 Hillis-Steele block scan,仅换 combine + 单位元(-inf/+inf);一般 monoid 不可逆,exclusive 前缀取线程 `tx-1` 的 inclusive 值而非 sum 专用的 incl-segTotal 相减。复杂递推用结构化 `loop`,见 [structured-loop.md](structured-loop.md)

**数据移动 / 重排族**(Layer A,纯重排)
- `transpose` / `reshape` / `broadcast` / `slice` / `concat` / `pad`

**间接寻址族**(Layer A 语义,地址作为 tensor operand —— 数据依赖的"地址",图形状不变)
- `gather` — `gather(data, indices, axis)`
- `scatter` — `scatter(data, indices, updates, reduce ∈ {none, add, mul, max})`
- `segment_reduce` — `segment_reduce(data, segment_ids, op)`(数据依赖的分段归约)

**结构控制族**(Layer A 语义 + Layer C tiling)
- `loop` — 带状态携带的结构化循环,见 [structured-loop.md](structured-loop.md)

## 归约结合性标注

并发归约 / atomic 必须显式标注,而不是 scope-out:

- `associative` — 数学上结合(整数加、max/min):归约顺序不影响结果,可 bit-exact 验证。
- `float_nondeterministic` — 浮点加法等不结合:并发顺序影响数值结果。验证用**合理 tolerance**,而非要求 bit-exact(见 [../verification/dynamic-feedback.md](../verification/dynamic-feedback.md))。

## 不做

- 不引入库级 named op(softmax/attention 当单算子)——起步阶段禁止,留作未来标准库扩展(见 [representation-decisions.md](representation-decisions.md))。
- 不引入语义无定义、不可执行的占位算子。
