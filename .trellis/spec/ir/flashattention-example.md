# Worked Example:FlashAttention 的 IntentIR

把 FlashAttention forward 从头走成 IntentIR JSON,验证"具体形态"能承载旗舰用例。下面是**带注解的示意 JSON**(真实 JSON 无 `//` 注释;注解只为讲解三层与因式分解)。

```jsonc
{
  "intent_function": {
    "name": "flash_attention_fwd",
    "inputs": [
      {"name": "Q", "dtype": "f16", "shape": ["B","H","Sq","D"]},
      {"name": "K", "dtype": "f16", "shape": ["B","H","Sk","D"]},
      {"name": "V", "dtype": "f16", "shape": ["B","H","Sk","D"]}
    ],
    "outputs": [{"name": "O", "dtype": "f16", "shape": ["B","H","Sq","D"]}],

    // ───── Layer C:非绑定 schedule(后端可重调)─────
    "layer_c": { "tile": {"Sk": "BK", "Sq": "BQ"}, "vector_width": 128 },

    "body": [
      // ───── 主体是一条带状态的 loop,沿 Sk 流式扫 K/V ─────
      { "op": "loop", "result": ["m","l","Oacc"],
        "carry": [                                  // Layer A:递推状态
          {"name": "m",    "init": "NEG_INF"},      // running max
          {"name": "l",    "init": "0"},            // running sum
          {"name": "Oacc", "init": "zeros"}         // 输出累加器
        ],
        "trip": { "axis": "Sk", "tile_hint": "BK" },// axis=被扫轴(语义);tile_hint=BK 属 Layer C
        "body": [
          {"op":"contraction","result":"S","operands":["Q","Kj"],
           "attrs":{"subscripts":"bhqd,bhkd->bhqk"},               // S = Q @ Kj^T
           "layer_b":{"bounds":["k < Sk"]}},                        // Layer B:K 块边界
          {"op":"reduce","result":"m_blk","operands":["S"],
           "attrs":{"fn":"max","axes":["k"],"assoc":"associative"}},
          {"op":"map.binary","result":"m_new","operands":["m","m_blk"],
           "attrs":{"fn":"max"}},
          {"op":"map.binary","result":"Sc","operands":["S","m_new"],
           "attrs":{"fn":"sub","broadcast":"row"}},
          {"op":"map.unary","result":"P","operands":["Sc"],
           "attrs":{"fn":"exp"}},                                   // P = exp(S - m_new)
          {"op":"map.binary","result":"corr","operands":["m","m_new"],
           "attrs":{"fn":"sub"}},
          {"op":"map.unary","result":"alpha","operands":["corr"],
           "attrs":{"fn":"exp"}},                                   // alpha = exp(m - m_new)
          {"op":"reduce","result":"p_sum","operands":["P"],
           "attrs":{"fn":"sum","axes":["k"],
                    "assoc":"float_nondeterministic"}},             // Layer B:浮点归约,tolerance 验证
          {"op":"map.binary","result":"l_scaled","operands":["l","alpha"],
           "attrs":{"fn":"mul"}},
          {"op":"map.binary","result":"l_new","operands":["l_scaled","p_sum"],
           "attrs":{"fn":"add"}},                                   // l_new = l*alpha + rowsum(P)
          {"op":"contraction","result":"PV","operands":["P","Vj"],
           "attrs":{"subscripts":"bhqk,bhkd->bhqd"}},               // P @ Vj
          {"op":"map.binary","result":"O_scaled","operands":["Oacc","alpha"],
           "attrs":{"fn":"mul","broadcast":"row"}},
          {"op":"map.binary","result":"O_new","operands":["O_scaled","PV"],
           "attrs":{"fn":"add"}}                                    // O_new = Oacc*alpha + P@Vj
        ],
        "yield": ["m_new","l_new","O_new"]
      },
      // ───── 收尾:O = Oacc / l ─────
      {"op":"map.binary","result":"O","operands":["Oacc","l"],
       "attrs":{"fn":"div","broadcast":"row"}, "out":"O"}           // destination-passing 写回 O
    ]
  }
}
```

## 这个例子证明了什么

- **递推 (A) 与 tiling (C) 被因式分解**:online-softmax 的状态 (`m`,`l`,`Oacc`) 在 `loop.carry` 里(Layer A 语义);K 切块大小 `BK` 在 `trip.tile_hint` 里(Layer C,可重调)。一个源 K 循环,拆成了两层,没有二选一。
- **B 层约束显式**:K 块边界、`p_sum` 的 `float_nondeterministic` 结合性都显式挂在 op 上——不被任何"大算子"吞掉。
- **粒度合适**:整个 FlashAttention 用约 14 个张量级原子表达,不爆炸成几百个标量 op,也没退化成一个不可验证的 `attention` 大算子。
- **可验证**:每个 op 都有可执行 reference 语义,整体可跑 IntentIR 解释器与源 kernel 做执行一致性;浮点归约用 tolerance。

## 与规范其它部分的对应

- 算子签名:[op-signatures.md](op-signatures.md) · 三层与因式分解:[layers-abc.md](layers-abc.md) · 循环:[structured-loop.md](structured-loop.md) · 验证 tolerance:[../verification/dynamic-feedback.md](../verification/dynamic-feedback.md)
