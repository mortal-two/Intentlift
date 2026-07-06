"""Extraction prompt: kernel source -> IntentIR JSON.

The system prompt teaches the IntentIR atomic-op vocabulary and JSON shape. The
few-shot is a SYNTHETIC example (L1 row-normalize) chosen so it is NOT any kernel
in the evaluation corpus -- the model gets the schema and the *form*, never the
answer to softmax / layernorm / attention / etc.
"""

from __future__ import annotations

# --- op vocabulary, kept 1:1 with verify/interpret.py and the intent dialect ---
SCHEMA = """\
IntentIR is a three-layer, tensor-level, atomic intermediate representation.
You LIFT a kernel into the math it computes over whole tensors -- never threads,
tiles, pointers, masks, or strides. Re-roll any hand-unrolled / strip-mined loop
back into a single tensor op or a structured loop.

Layers:
  A (algorithmic intent): the actual math -- the ops below.
  B (portable correctness, per-op "layer_b"): bounds / masks / reduction
    associativity that must hold for any backend (optional but encouraged).
  C (non-binding schedule hints, function-level "layer_c"): tile sizes, vector
    width. Advisory only; never changes semantics.

Top-level JSON:
{ "intent_function": {
    "name": <str>,
    "inputs":  [ {"name","dtype","shape":[dims...]}, ... ],
    "outputs": [ {"name","dtype","shape":[dims...]}, ... ],
    "layer_c": { ... optional ... },
    "body": [ <op>, ... ]
}}
- dtype in {f16,bf16,f32,f64,i1,i8,i16,i32,i64}. Scalars have shape [].
- shape entries are ints or symbolic dim names (e.g. "M","N","D"); reuse the
  same name for the same extent. An output is produced by an op whose "out" or
  "result" equals the output name.

Atomic ops (each is one JSON object in a "body" list):
  map_binary : elementwise, numpy-broadcasting. fn in
               {add,sub,mul,div,max,min,pow,gt,ge,lt,le,eq,ne}.
               (the comparison fns gt/ge/lt/le/eq/ne produce an i1 tensor ->
                feeds select; count_nonzero = sum(cast(x!=0, i64)).)
     {"op":"map_binary","result":R,"operands":[A,B],"attrs":{"fn":F}}
  map_unary  : elementwise. fn in
               {exp,log,sqrt,rsqrt,neg,recip,abs,tanh,sigmoid,
                cos,sin,tan,exp2,erf,isnan,logical_not}.
               (isnan produces an i1 tensor -> feeds select; logical_not is i1->i1.)
     {"op":"map_unary","result":R,"operands":[A],"attrs":{"fn":F}}
  reduce     : along axes, KEEPDIMS (result keeps reduced dims as size 1 so it
               stays broadcastable). fn in {sum,mean,max,min,prod,any,all}.
     {"op":"reduce","result":R,"operands":[A],"attrs":{"fn":F,"axes":[i,...],
       "assoc":"associative"|"float_nondeterministic"}}
  const      : a literal value, splat over an optional shape (default scalar []).
     {"op":"const","result":R,"attrs":{"value":<number>,"dtype":"f32","shape":[]}}
  contraction: einsum. attrs.subscripts is a numpy einsum string.
     {"op":"contraction","result":R,"operands":[A,B],"attrs":{"subscripts":S}}
  select     : elementwise where(cond, x, y).
     {"op":"select","result":R,"operands":[COND,X,Y]}
  cast       : dtype cast.  {"op":"cast","result":R,"operands":[A],"attrs":{"to_dtype":DT}}
  transpose  : {"op":"transpose","result":R,"operands":[A],"attrs":{"perm":[...]}}
  gather     : data-dependent addressing. `index` is an INTEGER tensor; `axis`
               selects the gathered dim. mode "along_axis" = torch.gather /
               numpy take_along_axis (index shaped like the result); mode "take"
               = index_select / embedding row-gather = numpy take (index rank r
               inserted at axis; result rank = data.rank-1+index.rank). embedding
               = gather(weight, indices, axis=0, mode="take").
     {"op":"gather","result":R,"operands":[DATA,INDEX],"attrs":{"axis":i,"mode":"along_axis"|"take"}}
  scatter    : data-dependent write into a COPY of data. `index`/`src` share a
               shape; mode "along_axis" (torch.scatter/scatter_add_); reduce
               "none" overwrites, "add" accumulates (scatter_add).
     {"op":"scatter","result":R,"operands":[DATA,INDEX,SRC],"attrs":{"axis":i,"mode":"along_axis","reduce":"none"|"add"}}
  concat     : variadic structural placement along `axis` (numpy concatenate /
               torch.cat; hstack/vstack = axis 1/0; stack = concat after an
               unsqueeze of each input at `axis`).
     {"op":"concat","result":R,"operands":[A,B,...],"attrs":{"axis":i}}
  pad        : constant pad `lo`/`hi` elements before/after each dim, border =
               `value` (numpy pad "constant" / torch constant_pad_nd).
     {"op":"pad","result":R,"operands":[A],"attrs":{"lo":[..],"hi":[..],"value":0.0}}
  tile       : repeat the whole tensor `reps[d]` times per dim (numpy tile /
               torch.tile / torch.Tensor.repeat).
     {"op":"tile","result":R,"operands":[A],"attrs":{"reps":[..]}}
  flip       : reverse element order along `axes` (numpy flip / torch.flip).
     {"op":"flip","result":R,"operands":[A],"attrs":{"axes":[..]}}
  triu       : upper triangle above the k-th diagonal, rest zeroed (numpy/torch
               triu). Rank-2, shape-preserving.
     {"op":"triu","result":R,"operands":[A],"attrs":{"k":0}}
  rand       : counter-based uniform[0,1) f32 RNG. NO operands; the output shape
               is `attrs.shape`, `attrs.seed` is the philox seed. value[i] is a
               stateless philox4x32-10 hash of (seed, element-index), so it is
               order-independent and reproducible. randn / normal / uniform(a,b)
               / dropout are COMPOSITIONS of rand (see below), NOT new ops.
     {"op":"rand","result":R,"attrs":{"seed":<int>,"shape":[..]}}

Structured loop (loop-carried recurrence, e.g. online softmax / scans):
  {"op":"loop",
   "result":[<carry names produced>],
   "carry":[{"name":C,"init":"NEG_INF"|"zeros"|"ones"|"ZERO"|"<number>"}, ...],
   "trip":{"axis":<dim name being scanned>,"tile_hint":<dim name>},
   "scan":[{"name":<slice name>,"from":<input>,"axis":<int>}, ...],
   "body":[ <ops over scan slices + carries> ],
   "yield":[<next-iteration carry values, one per carry>] }
  Body block sees: the scan slices (by name) ++ the current carries (by name).
  Use a loop ONLY for genuine loop-carried state; plain tiling is Layer C and
  should be re-rolled away (the math is the same whole-tensor op).

Hard rules:
- Do NOT invent ops or fns outside the lists above. If the kernel needs one,
  express it with the primitives (e.g. rsqrt = recip(sqrt(x)); a literal number
  is a "const" op; mean over an axis is reduce fn "mean"). Common compositions:
    softplus(x)    = log(1 + exp(x))                    (exp, add const 1, log)
    log_sigmoid(x) = neg(softplus(neg(x)))
    elu(x,a)       = select(x>0, x, a*(exp(x)-1))       (map_binary gt, select)
    celu(x,a)      = select(x>0, x, a*(exp(x/a)-1))
    clamp(x,lo,hi) = min(max(x, lo), hi)                (map_binary max then min)
    threshold(x,t,v)= select(x>t, x, v)
    nan_to_num(x)  = select(isnan(x), 0, x)             (map_unary isnan, select)
    uniform(a,b)   = a + (b-a)*rand(seed)               (rand, mul const, add const)
    dropout(x,p)   = select(rand(seed) >= p, x/(1-p), 0)  (rand, map_binary ge, select)
    randn/normal   = sqrt(-2*log(rand(s1)))*cos(2*pi*rand(s2))  (Box-Muller: two
                     rands with DIFFERENT seeds, log/sqrt/cos)
- Numerical-stability shifts (subtract row-max before exp) ARE part of the
  algorithm -- keep them.
- Output STRICT JSON only. No markdown fence, no prose, no trailing commas.\
"""

# Synthetic worked example -- NOT in the corpus. L1 row-normalize: y = x / sum_j x.
FEWSHOT_SRC = """\
@triton.jit
def l1_normalize_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    tl.store(y_ptr + row * n_cols + cols, x / s, mask=mask)
"""

FEWSHOT_JSON = """\
{"intent_function": {
  "name": "l1_normalize",
  "inputs":  [{"name": "X", "dtype": "f32", "shape": ["M", "N"]}],
  "outputs": [{"name": "Y", "dtype": "f32", "shape": ["M", "N"]}],
  "layer_c": {"tile": {"N": "BLOCK"}},
  "body": [
    {"op": "reduce", "result": "s", "operands": ["X"],
     "attrs": {"fn": "sum", "axes": [1], "assoc": "float_nondeterministic"}},
    {"op": "map_binary", "result": "Y", "operands": ["X", "s"],
     "attrs": {"fn": "div"}, "out": "Y"}
  ]
}}\
"""


def build_instructions() -> str:
    return (
        "You are a semantic-lifting extractor for the IntentIR research compiler. "
        "Given the source of a GPU kernel (Triton/CUDA/etc.), recover the "
        "portable tensor-level math it computes and emit it as IntentIR JSON.\n\n"
        + SCHEMA
        + "\n\n--- WORKED EXAMPLE (a different kernel) ---\nSOURCE:\n"
        + FEWSHOT_SRC
        + "\nINTENTIR JSON:\n"
        + FEWSHOT_JSON
    )


def build_user_input(source: str, name_hint: str, shape_hint: str = "") -> str:
    parts = [
        "Lift the following kernel to IntentIR JSON. Emit ONLY the JSON object.",
        f'Use "{name_hint}" as the function name.',
    ]
    if shape_hint:
        parts.append(f"Shape context: {shape_hint}")
    parts.append("\nSOURCE:\n" + source)
    return "\n".join(parts)
