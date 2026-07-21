"""Extraction prompt: kernel source -> IntentIR JSON.

The system prompt teaches the IntentIR atomic-op vocabulary and JSON shape. The
few-shot is a SYNTHETIC example (L1 row-normalize) chosen so it is NOT any kernel
in the evaluation corpus -- the model gets the schema and the *form*, never the
answer to softmax / layernorm / attention / etc.
"""

from __future__ import annotations

# --- provisional semantic vocabulary; every executor still fails closed ---
SCHEMA = """\
IntentIR J is a tensor-level, atomic semantic representation.
You LIFT a kernel into the math it computes over whole tensors -- never threads,
tiles, pointers, masks, or strides. Re-roll any hand-unrolled / strip-mined loop
back into a single tensor op or a structured loop.

Layers:
  A (algorithmic intent): the actual math -- the ops below.
  B (portable correctness): the single typed top-level J boundary supplied by
    the enclosing Lift schema. Never create per-op layer_b/out dictionaries.
  C is a separate non-semantic scheduling object and never appears inside J.

Top-level JSON:
{ "intent_function": {
    "name": <str>,
    "inputs":  [ {"name","dtype","shape":[dims...]}, ... ],
    "outputs": [ {"name","dtype","shape":[dims...]}, ... ],
    "returns": [ {"output":<declared output>,"value":<SSA/input value>}, ... ],
    "body": [ <op>, ... ]
}}
- dtype in {f16,bf16,f32,f64,i1,i8,i16,i32,i64}. Scalars have shape [].
- shape entries are ints or symbolic dim names (e.g. "M","N","D"); reuse the
  same name for the same extent. An output is produced by an op result with the
  output name, or by an explicit function-level `returns` binding.

Atomic ops (each is one JSON object in a "body" list):
  map_binary : elementwise, numpy-broadcasting. fn in
               {add,sub,mul,div,max,min,pow,atan2,gt,ge,lt,le,eq,ne}.
               (atan2(y,x): the complex angle primitive, angle=atan2(imag,real);
                the comparison fns gt/ge/lt/le/eq/ne produce an i1 tensor ->
                feeds select; count_nonzero = sum(cast(x!=0, i64)).)
     {"op":"map_binary","result":R,"operands":[A,B],"attrs":{"fn":F}}
  map_unary  : elementwise. fn in
               {exp,log,sqrt,rsqrt,neg,recip,abs,tanh,sigmoid,
                cos,sin,tan,exp2,atan,erf,isnan,logical_not}.
               (isnan produces an i1 tensor -> feeds select; logical_not is i1->i1.)
     {"op":"map_unary","result":R,"operands":[A],"attrs":{"fn":F}}
  reduce     : along axes. The declared result shape chooses either KEEPDIMS
               (reduced dims become size 1, for broadcast consumers) or
               COLLAPSED (reduced dims are removed, for an external ABI such as
               row-sum [M,N] -> [M]). fn in {sum,mean,max,min,prod,any,all}.
     {"op":"reduce","result":R,"operands":[A],"attrs":{"fn":F,"axes":[i,...],
       "assoc":"associative"|"float_nondeterministic"}}
  iota       : a rank-1 INTEGER vector [0,1,...,n-1]. `n` is either a static
               integer extent or the exact symbolic runtime extent name used
               by the tensor shape; dtype defaults to i32. Use cast/map ops to
               express affine arange(start,step). Never substitute a sampled
               runtime value, C tile size, program id, block base or lane
               decomposition into Layer A.
     {"op":"iota","result":R,"attrs":{"n":<int-or-symbol>,"dtype":"i32"|"i64"}}
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
               "none" overwrites, "add" accumulates (scatter_add). Every
               canonical scatter must also have one matching top-level
               boundary.index_bounds and boundary.updates record, plus an
               explicit result relation for every observable output.
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
  scan       : inclusive cumulative scan along ONE `axis`, shape-preserving. fn in
               {sum,max,min} (cumsum / cummax / cummin). torch.cummax/cummin also
               return argmax/argmin indices — emit the VALUE scan here (pair the
               index with arg_reduce if needed).
     {"op":"scan","result":R,"operands":[A],"attrs":{"fn":F,"axis":i}}
  arg_reduce : INDEX of the extremum along ONE `axis` (torch.argmax/argmin).
               Result is an INTEGER tensor (the axis collapsed); ties resolve to
               the FIRST/lowest index (matches torch). fn in {argmax,argmin}.
     {"op":"arg_reduce","result":R,"operands":[A],"attrs":{"fn":F,"axis":i}}
  sort       : full sort along the LAST `axis`, shape-preserving. Produces sorted
               VALUES and the argsort INDICES (i32). `descending` picks the
               direction. Name BOTH results (list) for torch.sort, ONE for a
               values-only sort, or ONE with {"return":"indices"} for argsort.
               topk = sort(descending) with `k` set (keep the first k). ties are
               unstable (like torch's default sort).
     {"op":"sort","result":[V,IDX],"operands":[A],"attrs":{"descending":false,"axis":i}}
     {"op":"sort","result":IDX,"operands":[A],"attrs":{"descending":false,"axis":i,"return":"indices"}}   // argsort
     {"op":"sort","result":[V,IDX],"operands":[A],"attrs":{"descending":true,"axis":i,"k":<int>}}          // topk
  pool       : 2-D windowed pooling over NCHW [N,C,H,W] -> [N,C,OH,OW]. `fn` "avg"
               (mean, torch count_include_pad divisor kH*kW) or "max". For
               max_pool2d_with_indices name BOTH results (list) — the 2nd is the
               argmax FLAT index ih*W+iw (i64).
     {"op":"pool","result":O,"operands":[A],"attrs":{"fn":"avg","kH":k,"kW":k,"sH":s,"sW":s,"padH":p,"padW":p}}
     {"op":"pool","result":[O,IDX],"operands":[A],"attrs":{"fn":"max","kH":k,"kW":k,"sH":s,"sW":s,"padH":p,"padW":p}}
  upsample   : nearest-neighbor resize over NCHW. out[.,oh,ow]=in[.,floor(oh*IH/OH),
               floor(ow*IW/OW)]. `scale`=[sh,sw] integer upscale factors.
     {"op":"upsample","result":O,"operands":[A],"attrs":{"mode":"nearest","scale":[sh,sw]}}
  polar      : complex build from magnitude+angle (torch.polar). The complex result
               is the trailing real/imag PAIR — name BOTH results (list):
               real=abs*cos(angle), imag=abs*sin(angle).
     {"op":"polar","result":[RE,IM],"operands":[ABS,ANGLE],"attrs":{}}
  conv       : forward conv1d/2d/3d (torch.nn.functional.conv{1,2,3}d, cross-
               correlation, NO kernel flip). input [N,Cin,*spatial], weight
               [Cout,Cin/groups,*kernel], optional bias [Cout] -> [N,Cout,*out].
               stride/padding/dilation are length-R spatial arrays (R=spatial
               rank 1/2/3); groups partitions channels. Omit the bias operand if
               there is none.
     {"op":"conv","result":O,"operands":[IN,W,BIAS],"attrs":{"stride":[s,s],"padding":[p,p],"dilation":[1,1],"groups":1}}
     {"op":"conv","result":O,"operands":[IN,W],"attrs":{"stride":[s],"padding":[p],"dilation":[1],"groups":1}}
  nonzero    : DYNAMIC-shape output. Row-major multi-indices of the nonzero
               elements (torch.nonzero / numpy.argwhere). Result [nnz, ndim] i64
               (ndim = cond rank); nnz is data-dependent. count_nonzero is NOT
               this — it is sum(cast(x!=0, i64)) (a reduce). masked_fill is NOT
               this — it is select(mask, value, x) (elementwise).
     {"op":"nonzero","result":R,"operands":[COND],"attrs":{}}
  masked_select : DYNAMIC-shape output. Elements where mask is true, flattened in
               row-major order (torch.masked_select / data[mask]). mask shares
               data's shape; result [nnz].
     {"op":"masked_select","result":R,"operands":[DATA,MASK],"attrs":{}}
  masked_scatter: fill the masked positions of data with `src` (rank-1) in
               row-major order (torch.Tensor.masked_scatter). Result is
               data-shaped: out[c]=mask[c]?src[rank(c)]:data[c].
     {"op":"masked_scatter","result":R,"operands":[DATA,MASK,SRC],"attrs":{}}

Structured loop (loop-carried recurrence, e.g. online softmax / scans):
  {"op":"loop",
   "scope_id":<opaque semantic scope id>,
   "result":[<post-loop result values>],
   "carry":[{
      "state_id":<opaque state id>,
      "init_value":<outer value or typed literal>,
      "body_argument":<loop-local current value>,
      "next_value":<value defined by the body>,
      "result_value":<post-loop value>,
      "type":{"dtype":<dtype>,"shape":[<semantic dims>]}
   }, ...],
   "trip":{"axis":<semantic dim name being scanned>},
   "scan":[{"name":<slice name>,"from":<input>,"axis":<int>}, ...],
   "body":[ <ops over scan slices + carries> ],
   "yield":[<next-iteration carry values, one per carry>] }
  result must exactly equal carry.result_value order; yield must exactly equal
  carry.next_value order. Body sees scan names and body_argument names only;
  body_argument must not escape the loop. The body op that defines each
  next_value must carry an explicit type exactly equal to the state type. Keep
  tile/block choices out of trip:
  they are Layer C, not J semantics.
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
- Complex tensors have NO element type: represent a complex value as its real
  and imaginary PLANES (two same-shaped real tensors; torch.view_as_real gives
  the trailing [...,2] pair). Complex ops are compositions over the planes:
    angle(re,im)      = atan2(im, re)                    (map_binary atan2)
    resolve_conj      -> (re, neg(im))                   (map_unary neg on imag)
    resolve_neg       -> (neg(re), neg(im))
    complex_add       -> (re1+re2, im1+im2)
    complex_mul       -> (re1*re2 - im1*im2, re1*im2 + im1*re2)
- Numerical-stability shifts (subtract row-max before exp) ARE part of the
  algorithm -- keep them.
- Per-op `layer_b`, destination `out`, and loop `trip.tile_hint` are forbidden.
  Correctness boundaries live only in the enclosing canonical J boundary;
  tile/block/stage choices live only in C.
- `boundary.index_bounds` exactly binds node/data/indices/axis and always means
  `0 <= indices < dim(data, axis)`; never emit proof prose or `proven=true`.
- `boundary.updates` records only logical base/indices/values/axis, combiner,
  duplicate policy and observable order. Atomic/CAS/thread ownership belongs to
  pi/backend realization. Result tolerance or one-of winners belong only to
  `boundary.result_relations`, never to the runtime harness.
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
  "body": [
    {"op": "reduce", "result": "s", "operands": ["X"],
     "attrs": {"fn": "sum", "axes": [1], "assoc": "float_nondeterministic"}},
    {"op": "map_binary", "result": "Y", "operands": ["X", "s"],
     "attrs": {"fn": "div"}}
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
