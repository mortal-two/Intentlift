"""Backward / gradient IntentIR authoring — the deferred backward cluster.

flag_gems exposes 24 ``*_backward`` entrypoints. Almost all are *compositions of
mechanisms IntentIR already lowers*, so this is authoring + verify (like the
extraction bulk), NOT new passes. Each builder returns a ``BwdKernel`` — a
``(pass_name, mlir_text, block_threads)`` spec that ONE existing pass lowers
unchanged, driven by :class:`intentir.benchmarks.row_reduction.RowReductionKernel`.

Three mechanism groups (reference gradient = torch autograd; also PK'd vs the
real ``flag_gems.<op>_backward``):

1. **Pointwise activation backward = pure map** (grad_in = grad_out * act'(x)) —
   ``convert-intent-elementwise-to-gpu``. One map_unary/map_binary/select graph:
   sigmoid/tanh/silu/gelu/elu/threshold/dropout backward.
2. **Softmax / norm backward = row-reduction + map** —
   ``convert-intent-row-reduction-to-gpu``. The row grad reduces over the
   normalized axis then maps: softmax/log_softmax/rms_norm/layer_norm backward
   (input gradient dx). The norm kernels recompute mean/rstd/inv_rms internally
   from x (self-contained, same shape as the forward corpus kernels).
3. **Scatter-grad backward = C3 scatter (reduce=add)** —
   ``convert-intent-gather-to-gpu``. The gradient scatters into a zero-initialized
   buffer: gather_backward (= scatter of grad), embedding_backward (scatter-add
   grad rows by index), nll_loss_backward (scatter -grad at the target class).

No new scalar-fn / derivative primitive was needed — every gradient composes from
the existing map (exp/erf/sigmoid/tanh/rsqrt/...), reduce (sum), select, and
scatter(add) atoms.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BwdKernel:
    """A backward kernel spec: the pass + intent-MLIR + block geometry.

    ``inputs`` documents the func-signature tensor order the caller must bind
    (followed by the appended output for the non-scatter passes; scatter is
    in-place on the first ``data`` arg).
    """

    name: str
    pass_name: str
    mlir_text: str
    block_threads: int
    inputs: list[str]


_ELT = "convert-intent-elementwise-to-gpu"
_RR = "convert-intent-row-reduction-to-gpu"
_GS = "convert-intent-gather-to-gpu"


# --------------------------------------------------------------------------- #
# Group 1 — pointwise activation backward = pure map (elementwise pass).       #
# --------------------------------------------------------------------------- #
def _shapes(M: int, N: int):
    return (f"tensor<{M}x{N}xf32>", "tensor<f32>", f"tensor<{M}x{N}xi1>")


def sigmoid_backward(M: int, N: int, T: int = 128) -> BwdKernel:
    """grad_in = grad_out * y*(1-y) = grad_out * (y - y*y).  (grad_out, y=output)"""
    R, _, _ = _shapes(M, N)
    mlir = f"""
func.func @sigmoid_backward(%g: {R}, %y: {R}) -> {R} {{
  %yy = intent.map_binary %y, %y {{fn = "mul"}} : ({R}, {R}) -> {R}
  %d = intent.map_binary %y, %yy {{fn = "sub"}} : ({R}, {R}) -> {R}
  %o = intent.map_binary %g, %d {{fn = "mul"}} : ({R}, {R}) -> {R}
  return %o : {R}
}}
"""
    return BwdKernel("sigmoid_backward", _ELT, mlir, T, ["grad_output", "output"])


def tanh_backward(M: int, N: int, T: int = 128) -> BwdKernel:
    """grad_in = grad_out * (1 - y*y) = grad_out - grad_out*y*y.  (grad_out, y)"""
    R, _, _ = _shapes(M, N)
    mlir = f"""
func.func @tanh_backward(%g: {R}, %y: {R}) -> {R} {{
  %yy = intent.map_binary %y, %y {{fn = "mul"}} : ({R}, {R}) -> {R}
  %gyy = intent.map_binary %g, %yy {{fn = "mul"}} : ({R}, {R}) -> {R}
  %o = intent.map_binary %g, %gyy {{fn = "sub"}} : ({R}, {R}) -> {R}
  return %o : {R}
}}
"""
    return BwdKernel("tanh_backward", _ELT, mlir, T, ["grad_output", "output"])


def silu_backward(M: int, N: int, T: int = 128) -> BwdKernel:
    """grad_in = grad_out * s*(1 + x*(1-s)), s = sigmoid(x).  (grad_out, self=x)"""
    R, S, _ = _shapes(M, N)
    mlir = f"""
func.func @silu_backward(%g: {R}, %x: {R}, %one: {S}) -> {R} {{
  %s = intent.map_unary %x {{fn = "sigmoid"}} : ({R}) -> {R}
  %ns = intent.map_unary %s {{fn = "neg"}} : ({R}) -> {R}
  %oms = intent.map_binary %ns, %one {{fn = "add"}} : ({R}, {S}) -> {R}
  %xoms = intent.map_binary %x, %oms {{fn = "mul"}} : ({R}, {R}) -> {R}
  %sx = intent.map_binary %s, %xoms {{fn = "mul"}} : ({R}, {R}) -> {R}
  %d = intent.map_binary %s, %sx {{fn = "add"}} : ({R}, {R}) -> {R}
  %o = intent.map_binary %g, %d {{fn = "mul"}} : ({R}, {R}) -> {R}
  return %o : {R}
}}
"""
    return BwdKernel("silu_backward", _ELT, mlir, T, ["grad_output", "self", "one"])


def gelu_backward(M: int, N: int, T: int = 128) -> BwdKernel:
    """grad_in = grad_out * gelu'(x) (exact/erf).  (grad_out, self=x, consts)

    gelu'(x) = 0.5*(1+erf(x/sqrt2)) + x * (1/sqrt(2pi)) * exp(-x^2/2).
    Consts bound by the caller: half=0.5, one=1, inv_sqrt2=1/sqrt2,
    inv_sqrt2pi=1/sqrt(2pi), neg_half=-0.5.
    """
    R, S, _ = _shapes(M, N)
    mlir = f"""
func.func @gelu_backward(%g: {R}, %x: {R}, %half: {S}, %one: {S},
    %inv_sqrt2: {S}, %inv_sqrt2pi: {S}, %neg_half: {S}) -> {R} {{
  %xs = intent.map_binary %x, %inv_sqrt2 {{fn = "mul"}} : ({R}, {S}) -> {R}
  %e = intent.map_unary %xs {{fn = "erf"}} : ({R}) -> {R}
  %e1 = intent.map_binary %e, %one {{fn = "add"}} : ({R}, {S}) -> {R}
  %cdf = intent.map_binary %e1, %half {{fn = "mul"}} : ({R}, {S}) -> {R}
  %x2 = intent.map_binary %x, %x {{fn = "mul"}} : ({R}, {R}) -> {R}
  %nx2h = intent.map_binary %x2, %neg_half {{fn = "mul"}} : ({R}, {S}) -> {R}
  %ex = intent.map_unary %nx2h {{fn = "exp"}} : ({R}) -> {R}
  %pdf = intent.map_binary %ex, %inv_sqrt2pi {{fn = "mul"}} : ({R}, {S}) -> {R}
  %xpdf = intent.map_binary %x, %pdf {{fn = "mul"}} : ({R}, {R}) -> {R}
  %d = intent.map_binary %cdf, %xpdf {{fn = "add"}} : ({R}, {R}) -> {R}
  %o = intent.map_binary %g, %d {{fn = "mul"}} : ({R}, {R}) -> {R}
  return %o : {R}
}}
"""
    return BwdKernel("gelu_backward", _ELT, mlir, T,
                     ["grad_output", "self", "half", "one", "inv_sqrt2",
                      "inv_sqrt2pi", "neg_half"])


def elu_backward(M: int, N: int, T: int = 128) -> BwdKernel:
    """grad_in = grad_out * (x>0 ? 1 : exp(x))  (alpha=1).  (grad_out, self=x)"""
    R, S, I1 = _shapes(M, N)
    mlir = f"""
func.func @elu_backward(%g: {R}, %x: {R}, %zero: {S}, %one: {S}) -> {R} {{
  %c = intent.map_binary %x, %zero {{fn = "gt"}} : ({R}, {S}) -> {I1}
  %ex = intent.map_unary %x {{fn = "exp"}} : ({R}) -> {R}
  %d = intent.select %c, %one, %ex : ({I1}, {S}, {R}) -> {R}
  %o = intent.map_binary %g, %d {{fn = "mul"}} : ({R}, {R}) -> {R}
  return %o : {R}
}}
"""
    return BwdKernel("elu_backward", _ELT, mlir, T,
                     ["grad_output", "self", "zero", "one"])


def threshold_backward(M: int, N: int, T: int = 128) -> BwdKernel:
    """grad_in = grad_out * (x > threshold).  (grad_out, self=x, threshold, zero)"""
    R, S, I1 = _shapes(M, N)
    mlir = f"""
func.func @threshold_backward(%g: {R}, %x: {R}, %thr: {S}, %zero: {S}) -> {R} {{
  %c = intent.map_binary %x, %thr {{fn = "gt"}} : ({R}, {S}) -> {I1}
  %o = intent.select %c, %g, %zero : ({I1}, {R}, {S}) -> {R}
  return %o : {R}
}}
"""
    return BwdKernel("threshold_backward", _ELT, mlir, T,
                     ["grad_output", "self", "threshold", "zero"])


def dropout_backward(M: int, N: int, T: int = 128) -> BwdKernel:
    """grad_in = grad_out * mask * scale.  (grad_out, mask f32, scale=1/(1-p))

    Reuses the C5 philox RNG mask (a 0/1 f32 keep-mask); scale = 1/(1-p).
    """
    R, S, _ = _shapes(M, N)
    mlir = f"""
func.func @dropout_backward(%g: {R}, %mask: {R}, %scale: {S}) -> {R} {{
  %gm = intent.map_binary %g, %mask {{fn = "mul"}} : ({R}, {R}) -> {R}
  %o = intent.map_binary %gm, %scale {{fn = "mul"}} : ({R}, {S}) -> {R}
  return %o : {R}
}}
"""
    return BwdKernel("dropout_backward", _ELT, mlir, T,
                     ["grad_output", "mask", "scale"])


# --------------------------------------------------------------------------- #
# Group 2 — softmax / norm backward = row-reduction + map (row-reduction pass).#
# --------------------------------------------------------------------------- #
def softmax_backward(M: int, N: int, T: int) -> BwdKernel:
    """grad_in = y*(g - sum(g*y)).  (grad_out g, output y)"""
    R = f"tensor<{M}x{N}xf32>"
    P = f"tensor<{M}x1xf32>"
    mlir = f"""
func.func @softmax_backward(%g: {R}, %y: {R}) -> {R} {{
  %gy = intent.map_binary %g, %y {{fn = "mul"}} : ({R}, {R}) -> {R}
  %s = intent.reduce %gy {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {P}
  %d = intent.map_binary %g, %s {{fn = "sub"}} : ({R}, {P}) -> {R}
  %o = intent.map_binary %y, %d {{fn = "mul"}} : ({R}, {R}) -> {R}
  return %o : {R}
}}
"""
    return BwdKernel("softmax_backward", _RR, mlir, T, ["grad_output", "output"])


def log_softmax_backward(M: int, N: int, T: int) -> BwdKernel:
    """grad_in = g - exp(y)*sum(g).  (grad_out g, output y=log_softmax)"""
    R = f"tensor<{M}x{N}xf32>"
    P = f"tensor<{M}x1xf32>"
    mlir = f"""
func.func @log_softmax_backward(%g: {R}, %y: {R}) -> {R} {{
  %s = intent.reduce %g {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {P}
  %ey = intent.map_unary %y {{fn = "exp"}} : ({R}) -> {R}
  %t = intent.map_binary %ey, %s {{fn = "mul"}} : ({R}, {P}) -> {R}
  %o = intent.map_binary %g, %t {{fn = "sub"}} : ({R}, {R}) -> {R}
  return %o : {R}
}}
"""
    return BwdKernel("log_softmax_backward", _RR, mlir, T,
                     ["grad_output", "output"])


def rms_norm_backward(M: int, N: int, T: int) -> BwdKernel:
    """rms_norm input-grad dx (recompute inv_rms from x).  (dy, x, w[N], eps, ninv)

    y_i = x_i * r * w_i, r = rsqrt(mean(x^2)+eps).  With c = sum(dy*w*x):
      dx_j = r*w_j*dy_j - r^3 * x_j * c / N.
    """
    R = f"tensor<{M}x{N}xf32>"
    P = f"tensor<{M}x1xf32>"
    C = f"tensor<{N}xf32>"
    S = "tensor<f32>"
    mlir = f"""
func.func @rms_norm_backward(%dy: {R}, %x: {R}, %w: {C}, %eps: {S}, %ninv: {S}) -> {R} {{
  %xx = intent.map_binary %x, %x {{fn = "mul"}} : ({R}, {R}) -> {R}
  %sx = intent.reduce %xx {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {P}
  %ms = intent.map_binary %sx, %ninv {{fn = "mul"}} : ({P}, {S}) -> {P}
  %mse = intent.map_binary %ms, %eps {{fn = "add"}} : ({P}, {S}) -> {P}
  %ir = intent.map_unary %mse {{fn = "rsqrt"}} : ({P}) -> {P}
  %dyw = intent.map_binary %dy, %w {{fn = "mul"}} : ({R}, {C}) -> {R}
  %dywx = intent.map_binary %dyw, %x {{fn = "mul"}} : ({R}, {R}) -> {R}
  %c = intent.reduce %dywx {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {P}
  %ir2 = intent.map_binary %ir, %ir {{fn = "mul"}} : ({P}, {P}) -> {P}
  %ir3 = intent.map_binary %ir2, %ir {{fn = "mul"}} : ({P}, {P}) -> {P}
  %t1 = intent.map_binary %dyw, %ir {{fn = "mul"}} : ({R}, {P}) -> {R}
  %xc = intent.map_binary %x, %c {{fn = "mul"}} : ({R}, {P}) -> {R}
  %xcir3 = intent.map_binary %xc, %ir3 {{fn = "mul"}} : ({R}, {P}) -> {R}
  %t2 = intent.map_binary %xcir3, %ninv {{fn = "mul"}} : ({R}, {S}) -> {R}
  %dx = intent.map_binary %t1, %t2 {{fn = "sub"}} : ({R}, {R}) -> {R}
  return %dx : {R}
}}
"""
    return BwdKernel("rms_norm_backward", _RR, mlir, T,
                     ["dy", "x", "weight", "eps", "ninv"])


def layer_norm_backward(M: int, N: int, T: int) -> BwdKernel:
    """layer_norm input-grad dx (recompute mean/rstd from x).  (dy, x, w[N], eps, ninv)

    xhat = (x-mean)*rstd; dxhat = dy*w;  a = mean(dxhat), b = mean(dxhat*xhat):
      dx = rstd * (dxhat - a - xhat*b).
    """
    R = f"tensor<{M}x{N}xf32>"
    P = f"tensor<{M}x1xf32>"
    C = f"tensor<{N}xf32>"
    S = "tensor<f32>"
    mlir = f"""
func.func @layer_norm_backward(%dy: {R}, %x: {R}, %w: {C}, %eps: {S}, %ninv: {S}) -> {R} {{
  %sx = intent.reduce %x {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {P}
  %mean = intent.map_binary %sx, %ninv {{fn = "mul"}} : ({P}, {S}) -> {P}
  %xc = intent.map_binary %x, %mean {{fn = "sub"}} : ({R}, {P}) -> {R}
  %xc2 = intent.map_binary %xc, %xc {{fn = "mul"}} : ({R}, {R}) -> {R}
  %sv = intent.reduce %xc2 {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {P}
  %var = intent.map_binary %sv, %ninv {{fn = "mul"}} : ({P}, {S}) -> {P}
  %vare = intent.map_binary %var, %eps {{fn = "add"}} : ({P}, {S}) -> {P}
  %rstd = intent.map_unary %vare {{fn = "rsqrt"}} : ({P}) -> {P}
  %xhat = intent.map_binary %xc, %rstd {{fn = "mul"}} : ({R}, {P}) -> {R}
  %dxhat = intent.map_binary %dy, %w {{fn = "mul"}} : ({R}, {C}) -> {R}
  %sa = intent.reduce %dxhat {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {P}
  %a = intent.map_binary %sa, %ninv {{fn = "mul"}} : ({P}, {S}) -> {P}
  %dxh_xhat = intent.map_binary %dxhat, %xhat {{fn = "mul"}} : ({R}, {R}) -> {R}
  %sb = intent.reduce %dxh_xhat {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {P}
  %b = intent.map_binary %sb, %ninv {{fn = "mul"}} : ({P}, {S}) -> {P}
  %i1 = intent.map_binary %dxhat, %a {{fn = "sub"}} : ({R}, {P}) -> {R}
  %xb = intent.map_binary %xhat, %b {{fn = "mul"}} : ({R}, {P}) -> {R}
  %i2 = intent.map_binary %i1, %xb {{fn = "sub"}} : ({R}, {R}) -> {R}
  %dx = intent.map_binary %i2, %rstd {{fn = "mul"}} : ({R}, {P}) -> {R}
  return %dx : {R}
}}
"""
    return BwdKernel("layer_norm_backward", _RR, mlir, T,
                     ["dy", "x", "weight", "eps", "ninv"])


# --------------------------------------------------------------------------- #
# Group 3 — scatter-grad backward = C3 scatter reduce=add (gather pass).       #
# --------------------------------------------------------------------------- #
def gather_backward(M: int, N: int, T: int) -> BwdKernel:
    """grad_self = zeros.scatter_add_(1, index, grad).  (grad_self=zeros, index, grad)

    In-place on the caller-cloned zero ``grad_self`` buffer.
    """
    R = f"tensor<{M}x{N}xf32>"
    IDX = f"tensor<{M}x{N}xi64>"
    mlir = f"""
func.func @gather_backward(%gself: {R}, %index: {IDX}, %grad: {R}) -> {R} {{
  %o = intent.scatter %gself, %index, %grad {{axis = 1 : i64, mode = "along_axis", reduce = "add"}} : ({R}, {IDX}, {R}) -> {R}
  return %o : {R}
}}
"""
    return BwdKernel("gather_backward", _GS, mlir, T,
                     ["grad_self_zeros", "index", "grad"])


def embedding_backward(V: int, D: int, Nidx: int, T: int) -> BwdKernel:
    """grad_weight[V,D] = zeros.index_add_(0, index, grad).  (gw=zeros, index[Ni,D], grad)

    Scatter-add along axis 0: gw[index[i], d] += grad[i, d]. ``index`` is the token
    row-index broadcast across the D feature columns ([Ni,D] i64), so the scatter's
    2-D (bx=token, col=feature) coordinate maps to the right weight row.
    """
    W = f"tensor<{V}x{D}xf32>"
    G = f"tensor<{Nidx}x{D}xf32>"
    IDX = f"tensor<{Nidx}x{D}xi64>"
    mlir = f"""
func.func @embedding_backward(%gw: {W}, %index: {IDX}, %grad: {G}) -> {W} {{
  %o = intent.scatter %gw, %index, %grad {{axis = 0 : i64, mode = "along_axis", reduce = "add"}} : ({W}, {IDX}, {G}) -> {W}
  return %o : {W}
}}
"""
    return BwdKernel("embedding_backward", _GS, mlir, T,
                     ["grad_weight_zeros", "index", "grad"])


def nll_loss_backward(M: int, C: int, T: int = 1) -> BwdKernel:
    """grad_input[i, target[i]] = -grad (mean: -1/M).  (gi=zeros, target[M,1], src[M,1])

    Scatter along axis 1 at the single target column per row. ``src`` = the
    per-sample gradient (e.g. -1/M for reduction=mean); ``target`` reshaped to
    [M,1]. Column extent is 1, so T=1.
    """
    GI = f"tensor<{M}x{C}xf32>"
    IDX = f"tensor<{M}x1xi64>"
    SRC = f"tensor<{M}x1xf32>"
    mlir = f"""
func.func @nll_loss_backward(%gi: {GI}, %target: {IDX}, %src: {SRC}) -> {GI} {{
  %o = intent.scatter %gi, %target, %src {{axis = 1 : i64, mode = "along_axis", reduce = "add"}} : ({GI}, {IDX}, {SRC}) -> {GI}
  return %o : {GI}
}}
"""
    return BwdKernel("nll_loss_backward", _GS, mlir, T,
                     ["grad_input_zeros", "target", "src"])


POINTWISE = ["sigmoid_backward", "tanh_backward", "silu_backward", "gelu_backward",
             "elu_backward", "threshold_backward", "dropout_backward"]
ROW_REDUCTION = ["softmax_backward", "log_softmax_backward", "rms_norm_backward",
                 "layer_norm_backward"]
SCATTER = ["gather_backward", "embedding_backward", "nll_loss_backward"]

__all__ = ["BwdKernel", *POINTWISE, *ROW_REDUCTION, *SCATTER,
           "POINTWISE", "ROW_REDUCTION", "SCATTER"]
