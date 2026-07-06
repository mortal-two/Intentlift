//===- ConvertIntentConvToGPU.cpp - forward convolution -> gpu.launch -----===//
//
// The C7 convolution pass. Lowers a func whose body is exactly ONE `intent.conv`
// (forward conv1d/conv2d/conv3d) over a rank-(2+R) f32 input `[N,Cin,*spatial]`
// with a `[Cout,Cin/groups,*kernel]` weight (+ optional `[Cout]` bias) to a
// single block-cooperative `gpu.launch`.
//
// The mechanism is the `intent.pool` windowed per-output-cell reduce EXTENDED
// with an input-channel sum and a per-tap weight multiply (direct
// windowed-weighted-accumulation; NO im2col). Grid = (N*Cout, prod(outer
// out-spatial), 1): one block per output row = a fixed `(n,co,*outer)`, with the
// innermost out-spatial axis split over T threads (K = O_last/T columns each,
// unrolled). For output cell `(n,co,*o)`:
//   acc = bias[co]
//   for ci_local in [0, Cin/groups):
//     ci = g*(Cin/groups) + ci_local ,  g = co/(Cout/groups)
//     for each kernel tap *k:
//       *src = *o*stride - pad + *k*dil
//       if all src coords in bounds: acc += in[n,ci,*src] * w[co,ci_local,*k]
// A single in-bounds `scf.if` per (ci_local, tap) guards every spatial coord
// (out-of-bounds = implicit zero pad). The accumulation core is parameterized by
// the spatial rank R so conv1d/2d/3d share one code path.
//
// Destination-passing: new signature = (input, weight[, bias]) ++ [output].
// Correctness-only (rel < 1e-3, fp accumulation). Static shapes, O_last % T == 0.
//
//===----------------------------------------------------------------------===//

#include "Intent/Conversion/Passes.h"

#include "MapEmit.h"
#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace intent {
#define GEN_PASS_DEF_CONVERTINTENTCONVTOGPU
#include "Intent/Conversion/Passes.h.inc"
} // namespace intent

using namespace mlir;
using namespace intent;

namespace {

struct ConvLowering {
  func::FuncOp fn;
  unsigned blockThreads;

  ConvOp conv;
  RankedTensorType inTy, wTy, outTy;
  int64_t R = 0;                        // spatial rank (1/2/3)
  int64_t N = 0, Cin = 0, Cout = 0, groups = 0;
  SmallVector<int64_t> H, kK, O;        // per-spatial-axis input/kernel/output
  SmallVector<int64_t> stride, pad, dil;
  int64_t T = 0, K = 0;                 // threads over O_last, cols per thread
  bool hasBias = false;

  ConvLowering(func::FuncOp fn, unsigned bt) : fn(fn), blockThreads(bt) {}

  static MemRefType memrefOf(RankedTensorType t) {
    return MemRefType::get(t.getShape(), t.getElementType());
  }

  static int64_t attrAt(ArrayAttr a, int64_t i) {
    return cast<IntegerAttr>(a[i]).getInt();
  }

  LogicalResult match() {
    if (!fn.getBody().hasOneBlock())
      return failure();
    Block &body = fn.getBody().front();
    auto ret = dyn_cast<func::ReturnOp>(body.getTerminator());
    if (!ret)
      return failure();
    Operation *only = nullptr;
    for (Operation &op : body.without_terminator()) {
      if (only)
        return failure();
      only = &op;
    }
    conv = dyn_cast_or_null<ConvOp>(only);
    if (!conv)
      return failure();
    inTy = cast<RankedTensorType>(conv.getInput().getType());
    wTy = cast<RankedTensorType>(conv.getWeight().getType());
    outTy = cast<RankedTensorType>(conv.getOutput().getType());
    int64_t rank = inTy.getRank();
    if (rank < 3 || rank > 5)
      return failure();
    if (wTy.getRank() != rank || outTy.getRank() != rank)
      return failure();
    if (!inTy.hasStaticShape() || !wTy.hasStaticShape() ||
        !outTy.hasStaticShape() || !inTy.getElementType().isF32())
      return failure();
    R = rank - 2;
    N = inTy.getDimSize(0);
    Cin = inTy.getDimSize(1);
    Cout = wTy.getDimSize(0);
    groups = conv.getGroups();
    if (groups < 1 || Cin % groups != 0 || Cout % groups != 0)
      return failure();
    if (wTy.getDimSize(1) != Cin / groups)
      return failure();
    for (int64_t j = 0; j < R; ++j) {
      H.push_back(inTy.getDimSize(2 + j));
      kK.push_back(wTy.getDimSize(2 + j));
      O.push_back(outTy.getDimSize(2 + j));
      stride.push_back(attrAt(conv.getStride(), j));
      pad.push_back(attrAt(conv.getPadding(), j));
      dil.push_back(attrAt(conv.getDilation(), j));
    }
    hasBias = conv.getBias() != nullptr;
    int64_t Olast = O[R - 1];
    T = blockThreads ? (int64_t)blockThreads : autoThreads(Olast);
    if (T <= 0 || Olast % T != 0)
      return failure();
    K = Olast / T;
    return success();
  }

  LogicalResult rewrite() {
    MLIRContext *ctx = fn.getContext();
    Location loc = fn.getLoc();
    OpBuilder b(ctx);
    Type f32 = b.getF32Type();

    // New signature: (input, weight[, bias]) ++ [output].
    SmallVector<Type> inTys;
    inTys.push_back(memrefOf(inTy));
    inTys.push_back(memrefOf(wTy));
    if (hasBias)
      inTys.push_back(memrefOf(cast<RankedTensorType>(conv.getBias().getType())));
    inTys.push_back(memrefOf(outTy));
    auto newFty = FunctionType::get(ctx, inTys, {});

    std::string name = fn.getName().str();
    fn.setName(name + "_intent_src");
    b.setInsertionPoint(fn);
    auto gpuFn = b.create<func::FuncOp>(loc, name, newFty);
    Block *entry = gpuFn.addEntryBlock();
    b.setInsertionPointToStart(entry);
    unsigned ai = 0;
    Value inMem = entry->getArgument(ai++);
    Value wMem = entry->getArgument(ai++);
    Value biasMem = hasBias ? entry->getArgument(ai++) : nullptr;
    Value outMem = entry->getArgument(ai++);

    // Product of the OUTER out-spatial dims (all but the innermost) -> grid.y.
    int64_t outerProd = 1;
    for (int64_t j = 0; j < R - 1; ++j)
      outerProd *= O[j];

    Value cNCo = b.create<arith::ConstantIndexOp>(loc, N * Cout);
    Value cOuter = b.create<arith::ConstantIndexOp>(loc, outerProd);
    Value cT = b.create<arith::ConstantIndexOp>(loc, T);
    Value c1 = b.create<arith::ConstantIndexOp>(loc, 1);
    auto launch = b.create<gpu::LaunchOp>(loc, cNCo, cOuter, c1, cT, c1, c1);
    b.setInsertionPointToStart(&launch.getBody().front());
    Value bx = launch.getBlockIds().x;   // n*Cout + co
    Value by = launch.getBlockIds().y;   // flattened outer out-spatial coord
    Value tx = launch.getThreadIds().x;

    auto cidx = [&](int64_t v) {
      return b.create<arith::ConstantIndexOp>(loc, v).getResult();
    };

    // Decode (n, co) and the group's input-channel base.
    Value cCout = cidx(Cout);
    Value nIdx = b.create<arith::DivUIOp>(loc, bx, cCout);
    Value co = b.create<arith::RemUIOp>(loc, bx, cCout);
    int64_t ipg = Cin / groups;          // input channels per group
    int64_t opg = Cout / groups;         // output channels per group
    Value g = b.create<arith::DivUIOp>(loc, co, cidx(opg));
    Value ciBase = b.create<arith::MulIOp>(loc, g, cidx(ipg));

    // Decode the outer out-spatial coords from `by` (row-major, last outer
    // varies fastest). oCoord[R-1] is filled per-thread below.
    SmallVector<Value> oCoord(R);
    Value rem = by;
    for (int64_t j = R - 2; j >= 0; --j) {
      oCoord[j] = b.create<arith::RemUIOp>(loc, rem, cidx(O[j]));
      if (j > 0)
        rem = b.create<arith::DivUIOp>(loc, rem, cidx(O[j]));
    }

    for (int64_t k = 0; k < K; ++k) {
      Value oLast = tx;
      if (k != 0)
        oLast = b.create<arith::AddIOp>(loc, tx, cidx(k * T));
      oCoord[R - 1] = oLast;

      // Accumulator init = bias[co] (or 0).
      Value acc;
      if (hasBias)
        acc = b.create<memref::LoadOp>(loc, biasMem, ValueRange{co});
      else
        acc = b.create<arith::ConstantOp>(loc, b.getFloatAttr(f32, 0.0));

      for (int64_t ci = 0; ci < ipg; ++ci) {
        Value ciLocal = cidx(ci);
        Value ciIdx = b.create<arith::AddIOp>(loc, ciBase, ciLocal);

        // Enumerate every kernel tap (cartesian product over R axes).
        int64_t taps = 1;
        for (int64_t j = 0; j < R; ++j)
          taps *= kK[j];
        SmallVector<int64_t> kk(R, 0);
        for (int64_t t = 0; t < taps; ++t) {
          // src[j] = o[j]*stride[j] - pad[j] + kk[j]*dil[j] (i64, may be < 0 ->
          // guarded). Build the in-bounds condition over all spatial axes.
          Type i64 = b.getI64Type();
          auto ci64 = [&](int64_t v) {
            return b.create<arith::ConstantIntOp>(loc, v, 64).getResult();
          };
          Value zero64 = ci64(0);
          Value cond = nullptr;
          SmallVector<Value> srcI64(R);
          for (int64_t j = 0; j < R; ++j) {
            Value oj = b.create<arith::IndexCastOp>(loc, i64, oCoord[j]);
            Value sj = b.create<arith::AddIOp>(
                loc,
                b.create<arith::SubIOp>(
                    loc, b.create<arith::MulIOp>(loc, oj, ci64(stride[j])),
                    ci64(pad[j])),
                ci64(kk[j] * dil[j]));
            srcI64[j] = sj;
            Value ge = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                               sj, zero64);
            Value lt = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               sj, ci64(H[j]));
            Value inb = b.create<arith::AndIOp>(loc, ge, lt);
            cond = cond ? b.create<arith::AndIOp>(loc, cond, inb) : inb;
          }

          auto ifOp = b.create<scf::IfOp>(loc, TypeRange{f32}, cond,
                                          /*withElse=*/true);
          {
            OpBuilder tb = ifOp.getThenBodyBuilder();
            // input index: [n, ci, *src]
            SmallVector<Value> inIdx{nIdx, ciIdx};
            for (int64_t j = 0; j < R; ++j)
              inIdx.push_back(
                  tb.create<arith::IndexCastOp>(loc, tb.getIndexType(),
                                                srcI64[j]));
            Value iv = tb.create<memref::LoadOp>(loc, inMem, inIdx);
            // weight index: [co, ci_local, *kk]
            SmallVector<Value> wIdx{co, ciLocal};
            for (int64_t j = 0; j < R; ++j)
              wIdx.push_back(tb.create<arith::ConstantIndexOp>(loc, kk[j]));
            Value wv = tb.create<memref::LoadOp>(loc, wMem, wIdx);
            Value prod = tb.create<arith::MulFOp>(loc, iv, wv);
            Value sum = tb.create<arith::AddFOp>(loc, acc, prod);
            tb.create<scf::YieldOp>(loc, ValueRange{sum});
          }
          {
            OpBuilder eb = ifOp.getElseBodyBuilder();
            eb.create<scf::YieldOp>(loc, ValueRange{acc});
          }
          acc = ifOp.getResult(0);

          // Increment the mixed-radix kernel index (last axis fastest).
          for (int64_t j = R - 1; j >= 0; --j) {
            if (++kk[j] < kK[j])
              break;
            kk[j] = 0;
          }
        }
      }

      // Store out[n, co, *o].
      SmallVector<Value> outIdx{nIdx, co};
      for (int64_t j = 0; j < R; ++j)
        outIdx.push_back(oCoord[j]);
      b.create<memref::StoreOp>(loc, acc, outMem, outIdx);
    }

    b.create<gpu::TerminatorOp>(loc);
    b.setInsertionPointAfter(launch);
    b.create<func::ReturnOp>(loc);
    fn.erase();
    return success();
  }
};

struct ConvertIntentConvToGPUPass
    : ::intent::impl::ConvertIntentConvToGPUBase<ConvertIntentConvToGPUPass> {
  using ConvertIntentConvToGPUBase::ConvertIntentConvToGPUBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    SmallVector<func::FuncOp> targets;
    for (auto fn : mod.getOps<func::FuncOp>()) {
      ConvLowering probe(fn, blockThreads);
      if (succeeded(probe.match()))
        targets.push_back(fn);
    }
    for (func::FuncOp fn : targets) {
      ConvLowering low(fn, blockThreads);
      if (failed(low.match()) || failed(low.rewrite())) {
        fn.emitError("conv lowering failed after match");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
