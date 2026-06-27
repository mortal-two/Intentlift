#include "Intent/IntentOps.h"
#include "Intent/IntentDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace intent;

#define GET_OP_CLASSES
#include "Intent/IntentOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

LogicalResult MapBinaryOp::verify() {
  static constexpr llvm::StringRef kFns[] = {
      "add", "sub", "mul", "div", "max", "min", "pow", "cmp"};
  // Comparison fns produce an i1-typed result tensor (the boolean condition for
  // intent.select); every other fn is element-type-preserving.
  static constexpr llvm::StringRef kCmpFns[] = {"gt", "ge", "lt", "le", "eq"};
  bool isCmp = false;
  for (llvm::StringRef f : kCmpFns)
    if (getFn() == f)
      isCmp = true;
  if (isCmp) {
    auto rTy = cast<RankedTensorType>(getResult().getType());
    if (!rTy.getElementType().isInteger(1))
      return emitOpError("comparison fn '")
             << getFn() << "' must produce an i1-typed result tensor";
    return success();
  }
  for (llvm::StringRef f : kFns)
    if (getFn() == f)
      return success();
  return emitOpError("unknown fn '") << getFn() << "'";
}

LogicalResult SelectOp::verify() {
  auto cTy = cast<RankedTensorType>(getCond().getType());
  if (!cTy.getElementType().isInteger(1))
    return emitOpError("cond must be an i1-typed tensor");
  auto xTy = cast<RankedTensorType>(getX().getType());
  auto yTy = cast<RankedTensorType>(getY().getType());
  auto rTy = cast<RankedTensorType>(getResult().getType());
  // Element types only (shapes broadcast numpy-style, e.g. an [M,N] i1 condition
  // selecting between two rank-0 scalars).
  if (xTy.getElementType() != yTy.getElementType())
    return emitOpError("x and y must share one element type");
  if (rTy.getElementType() != xTy.getElementType())
    return emitOpError("result element type must equal x/y element type");
  return success();
}

LogicalResult IotaOp::verify() {
  auto rTy = cast<RankedTensorType>(getResult().getType());
  if (rTy.getRank() != 1 || !rTy.hasStaticShape())
    return emitOpError("result must be a rank-1 static-shape tensor");
  if (!rTy.getElementType().isIntOrIndex())
    return emitOpError("result element type must be integer/index");
  return success();
}

LogicalResult MapUnaryOp::verify() {
  static constexpr llvm::StringRef kFns[] = {
      "exp", "log", "neg", "abs", "sqrt", "rsqrt", "recip", "tanh", "sigmoid"};
  bool ok = false;
  for (llvm::StringRef f : kFns)
    if (getFn() == f)
      ok = true;
  if (!ok)
    return emitOpError("unknown fn '") << getFn() << "'";
  // Shape-preserving: result type matches the input type.
  if (getInput().getType() != getResult().getType())
    return emitOpError("result type must equal input type");
  return success();
}

LogicalResult ScanOp::verify() {
  static constexpr llvm::StringRef kFns[] = {"sum"};
  bool ok = false;
  for (llvm::StringRef f : kFns)
    if (getFn() == f)
      ok = true;
  if (!ok)
    return emitOpError("unknown scan fn '") << getFn() << "'";
  // Shape-preserving: result type matches the input type.
  if (getInput().getType() != getResult().getType())
    return emitOpError("result type must equal input type");
  auto inTy = cast<RankedTensorType>(getInput().getType());
  if (getAxis() < 0 || getAxis() >= inTy.getRank())
    return emitOpError("axis ")
           << getAxis() << " out of range for rank-" << inTy.getRank()
           << " input";
  return success();
}

LogicalResult ReduceOp::verify() {
  static constexpr llvm::StringRef kFns[] = {"sum", "mean", "max", "min",
                                             "prod", "any", "all"};
  bool ok = false;
  for (llvm::StringRef f : kFns)
    if (getFn() == f)
      ok = true;
  if (!ok)
    return emitOpError("unknown reduce fn '") << getFn() << "'";
  if (getAssoc() != "associative" && getAssoc() != "float_nondeterministic")
    return emitOpError("assoc must be 'associative' or 'float_nondeterministic'");
  return success();
}

LogicalResult ContractionOp::verify() {
  // Parse einsum-style "lhs,rhs->out" (exactly two inputs).
  llvm::StringRef sub = getSubscripts();
  size_t arrow = sub.find("->");
  if (arrow == llvm::StringRef::npos)
    return emitOpError("subscripts must contain '->'");
  llvm::StringRef ins = sub.substr(0, arrow);
  llvm::StringRef outLabels = sub.substr(arrow + 2);
  size_t comma = ins.find(',');
  if (comma == llvm::StringRef::npos || ins.substr(comma + 1).contains(','))
    return emitOpError("contraction expects exactly two inputs (one ',')");
  llvm::StringRef aLabels = ins.substr(0, comma);
  llvm::StringRef bLabels = ins.substr(comma + 1);

  auto aTy = cast<RankedTensorType>(getA().getType());
  auto bTy = cast<RankedTensorType>(getB().getType());
  auto rTy = cast<RankedTensorType>(getResult().getType());

  if (aLabels.size() != static_cast<size_t>(aTy.getRank()))
    return emitOpError("lhs labels '")
           << aLabels << "' (" << aLabels.size() << ") != operand rank "
           << aTy.getRank();
  if (bLabels.size() != static_cast<size_t>(bTy.getRank()))
    return emitOpError("rhs labels '")
           << bLabels << "' (" << bLabels.size() << ") != operand rank "
           << bTy.getRank();
  if (outLabels.size() != static_cast<size_t>(rTy.getRank()))
    return emitOpError("result labels '")
           << outLabels << "' (" << outLabels.size() << ") != result rank "
           << rTy.getRank();

  // Each index label must have one consistent static extent everywhere it
  // appears (dynamic dims are wildcards).
  llvm::DenseMap<char, int64_t> dims;
  auto bind = [&](llvm::StringRef labels, RankedTensorType ty) -> LogicalResult {
    for (size_t i = 0; i < labels.size(); ++i) {
      int64_t d = ty.getDimSize(i);
      if (ShapedType::isDynamic(d))
        continue;
      char c = labels[i];
      auto it = dims.find(c);
      if (it == dims.end())
        dims[c] = d;
      else if (it->second != d)
        return emitOpError("index '")
               << llvm::Twine(c) << "' has conflicting extents " << it->second
               << " and " << d;
    }
    return success();
  };
  if (failed(bind(aLabels, aTy)) || failed(bind(bLabels, bTy)))
    return failure();

  for (size_t i = 0; i < outLabels.size(); ++i) {
    char c = outLabels[i];
    auto it = dims.find(c);
    if (it == dims.end())
      return emitOpError("result index '")
             << llvm::Twine(c) << "' does not appear in any input";
    int64_t d = rTy.getDimSize(i);
    if (!ShapedType::isDynamic(d) && it->second != d)
      return emitOpError("result index '")
             << llvm::Twine(c) << "' extent " << d << " != input extent "
             << it->second;
  }
  return success();
}

LogicalResult LoopOp::verify() {
  unsigned nScan = getScanInputs().size();
  unsigned nCarry = getInits().size();
  if (nCarry != getResults().size())
    return emitOpError("number of inits (")
           << nCarry << ") must equal number of results (" << getResults().size()
           << ") — carry types are loop-invariant";
  if (getScanAxes().size() != nScan)
    return emitOpError("scan_axes (")
           << getScanAxes().size() << ") must have one entry per scan input ("
           << nScan << ")";
  for (auto [i, in] : llvm::enumerate(getScanInputs())) {
    int64_t ax = cast<IntegerAttr>(getScanAxes()[i]).getInt();
    int64_t rank = cast<RankedTensorType>(in.getType()).getRank();
    if (ax < 0 || ax >= rank)
      return emitOpError("scan_axes[")
             << i << "] = " << ax << " out of range for rank-" << rank
             << " scan input";
  }

  Block &body = getBody().front();
  if (body.getNumArguments() != nScan + nCarry)
    return emitOpError("body must have ")
           << (nScan + nCarry) << " block args (scan slices + carries), got "
           << body.getNumArguments();
  // Carry block args (after the scan-slice args) and results must match inits.
  for (auto [i, init] : llvm::enumerate(getInits())) {
    Type carryArgTy = body.getArgument(nScan + i).getType();
    if (carryArgTy != init.getType())
      return emitOpError("carry block arg ")
             << i << " type " << carryArgTy << " != init type " << init.getType();
    if (getResult(i).getType() != init.getType())
      return emitOpError("result ")
             << i << " type " << getResult(i).getType() << " != init type "
             << init.getType();
  }
  // The implicit terminator is a YieldOp (trait-enforced); check its arity/types.
  auto yield = cast<YieldOp>(body.getTerminator());
  if (yield.getResults().size() != nCarry)
    return emitOpError("yield yields ")
           << yield.getResults().size() << " values but loop has " << nCarry
           << " carries";
  for (auto [i, y] : llvm::enumerate(yield.getResults()))
    if (y.getType() != getResult(i).getType())
      return emitOpError("yield operand ")
             << i << " type " << y.getType() << " != result type "
             << getResult(i).getType();
  return success();
}
