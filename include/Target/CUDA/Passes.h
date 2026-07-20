#ifndef INTENT_TARGET_CUDA_PASSES_H
#define INTENT_TARGET_CUDA_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace intent {

#define GEN_PASS_DECL
#include "Target/CUDA/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Target/CUDA/Passes.h.inc"

} // namespace intent

#endif // INTENT_TARGET_CUDA_PASSES_H
