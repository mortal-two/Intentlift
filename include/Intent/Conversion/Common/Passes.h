#ifndef INTENT_CONVERSION_COMMON_PASSES_H
#define INTENT_CONVERSION_COMMON_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace intent {

#define GEN_PASS_DECL
#include "Intent/Conversion/Common/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Intent/Conversion/Common/Passes.h.inc"

} // namespace intent

#endif // INTENT_CONVERSION_COMMON_PASSES_H
