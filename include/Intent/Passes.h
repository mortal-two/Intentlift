#ifndef INTENT_PASSES_H
#define INTENT_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace intent {

#define GEN_PASS_DECL
#include "Intent/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Intent/Passes.h.inc"

} // namespace intent

#endif // INTENT_PASSES_H
