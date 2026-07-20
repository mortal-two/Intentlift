#ifndef INTENT_TARGET_PASSES_H
#define INTENT_TARGET_PASSES_H

#include "Target/CUDA/Passes.h"

namespace intent {

/// Compatibility umbrella for clients that still register every target.
inline void registerIntentTargetPasses() { registerIntentTargetCUDAPasses(); }

} // namespace intent

#endif // INTENT_TARGET_PASSES_H
