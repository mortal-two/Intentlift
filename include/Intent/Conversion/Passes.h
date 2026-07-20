#ifndef INTENT_CONVERSION_PASSES_H
#define INTENT_CONVERSION_PASSES_H

#include "Intent/Conversion/Common/Passes.h"
#include "Intent/Conversion/CUDA/Passes.h"

namespace intent {

/// Compatibility umbrella for clients that still register every backend.
inline void registerIntentConversionPasses() {
  registerIntentConversionCommonPasses();
  registerIntentConversionCUDAPasses();
}

} // namespace intent

#endif // INTENT_CONVERSION_PASSES_H
