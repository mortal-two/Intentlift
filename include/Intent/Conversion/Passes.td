#ifndef INTENT_CONVERSION_PASSES
#define INTENT_CONVERSION_PASSES

// Compatibility umbrella. New target-selective users should include the
// Common or CUDA pass definition file directly.
include "Intent/Conversion/Common/Passes.td"
include "Intent/Conversion/CUDA/Passes.td"

#endif // INTENT_CONVERSION_PASSES
