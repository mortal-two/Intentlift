#ifndef INTENT_DIALECT
#define INTENT_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IntentIR dialect definition.
//===----------------------------------------------------------------------===//

def Intent_Dialect : Dialect {
  let name = "intent";
  let summary = "IntentIR: certified semantic lifting dialect.";
  let description = [{
    The `intent` dialect is the MLIR form of IntentIR's three-layer
    representation (Layer A algorithmic intent / Layer B portable
    correctness structure / Layer C non-binding schedule hints). Ops are
    tensor-level atomic primitives produced 1:1 from IntentIR JSON.
  }];
  let cppNamespace = "::intent";
}

// Base class for all IntentIR ops.
class Intent_Op<string mnemonic, list<Trait> traits = []>
    : Op<Intent_Dialect, mnemonic, traits>;

#endif // INTENT_DIALECT
