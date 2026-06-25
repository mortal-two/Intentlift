#include "Intent/IntentDialect.h"
#include "Intent/IntentOps.h"

using namespace mlir;
using namespace intent;

#include "Intent/IntentDialect.cpp.inc"

void IntentDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Intent/IntentOps.cpp.inc"
      >();
}
