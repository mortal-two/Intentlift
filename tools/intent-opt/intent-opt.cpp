//===- intent-opt.cpp - IntentIR optimizer driver -------------------------===//
//
// Registers the IntentIR `intent` dialect alongside all upstream MLIR
// dialects and passes, and exposes them through the standard mlir-opt driver.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Intent/Conversion/Common/Passes.h"
#include "Intent/Conversion/CUDA/Passes.h"
#include "Intent/IntentDialect.h"
#include "Target/CUDA/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  intent::registerIntentConversionCommonPasses();
  intent::registerIntentConversionCUDAPasses();
  intent::registerIntentTargetCUDAPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<intent::IntentDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "IntentIR optimizer driver\n", registry));
}
