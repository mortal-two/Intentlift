set(INTENT_TOOLS_DIR ${CMAKE_BINARY_DIR}/tools/intent-opt)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# FileCheck/count/not come from the system LLVM install, not this build, so the
# only build dependency is our own opt driver.
set(INTENT_TEST_DEPENDS intent-opt)

add_lit_testsuite(check-intent "Running the Intent regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${INTENT_TEST_DEPENDS}
)
set_target_properties(check-intent PROPERTIES FOLDER "Tests")

add_lit_testsuites(INTENT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${INTENT_TEST_DEPENDS})
