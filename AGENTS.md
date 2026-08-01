# Repository execution constraint

- `python/tests` was archived to `archive/2026-07-29-python-tests` on 2026-07-29.
- Do not add, regenerate, or restore test files or test directories in the active tree.
- Do not replace real SCRC closure work with mocked unit, regression, smoke, or protocol tests.
- Validate changes through the real Kernel closure workflow and its actual Gate/closure artifacts.
- The archived test suite is historical material only; it is not part of the active workflow.

# Final-task-only development

- Do not create, bump, rename, synchronize, or discuss `vN`-style versions for
  internal implementations, Agent backends, prompts, tools, protocols, observer
  profiles, or intermediate specifications. Existing compatibility tags may stay
  only where current parsers require them; ordinary changes must not introduce a
  new tag or turn version bookkeeping into work.
- Do not create a new Markdown design/status/changelog document after each code
  change. Do not append per-commit or per-version implementation diaries to the
  active task.
- Maintain one active final task as the sole execution record. It should contain
  only the current contract, unresolved blockers, next actions, and authoritative
  real-Kernel closure evidence. Replace stale current-state text instead of
  accumulating historical narratives; archive genuinely superseded material.
- Measure progress only by real Gate1 closure, all three Gate2 comparisons,
  closed-Kernel count, and the final task's deliverables. TypeCheck is only a
  mechanical prerequisite/diagnostic and must never be reported as progress.
  Likewise, a new protocol label, document, receipt format, mock check, tool
  call, or implementation milestone is not progress.
