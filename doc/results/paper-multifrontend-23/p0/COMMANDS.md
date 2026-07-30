# Task 23 P0 machine-data receipt

- measured_at: `2026-07-31T00:27:07+08:00`
- commit: `3a9fe4aa814736fdff07aed63349c02c35d39de9`
- Python: `Python 3.10.12`

## Commands

```bash
PYTHONPATH=python python tools/frontend_dataset.py --validate
PYTHONPATH=python python tools/scrc_frontend.py --list --json
python tools/measure_multifrontend_onboarding.py
```

## Interpretation boundary

- Dataset validation checks catalog/manifest/digests only.
- Runnable projection means a case has a usable Harness projection; it does not mean Source compilation, Gate1, Gate2, or SCRC certification.
- Pairing is derived from generated CUDA `origin_case_id`, not from name similarity.
- LoC is measured at three commit-level checkpoints; historical tests/docs are excluded from the production headline, and shared code is not assigned to TileLang or CUDA.
