# XAgentic Clean Segregated Project

This package was reorganized from `New-Xagentic-Slicing.zip` without altering the authoritative final evidence.

## What to check into Git

Recommended minimum check-in:

- `01_NOTEBOOKS/REQUIRED_CHECKIN/E1_E4_Final_5Seed_Experiments.ipynb`
- `01_NOTEBOOKS/REQUIRED_CHECKIN/E5_Final_Qwen37_N100_and_Final_Audit.ipynb`
- `03_SOURCE_CODE/`
- `04_REPRODUCIBILITY/README.md` and `requirements.txt`

If repository size is a concern, keep `02_RESULTS/FINAL/` as release/evidence artifacts rather than normal Git history.

## Authoritative results

- `02_RESULTS/FINAL/E1_E4_5SEEDS/` — final five-seed E1-E4 evidence (seeds 42-46).
- `02_RESULTS/FINAL/E5_QWEN37_N100_SEED52/` — final E5 Qwen3.7-Max, N=100, seed52 evidence.

## Important

The final E5 run used the authoritative `03_SOURCE_CODE/agents.py` from the frozen final-evidence archive, not the older top-level `experiment/agents.py` found in the uploaded bundle.
