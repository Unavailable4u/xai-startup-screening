# XAI Startup Screening and Capital Allocation (Bangladesh pilot)

Can explainable AI improve startup screening and simulated capital allocation in emerging-market venture ecosystems?

**Status:** pilot / pre-registration phase. Results are not final.

## Quick start (WSL2 Ubuntu)
```bash
git clone git@github.com:<user>/xai-startup-screening.git
cd xai-startup-screening
make setup
cp /path/to/Startup_Data_MODEL_READY_BD-ST-001_to_100.xlsx data/raw/
source .venv/bin/activate
make all
```
Outputs: `results/tables/`, `results/figures/`.

## Layout
`src/xai_capital/` package, `scripts/` one script per analysis step, `tests/` leakage guards, `docs/` analysis plan and logs, `manuscript/` paper files.

## Data
Raw data is not redistributed here. See `docs/data_collection_protocol.md`. Founder-level personal data must not be released.

## License
Code: MIT. Data: see source terms.
