
# Model Card — Green AI Optimizer

**Intended Use.** Demonstrate carbon-aware training/inference with SCI-style evidence for Hack4Earth Green AI.

**Data.** `/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai` (train/test/metaData).

**Model.** GradientBoostingRegressor (baseline and optimized variants) in an sklearn Pipeline.

**Metrics.** MAE for accuracy; runtime (s), energy (kWh, proxy), CO₂e (kg, proxy) for footprint.

**Carbon-aware.** Lowest carbon-intensity window selected from `metaData.csv` and logged as `picked_region`, `picked_utc_hr`.

**Limitations.**
- Energy/CO₂ proxies approximate actual hardware draw. Enable CodeCarbon for real emissions.
- Synthetic/derived target may be used for scaffold; use domain targets for production.

**Ethics & Risks.**
- Prefer energy-efficient hardware/regions; document operational windows; avoid rebound effects.

**License.** MIT.
