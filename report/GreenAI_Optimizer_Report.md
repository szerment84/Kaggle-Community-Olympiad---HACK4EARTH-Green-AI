# Green AI Optimizer — SCI-style Evidence

## 1. Goal
Compare baseline vs green-optimized runs; log runtime, energy (kWh), CO₂e (kg), MAE; show carbon-aware proof; estimate annual Green Impact.

## 2. Methods
- sklearn Pipeline (preprocessing + GBDT),
- baseline vs optimized (lighter config + carbon-aware slot from metaData),
- energy/CO₂ **proxy** (0.1 kW * time; gCO₂/kWh from metaData mean),
- optional CodeCarbon for real measurements.

## 3. Results (excerpt)
- Before/After table: `metrics_before_after.csv`,
- Carbon-aware proof: `picked_region`, `picked_utc_hr`,
- Charts: CO₂e and Energy (notebook).

## 4. Green Impact
Yearly savings computed for low/medium/high workloads with ±20% sensitivity.  
Applications: **OmniEnergy EMS/MES** and data-center schedulers.

## 5. Reproducibility
Seed=42, package versions in notebook, deterministic preprocessing.

## 6. License
MIT.
