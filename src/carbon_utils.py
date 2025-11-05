import time, platform
import pandas as pd

def pick_low_ci_window(meta: pd.DataFrame, region: str | None = None) -> dict:
    dfm = meta if (region is None or 'region' not in meta.columns) else meta[meta['region'].eq(region)]
    if 'carbon_intensity_gco2_per_kwh' not in dfm.columns or len(dfm) == 0:
        return {"region": region, "utc_hour": None, "carbon_intensity_gco2_per_kwh": None}
    row = dfm.sort_values('carbon_intensity_gco2_per_kwh').head(1)
    return dict(
        region=(row['region'].iloc[0] if 'region' in row.columns else region),
        utc_hour=(int(row['UTC_hour'].iloc[0]) if 'UTC_hour' in row.columns else None),
        carbon_intensity_gco2_per_kwh=float(row['carbon_intensity_gco2_per_kwh'].iloc[0])
    )

def energy_co2_proxy(runtime_s: float, mean_ci: float, assumed_kw: float = 0.1) -> tuple[float, float]:
    """
    Proxy: energy_kwh = P[kW] * runtime[h]. We assume 0.1 kW (100 W) for a conservative CPU baseline.
    CO2e (kg) = energy_kwh * (carbon_intensity[gCO2/kWh] / 1000).
    """
    energy_kwh = assumed_kw * (runtime_s / 3600.0)
    co2e_kg = energy_kwh * (mean_ci / 1000.0)
    return energy_kwh, co2e_kg

def measure_block(fn, mean_ci: float, label: str, meta_slot: dict | None = None) -> dict:
    t0 = time.time()
    out = fn()
    dt = time.time() - t0
    e_kwh, co2_kg = energy_co2_proxy(dt, mean_ci)
    rec = {
        "Scenario": label,
        "Runtime_s": dt,
        "Energy_kWh": e_kwh,
        "CO2e_kg": co2_kg,
        "picked_region": None if not meta_slot else meta_slot.get("region"),
        "picked_utc_hr": None if not meta_slot else meta_slot.get("utc_hour"),
        "hardware": platform.platform(),
    }
    return out, rec

def try_codecarbon(run_fn):
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(save_to_file=True, log_level="error")
        tracker.start()
        out = run_fn()
        kg = tracker.stop()
        return out, kg
    except Exception:
        return run_fn(), None
