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

def energy_co2_proxy(runtime_s: float, mean_ci: float | None, assumed_kw: float = 0.1) -> tuple[float, float]:
    """
    Prosty proxy: energia [kWh] = moc[kW] * czas[h]; CO2e [kg] = energia[kWh] * CI[gCO2/kWh]/1000.
    Gdy brak CI -> zwracamy (energia, 0).
    """
    energy_kwh = max(runtime_s, 0.0) * assumed_kw / 3600.0
    if mean_ci is None:
        return energy_kwh, 0.0
    co2e_kg = energy_kwh * (mean_ci / 1000.0)
    return energy_kwh, co2e_kg

def measure_block(fn, ci_gco2_per_kwh: float | None):
    t0 = time.perf_counter()
    out = fn()
    runtime_s = time.perf_counter() - t0
    energy_kwh, co2e_kg = energy_co2_proxy(runtime_s, ci_gco2_per_kwh)
    rec = {
        "Runtime_s": runtime_s,
        "Energy_kWh": energy_kwh,
        "CO2e_kg": co2e_kg,
        "carbon_intensity_gco2_per_kwh": ci_gco2_per_kwh,
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
