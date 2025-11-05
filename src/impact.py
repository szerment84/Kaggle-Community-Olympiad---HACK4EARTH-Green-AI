import argparse, pandas as pd

def annual_impact(tasks_per_month, co2e_before_kg, co2e_after_kg):
    runs_per_year = tasks_per_month * 12
    before_year_kg = runs_per_year * co2e_before_kg
    after_year_kg  = runs_per_year * co2e_after_kg
    saved_year_kg  = before_year_kg - after_year_kg
    return before_year_kg, after_year_kg, saved_year_kg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="metrics_before_after.csv")
    ap.add_argument("--low", type=int, default=50)
    ap.add_argument("--med", type=int, default=200)
    ap.add_argument("--high", type=int, default=500)
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    base = float(df.loc[df["Scenario"]=="Baseline","CO2e_kg"].iloc[0])
    opt  = float(df.loc[df["Scenario"]=="Optimized","CO2e_kg"].iloc[0])

    rows=[]
    for name, tpm in [("low",args.low), ("medium",args.med), ("high",args.high)]:
        b,a,s = annual_impact(tpm, base, opt)
        rows.append([name, tpm, b/1000.0, a/1000.0, s/1000.0])  # tons
    out = pd.DataFrame(rows, columns=["scenario","tasks_per_month","tCO2_year_before","tCO2_year_after","tCO2_year_saved"])
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
