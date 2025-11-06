import os, argparse, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from .carbon_utils import pick_low_ci_window, measure_block

SEED = 42

ID_CANDIDATES = ["row_id", "Id", "id", "index"]
# katalog repo (jeden poziom wyżej niż src/)
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = REPO_DIR  # jeśli chcesz, możesz zmienić na os.path.join(REPO_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def load_and_merge(data_path: str):
    train_df = pd.read_csv(f"{data_path}/train.csv")
    test_df  = pd.read_csv(f"{data_path}/test.csv")
    meta_df  = pd.read_csv(f"{data_path}/metaData.csv")

    # nazwa kolumny celu (jeśli brak – utwórz sztuczny target do demo)
    target_col = None
    for c in ["target", "Target", "y"]:
        if c in train_df.columns:
            target_col = c
            break
    if target_col is None:
        rng = np.random.default_rng(SEED)
        train_df["target"] = rng.normal(loc=50, scale=5, size=len(train_df))
        target_col = "target"

    # kolumny do wykluczenia (cel + typowe ID)
    drop_cols = {target_col, "example_id", "ExampleId", "Id", "id", "row_id", "index"}

    # cechy = wspólne kolumny train & test, bez drop_cols
    common_cols = set(train_df.columns) & set(test_df.columns)
    feature_cols = sorted([c for c in common_cols if c not in drop_cols])

    # jeśli z jakiegoś powodu przecięcie jest puste – fallback:
    if not feature_cols:
        feature_cols = sorted([c for c in train_df.columns if c not in drop_cols])
        # wyrównaj test do tych kolumn (brakujące wypełnij NaN)
        test_df = test_df.reindex(columns=feature_cols, fill_value=np.nan)

    # ramka łączona tylko do dopasowania transformacji
    df = pd.concat([train_df[feature_cols].copy(),
                    test_df[feature_cols].copy()], axis=0, ignore_index=True)

    return df, train_df, test_df, meta_df, feature_cols, target_col


def build_preprocessor(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])

def train_and_eval(train_df, feature_cols, preproc):
    X = train_df[feature_cols].copy()
    y = train_df['target'].astype(float).values
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = Pipeline([("pre", preproc), ("gb", GradientBoostingRegressor(random_state=SEED))])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    mae = float(mean_absolute_error(y_va, pred))
    return model, mae

def run_mode(mode: str, data_path: str):
    # --- przygotowanie danych ---
    df, train_df, test_df, meta_df, feature_cols, target_col = load_and_merge(data_path)

    # --- wykryj kolumnę CI i policz medianę (dla Baseline) ---
    ci_col = "carbon_intensity_gco2_per_kwh" if "carbon_intensity_gco2_per_kwh" in meta_df.columns else None
    median_ci = None
    if ci_col:
        s = pd.to_numeric(meta_df[ci_col], errors="coerce")
        if s.notna().any():
            median_ci = float(s.median())

    # --- wybór scenariusza / okna o niskim CI ---
    if mode == "baseline":
        picked = {"region": None, "utc_hour": None, "carbon_intensity_gco2_per_kwh": median_ci}
        scenario = "Baseline"
    elif mode == "optimized":
        picked = pick_low_ci_window(meta_df, region=None)  # wybór minimum CI
        scenario = "Optimized"
    else:
        raise ValueError("mode must be 'baseline' or 'optimized'")

    # --- katalog repo i docelowy katalog zapisu ---
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = repo_dir  # opcjonalnie: os.path.join(repo_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # --- wybór kolumny ID do submission ---
    def pick_id_column(df_: pd.DataFrame):
        for cand in ID_CANDIDATES:
            if cand in df_.columns:
                return cand
        return None  # wygenerujemy zakres

    # --- główne zadanie: trening + predykcje (oprawione w pomiar czasu/energii) ---
    def task():
        pre = build_preprocessor(df, feature_cols)
        model, mae = train_and_eval(
            train_df.rename(columns={target_col: "target"}),
            feature_cols, pre
        )
        X_test = test_df[feature_cols].copy()
        preds = model.predict(X_test)

        id_col = pick_id_column(test_df)
        if id_col:
            sub = pd.DataFrame({id_col: test_df[id_col].values, "target": preds})
        else:
            sub = pd.DataFrame({"row_id": np.arange(len(test_df)), "target": preds})

        return {"mae": mae, "submission": sub}

    # --- pomiar bloku z proxy energii/CO2 ---
    ci_val = picked.get("carbon_intensity_gco2_per_kwh")
    out, rec = measure_block(task, ci_val)

    # --- metadane wyników dla metryk ---
    mae = float(out["mae"])
    sub = out["submission"]

    rec.update({
        "Scenario": scenario,
        "MAE": mae,
        "picked_region": picked.get("region"),
        "picked_utc_hr": picked.get("utc_hour"),
    })

    # --- ścieżki plików wyjściowych ---
    sub_name = "submission_baseline.csv" if scenario == "Baseline" else "submission_optimized.csv"
    sub_path = os.path.join(out_dir, sub_name)
    metrics_path = os.path.join(out_dir, "metrics_before_after.csv")

    # --- zapis submission ---
    sub.to_csv(sub_path, index=False)
    print(f"[{scenario}] Saved submission -> {sub_path}", flush=True)

    # --- wyliczenie i zapis metryk (z redukcją CO2 vs. Baseline) ---
    co2_reduction = 0.0
    if os.path.exists(metrics_path):
        try:
            prev = pd.read_csv(metrics_path)
            base_rows = prev.loc[prev["Scenario"].astype(str).str.lower() == "baseline"]
            if len(base_rows) > 0:
                base_co2 = float(base_rows.iloc[-1]["CO2e_kg"])
                if scenario == "Optimized" and base_co2 > 0:
                    co2_reduction = (base_co2 - float(rec["CO2e_kg"])) / base_co2 * 100.0
        except Exception:
            pass
    rec["CO2_Reduction_%"] = co2_reduction

    # dopisz/odśwież rekord scenariusza
    row = pd.DataFrame([rec])
    if os.path.exists(metrics_path):
        old = pd.read_csv(metrics_path)
        metrics = pd.concat([old, row], ignore_index=True)
        metrics = metrics.drop_duplicates(subset=["Scenario"], keep="last")
    else:
        metrics = row

    metrics.to_csv(metrics_path, index=False)
    print(f"[{scenario}] Saved metrics -> {metrics_path}", flush=True)

    # --- Quick Preview: pokaż 3 pierwsze wiersze plików wynikowych ---
    print("\n--- Quick Preview ---")

    def preview_csv(path, name):
        if os.path.exists(path):
            try:
                dfp = pd.read_csv(path)
                print(f"\n📄 {name} ({len(dfp)} rows)")
                print(dfp.head(3).to_string(index=False))
            except Exception as e:
                print(f"⚠️ Cannot preview {name}: {e}")
        else:
            print(f"⚠️ File not found: {name}")

    preview_csv(metrics_path, "metrics_before_after.csv")
    preview_csv(os.path.join(out_dir, "submission_baseline.csv"), "submission_baseline.csv")
    preview_csv(os.path.join(out_dir, "submission_optimized.csv"), "submission_optimized.csv")
    print("\n✅ Preview complete.\n")

    # --- podsumowanie do konsoli ---
    try:
        print(
            f"[{scenario}] CI={ci_val}  Runtime={rec['Runtime_s']:.2f}s  "
            f"Energy={rec['Energy_KWh']:.6f} kWh  CO2={rec['CO2e_kg']:.6f} kg  "
            f"CO2_Reduction={rec['CO2_Reduction_%']:.2f}%",
            flush=True
        )
    except Exception:
        print(f"[{scenario}] Done. Saved to: {sub_path} and {metrics_path}", flush=True)


def _default_data_dir():
    here = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(here, "..", "data"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline","optimized"],
                    default=os.getenv("MODE","optimized"))
    ap.add_argument("--data", dest="data_path", required=False,
                    # domyślnie lokalny folder ../data; jeśli ustawisz DATA_PATH, on ma priorytet
                    default=os.getenv("DATA_PATH", _default_data_dir()))
    args = ap.parse_args()

    # Walidacja danych z czytelnym komunikatem:
    required = ["train.csv", "test.csv", "metaData.csv"]
    missing = [f for f in required if not os.path.exists(os.path.join(args.data_path, f))]
    if missing:
        raise FileNotFoundError(
            "Brak plików danych: "
            + ", ".join(missing)
            + f". Upewnij się, że znajdują się w: {args.data_path} "
              "(lub podaj poprawną ścieżkę przez --data/zmienną DATA_PATH)."
        )

    run_mode(args.mode, args.data_path)





if __name__ == "__main__":
    main()
