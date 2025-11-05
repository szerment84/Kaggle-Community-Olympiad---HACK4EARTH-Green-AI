import os, sys, json, argparse
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

def load_and_merge(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train_df = pd.read_csv(f"{data_path}/train.csv")
    test_df  = pd.read_csv(f"{data_path}/test.csv")
    meta_df  = pd.read_csv(f"{data_path}/metaData.csv")

    if 'region' in train_df.columns and 'region' in meta_df.columns:
        df = train_df.merge(meta_df, on='region', how='left')
    else:
        meta_s = meta_df.sample(len(train_df), replace=True, random_state=SEED).reset_index(drop=True)
        df = pd.concat([train_df.reset_index(drop=True), meta_s], axis=1)

    if 'target' not in df.columns:
        if 'carbon_intensity_gco2_per_kwh' in df.columns:
            df['target'] = 100.0 / (df['carbon_intensity_gco2_per_kwh'].astype(float) + 1.0)
        else:
            rng = np.random.default_rng(SEED)
            df['target'] = rng.normal(loc=50, scale=5, size=len(df))

    drop_cols = {'target', 'example_id', 'Id'}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    return df, train_df, test_df, meta_df, feature_cols

def build_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    X = df[feature_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ])

def fit_and_eval(df: pd.DataFrame, pre, mode: str, meta_df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X_full = df.drop(columns=['target', 'example_id'], errors='ignore')
    y_full = df['target'].copy()
    feature_cols = X_full.columns.tolist()
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=SEED)

    mean_ci = float(df['carbon_intensity_gco2_per_kwh'].mean()) if 'carbon_intensity_gco2_per_kwh' in df.columns else 400.0

    if mode == "baseline":
        model = GradientBoostingRegressor(random_state=SEED)
        pipe = Pipeline([("prep", pre), ("model", model)])
        def run(): pipe.fit(X_train, y_train); return pipe
        pipe, rec = measure_block(run, mean_ci, "Baseline", None)
    else:
        model = GradientBoostingRegressor(n_estimators=80, learning_rate=0.08, max_depth=3, subsample=0.7, random_state=SEED)
        pipe = Pipeline([("prep", pre), ("model", model)])
        slot = pick_low_ci_window(meta_df, region=None)
        def run(): pipe.fit(X_train, y_train); return pipe
        pipe, rec = measure_block(run, mean_ci, "Optimized", slot)

    y_pred = pipe.predict(X_val)
    rec["MAE"] = float(mean_absolute_error(y_val, y_pred))
    return pipe, rec

def prepare_test_features(test_df: pd.DataFrame, meta_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    if 'region' in test_df.columns and 'region' in meta_df.columns:
        test_features = test_df.merge(meta_df, on='region', how='left')
    else:
        meta_s_test = meta_df.sample(len(test_df), replace=True, random_state=SEED).reset_index(drop=True)
        test_features = pd.concat([test_df.reset_index(drop=True), meta_s_test], axis=1)
    return test_features.reindex(columns=feature_cols)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline","optimized"], default=os.environ.get("MODE","optimized"))
    ap.add_argument("--data", default=os.environ.get("DATA_PATH","/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai"))
    args = ap.parse_args()

    df, train_df, test_df, meta_df, feature_cols = load_and_merge(args.data)
    pre = build_preprocessor(df, feature_cols)

    pipe, rec = fit_and_eval(df, pre, args.mode, meta_df)
    print(json.dumps(rec, indent=2))

    # submission
    test_features = prepare_test_features(test_df, meta_df, feature_cols)
    id_col = "example_id" if "example_id" in test_df.columns else ("Id" if "Id" in test_df.columns else test_df.columns[0])
    preds = pipe.predict(test_features)

    fn = "submission_baseline.csv" if args.mode == "baseline" else "submission_optimized.csv"
    pd.DataFrame({"Id": test_df[id_col], "GreenScore": preds}).to_csv(fn, index=False)
    print("Saved", fn)

    # metrics file (append or create)
    rec_row = pd.DataFrame([rec])
    if os.path.exists("metrics_before_after.csv"):
        old = pd.read_csv("metrics_before_after.csv")
        out = pd.concat([old, rec_row], ignore_index=True).drop_duplicates(subset=["Scenario"], keep="last")
    else:
        out = rec_row
    out.to_csv("metrics_before_after.csv", index=False)
    print("Saved metrics_before_after.csv")

if __name__ == "__main__":
    main()
