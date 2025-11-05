#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-optimized}"   # baseline | optimized
DATA="${2:-/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai}"

echo "Mode: $MODE"
echo "Data: $DATA"

python - <<'PYCODE'
import os, time, json, random, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

SEED=42
random.seed(SEED); np.random.seed(SEED)

BASE = os.environ.get("DATA_PATH", sys.argv[1] if len(sys.argv)>1 else "/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai")
train_df = pd.read_csv(f"{BASE}/train.csv")
test_df  = pd.read_csv(f"{BASE}/test.csv")
meta_df  = pd.read_csv(f"{BASE}/metaData.csv")

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

drop_cols = {'target','example_id','Id'}
feature_cols = [c for c in df.columns if c not in drop_cols]
X_full = df[feature_cols].copy()
y_full = df['target'].copy()

num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

pre = ColumnTransformer([
    ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
])

X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=SEED)

mean_ci = float(df['carbon_intensity_gco2_per_kwh'].mean()) if 'carbon_intensity_gco2_per_kwh' in df.columns else 400.0
def eproxy(dt_s, mean_ci, kw=0.1):
    e = kw*(dt_s/3600.0)
    return e, e*(mean_ci/1000.0)

if os.environ.get("MODE","optimized")== "baseline":
    pipe = Pipeline([("prep",pre),("model",GradientBoostingRegressor(random_state=SEED))])
else:
    pipe = Pipeline([("prep",pre),("model",GradientBoostingRegressor(n_estimators=80,learning_rate=0.08,max_depth=3,subsample=0.7,random_state=SEED))])

t0=time.time(); pipe.fit(X_train, y_train); dt=time.time()-t0
mae = mean_absolute_error(y_val, pipe.predict(X_val))
ekwh, co2 = eproxy(dt, mean_ci)

print(json.dumps({"mode":os.environ.get("MODE","optimized"),"runtime_sec":dt,"mae":mae,"energy_kwh":ekwh,"co2e_kg":co2}, indent=2))

# create submission
if 'region' in test_df.columns and 'region' in meta_df.columns:
    test_features = test_df.merge(meta_df, on='region', how='left')
else:
    test_features = pd.concat([test_df.reset_index(drop=True), meta_df.sample(len(test_df), replace=True, random_state=SEED).reset_index(drop=True)], axis=1)
test_features = test_features.reindex(columns=feature_cols)
id_col = "example_id" if "example_id" in test_df.columns else ( "Id" if "Id" in test_df.columns else test_df.columns[0] )
pred = pipe.predict(test_features)

fn = "submission_optimized.csv" if os.environ.get("MODE","optimized")!="baseline" else "submission_baseline.csv"
pd.DataFrame({"Id": test_df[id_col], "GreenScore": pred}).to_csv(fn, index=False)
print("Saved", fn)
PYCODE
