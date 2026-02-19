import os
import joblib
import numpy as np
import pandas as pd
import optuna
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# =============================
# CONFIG
# =============================
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

COST_CALL = 100
GAIN_SAVE = 2000
RANDOM_STATE = 42

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

# ไม่ drop_first เพื่อไม่ให้ category หาย
X = pd.get_dummies(X, drop_first=False)
columns = X.columns.tolist()

# =============================
# TRAIN / TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=columns
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=columns
)

# =============================
# OPTUNA (XGB on TRAIN ONLY)
# =============================
def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }

    model = XGBClassifier(**params)

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    aucs = []

    for train_idx, val_idx in skf.split(X_train_scaled, y_train):
        model.fit(
            X_train_scaled.iloc[train_idx],
            y_train.iloc[train_idx]
        )
        preds = model.predict_proba(
            X_train_scaled.iloc[val_idx]
        )[:, 1]

        aucs.append(
            roc_auc_score(
                y_train.iloc[val_idx],
                preds
            )
        )

    return np.mean(aucs)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# =============================
# Train Best Models
# =============================
best_xgb = XGBClassifier(
    **study.best_params,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
best_xgb.fit(X_train_scaled, y_train)

lgbm = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
lgbm.fit(X_train_scaled, y_train)

# =============================
# EVALUATE ON TEST
# =============================
xgb_probs = best_xgb.predict_proba(X_test_scaled)[:, 1]
lgb_probs = lgbm.predict_proba(X_test_scaled)[:, 1]
ensemble_probs = (xgb_probs + lgb_probs) / 2

auc = roc_auc_score(y_test, ensemble_probs)
print("REAL TEST ROC-AUC:", auc)

# =============================
# PROFIT SIMULATION
# =============================
best_profit = -np.inf
best_threshold = 0.5

for t in np.linspace(0.1, 0.9, 100):
    preds = (ensemble_probs >= t).astype(int)
    tp = np.sum((preds == 1) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))

    profit = (tp * GAIN_SAVE) - ((tp + fp) * COST_CALL)

    if profit > best_profit:
        best_profit = profit
        best_threshold = t

print("Best Profit Threshold:", best_threshold)
print("Expected Profit (Test):", best_profit)

# =============================
# SHAP
# =============================
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_train_scaled.iloc[:500])

plt.figure()
shap.summary_plot(
    shap_values,
    X_train_scaled.iloc[:500],
    show=False
)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "shap_summary.png"))
plt.close()

# =============================
# RETRAIN ON FULL DATA
# =============================
X_full_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=columns
)

final_xgb = XGBClassifier(
    **study.best_params,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
final_xgb.fit(X_full_scaled, y)

final_lgbm = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
final_lgbm.fit(X_full_scaled, y)

# =============================
# SAVE
# =============================
joblib.dump(final_xgb, os.path.join(MODEL_DIR, "xgb.pkl"))
joblib.dump(final_lgbm, os.path.join(MODEL_DIR, "lgbm.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(columns, os.path.join(MODEL_DIR, "columns.pkl"))
joblib.dump(best_threshold, os.path.join(MODEL_DIR, "threshold.pkl"))

print("ALL MODELS SAVED SUCCESSFULLY")
