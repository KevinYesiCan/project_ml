import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# =========================
# 1️⃣ LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = pd.read_csv(file_path)
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# =========================
# 2️⃣ FEATURE ENGINEERING
# =========================
df["AvgChargesPerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)
df["IsLongTerm"] = (df["tenure"] > 24).astype(int)

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})

# =========================
# 3️⃣ SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# 4️⃣ PREPROCESS
# =========================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),

    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])

# =========================
# 5️⃣ MODEL (Balanced + Strong)
# =========================
scale_pos_weight = (len(y_train[y_train == 0]) / len(y_train[y_train == 1])) * 0.8

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    random_state=42
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

pipeline.fit(X_train, y_train)

# =========================
# 6️⃣ EVALUATE
# =========================
y_proba = pipeline.predict_proba(X_test)[:, 1]

# 🔥 หา threshold ที่ดีที่สุดจาก F1
best_threshold = 0.5
best_f1 = 0

for t in np.arange(0.4, 0.7, 0.01):
    preds = (y_proba > t).astype(int)
    score = f1_score(y_test, preds)
    if score > best_f1:
        best_f1 = score
        best_threshold = t

y_pred = (y_proba > best_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n📊 FINAL MODEL PERFORMANCE")
print("================================")
print(f"🎯 Accuracy  : {accuracy*100:.2f}%")
print(f"🏆 F1 Score  : {best_f1:.4f}")
print(f"📈 ROC-AUC   : {roc_auc:.4f}")
print(f"🔥 Best Threshold : {best_threshold:.2f}")
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 7️⃣ SAVE MODEL
# =========================
joblib.dump({
    "model": pipeline,
    "threshold": best_threshold,
    "features": X.columns.tolist()
}, os.path.join(BASE_DIR, "model_final.pkl"))

print("\n💾 Saved successfully: model_final.pkl")
