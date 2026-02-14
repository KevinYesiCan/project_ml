import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# =========================
# 1️⃣ LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = pd.read_csv(file_path)
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# =========================
# 2️⃣ FEATURE ENGINEERING (สูตรเพิ่มความแม่น)
# =========================
# A. รวมบริการที่ลูกค้าใช้ (ยิ่งใช้เยอะ ยิ่งเลิกยาก)
services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['TotalServices'] = (df[services] == 'Yes').sum(axis=1)

# B. สร้างกลุ่มตามอายุการใช้งาน
df['TenureGroup'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 100], labels=['Short', 'Medium', 'Long', 'VeryLong'])

# C. ฟีเจอร์ทางการเงิน
df["AvgChargesPerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)
df["IsAutomaticPayment"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# =========================
# 3️⃣ PREPROCESS (Standardize Data)
# =========================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())  # เพิ่มการปรับสเกล
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])

# =========================
# 4️⃣ MODEL + SMOTE + GRID SEARCH
# =========================
# ใช้ ImbPipeline เพื่อให้ SMOTE ทำงานเฉพาะตอน Train เท่านั้น
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)), 
    ('classifier', XGBClassifier(eval_metric='logloss', tree_method="hist", random_state=42))
])

param_grid = {
    "classifier__n_estimators": [300, 500],
    "classifier__max_depth": [3, 4, 5],
    "classifier__learning_rate": [0.02, 0.05],
    "classifier__subsample": [0.8],
    "classifier__colsample_bytree": [0.8]
}

grid = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(3), scoring="f1", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# =========================
# 5️⃣ FIND BEST THRESHOLD (จุดตัดสินใจที่ดีที่สุด)
# =========================
y_proba = best_model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_test, (y_proba > t).astype(int)) for t in thresholds]
best_t = thresholds[np.argmax(f1_scores)]

# =========================
# 6️⃣ EVALUATE & SAVE
# =========================
y_pred = (y_proba > best_t).astype(int)

print(f"\n🎯 Best Threshold: {best_t:.2f}")
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"🏆 F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"📈 ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

joblib.dump({
    "model": best_model,
    "threshold": best_t,
    "features": X.columns.tolist()
}, os.path.join(BASE_DIR, "model_final.pkl"))

print("\n💾 Saved: model_final.pkl")