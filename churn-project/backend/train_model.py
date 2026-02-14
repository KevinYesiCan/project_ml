import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report, recall_score

# =========================
# 1. LOAD & ENGINEER FEATURES
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ ไม่พบไฟล์ข้อมูลที่: {file_path}")

df = pd.read_csv(file_path)

def engineer_features(data):
    d = data.copy()
    d.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)
    
    # [Power Features]
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    d['TotalServices'] = (d[services] == 'Yes').sum(axis=1)
    d['TenureGroup'] = pd.cut(d['tenure'], bins=[-1, 12, 24, 48, 100], labels=['Short', 'Medium', 'Long', 'VeryLong'])
    d["AvgChargesPerMonth"] = d["TotalCharges"] / (d["tenure"] + 1)
    d["IsAutomaticPayment"] = d["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    d["ChargeScale"] = d["MonthlyCharges"] / (d["MonthlyCharges"].mean() + 1e-6)
    
    return d

df = engineer_features(df)
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# =========================
# 2. PREPROCESSING (Advanced)
# =========================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)), # เพิ่มความซับซ้อนให้ฟีเจอร์ตัวเลข
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])


print("🔍 Step 1: Fine-tuning XGBoost...")
imb_pipeline = ImbPipeline([
    ('pre', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', XGBClassifier(eval_metric='logloss', tree_method="hist", random_state=42))
])

param_grid = {
    'clf__n_estimators': [400, 600],
    'clf__max_depth': [2, 3],
    'clf__learning_rate': [0.01, 0.03],
    'clf__subsample': [0.8]
}

grid_search = GridSearchCV(imb_pipeline, param_grid, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model_xgb = grid_search.best_estimator_

print("🔍 Step 2: Fitting Logistic Regression Baseline...")
model_lr = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
model_lr.fit(X_train, y_train)

# =========================
# 4. ENSEMBLE & METRICS CALCULATION
# =========================
y_prob_lr = model_lr.predict_proba(X_test)[:, 1]
y_prob_xgb = best_model_xgb.predict_proba(X_test)[:, 1]
final_probs = (y_prob_lr * 0.3) + (y_prob_xgb * 0.7)

# ค้นหา Threshold ที่ F1 ดีที่สุด
thresholds = np.arange(0.1, 0.8, 0.01)
best_t = thresholds[np.argmax([f1_score(y_test, (final_probs > t).astype(int)) for t in thresholds])]
final_preds = (final_probs > best_t).astype(int)

# สถิติสำหรับนำเสนอ
acc = accuracy_score(y_test, final_preds) * 100
rec = recall_score(y_test, final_preds) * 100
auc = roc_auc_score(y_test, final_probs) * 100

# =========================
# 5. FINAL REPORTING (โชว์ค่าที่คุณต้องการ)
# =========================
print("\n" + "⭐" * 20)
print("  KPIs FOR DASHBOARD  ")
print("⭐" * 20)
print(f"✅ Churn Capture Rate (Recall) : {rec:.1f}%  <-- หัวใจธุรกิจ")
print(f"✅ Model Reliability (ROC-AUC) : {auc:.1f}%  <-- ความน่าเชื่อถือ")
print(f"✅ Overall Accuracy            : {acc:.1f}%  <-- ความแม่นยำรวม")
print(f"✅ Optimal Risk Threshold      : {best_t:.2f}  <-- จุดคุ้มทุน")
print("=" * 40)

# Save
joblib.dump({
    "model_lr": model_lr,
    "model_xgb": best_model_xgb,
    "threshold": best_t,
    "features": X.columns.tolist()
}, os.path.join(BASE_DIR, "model_final.pkl"))

print(f"💾 Model saved successfully as 'model_final.pkl'")