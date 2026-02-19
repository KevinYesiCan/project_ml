import os
import io
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# =========================
# App Init
# =========================
app = FastAPI(title="Churn Ensemble API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Load Models
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb.pkl"))
lgbm_model = joblib.load(os.path.join(MODEL_DIR, "lgbm.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))
threshold = float(joblib.load(os.path.join(MODEL_DIR, "threshold.pkl")))

@app.get("/")
def home():
    return {"message": "Churn Ensemble Model Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return JSONResponse(status_code=400, content={"error": "Please upload a CSV file."})

    try:
        contents = await file.read()
        df_input = pd.read_csv(io.BytesIO(contents))
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid CSV format."})

    if df_input.empty:
        return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})

    original_df = df_input.copy()

    # =====================
    # Preprocessing
    # =====================
    if "TotalCharges" in df_input.columns:
        df_input["TotalCharges"] = pd.to_numeric(df_input["TotalCharges"], errors="coerce")
        df_input.fillna(0, inplace=True)

    # get dummies
    df_proc = pd.get_dummies(df_input, drop_first=False)
    for col in columns:
        if col not in df_proc.columns:
            df_proc[col] = 0
    df_proc = df_proc[[col for col in columns]]

    # Scaling
    scaled_data = scaler.transform(df_proc)

    # =====================
    # Ensemble Prediction
    # =====================
    xgb_prob = xgb_model.predict_proba(scaled_data)[:, 1]
    lgb_prob = lgbm_model.predict_proba(scaled_data)[:, 1]
    
    final_prob = (xgb_prob + lgb_prob) / 2
    prediction = (final_prob >= threshold).astype(int)

    # Attach Results
    original_df["churn_prob"] = final_prob
    original_df["churn_prediction"] = prediction

    # =====================
    # Aggregation for Dashboard
    # =====================
    # 1. Risk by Contract (แก้จุดที่ตารางไม่ขึ้น)
    def calculate_group_risk(df, group_col):
        if group_col not in df.columns: return []
        stats = df.groupby(group_col).agg(
            total=(group_col, 'count'),
            churn=('churn_prediction', 'sum')
        ).reset_index()
        stats['churn_rate'] = ((stats['churn'] / stats['total']) * 100).round(1)
        return stats.rename(columns={group_col: "type"}).to_dict(orient="records")

    risk_by_contract = calculate_group_risk(original_df, "Contract")

    # 2. Feature Importance (ปัจจัยสำคัญ)
    # ใช้ค่าเฉลี่ยความสำคัญจากทั้ง 2 โมเดล
    importance = (xgb_model.feature_importances_ + lgbm_model.feature_importances_) / 2
    feat_imp = pd.DataFrame({'feature': columns, 'importance': importance})
    feat_imp = feat_imp.sort_values('importance', ascending=False).head(10).to_dict(orient="records")

    total_customers = len(original_df)
    churn_count = int(prediction.sum())
    churn_rate = round((churn_count / total_customers) * 100, 2)

    return {
        "total_customers": total_customers,
        "churn_count": churn_count,
        "churn_rate": churn_rate,
        "threshold": threshold,
        "risk_by_contract": risk_by_contract,  # ✅ ส่งตัวนี้ไปตารางถึงจะขึ้น
        "feature_importance": feat_imp,
        "details": original_df.to_dict(orient="records")
    }