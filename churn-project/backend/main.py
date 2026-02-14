from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io, joblib, os, traceback

app = FastAPI(title="Churn Dual-Engine API")

# ---------------------------------------------------------
# 1. INITIALIZATION & CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_final.pkl")

# โหลดโมเดลและพารามิเตอร์ที่บันทึกไว้
if not os.path.exists(MODEL_PATH):
    print("⚠️ Warning: Model file not found. Please run train_model.py first.")
else:
    artifacts = joblib.load(MODEL_PATH)
    model_lr = artifacts["model_lr"]
    model_xgb = artifacts["model_xgb"]
    best_threshold = artifacts["threshold"]
    feature_cols = artifacts["features"]
    print(f"✅ Model Loaded Successfully (Threshold: {best_threshold:.2f})")

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def process_data(df_input):
    """ทำ Feature Engineering ให้เหมือนกับขั้นตอนการ Train ทุกประการ"""
    X = df_input.copy()
    
    # ล้างข้อมูลเบื้องต้น
    X.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)
    
    # [Power Feature 1] Total Services
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    if all(s in X.columns for s in services):
        X['TotalServices'] = (X[services] == 'Yes').sum(axis=1)
    
    # [Power Feature 2] Is Automatic Payment
    if 'PaymentMethod' in X.columns:
        X['IsAutomaticPayment'] = X['PaymentMethod'].str.contains("automatic", case=False).astype(int)
        
    # [Power Feature 3] Tenure Groups & Avg Charges
    if 'tenure' in X.columns:
        X['TenureGroup'] = pd.cut(X['tenure'], bins=[-1, 12, 24, 48, 100], labels=['Short', 'Medium', 'Long', 'VeryLong'])
        X["AvgChargesPerMonth"] = X["TotalCharges"] / (X["tenure"] + 1)
        
    # [Power Feature 4] Charge Scale
    if 'MonthlyCharges' in X.columns:
        avg_monthly = X['MonthlyCharges'].mean() if not X['MonthlyCharges'].empty else 0
        X["ChargeScale"] = X["MonthlyCharges"] / (avg_monthly + 1e-6)
        
    # จัดเรียงคอลัมน์ให้ตรงตามชุด Train (Reindex)
    return X.reindex(columns=feature_cols, fill_value=0)

# ---------------------------------------------------------
# 3. ENDPOINTS
# ---------------------------------------------------------
@app.get("/")
def health():
    return {"status": "online", "engine": "Dual-Model Ensemble (XGBoost 70% + LR 30%)"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. อ่านไฟล์ข้อมูล
        content = await file.read()
        if file.filename.endswith('.csv'):
            df_raw = pd.read_csv(io.BytesIO(content))
        else:
            df_raw = pd.read_excel(io.BytesIO(content))
        
        # 2. เตรียมข้อมูล (Preprocessing)
        X_processed = process_data(df_raw)
        
        # 3. การทำนายผล (Inference)
        # ดึงความน่าจะเป็นจากทั้ง 2 โมเดล
        prob_lr = model_lr.predict_proba(X_processed)[:, 1]
        prob_xgb = model_xgb.predict_proba(X_processed)[:, 1]
        
        # รวมคะแนนแบบถ่วงน้ำหนัก (Weighted Ensemble) ตามสูตรที่เทรนมา
        final_probs = (prob_lr * 0.3) + (prob_xgb * 0.7)
        
        # ตัดเกณฑ์ที่ Threshold ที่ดีที่สุดจากตอน Train
        final_preds = (final_probs > best_threshold).astype(int)
        
        # 4. รวมผลลัพธ์กลับเข้า DataFrame หลัก
        df_raw["churn_prediction"] = final_preds
        df_raw["churn_probability"] = np.round(final_probs * 100, 2)
        
        # 5. สรุปข้อมูลราย Contract สำหรับแสดงกราฟใน UI
        risk_summary = []
        if "Contract" in df_raw.columns:
            # คำนวณ % ความเสี่ยงแยกตามประเภทสัญญา
            stats = df_raw.groupby("Contract")["churn_prediction"].mean() * 100
            risk_summary = [{"type": k, "churn_rate": round(v, 2)} for k, v in stats.items()]

        # 6. ส่งค่ากลับให้ Frontend (โครงสร้าง JSON ตรงตามหน้า UI)
        return {
            "total_customers": len(df_raw),
            "churn_count": int(final_preds.sum()),
            "churn_rate": round(float(final_preds.mean() * 100), 2),
            "summary": {
                "avg_risk": round(float(final_probs.mean() * 100), 2),
                "threshold_used": float(best_threshold)
            },
            "risk_by_contract": risk_summary,
            "details": df_raw.replace({np.nan: None}).to_dict(orient="records")
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)