from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import joblib
import os
import traceback

app = FastAPI(title="Churn Prediction API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Load Model ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_final.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"❌ ไม่พบไฟล์โมเดลที่: {MODEL_PATH}")

# โหลดข้อมูล Model Pipeline และ Metadata
saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]
threshold = saved_data["threshold"]
features_names = saved_data["features"]

@app.get("/")
def root():
    return {"status": "online", "model_info": "XGBoost + SMOTE Enhanced"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # 1. อ่านไฟล์ข้อมูล
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="รองรับเฉพาะ .csv หรือ .xlsx")

        # 2. เตรียมข้อมูล X (Feature Engineering) - ต้องเหมือนตอน Train 100%
        X = df.copy()

        # จัดการค่าว่างและแปลงประเภทข้อมูล
        X.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)

        # 🚀 [สูตรลับ] สร้างฟีเจอร์ใหม่เพื่อความแม่นยำ
        # A. นับจำนวนบริการเสริมที่ใช้
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        if all(col in X.columns for col in service_cols):
            X['TotalServices'] = (X[service_cols] == 'Yes').sum(axis=1)

        # B. เช็คว่าเป็นการจ่ายอัตโนมัติหรือไม่
        if 'PaymentMethod' in X.columns:
            X['IsAutomaticPayment'] = X['PaymentMethod'].str.contains('automatic', case=False).astype(int)

        # C. แบ่งกลุ่ม Tenure และคำนวณค่าเฉลี่ย
        if 'tenure' in X.columns:
            X['TenureGroup'] = pd.cut(
                X['tenure'], 
                bins=[-1, 12, 24, 48, 100], 
                labels=['Short', 'Medium', 'Long', 'VeryLong']
            )
            X["AvgChargesPerMonth"] = X["TotalCharges"] / (X["tenure"] + 1)
            X["IsLongTerm"] = (X["tenure"] > 24).astype(int)

        # D. ฟีเจอร์เพิ่มเติม (ถ้ามีในตอนเทรน)
        if "OnlineSecurity" in X.columns:
            X["HasSecurity"] = (X["OnlineSecurity"] == "Yes").astype(int)
        if "TechSupport" in X.columns:
            X["HasTechSupport"] = (X["TechSupport"] == "Yes").astype(int)

        # ลบคอลัมน์ที่ไม่เกี่ยวข้องออก
        drop_cols = ["customerID", "Churn", "churn_prediction", "churn_prob"]
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')

        # ✅ จัดเรียงคอลัมน์ให้ตรงกับโมเดล (สำคัญที่สุด)
        X = X.reindex(columns=features_names, fill_value=0)

        # 3. พยากรณ์ (ใช้ Best Threshold จากตอนเทรน)
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities > threshold).astype(int)

        # 4. ใส่ผลลัพธ์กลับเข้า DataFrame ต้นฉบับ
        df["churn_prediction"] = predictions.tolist()
        df["churn_prob"] = np.round(probabilities * 100, 2)

        # 5. สรุปผลทางสถิติ
        total = len(df)
        churn_count = int(np.sum(predictions))
        
        risk_by_contract = []
        if "Contract" in df.columns:
            grouped = df.groupby("Contract")["churn_prediction"].mean() * 100
            for contract, rate in grouped.items():
                risk_by_contract.append({
                    "type": str(contract), 
                    "churn_rate": round(float(rate), 2)
                })

        return {
            "total_customers": total,
            "churn_count": churn_count,
            "non_churn_count": total - churn_count,
            "churn_rate": round((churn_count / total) * 100, 2),
            "best_threshold_used": round(float(threshold), 2),
            "risk_by_contract": risk_by_contract,
            "details": df.replace({np.nan: None}).to_dict(orient="records")
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))