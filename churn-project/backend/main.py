from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import joblib
import os
import traceback

app = FastAPI(title="Churn Prediction API")

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

# โหลดข้อมูลที่เซฟไว้
saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]
threshold = saved_data["threshold"]
features_names = saved_data["features"]
# เพิ่มในส่วนเตรียมข้อมูล (X) ของ main.py
X = df.copy()

# 1. จัดการค่าตัวเลข
X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)

# 2. สร้าง Feature ใหม่ (ต้องเหมือนใน train_model.py เป๊ะๆ)
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
if all(col in X.columns for col in service_cols):
    X['TotalServices'] = (X[service_cols] == 'Yes').sum(axis=1)

if 'PaymentMethod' in X.columns:
    X['IsAutomaticPayment'] = X['PaymentMethod'].str.contains('automatic').astype(int)

if 'tenure' in X.columns:
    X['TenureGroup'] = pd.cut(X['tenure'], bins=[0, 12, 24, 48, 100], labels=['Short', 'Medium', 'Long', 'VeryLong'])
    X["AvgChargesPerMonth"] = X["TotalCharges"] / (X["tenure"] + 1)

# 3. จัดการคอลัมน์ให้ตรง
X = X.reindex(columns=features_names, fill_value=0)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="รองรับเฉพาะ .csv หรือ .xlsx")

        # 1. เตรียมข้อมูล X (Preprocessing)
        X = df.copy()

        # แปลง TotalCharges เป็นตัวเลข
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)

        # 🔥 สร้าง Feature ใหม่ให้เหมือนตอน Train (สำคัญมาก)
        if "tenure" in X.columns and "TotalCharges" in X.columns:
            X["AvgChargesPerMonth"] = X["TotalCharges"] / (X["tenure"] + 1)
            X["IsLongTerm"] = (X["tenure"] > 24).astype(int)
        
        if "OnlineSecurity" in X.columns:
            X["HasSecurity"] = (X["OnlineSecurity"] == "Yes").astype(int)
        
        if "TechSupport" in X.columns:
            X["HasTechSupport"] = (X["TechSupport"] == "Yes").astype(int)

        # ลบคอลัมน์ที่ไม่เกี่ยวข้อง (ลบแบบปลอดภัย แม้ไม่มี Churn ก็ไม่พัง)
        drop_cols = ["customerID", "Churn", "churn_prediction", "churn_prob"]
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')

        # จัดเรียงคอลัมน์ให้ตรงกับตอนเทรน 100%
        X = X.reindex(columns=features_names, fill_value=0)

        # 2. พยากรณ์
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities > threshold).astype(int)

        # 3. ใส่ผลกลับเข้า df หลัก
        df["churn_prediction"] = predictions.tolist()
        df["churn_prob"] = np.round(probabilities * 100, 2)

        # 4. สรุปผล
        total = len(df)
        churn_count = int(np.sum(predictions))
        
        risk_by_contract = []
        if "Contract" in df.columns:
            grouped = df.groupby("Contract")["churn_prediction"].mean() * 100
            for contract, rate in grouped.items():
                risk_by_contract.append({"type": str(contract), "churn_rate": round(float(rate), 2)})

        return {
            "total_customers": total,
            "churn_count": churn_count,
            "non_churn_count": total - churn_count,
            "churn_rate": round((churn_count / total) * 100, 2),
            "risk_by_contract": risk_by_contract,
            "details": df.replace({np.nan: None}).to_dict(orient="records")
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))