from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

S3_BUCKET    = os.environ["S3_BUCKET"]
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH   = os.path.expanduser("~/models/model.pkl")


def download_model():
    # TODO 1: Tao S3 client
    client = boto3.client("s3")

    # TODO 2 & 3: Tai file model ve may
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    client.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)

    # TODO 4: In thong bao thanh cong
    print(f"Model da duoc tai xuong tu s3://{S3_BUCKET}/{S3_MODEL_KEY}")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    # TODO 5: Tra ve {"status": "ok"}
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    # TODO 6: Kiem tra so luong dac trung
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Can dung 12 dac trung")

    # TODO 7: Goi model du doan
    pred = model.predict([req.features])[0]

    # TODO 8: Tra ve ket qua
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": int(pred), "label": labels[int(pred)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
