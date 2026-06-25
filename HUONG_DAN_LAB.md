# Hướng Dẫn Thực Hành - Day 21: CI/CD for AI Systems

> **Cầm tay chỉ việc** - Làm từng bước một theo thứ tự, không bỏ qua.

---

## Tổng Quan Bài Lab

Bài lab mô phỏng một hệ thống MLOps hoàn chỉnh:

```
Local Machine                GitHub                  Cloud
─────────────────      ─────────────────────      ──────────────────
Viết code         ──►  GitHub Actions:            GCS Bucket:
Chạy MLflow            1. Unit Test          ──►  - Dữ liệu (DVC)
Thêm dữ liệu ──►  ──►  2. Train                   - Model mới nhất
                        3. Eval Gate (≥0.70)
                        4. Deploy via SSH     ──►  VM (FastAPI :8000)
```

**Dataset:** Wine Quality (UCI) - phân loại chất lượng rượu vang thành 3 mức (thấp/trung bình/cao)

**Bạn cần hoàn thành:**
- `src/train.py` - 10 TODO
- `src/serve.py` - 8 TODO
- `tests/test_train.py` - 9 TODO
- `.github/workflows/mlops.yml` - 8 TODO

---

## Chuẩn Bị Ban Đầu

### 1. Clone và tạo môi trường

```bash
cd Day21-Track2-CI-CD-for-AI-Systems

# Tạo virtual environment
python -m venv .venv

# Kích hoạt (Windows)
.venv\Scripts\activate

# Kích hoạt (Mac/Linux)
source .venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Tạo dữ liệu

```bash
python generate_data.py
```

Kết quả mong đợi:
```
train_phase1.csv : 2998 mau
eval.csv         :  500 mau
train_phase2.csv : 2998 mau
```

---

## BƯỚC 1: MLflow Local (2-3 giờ)

**Mục tiêu:** Chạy ít nhất 3 thí nghiệm với hyperparameter khác nhau, ghi lại bằng MLflow.

### 1.1 Hoàn thành `src/train.py`

Mở file `src/train.py` và thay thế toàn bộ nội dung bằng code sau:

```python
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    # TODO 1: Đọc dữ liệu
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    # TODO 2: Tách đặc trưng và nhãn
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    with mlflow.start_run():

        # TODO 3: Ghi nhận siêu tham số
        mlflow.log_params(params)

        # TODO 4: Khởi tạo và huấn luyện
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Dự đoán và tính chỉ số
        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        # TODO 6: Ghi nhận chỉ số vào MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In kết quả
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # TODO 8: Lưu metrics ra file
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        # TODO 9: Lưu model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Trả về accuracy
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
```

### 1.2 Set biến môi trường MLflow

```bash
# Windows PowerShell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
$env:MLFLOW_ARTIFACT_ROOT = "./mlartifacts"

# Mac/Linux
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export MLFLOW_ARTIFACT_ROOT=./mlartifacts
```

### 1.3 Chạy 3 thí nghiệm (bắt buộc ≥3 lần)

**Lần 1** - Mặc định (params.yaml hiện tại: n_estimators=100, max_depth=5):
```bash
python src/train.py
```

**Lần 2** - Sửa `params.yaml`:
```yaml
n_estimators: 50
max_depth: 3
min_samples_split: 5
```
```bash
python src/train.py
```

**Lần 3** - Sửa lại `params.yaml`:
```yaml
n_estimators: 200
max_depth: 10
min_samples_split: 2
```
```bash
python src/train.py
```

Sau khi chạy xong 3 lần, **đặt lại params.yaml về bộ tốt nhất** (thường là n_estimators=200, max_depth=10).

### 1.4 Xem kết quả trên MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Truy cập: http://localhost:5000

- So sánh các run, chọn run có accuracy cao nhất
- **Chụp screenshot** giao diện MLflow (cần nộp)

### Kiểm tra Bước 1 xong chưa

```bash
# Phải có 2 file này
ls outputs/metrics.json
ls models/model.pkl
```

---

## BƯỚC 2: CI/CD Pipeline + Cloud Deploy (4-5 giờ)

### 2.1 Chọn Cloud Provider và tạo bucket

> Chọn 1 trong 3: **GCP** (khuyên dùng nếu có tài khoản Google), AWS, hoặc Azure.

#### Option A - GCP (Google Cloud Platform)

```bash
# Cài Google Cloud SDK nếu chưa có: https://cloud.google.com/sdk/docs/install

# Đăng nhập
gcloud auth login

# Tạo bucket (đặt tên duy nhất, ví dụ: mlops-lab-yourname-2024)
export PROJECT=$(gcloud config get-value project)
export BUCKET="mlops-lab-yourname-2024"   # <-- ĐỔI THÀNH TÊN CỦA BẠN
gsutil mb -p $PROJECT -l us-central1 gs://$BUCKET
```

Tạo Service Account để GitHub Actions dùng:
```bash
# Tạo service account
gcloud iam service-accounts create mlops-sa \
    --display-name="MLOps Service Account"

# Cấp quyền storage
gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:mlops-sa@$PROJECT.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Tải key về
gcloud iam service-accounts keys create sa-key.json \
    --iam-account=mlops-sa@$PROJECT.iam.gserviceaccount.com
```

#### Option B - AWS

```bash
export BUCKET="mlops-lab-yourname-2024"
aws s3 mb s3://$BUCKET --region us-east-1
# Credentials: tạo IAM User có quyền S3FullAccess, tải Access Key
```

#### Option C - Azure

```bash
az storage container create --name mlops-data --account-name <your-storage-account>
# Lấy Connection String từ Azure Portal → Storage Account → Access Keys
```

### 2.2 Cài đặt DVC và đẩy dữ liệu lên cloud

```bash
# Khởi tạo DVC
dvc init

# --- GCP ---
dvc remote add -d myremote gs://$BUCKET/dvc
dvc remote modify myremote credentialpath sa-key.json

# --- AWS ---
# dvc remote add -d myremote s3://$BUCKET/dvc

# --- Azure ---
# dvc remote add -d myremote azure://mlops-data/dvc
# dvc remote modify myremote connection_string "<your-connection-string>"

# Thêm dữ liệu vào DVC (tạo file .dvc thay vì lưu CSV vào git)
dvc add data/train_phase1.csv
dvc add data/eval.csv
dvc add data/train_phase2.csv

# Commit các file .dvc vào git
git add data/train_phase1.csv.dvc data/eval.csv.dvc data/train_phase2.csv.dvc
git add .dvc/config .gitignore
git commit -m "feat: track datasets with DVC"

# Đẩy dữ liệu lên cloud
dvc push
```

Kiểm tra thành công: lên GCS Console / S3 Console / Azure Portal xem thư mục `dvc/` đã xuất hiện chưa.

### 2.3 Tạo Cloud VM

#### GCP VM

```bash
# Tạo VM
gcloud compute instances create mlops-serve \
    --zone=us-central1-a \
    --machine-type=e2-small \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=20GB

# Mở port 8000
gcloud compute firewall-rules create allow-mlops \
    --allow=tcp:8000 \
    --target-tags=mlops-serve \
    --description="Allow MLOps inference API"

gcloud compute instances add-tags mlops-serve \
    --tags=mlops-serve --zone=us-central1-a

# Lấy IP
gcloud compute instances describe mlops-serve \
    --zone=us-central1-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

#### AWS EC2 (nếu dùng AWS)

```bash
aws ec2 run-instances \
    --image-id ami-0c7217cdde317cfec \
    --instance-type t3.micro \
    --key-name <your-key-pair> \
    --security-group-ids <sg-với-port-8000-mở> \
    --region us-east-1
```

### 2.4 Cài đặt môi trường trên VM

```bash
# SSH vào VM (GCP)
gcloud compute ssh mlops-serve --zone=us-central1-a

# Trên VM, chạy:
sudo apt update && sudo apt install -y python3-pip python3-venv
pip3 install fastapi uvicorn scikit-learn joblib google-cloud-storage
mkdir -p ~/models ~/src
exit
```

Copy credentials lên VM:
```bash
# GCP
gcloud compute scp sa-key.json mlops-serve:~/sa-key.json --zone=us-central1-a
```

### 2.5 Hoàn thành `src/serve.py`

Mở file `src/serve.py` và thay thế toàn bộ bằng:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

GCS_BUCKET    = os.environ["GCS_BUCKET"]
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH    = os.path.expanduser("~/models/model.pkl")


def download_model():
    # TODO 1: Tạo storage client
    client = storage.Client()

    # TODO 2: Lấy bucket và blob
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(GCS_MODEL_KEY)

    # TODO 3: Tải file model về máy
    blob.download_to_filename(MODEL_PATH)

    # TODO 4: In thông báo
    print("Model da duoc tai xuong tu GCS.")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    # TODO 5: Trả về {"status": "ok"}
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    # TODO 6: Kiểm tra số lượng đặc trưng
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Can dung 12 dac trung")

    # TODO 7: Gọi model dự đoán
    pred = model.predict([req.features])[0]

    # TODO 8: Trả về kết quả
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": int(pred), "label": labels[int(pred)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Copy serve.py lên VM:
```bash
gcloud compute scp src/serve.py mlops-serve:~/src/serve.py --zone=us-central1-a
```

### 2.6 Tạo systemd service trên VM

SSH vào VM:
```bash
gcloud compute ssh mlops-serve --zone=us-central1-a
```

Chạy lệnh sau trên VM (thay `YOUR_BUCKET_NAME`):
```bash
# Lấy username hiện tại
echo $USER    # ghi nhớ giá trị này cho GitHub Secret VM_USER

# Tạo service file
sudo tee /etc/systemd/system/mlops-serve.service > /dev/null <<EOF
[Unit]
Description=MLOps Model Inference Server
After=network.target

[Service]
User=$USER
WorkingDirectory=/home/$USER
Environment="GCS_BUCKET=YOUR_BUCKET_NAME"
Environment="GOOGLE_APPLICATION_CREDENTIALS=/home/$USER/sa-key.json"
ExecStart=/usr/bin/python3 /home/$USER/src/serve.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mlops-serve
exit
```

### 2.7 Tạo SSH key để GitHub Actions deploy

Chạy trên máy LOCAL:
```bash
# Tạo key pair
ssh-keygen -t ed25519 -f ~/.ssh/mlops_deploy -N "" -C "github-actions-deploy"

# Thêm public key vào VM
gcloud compute ssh mlops-serve --zone=us-central1-a \
    --command "echo '$(cat ~/.ssh/mlops_deploy.pub)' >> ~/.ssh/authorized_keys"

# Xem nội dung private key (sẽ dùng cho GitHub Secret)
cat ~/.ssh/mlops_deploy
```

### 2.8 Thêm GitHub Secrets

Vào repo GitHub → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Thêm 5 secrets:

| Secret Name | Giá trị |
|-------------|---------|
| `CLOUD_CREDENTIALS` | Toàn bộ nội dung file `sa-key.json` |
| `CLOUD_BUCKET` | Tên bucket, ví dụ `mlops-lab-yourname-2024` |
| `VM_HOST` | IP public của VM |
| `VM_USER` | Username trên VM (kết quả `echo $USER` trên VM) |
| `VM_SSH_KEY` | Toàn bộ nội dung file `~/.ssh/mlops_deploy` (private key) |

### 2.9 Hoàn thành `tests/test_train.py`

Thay thế toàn bộ nội dung file:

```python
import os
import json
import numpy as np
import pandas as pd
from src.train import train

FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    rng = np.random.default_rng(0)
    n   = 200

    # TODO 1: Tạo ma trận đặc trưng
    X = rng.random((n, len(FEATURE_NAMES)))

    # TODO 2: Tạo nhãn ngẫu nhiên
    y = rng.integers(0, 3, size=n)

    # TODO 3: Tạo DataFrame
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    # TODO 4: Lưu file train/eval
    train_path = str(tmp_path / "train.csv")
    eval_path  = str(tmp_path / "eval.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path,  index=False)

    # TODO 5: Trả về paths
    return train_path, eval_path


def test_train_returns_float(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)

    # TODO 6: Gọi hàm train()
    acc = train({"n_estimators": 10, "max_depth": 3},
                data_path=train_path, eval_path=eval_path)

    # TODO 7: Kiểm tra kiểu và khoảng giá trị
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    train({"n_estimators": 10, "max_depth": 3},
          data_path=train_path, eval_path=eval_path)

    # TODO 8: Kiểm tra file metrics
    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics


def test_model_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    train({"n_estimators": 10, "max_depth": 3},
          data_path=train_path, eval_path=eval_path)

    # TODO 9: Kiểm tra file model
    assert os.path.exists("models/model.pkl")
```

Kiểm tra test chạy được trên máy local:
```bash
pytest tests/ -v
```

Kết quả mong đợi:
```
tests/test_train.py::test_train_returns_float PASSED
tests/test_train.py::test_metrics_file_created PASSED
tests/test_train.py::test_model_file_created PASSED
3 passed
```

### 2.10 Hoàn thành `.github/workflows/mlops.yml`

Thay thế toàn bộ nội dung file:

```yaml
name: MLOps Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'data/**.dvc'
      - 'src/**.py'
      - 'params.yaml'
  workflow_dispatch:

jobs:

  # -------------------------------------------------------------------------
  # JOB 1 - UNIT TEST
  # -------------------------------------------------------------------------
  test:
    name: Unit Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run unit tests
        # TODO 1: Chạy pytest
        run: pytest tests/ -v


  # -------------------------------------------------------------------------
  # JOB 2 - TRAIN
  # -------------------------------------------------------------------------
  train:
    name: Train
    needs: test
    runs-on: ubuntu-latest
    outputs:
      accuracy: ${{ steps.read_metrics.outputs.accuracy }}

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Authenticate to Cloud Storage
        # TODO 2: Xác thực với GCS
        run: |
          echo '${{ secrets.CLOUD_CREDENTIALS }}' > /tmp/sa-key.json
          echo "GOOGLE_APPLICATION_CREDENTIALS=/tmp/sa-key.json" >> $GITHUB_ENV

      - name: Pull data with DVC
        # TODO 3: Pull dữ liệu
        run: dvc pull data/train_phase1.csv.dvc data/eval.csv.dvc

      - name: Train model
        run: python src/train.py

      - name: Read metrics
        id: read_metrics
        # TODO 4: Đọc accuracy và xuất ra output
        run: |
          ACC=$(python -c "import json; d=json.load(open('outputs/metrics.json')); print(d['accuracy'])")
          echo "accuracy=$ACC" >> $GITHUB_OUTPUT

      - name: Upload model to Cloud Storage
        # TODO 5: Upload model lên GCS
        run: |
          python - <<'EOF'
          from google.cloud import storage
          import os
          client = storage.Client()
          bucket = client.bucket("${{ secrets.CLOUD_BUCKET }}")
          blob   = bucket.blob("models/latest/model.pkl")
          blob.upload_from_filename("models/model.pkl")
          print("Model uploaded to GCS.")
          EOF

      - name: Save metrics as artifact
        uses: actions/upload-artifact@v4
        with:
          name: metrics
          path: outputs/metrics.json


  # -------------------------------------------------------------------------
  # JOB 3 - EVAL (Quality Gate)
  # -------------------------------------------------------------------------
  eval:
    name: Eval
    needs: train
    runs-on: ubuntu-latest
    steps:

      - name: Check eval gate
        # TODO 6: Kiểm tra accuracy >= 0.70
        run: |
          python - <<'EOF'
          acc = float("${{ needs.train.outputs.accuracy }}")
          if acc < 0.70:
              raise SystemExit(f"FAILED: accuracy {acc:.4f} < 0.70. Huy deploy.")
          print(f"PASSED: accuracy {acc:.4f} >= 0.70. Dang trien khai model.")
          EOF


  # -------------------------------------------------------------------------
  # JOB 4 - DEPLOY
  # -------------------------------------------------------------------------
  deploy:
    name: Deploy
    needs: eval
    runs-on: ubuntu-latest
    steps:

      - name: SSH deploy to VM
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.VM_HOST }}
          username: ${{ secrets.VM_USER }}
          key: ${{ secrets.VM_SSH_KEY }}
          script: |
            # TODO 7: Restart service
            sudo systemctl restart mlops-serve
            # TODO 8: Health check
            sleep 5
            curl -sf http://localhost:8000/health && echo "Health check passed." || exit 1
```

### 2.11 Push lên GitHub và theo dõi pipeline

```bash
git add src/serve.py src/train.py tests/test_train.py .github/workflows/mlops.yml
git commit -m "feat: complete train, serve, tests, and CI/CD pipeline"
git push origin main
```

Vào tab **Actions** trên GitHub để theo dõi. Cần thấy 4 jobs đều xanh.

### 2.12 Khởi động service lần đầu trên VM

Sau khi pipeline chạy thành công lần đầu (model đã upload lên GCS):
```bash
gcloud compute ssh mlops-serve --zone=us-central1-a \
    --command "sudo systemctl start mlops-serve"
```

Kiểm tra hoạt động:
```bash
VM_IP=$(gcloud compute instances describe mlops-serve \
    --zone=us-central1-a \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

# Health check
curl http://$VM_IP:8000/health

# Dự đoán thử
curl -X POST http://$VM_IP:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
```

Kết quả mong đợi:
```json
{"status": "ok"}
{"prediction": 0, "label": "thap"}
```

---

## BƯỚC 3: Continuous Training (1-2 giờ)

**Mục tiêu:** Chỉ cần commit dữ liệu mới, toàn bộ pipeline tự chạy lại mà không cần làm gì thêm.

### 3.1 Thêm dữ liệu mới

```bash
# Gộp train_phase2.csv vào train_phase1.csv (2998 → 5996 mẫu)
python add_new_data.py
```

Kết quả:
```
Cập nhật dữ liệu: 2998 -> 5996 mẫu
```

### 3.2 Version dữ liệu và trigger pipeline

**Thứ tự quan trọng: dvc push TRƯỚC, git push SAU.**

```bash
# 1. Cập nhật DVC tracking
dvc add data/train_phase1.csv

# 2. Commit file .dvc (không commit CSV)
git add data/train_phase1.csv.dvc
git commit -m "data: bo sung 2998 mau moi (train_phase2)"

# 3. Đẩy dữ liệu lên cloud TRƯỚC
dvc push

# 4. Đẩy git lên GitHub - pipeline tự kích hoạt
git push origin main
```

### 3.3 Theo dõi pipeline tự chạy

Vào **GitHub Actions** - thấy pipeline mới bắt đầu tự động.

Pipeline chạy trên 5996 mẫu (nhiều gấp đôi lần trước), mô hình mới được deploy lên VM mà không cần thao tác gì thêm.

### 3.4 Kiểm tra và so sánh kết quả

```bash
# Kiểm tra model mới trên VM
curl http://$VM_IP:8000/health
curl -X POST http://$VM_IP:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
```

Tải file metrics để so sánh (từ GitHub Actions → Actions tab → chọn run → Artifacts → metrics):

| Metric | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) |
|--------|-------------------|-------------------|
| accuracy | ? | ? |
| f1_score | ? | ? |

---

## Danh Sách Kiểm Tra Nộp Bài

### Bước 1
- [ ] `outputs/metrics.json` tồn tại
- [ ] `models/model.pkl` tồn tại
- [ ] MLflow UI hiển thị ≥3 run với hyperparameter khác nhau
- [ ] Mỗi run có `accuracy` và `f1_score`
- [ ] `params.yaml` đã cập nhật về bộ hyperparameter tốt nhất
- [ ] Screenshot MLflow UI

### Bước 2
- [ ] DVC remote đã cấu hình, `dvc push` thành công
- [ ] Dữ liệu hiển thị trong cloud storage
- [ ] Tất cả 4 GitHub Actions jobs màu xanh (Test / Train / Eval / Deploy)
- [ ] VM trả về `{"status": "ok"}` tại `/health`
- [ ] VM trả về dự đoán hợp lệ tại `/predict`
- [ ] Screenshot Actions tab với 4 jobs xanh

### Bước 3
- [ ] Screenshot Actions tab: pipeline kích hoạt bởi commit dữ liệu
- [ ] Tất cả 4 jobs xanh
- [ ] Bảng so sánh metrics trước/sau

---

## Xử Lý Sự Cố Thường Gặp

### `dvc push` thất bại xác thực
```bash
export GOOGLE_APPLICATION_CREDENTIALS=sa-key.json
dvc push  # thử lại
```

### GitHub Actions `dvc pull` thất bại
- Kiểm tra secret `CLOUD_CREDENTIALS` không có khoảng trắng thừa ở đầu/cuối
- Xem log chi tiết trong Actions tab

### Eval job thất bại vì accuracy < 0.70
- Thường do dữ liệu test ngẫu nhiên, chạy lại với n_estimators=200, max_depth=10
- Kiểm tra `params.yaml` đã cập nhật về bộ tốt nhất chưa

### VM service không khởi động
```bash
# Xem log lỗi
gcloud compute ssh mlops-serve --zone=us-central1-a \
    --command "sudo journalctl -u mlops-serve -n 50"
```

Lỗi thường gặp:
- `GCS_BUCKET` chưa đặt đúng trong systemd service file
- `model.pkl` chưa được upload lên GCS (pipeline chưa chạy xong)
- Thiếu `sa-key.json` trên VM

### TypeError khi chạy test
```bash
# Đảm bảo đang ở đúng thư mục
cd Day21-Track2-CI-CD-for-AI-Systems
pytest tests/ -v
```

---

## Thang Điểm

| Tiêu chí | Điểm |
|----------|------|
| Bước 1: ≥3 MLflow runs với hyperparameter khác nhau | 12 |
| Bước 1: Mỗi run log accuracy + f1_score | 8 |
| Bước 1: Phân tích chọn hyperparameter tốt nhất | 4 |
| Bước 2: DVC remote + data trên cloud | 12 |
| Bước 2: 4 GitHub Actions jobs đều xanh | 16 |
| Bước 2: Eval gate chặn deploy khi accuracy < 0.70 | 4 |
| Bước 2: VM `/predict` trả kết quả đúng | 12 |
| Bước 3: 1 data commit → toàn bộ pipeline tự chạy | 12 |
| **Tổng** | **80** |

**Bonus (mỗi mục +4 điểm):**
1. MLflow tracking qua DagsHub (remote tracking)
2. Thử nhiều thuật toán (không chỉ RandomForest)
3. Tự động tạo báo cáo performance sau mỗi run
4. Rollback model tự động nếu accuracy giảm
5. Phát hiện data drift khi thêm dữ liệu mới
