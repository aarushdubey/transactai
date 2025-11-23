# 🚀 TransactAI — Autonomous Financial Transaction Categorisation  
**Team Name:** Parasite  
**Team Member:** Aarush Dubey  
**Submission for:** GHCI 2025 Hackathon – Round 2  

TransactAI is an **end-to-end, fully local, AI-powered system** for categorising financial transactions without relying on expensive third-party APIs.  
It is fast, explainable, secure, and completely customizable — built for real-world fintech use cases.

---

# 🌟 Key Features

### 🔒 1. Fully Local Inference  
No external APIs. All predictions run inside the user environment — ensuring privacy, zero latency, and zero recurring costs.

### ⚡ 2. Fast & Lightweight ML  
- TF-IDF vectorizer  
- Logistic Regression classifier  
- Microsecond-level inference  
- Suitable for large-scale batch categorisation

### 🧠 3. Explainable AI  
Each prediction includes:
- Category  
- Confidence score  
- Top contributing keywords

### 🔁 4. Human-in-the-Loop Feedback  
Incorrect predictions can be corrected using `/feedback`.  
All corrections are stored and can be used for model retraining.

### 🗂 5. Config-Driven Taxonomy  
Categories are stored in `config/taxonomy_config.yaml`  
Admins can add, edit, or remove categories without touching code.

### 🛡 6. Responsible & Robust AI  
- Bias-aware synthetic dataset  
- Region + merchant + noise variations  
- No personal/transactional PII  
- High transparency and fairness

---

# 📁 Project Structure

```
transactai/
│
├── service/
│   ├── main.py               # FastAPI routes
│   └── utils.py              # Prediction, explanation utilities
│
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_train_model.py
│   └── 03_evaluate_model.py
│
├── models/
│   ├── model.joblib
│   └── vectorizer.joblib
│
├── data/
│   ├── transactions_dataset_1000.csv
│   └── processed/            # Train/val/test splits
│
├── config/
│   └── taxonomy_config.yaml
│
├── reports/
│   ├── metrics_report.txt
│   └── confusion_matrix.png
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Technology Stack

- **Python 3.11**
- **FastAPI** (backend service)
- **Scikit-Learn** (ML model)
- **TF-IDF Vectorizer + Logistic Regression**
- **Joblib** (model persistence)
- **Pandas, NumPy**
- **Uvicorn** (server)
- **YAML config** for taxonomy

---

# 🧠 Model Training Pipeline

### **1️⃣ Prepare Data**
```bash
python scripts/01_prepare_data.py
```
- Cleans data  
- Normalizes text  
- Splits into train/val/test  

### **2️⃣ Train Model**
```bash
python scripts/02_train_model.py
```
- Trains TF-IDF + Logistic Regression  
- Saves `model.joblib` & `vectorizer.joblib`

### **3️⃣ Evaluate**
```bash
python scripts/03_evaluate_model.py
```
Produces:
- `reports/metrics_report.txt`  
- `reports/confusion_matrix.png`

**Test Macro F1 Score:** ~1.00 (synthetic dataset)  
**Per-class F1:** ~1.00 across all 8 categories  

Confusion matrix example:  
*(Insert your confusion matrix image in the PDF)*

---

# 🚀 Running the Backend (FastAPI)

Start the API:

```bash
uvicorn service.main:app --reload
```

Open Swagger UI:  
👉 http://127.0.0.1:8000/docs

### **Endpoints**

#### 🔹 POST `/predict`
Request:
```json
{
  "raw_description": "Starbucks Coffee BLR IN #0421"
}
```

Response:
```json
{
  "category": "Dining",
  "confidence": 0.97,
  "explanation": ["starbucks", "coffee"]
}
```

#### 🔹 POST `/feedback`
Logs corrections to:
```
data/feedback.csv
```

#### 🔹 GET `/categories`
Returns YAML-driven taxonomy.

---

# 🧪 Dataset Details (1000 Rows)

The dataset is **synthetic**, balanced, and noise-augmented:
- Merchant name variability  
- City/region differences  
- UPI/POS/ATM patterns  
- Amount bands  
- Typos + random noise  

Columns:
- transaction_id  
- date  
- raw_description  
- normalized_description  
- category  
- region  
- amount_band  

---

# 🧱 Architecture Diagram
*(Insert the architecture PNG here in your PDF submission)*

**Layers**:
- **API Layer** → FastAPI  
- **ML Engine** → Vectorizer + Classifier  
- **Explainability Engine** → Keyword weighting  
- **Feedback Loop** → Data correction layer  
- **Config Module** → YAML taxonomy  

---

# 🔍 Responsible AI Checklist

✔ No PII  
✔ Local inference only  
✔ Synthetic dataset  
✔ Bias mitigation (region + merchant variation)  
✔ Explainability included  
✔ Human correction loop  

---

# 📹 Demo Video
👉 *(Add your YouTube / Google Drive link here after recording)*

Your video should include:
1. Data prep  
2. Training  
3. Evaluation  
4. Confusion matrix  
5. Running FastAPI  
6. Predict + Feedback demo  
7. Architecture overview  

---

# 🔗 GitHub Repository
👉 https://github.com/aarushdubey/transactai

---

# 🏁 Conclusion

TransactAI bridges the gap between **accuracy, transparency, and control** in financial transaction categorisation.  
It is cost-effective, fast, explainable, customizable, and built entirely in-house — making it a powerful replacement for expensive third-party APIs.

---

# 👤 Author
**Aarush Dubey**  
Team: Parasite  
GHCI 2025 Hackathon Participant  
