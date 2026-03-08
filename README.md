# 🛡️ Insurance Fraud Detection Using Machine Learning

A full-stack machine learning web application that predicts whether an auto insurance claim is **fraudulent or legitimate** based on claim details.

---

## 📌 Project Overview

Insurance fraud costs companies billions annually. This project applies supervised machine learning to detect fraudulent claims from the **Auto Insurance Claims dataset** (Kaggle).

**Live flow:** User enters claim details → Flask backend → Trained ML model → Fraud/Legitimate prediction with confidence score.

---

## 🗂️ Project Structure

```
insurance_fraud_detection/
│
├── app.py                  # Flask web application
├── model.py                # Model training script
├── insurance_claims.csv    # Dataset (1,000 records)
├── model.pkl               # Saved best ML model (Logistic Regression)
├── scaler.pkl              # Saved StandardScaler
├── features.json           # Feature list used for prediction
├── requirements.txt        # Python dependencies
│
└── templates/
    ├── index.html          # Home page
    ├── about.html          # About/methodology page
    └── predict.html        # Prediction form
```

---

## 🤖 Machine Learning Pipeline

### 1. Data Collection
- Dataset: [Auto Insurance Claims — Kaggle](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data)
- 1,000 records, 39 features, binary target: `fraud_reported` (Y/N)

### 2. Data Preprocessing
- Replaced `?` values with column **mode**
- Dropped non-informative columns: `policy_number`, `policy_bind_date`, `incident_date`, `incident_location`, `insured_zip`, `auto_model`
- Dropped **highly correlated** features: `injury_claim`, `property_claim`, `vehicle_claim`, `months_as_customer`

### 3. Feature Engineering
- **Label Encoding** for all categorical features
- **Standard Scaling** (mean=0, std=1) for all numeric features

### 4. Model Training & Evaluation

| Algorithm            | Test Accuracy | CV Score (5-fold) |
|----------------------|--------------|-------------------|
| **Logistic Regression** | 72.0%     | **77.7%** ✅ Deployed |
| Decision Tree        | 72.5%        | 77.3%             |
| Random Forest        | 72.0%        | 77.2%             |
| Naive Bayes          | 70.5%        | 76.6%             |
| SVM                  | 70.0%        | 74.5%             |
| KNN                  | 70.5%        | 72.4%             |

Best model selected by **5-fold cross-validation score**.

### 5. Model Deployment
- Flask REST application with 3 routes: `/`, `/about`, `/predict`
- Model + scaler serialised with `joblib`

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/insurance_fraud_detection.git
cd insurance_fraud_detection

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (generates model.pkl, scaler.pkl, features.json)
python model.py

# 5. Run the Flask app
python app.py

# 6. Open browser at:
#    http://127.0.0.1:5000
```

---

## 📊 Key EDA Findings

- **~24.7% fraud rate** — 247 out of 1,000 claims are fraudulent
- **Age distribution** — mostly 30–50 years old (near-normal)
- **High correlation (0.92)** between `months_as_customer` and `age` → one dropped
- Claim sub-totals (`injury_claim` + `property_claim` + `vehicle_claim`) sum to `total_claim_amount` → dropped to avoid data leakage
- `incident_severity` and `total_claim_amount` are among the most predictive features

---

## 🛠️ Technologies Used

| Layer       | Technology |
|-------------|-----------|
| Language    | Python 3.x |
| ML Library  | scikit-learn |
| Web Framework | Flask |
| Data Processing | pandas, numpy |
| Model Serialisation | joblib |
| Frontend    | HTML5, CSS3 (Jinja2 templates) |

---

## 📁 Dataset

- **Source:** [Kaggle — buntyshah/auto-insurance-claims-data](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data)
- **Records:** 1,000
- **Features:** 39 (policy info, incident details, claimant demographics, vehicle info)
- **Target:** `fraud_reported` (Y = Fraud, N = Legitimate)

---

## 📝 License

This project is for educational purposes as part of an Insurance Fraud Detection Machine Learning course project.
