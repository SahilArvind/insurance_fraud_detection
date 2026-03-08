import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Load Dataset ───────────────────────────────────────────────────────────
df = pd.read_csv('insurance_claims.csv')
print("Dataset shape:", df.shape)

# Drop unnamed/empty last column if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed|^_c')]

# ─── 2. Feature Selection ──────────────────────────────────────────────────────
# Drop columns not useful for prediction
drop_cols = ['policy_number', 'policy_bind_date', 'incident_date',
             'incident_location', 'insured_zip', 'auto_model']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Replace '?' with NaN then fill with mode
df.replace('?', np.nan, inplace=True)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("Null values after handling:", df.isnull().sum().sum())

# ─── 3. Encode Categorical Features ───────────────────────────────────────────
label_encoders = {}
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols = [c for c in cat_cols if c != 'fraud_reported']

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Encode target
df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})

# ─── 4. Drop Highly Correlated Features (from multivariate analysis) ──────────
# injury_claim, property_claim, vehicle_claim are highly correlated with total_claim_amount
# months_as_customer and age are also highly correlated (0.92)
high_corr = ['injury_claim', 'property_claim', 'vehicle_claim', 'months_as_customer']
df.drop(columns=[c for c in high_corr if c in df.columns], inplace=True)

# ─── 5. Split Features and Target ─────────────────────────────────────────────
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

print("Features used:", list(X.columns))

# ─── 6. Scale Features ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── 7. Train/Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# ─── 8. Train Multiple Models ──────────────────────────────────────────────────
models = {
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes':         GaussianNB(),
    'SVM':                 SVC(kernel='rbf', probability=True)
}

results = {}
print("\n── Model Accuracy Comparison ──")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cv  = cross_val_score(model, X_scaled, y, cv=5).mean()
    results[name] = {'accuracy': acc, 'cv_score': cv}
    print(f"{name:22s} | Test Acc: {acc:.4f} | CV Score: {cv:.4f}")

# ─── 9. Pick Best Model & Save ────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['cv_score'])
best_model = models[best_name]
print(f"\nBest model: {best_name} (CV={results[best_name]['cv_score']:.4f})")

# Print full report for best model
best_preds = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, best_preds, target_names=['Not Fraud', 'Fraud']))

joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler,     'scaler.pkl')
print("\nSaved: model.pkl, scaler.pkl")

# Save feature names for app.py
import json
with open('features.json', 'w') as f:
    json.dump(list(X.columns), f)
print("Saved: features.json")
