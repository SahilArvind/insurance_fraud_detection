from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load saved model, scaler, and feature list
model   = joblib.load('model.pkl')
scaler  = joblib.load('scaler.pkl')
with open('features.json') as f:
    FEATURES = json.load(f)

# ── Encoding maps (same mappings used during training) ────────────────────────
# These are derived from the training LabelEncoder fit order (alphabetical)
ENCODE_MAP = {
    'policy_state':            {'IL': 0, 'IN': 1, 'OH': 2},
    'policy_csl':              {'100/300': 0, '250/500': 1, '500/1000': 2},
    'insured_sex':             {'FEMALE': 0, 'MALE': 1},
    'insured_education_level': {'Associate': 0, 'College': 1, 'High School': 2, 'JD': 3, 'MD': 4, 'Masters': 5, 'PhD': 6},
    'insured_occupation':      {
        'adm-clerical': 0, 'armed-forces': 1, 'craft-repair': 2, 'exec-managerial': 3,
        'farming-fishing': 4, 'handlers-cleaners': 5, 'machine-op-inspct': 6,
        'other-service': 7, 'priv-house-serv': 8, 'prof-specialty': 9,
        'protective-serv': 10, 'sales': 11, 'tech-support': 12, 'transport-moving': 13
    },
    'insured_hobbies':         {
        'base-jumping': 0, 'basketball': 1, 'board-games': 2, 'bungie-jumping': 3,
        'camping': 4, 'chess': 5, 'cross-fit': 6, 'dancing': 7, 'golf': 8,
        'hiking': 9, 'kayaking': 10, 'movies': 11, 'paintball': 12,
        'polo': 13, 'reading': 14, 'skydiving': 15, 'sleeping': 16,
        'video-games': 17, 'yachting': 18
    },
    'insured_relationship':    {
        'husband': 0, 'not-in-family': 1, 'other-relative': 2,
        'own-child': 3, 'unmarried': 4, 'wife': 5
    },
    'incident_type':           {
        'Multi-vehicle Collision': 0, 'Parked Car': 1,
        'Single Vehicle Collision': 2, 'Vehicle Theft': 3
    },
    'collision_type':          {
        'Front Collision': 0, 'Rear Collision': 1, 'Side Collision': 2
    },
    'incident_severity':       {
        'Major Damage': 0, 'Minor Damage': 1, 'Total Loss': 2, 'Trivial Damage': 3
    },
    'authorities_contacted':   {
        'Ambulance': 0, 'Fire': 1, 'None': 2, 'Other': 3, 'Police': 4
    },
    'incident_state':          {
        'NC': 0, 'NY': 1, 'OH': 2, 'PA': 3, 'SC': 4, 'VA': 5, 'WV': 6
    },
    'incident_city':           {
        'Arlington': 0, 'Columbus': 1, 'Hillsdale': 2, 'Northbend': 3,
        'Northbrook': 4, 'Riverwood': 5, 'Springfield': 6
    },
    'property_damage':         {'NO': 0, 'YES': 1},
    'police_report_available': {'NO': 0, 'YES': 1},
    'auto_make':               {
        'Accura': 0, 'Audi': 1, 'BMW': 2, 'Chevrolet': 3, 'Dodge': 4,
        'Ford': 5, 'Honda': 6, 'Jeep': 7, 'Mercedes': 8, 'Nissan': 9,
        'Saab': 10, 'Suburu': 11, 'Toyota': 12, 'Volkswagen': 13
    }
}

DEFAULTS = {
    'age': 38, 'policy_state': 'OH', 'policy_csl': '250/500',
    'policy_deductable': 1000, 'policy_annual_premium': 1200,
    'umbrella_limit': 0, 'insured_sex': 'MALE',
    'insured_education_level': 'College', 'insured_occupation': 'craft-repair',
    'insured_hobbies': 'reading', 'insured_relationship': 'husband',
    'capital-gains': 0, 'capital-loss': 0,
    'incident_type': 'Single Vehicle Collision', 'collision_type': 'Side Collision',
    'incident_severity': 'Minor Damage', 'authorities_contacted': 'Police',
    'incident_state': 'OH', 'incident_city': 'Columbus',
    'incident_hour_of_the_day': 12, 'number_of_vehicles_involved': 1,
    'property_damage': 'NO', 'bodily_injuries': 0, 'witnesses': 1,
    'police_report_available': 'NO', 'total_claim_amount': 50000,
    'auto_make': 'Toyota', 'auto_year': 2015
}


def encode_value(feature, value):
    """Encode a single feature value."""
    if feature in ENCODE_MAP:
        mapping = ENCODE_MAP[feature]
        return mapping.get(str(value), 0)
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0


def build_feature_vector(form_data):
    """Build the full feature vector from form input."""
    row = []
    for feat in FEATURES:
        val = form_data.get(feat, DEFAULTS.get(feat, 0))
        row.append(encode_value(feat, val))
    return np.array(row).reshape(1, -1)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = ''
    if request.method == 'POST':
        try:
            features = build_feature_vector(request.form)
            scaled   = scaler.transform(features)
            result   = model.predict(scaled)[0]
            prob     = model.predict_proba(scaled)[0] if hasattr(model, 'predict_proba') else None

            if result == 1:
                label = '⚠️ FRAUDULENT CLAIM DETECTED'
                conf  = f'{prob[1]*100:.1f}%' if prob is not None else 'N/A'
                prediction_text = f'{label} — Confidence: {conf}'
            else:
                label = '✅ LEGITIMATE CLAIM'
                conf  = f'{prob[0]*100:.1f}%' if prob is not None else 'N/A'
                prediction_text = f'{label} — Confidence: {conf}'
        except Exception as e:
            prediction_text = f'Error during prediction: {str(e)}'

    return render_template('predict.html', prediction_text=prediction_text)


if __name__ == '__main__':
    app.run(debug=True)
