from flask import Flask, request, jsonify, render_template
import joblib, os, traceback
import pandas as pd, numpy as np

app = Flask(__name__)

# Hardcoded exact feature lists provided by the user
FEATURE_SETS = {
    "Parkinson": [
        "mean_MFCC_2nd_coef",
        "tqwt_minValue_dec_12",
        "tqwt_stdValue_dec_12",
        "tqwt_maxValue_dec_12",
        "tqwt_stdValue_dec_11",
        "tqwt_entropy_log_dec_12",
        "tqwt_maxValue_dec_11",
        "tqwt_minValue_dec_11",
        "tqwt_minValue_dec_13",
        "std_9th_delta_delta",
        "std_8th_delta_delta",
        "tqwt_maxValue_dec_13"
    ],
    "Breast Cancer": [
        "concave points_worst",
        "perimeter_worst",
        "concave points_mean",
        "radius_worst",
        "perimeter_mean",
        "area_worst",
        "radius_mean",
        "area_mean",
        "concavity_mean",
        "concavity_worst",
        "compactness_mean",
        "compactness_worst"
    ],
    "Heart": [
        "male",
        "age",
        "currentSmoker",
        "cigsPerDay",
        "BPMeds",
        "prevalentStroke",
        "prevalentHyp",
        "diabetes",
        "totChol",
        "sysBP",
        "diaBP",
        "BMI",
        "heartRate",
        "glucose"
    ]
}

MODEL_PATHS = {
    "Parkinson": "models/parkinson_pipeline.model",
    "Breast Cancer": "models/breast_cancer_pipeline.model",
    "Heart": "models/heart_disease_logistic.model"
}

MODELS = {}
LOAD_ERRORS = {}

def load_models():
    for name, path in MODEL_PATHS.items():
        try:
            full = os.path.join(os.path.dirname(__file__), path) if not os.path.isabs(path) else path
            if os.path.exists(full):
                with open(full, "rb") as f:
                    MODELS[name] = joblib.load(f)
                    LOAD_ERRORS[name] = None
            else:
                MODELS[name] = None
                LOAD_ERRORS[name] = f"Model file not found at {full}"
        except Exception as e:
            MODELS[name] = None
            LOAD_ERRORS[name] = f"Failed to load model: {e}"

load_models()

@app.route('/')
def index():
    diseases = list(FEATURE_SETS.keys())
    return render_template('index.html', diseases=diseases)

@app.route('/features/<disease>')
def features(disease):
    if disease not in FEATURE_SETS:
        return jsonify({"error":"Unknown disease"}), 404
    return jsonify({"features": FEATURE_SETS[disease], "model_loaded": MODELS.get(disease) is not None, "load_error": LOAD_ERRORS.get(disease)})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    disease = data.get('disease')
    inputs = data.get('inputs', {})
    if disease not in FEATURE_SETS:
        return jsonify({"error":"Unknown disease"}), 400
    features = FEATURE_SETS[disease]
    row = {}
    for f in features:
        # accept keys with spaces or underscores
        val = None
        if f in inputs:
            val = inputs[f]
        else:
            alt = f.replace(" ", "_")
            val = inputs.get(alt, 0)
        row[f] = val
    df = pd.DataFrame([row], columns=features)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    model = MODELS.get(disease)
    if model is None:
        return jsonify({"error": "Model not loaded on server", "load_error": LOAD_ERRORS.get(disease)}), 400
    try:
        pred = model.predict(df)[0]
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0].tolist()
        classes = getattr(model, 'classes_', None)
        return jsonify({"prediction": int(pred) if (isinstance(pred, (np.integer, int))) else str(pred), "probability": proba, "classes": classes.tolist() if classes is not None else None})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
