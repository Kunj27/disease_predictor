Disease Predictor — Exact Feature Forms
-------------------------------------
This project builds a single-page Flask app where each disease form exactly matches the features you provided.

Instructions:
1. Unzip the project folder.
2. Open it in VS Code.
3. Create and activate a venv (PowerShell users: see README of main project about execution policy):
   - Windows (cmd): venv\Scripts\activate
   - PowerShell: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\venv\Scripts\Activate.ps1
4. Install requirements: pip install -r requirements.txt
5. Ensure your model files are placed in the models/ directory with these names:
   - parkinson_pipeline.model
   - breast_cancer_pipeline.model
   - heart_disease_logistic.model
6. Run: python app.py
7. Open http://127.0.0.1:5000 in your browser.
