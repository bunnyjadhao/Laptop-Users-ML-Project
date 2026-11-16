import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

# ------------------------------------------------------------------
# Load model (path is relative to this file so it works on Render)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svm.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = ["Age", "Gender", "Region", "Occupation", "Income"]

# ------------------------------------------------------------------
# Label mappings: the model was trained on numeric codes.
# Edit the TEXT on the right so it matches how you encoded your data.
# (The numbers on the left must stay the same.)
# ------------------------------------------------------------------
GENDER = {0: "Female", 1: "Male"}
REGION = {0: "Rural", 1: "Urban"}
OCCUPATION = {0: "Occupation 0", 1: "Occupation 1", 2: "Occupation 2", 3: "Occupation 3"}

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SVM Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Poppins', sans-serif;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    color: #fff;
    background: linear-gradient(-45deg, #1a0b3b, #5b21b6, #db2777, #f97316, #0ea5e9);
    background-size: 400% 400%;
    animation: gradientShift 16s ease infinite;
    overflow-x: hidden;
  }
  @keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }

  /* floating blurred blobs */
  .blob { position: fixed; border-radius: 50%; filter: blur(70px); opacity: .45; z-index: 0; }
  .b1 { width: 320px; height: 320px; background: #22d3ee; top: -80px; left: -80px; }
  .b2 { width: 380px; height: 380px; background: #f472b6; bottom: -120px; right: -100px; }

  .card {
    position: relative;
    z-index: 1;
    width: 100%;
    max-width: 480px;
    padding: 38px 34px;
    border-radius: 28px;
    background: rgba(255, 255, 255, 0.13);
    border: 1px solid rgba(255, 255, 255, 0.28);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    box-shadow: 0 25px 60px rgba(0, 0, 0, 0.35);
  }

  .logo {
    width: 64px; height: 64px; margin: 0 auto 14px;
    border-radius: 20px;
    display: flex; align-items: center; justify-content: center;
    font-size: 30px;
    background: linear-gradient(135deg, #22d3ee, #a855f7, #ec4899);
    box-shadow: 0 10px 25px rgba(168, 85, 247, .5);
  }
  h1 { text-align: center; font-size: 1.7rem; font-weight: 700; letter-spacing: .3px; }
  .sub { text-align: center; font-weight: 300; font-size: .9rem; opacity: .85; margin: 6px 0 26px; }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .full { grid-column: 1 / -1; }

  label { display: block; font-size: .78rem; font-weight: 500; margin-bottom: 6px; opacity: .9; letter-spacing: .4px; text-transform: uppercase; }

  input, select {
    width: 100%;
    padding: 13px 14px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    background: rgba(255, 255, 255, 0.16);
    color: #fff;
    font-family: inherit;
    font-size: .95rem;
    outline: none;
    transition: all .25s ease;
  }
  input::placeholder { color: rgba(255, 255, 255, .6); }
  select option { color: #1e1b4b; }
  input:focus, select:focus {
    background: rgba(255, 255, 255, 0.26);
    border-color: #fff;
    box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.18);
  }

  button {
    margin-top: 24px;
    width: 100%;
    padding: 15px;
    border: none;
    border-radius: 16px;
    font-family: inherit;
    font-size: 1.02rem;
    font-weight: 600;
    color: #fff;
    cursor: pointer;
    background: linear-gradient(90deg, #06b6d4, #8b5cf6, #ec4899);
    background-size: 200% 100%;
    box-shadow: 0 12px 28px rgba(139, 92, 246, .55);
    transition: transform .2s ease, background-position .5s ease, box-shadow .2s ease;
  }
  button:hover { transform: translateY(-2px); background-position: 100% 0; box-shadow: 0 16px 34px rgba(236, 72, 153, .55); }
  button:active { transform: translateY(0); }
  button:disabled { opacity: .7; cursor: wait; }

  .result {
    margin-top: 22px;
    padding: 18px;
    border-radius: 18px;
    text-align: center;
    font-weight: 600;
    font-size: 1.15rem;
    display: none;
    animation: pop .45s ease;
  }
  .result small { display: block; font-weight: 300; font-size: .78rem; opacity: .9; margin-bottom: 4px; letter-spacing: 1px; text-transform: uppercase; }
  .result.yes { display: block; background: linear-gradient(135deg, #10b981, #34d399); box-shadow: 0 10px 25px rgba(16, 185, 129, .45); }
  .result.no  { display: block; background: linear-gradient(135deg, #f43f5e, #fb923c); box-shadow: 0 10px 25px rgba(244, 63, 94, .45); }
  .result.err { display: block; background: rgba(0, 0, 0, .35); }
  @keyframes pop { from { transform: scale(.85); opacity: 0; } to { transform: scale(1); opacity: 1; } }

  footer { text-align: center; margin-top: 20px; font-size: .72rem; opacity: .7; font-weight: 300; }

  @media (max-width: 480px) {
    .card { padding: 28px 20px; }
    .grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
  <div class="blob b1"></div>
  <div class="blob b2"></div>

  <main class="card">
    <div class="logo">&#129302;</div>
    <h1>SVM Predictor</h1>
    <p class="sub">Enter the details below to get an instant prediction</p>

    <div class="grid">
      <div>
        <label for="age">Age</label>
        <input type="number" id="age" placeholder="e.g. 28" min="1" max="120">
      </div>
      <div>
        <label for="gender">Gender</label>
        <select id="gender">
          {% for k, v in gender.items() %}<option value="{{ k }}">{{ v }}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label for="region">Region</label>
        <select id="region">
          {% for k, v in region.items() %}<option value="{{ k }}">{{ v }}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label for="occupation">Occupation</label>
        <select id="occupation">
          {% for k, v in occupation.items() %}<option value="{{ k }}">{{ v }}</option>{% endfor %}
        </select>
      </div>
      <div class="full">
        <label for="income">Income</label>
        <input type="number" id="income" placeholder="e.g. 20000" min="0">
      </div>
    </div>

    <button id="btn" onclick="predict()">Predict</button>
    <div id="result" class="result"></div>
    <footer>Powered by Scikit-learn SVM &bull; Flask</footer>
  </main>

<script>
async function predict() {
  const btn = document.getElementById('btn');
  const box = document.getElementById('result');
  const payload = {
    Age: document.getElementById('age').value,
    Gender: document.getElementById('gender').value,
    Region: document.getElementById('region').value,
    Occupation: document.getElementById('occupation').value,
    Income: document.getElementById('income').value
  };

  if (payload.Age === '' || payload.Income === '') {
    box.className = 'result err';
    box.innerHTML = 'Please fill in Age and Income.';
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Predicting...';
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Something went wrong');

    const isYes = String(data.prediction).toLowerCase() === 'yes';
    box.className = 'result ' + (isYes ? 'yes' : 'no');
    box.innerHTML = '<small>Prediction</small>' + (isYes ? '\u2705 ' : '\u274C ') + String(data.prediction).toUpperCase();
  } catch (err) {
    box.className = 'result err';
    box.innerHTML = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Predict';
  }
}
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE, gender=GENDER, region=REGION, occupation=OCCUPATION)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        row = pd.DataFrame(
            [[
                float(data["Age"]),
                float(data["Gender"]),
                float(data["Region"]),
                float(data["Occupation"]),
                float(data["Income"]),
            ]],
            columns=FEATURES,
        )
        pred = model.predict(row)[0]
        if isinstance(pred, (np.generic,)):
            pred = pred.item()
        return jsonify({"prediction": str(pred)})
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
