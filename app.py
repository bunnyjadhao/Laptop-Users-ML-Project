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
<title>SVM Predictor | AI &amp; ML</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
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
    background: linear-gradient(-45deg, #050816, #1e1b8f, #5b21b6, #0e7490, #0b1030);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
    overflow-x: hidden;
  }
  @keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }

  /* SVM decision-boundary animation */
  #bg { position: fixed; inset: 0; width: 100%; height: 100%; z-index: 0; }

  .card {
    position: relative;
    z-index: 1;
    width: 100%;
    max-width: 500px;
    padding: 36px 34px 30px;
    border-radius: 28px;
    background: rgba(10, 14, 40, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 25px 70px rgba(0, 0, 0, 0.55), 0 0 60px rgba(139, 92, 246, 0.25);
  }
  .card::before {
    content: "";
    position: absolute; inset: -1px; z-index: -1;
    border-radius: 29px;
    background: linear-gradient(135deg, #22d3ee, #8b5cf6, #ec4899);
    opacity: .55;
    filter: blur(14px);
  }

  .tags { display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
  .tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: .68rem;
    padding: 5px 11px;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, .3);
    background: rgba(255, 255, 255, .08);
    letter-spacing: .5px;
  }
  .tag.a { border-color: #22d3ee; color: #67e8f9; }
  .tag.b { border-color: #a78bfa; color: #c4b5fd; }
  .tag.c { border-color: #f472b6; color: #f9a8d4; }

  h1 {
    text-align: center; font-size: 1.85rem; font-weight: 700; letter-spacing: .3px;
    background: linear-gradient(90deg, #67e8f9, #c4b5fd, #f9a8d4);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .sub { text-align: center; font-weight: 300; font-size: .88rem; opacity: .85; margin: 6px 0 24px; }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .full { grid-column: 1 / -1; }

  label { display: block; font-size: .74rem; font-weight: 500; margin-bottom: 6px; opacity: .9; letter-spacing: .6px; text-transform: uppercase; color: #bae6fd; }

  input, select {
    width: 100%;
    padding: 13px 14px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.22);
    background: rgba(255, 255, 255, 0.08);
    color: #fff;
    font-family: inherit;
    font-size: .95rem;
    outline: none;
    transition: all .25s ease;
  }
  input::placeholder { color: rgba(255, 255, 255, .45); }
  select option { color: #1e1b4b; }
  input:focus, select:focus {
    background: rgba(255, 255, 255, 0.15);
    border-color: #22d3ee;
    box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.2);
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
    --c: #fff; --g: rgba(255,255,255,.3); --g2: rgba(255,255,255,.05);
    position: relative;
    overflow: hidden;
    margin-top: 22px;
    padding: 22px 16px 20px;
    border-radius: 22px;
    text-align: center;
    display: none;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid var(--c);
    box-shadow: 0 0 34px var(--g), inset 0 0 34px var(--g2);
  }
  .result.yes { --c: #22d3ee; --g: rgba(34,211,238,.5); --g2: rgba(16,185,129,.18); display: block; animation: pop .55s cubic-bezier(.2, 1.4, .4, 1); }
  .result.no  { --c: #f472b6; --g: rgba(244,114,182,.5); --g2: rgba(249,115,22,.18); display: block; animation: shake .55s ease; }
  .result.err { display: block; background: rgba(0, 0, 0, .4); border-color: rgba(255,255,255,.25); box-shadow: none; padding: 16px; font-weight: 500; font-size: .95rem; animation: pop .4s ease; }

  .badge { position: relative; width: 92px; height: 92px; margin: 0 auto 10px; color: var(--c); }
  .badge svg { width: 100%; height: 100%; filter: drop-shadow(0 0 10px var(--c)); }
  .badge .ring  { fill: rgba(255,255,255,.06); stroke: currentColor; stroke-width: 4; stroke-linecap: round; stroke-dasharray: 227; stroke-dashoffset: 227; transform: rotate(-90deg); transform-origin: 50% 50%; animation: draw .7s .05s ease forwards; }
  .badge .mark  { fill: none; stroke: currentColor; stroke-width: 6; stroke-linecap: round; stroke-linejoin: round; stroke-dasharray: 70; stroke-dashoffset: 70; animation: draw .45s .6s ease forwards; }
  .badge::after { content: ""; position: absolute; inset: 0; border-radius: 50%; border: 2px solid var(--c); animation: pulse 1.8s .9s ease-out infinite; }

  .label { font-family: 'JetBrains Mono', monospace; font-size: .66rem; letter-spacing: 3px; text-transform: uppercase; opacity: .75; }
  .word {
    font-size: 2.7rem; font-weight: 800; letter-spacing: 8px; line-height: 1.1; margin: 4px 0 2px; padding-left: 8px;
    background: linear-gradient(90deg, #67e8f9, #34d399, #67e8f9);
    background-size: 200% 100%;
    -webkit-background-clip: text; background-clip: text; color: transparent;
    filter: drop-shadow(0 0 14px var(--g));
    animation: shine 2.5s linear infinite, rise .5s .35s both;
  }
  .result.no .word { background-image: linear-gradient(90deg, #f9a8d4, #fb923c, #f9a8d4); background-size: 200% 100%; }
  .note { font-size: .78rem; font-weight: 300; opacity: .85; animation: rise .5s .5s both; }

  .spark { position: absolute; left: 50%; top: 56px; width: 8px; height: 8px; border-radius: 50%; pointer-events: none; animation: fly .95s ease-out forwards; }

  @keyframes pop   { from { transform: scale(.85); opacity: 0; } to { transform: scale(1); opacity: 1; } }
  @keyframes shake { 0%,100% { transform: translateX(0); } 20% { transform: translateX(-10px); } 40% { transform: translateX(9px); } 60% { transform: translateX(-6px); } 80% { transform: translateX(4px); } }
  @keyframes draw  { to { stroke-dashoffset: 0; } }
  @keyframes pulse { 0% { transform: scale(1); opacity: .7; } 100% { transform: scale(1.7); opacity: 0; } }
  @keyframes shine { to { background-position: 200% 0; } }
  @keyframes rise  { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes fly   { 0% { transform: translate(-50%, 0) scale(1); opacity: 1; } 100% { transform: translate(calc(-50% + var(--dx)), var(--dy)) scale(0); opacity: 0; } }

  .meta { display: flex; justify-content: space-between; margin-top: 20px; font-family: 'JetBrains Mono', monospace; font-size: .66rem; opacity: .65; }
  footer { text-align: center; margin-top: 8px; font-size: .7rem; opacity: .6; font-weight: 300; }

  @media (max-width: 480px) {
    .card { padding: 28px 20px 24px; }
    .grid { grid-template-columns: 1fr; }
    .meta { flex-direction: column; gap: 4px; align-items: center; }
  }
</style>
</head>
<body>
  <canvas id="bg"></canvas>

  <main class="card">
    <div class="tags">
      <span class="tag a">SVM</span>
      <span class="tag b">{{ kernel|upper }} KERNEL</span>
      <span class="tag c">AI / ML</span>
    </div>
    <h1>SVM Predictor</h1>
    <p class="sub">Support Vector Machine classifier &mdash; enter details to get a prediction</p>

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

    <div class="meta">
      <span>classes: {{ classes }}</span>
      <span>support vectors: {{ n_sv }}</span>
      <span>features: {{ n_features }}</span>
    </div>
    <footer>Built with Scikit-learn &bull; Flask</footer>
  </main>

<script>
/* ---------- SVM decision boundary background ---------- */
(function () {
  const cv = document.getElementById('bg');
  const ctx = cv.getContext('2d');
  let W, H, pts = [];

  function resize() {
    W = cv.width = window.innerWidth;
    H = cv.height = window.innerHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  // random points; class decided by side of a base line (+ a gap so a margin is visible)
  for (let i = 0; i < 90; i++) {
    const x = Math.random(), y = Math.random();
    const d = (x - 0.5) * 0.9 + (y - 0.5) * 0.45;
    if (Math.abs(d) < 0.07) { i--; continue; }
    pts.push({ x: x, y: y, c: d > 0 ? 1 : 0, ph: Math.random() * 6.28 });
  }

  function draw(t) {
    ctx.clearRect(0, 0, W, H);
    const cx = W / 2, cy = H / 2;
    const ang = Math.atan2(-0.9, 0.45) + Math.sin(t / 4000) * 0.12;
    const nx = Math.cos(ang + Math.PI / 2), ny = Math.sin(ang + Math.PI / 2);   // normal
    const tx = Math.cos(ang), ty = Math.sin(ang);                               // along line
    const L = Math.max(W, H);
    const margin = 70;

    // hyperplane + margins
    function line(off, dash, alpha, width) {
      ctx.beginPath();
      ctx.setLineDash(dash);
      ctx.lineWidth = width;
      ctx.strokeStyle = 'rgba(255,255,255,' + alpha + ')';
      ctx.moveTo(cx + nx * off - tx * L, cy + ny * off - ty * L);
      ctx.lineTo(cx + nx * off + tx * L, cy + ny * off + ty * L);
      ctx.stroke();
    }
    line(0, [], 0.45, 2);
    line(margin, [8, 8], 0.22, 1.5);
    line(-margin, [8, 8], 0.22, 1.5);
    ctx.setLineDash([]);

    // points
    pts.forEach(function (p) {
      const px = p.x * W + Math.sin(t / 1500 + p.ph) * 6;
      const py = p.y * H + Math.cos(t / 1700 + p.ph) * 6;
      const dist = (px - cx) * nx + (py - cy) * ny;
      const col = p.c ? '244,114,182' : '34,211,238';

      ctx.beginPath();
      ctx.arc(px, py, 4.5, 0, 6.283);
      ctx.fillStyle = 'rgba(' + col + ',0.85)';
      ctx.fill();

      // support vectors = points near the margin
      if (Math.abs(Math.abs(dist) - margin) < 26) {
        ctx.beginPath();
        ctx.arc(px, py, 10, 0, 6.283);
        ctx.strokeStyle = 'rgba(255,255,255,0.7)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
})();

/* ---------- cool result display ---------- */
function showResult(box, value) {
  const isYes = value.toLowerCase() === 'yes';
  box.className = 'result';
  void box.offsetWidth;                       // restart animations
  box.className = 'result ' + (isYes ? 'yes' : 'no');

  const mark = isYes
    ? '<path class="mark" d="M30 50 L43 63 L68 35"/>'
    : '<path class="mark" d="M35 35 L65 65"/><path class="mark" d="M65 35 L35 65"/>';

  box.innerHTML =
    '<div class="badge"><svg viewBox="0 0 100 100"><circle class="ring" cx="50" cy="50" r="36"/>' + mark + '</svg></div>' +
    '<div class="label">Prediction</div>' +
    '<div class="word">' + value.toUpperCase() + '</div>' +
    '<div class="note">' + (isYes ? 'Positive class detected' : 'Negative class detected') + '</div>';

  // particle burst
  const colors = isYes ? ['#22d3ee', '#34d399', '#a78bfa', '#ffffff'] : ['#f472b6', '#fb923c', '#fda4af'];
  const n = isYes ? 22 : 12;
  for (let i = 0; i < n; i++) {
    const sp = document.createElement('span');
    sp.className = 'spark';
    const ang = Math.random() * Math.PI * 2;
    const dist = 60 + Math.random() * 90;
    sp.style.setProperty('--dx', Math.cos(ang) * dist + 'px');
    sp.style.setProperty('--dy', (isYes ? Math.sin(ang) * dist : Math.abs(Math.sin(ang)) * dist) + 'px');
    sp.style.background = colors[i % colors.length];
    sp.style.animationDelay = (isYes ? 0.55 : 0.1) + 's';
    box.appendChild(sp);
  }
}

/* ---------- prediction ---------- */
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

    showResult(box, String(data.prediction));
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
    return render_template_string(
        PAGE,
        gender=GENDER,
        region=REGION,
        occupation=OCCUPATION,
        kernel=getattr(model, "kernel", "svm"),
        classes=" / ".join(str(c) for c in getattr(model, "classes_", [])),
        n_sv=int(sum(getattr(model, "n_support_", [0]))),
        n_features=len(FEATURES),
    )


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
