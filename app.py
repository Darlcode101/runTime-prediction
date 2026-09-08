from flask import Flask, jsonify, render_template, request

from predict import DISTANCE_CHOICES, predict_5k

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", distances=list(DISTANCE_CHOICES.keys()))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form

    races = []
    distances = data.getlist("distance[]") if hasattr(data, "getlist") else data.get("races", [])
    if hasattr(data, "getlist"):
        times = data.getlist("time[]")
        dates = data.getlist("date[]")
        for dist_label, time_str, date_str in zip(distances, times, dates):
            if not dist_label or not time_str or not date_str:
                continue
            minutes, seconds = time_str.split(":")
            races.append({
                "date": date_str,
                "distance_m": DISTANCE_CHOICES[dist_label],
                "time_seconds": float(minutes) * 60 + float(seconds),
            })
        gender = data.get("gender") or None
        target_date = data.get("target_date") or None
    else:
        races = distances
        gender = data.get("gender")
        target_date = data.get("target_date")

    if not races:
        return jsonify({"error": "Enter at least one prior race."}), 400

    try:
        result = predict_5k(races, gender=gender, target_date=target_date)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import os
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
