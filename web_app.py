from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# URL de ton API FastAPI
API_URL = "http://127.0.0.1:8000/predict"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Récupérer les données entrées par l'utilisateur
        features = request.form.get("features")
        try:
            # Convertir les valeurs en liste de nombres
            features_list = [float(x) for x in features.split(",")]

            # Envoyer la requête à l'API FastAPI
            response = requests.post(API_URL, json={"features": features_list})
            if response.status_code == 200:
                prediction = response.json().get("prediction")
            else:
                prediction = f"Erreur API: {response.text}"
        except Exception as e:
            prediction = f"Erreur : {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
