from prefect import flow, task
import pandas as pd
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import logging
from sklearn.feature_selection import SelectKBest, f_classif
import mlflow
from elasticsearch import Elasticsearch
from prefect.artifacts import create_markdown_artifact
import psutil
import time

# Configure logging if needed (optional, can be done globally in your script)
logging.basicConfig(level=logging.INFO)

# Connexion à Elasticsearch (ajuste l'URL si nécessaire)
es = Elasticsearch(["http://localhost:9200"])  # Utilise localhost


# Vérifier la connexion
if es.ping():
    print("✅ Connexion à Elasticsearch réussie !")
else:
    print("❌ Impossible de se connecter à Elasticsearch")
    exit(1)  # Arrêter le script si Elasticsearch n'est pas accessible


def log_to_elasticsearch(index, data):
    """Fonction pour envoyer les logs vers Elasticsearch (sans doc_type car déprécié)"""
    try:
        es.index(index=index, body=data)
        print(f"✅ Log envoyé à Elasticsearch : {data}")
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi des logs : {e}")


@task
def compute_bounds(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound


@task
def replace_outliers(df, column, method="mean"):
    lower_bound, upper_bound = compute_bounds(df, column)
    replacement_value = df[column].median() if method == "median" else df[column].mean()
    df[column] = df[column].where(
        (df[column] >= lower_bound) & (df[column] <= upper_bound), replacement_value
    )
    return df


@task
def prepare_data(
    train_path="~/wejden_nasfi_4DS8_ml_project_1/Churn-bigml-80.csv",
    test_path="~/wejden_nasfi_4DS8_ml_project_1/Churn-bigml-20.csv",
    nbr_features=10,
):
    mlflow.set_tracking_uri("http://localhost:5000")
    training_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)

    # Handle outliers
    for col in training_dataset.select_dtypes(include=[np.number]).columns:
        method = "mean" if training_dataset[col].nunique() > 2 else "median"
        training_dataset = replace_outliers(training_dataset, col, method)
        test_dataset = replace_outliers(test_dataset, col, method)

    # Encode categorical features
    label_encoders = {}
    for col in training_dataset.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        training_dataset[col] = le.fit_transform(training_dataset[col])
        test_dataset[col] = le.transform(test_dataset[col])
        label_encoders[col] = le

    # Splitting features and target
    X_train = training_dataset.drop(columns=["Churn"])
    y_train = training_dataset["Churn"]
    X_test = test_dataset.drop(columns=["Churn"])
    y_test = test_dataset["Churn"]

    # Sélection des meilleures caractéristiques
    selector = SelectKBest(score_func=f_classif, k=nbr_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Récupérer les noms des colonnes sélectionnées
    selected_columns = X_train.columns[selector.get_support()]

    # Convertir en DataFrame
    X_train = pd.DataFrame(X_train_selected, columns=selected_columns)
    X_test = pd.DataFrame(X_test_selected, columns=selected_columns)

    # Normalisation après sélection des features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reconversion en DataFrame pour éviter l'erreur AttributeError sur iloc
    X_train = pd.DataFrame(X_train, columns=selected_columns)
    X_test = pd.DataFrame(X_test, columns=selected_columns)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Sauvegarde des données préparées
    joblib.dump((X_train, X_test, y_train, y_test, scaler), "prepared_data.joblib")
    joblib.dump(scaler, "scaler.joblib")

    # Enregistrement des artefacts avec MLflow
    mlflow.log_artifact("prepared_data.joblib")
    mlflow.log_artifact("scaler.joblib")

    return X_train, X_test, y_train, y_test, scaler


@task
def train_model(X_train, y_train):
    # Configurer l'URI de suivi MLflow
    mlflow.set_tracking_uri("http://localhost:5000")  # Remplacez par l'URI correcte

    # Assurer qu'aucun autre run n'est actif
    if mlflow.active_run() is not None:
        mlflow.end_run()
    # Démarrer une nouvelle exécution MLflow
    with mlflow.start_run():
        # Définir et suivre les hyperparamètres
        kernel = "linear"
        probability = True
        random_state = 42
        class_weight = "balanced"

        mlflow.log_param("kernel", kernel)
        mlflow.log_param("probability", probability)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("class_weight", class_weight)

        # Entraîner le modèle
        model = SVC(
            kernel=kernel,
            probability=probability,
            random_state=random_state,
            class_weight=class_weight,
        )
        model.fit(X_train, y_train)

        # Suivre la précision du modèle
        accuracy = model.score(X_train, y_train)
        mlflow.log_metric("accuracy", accuracy)

        # Sauvegarde du modèle
        joblib.dump(model, "svm_model.joblib")

        # Enregistrer le modèle dans MLflow
        mlflow.sklearn.log_model(model, "svm_model")
        print("Modèle enregistré dans MLflow avec une train_accurancy de", accuracy)

        # Enregistrer le modèle comme artefact
        mlflow.log_artifact("svm_model.joblib")

        # Envoyer les logs à Elasticsearch
        # log_data = {"metric": "accuracy", "value": 0.92}
        log_data = {
            "metric": "accuracy",
            "value": accuracy,
            "@timestamp": pd.Timestamp.now().isoformat(),
        }
        log_to_elasticsearch(index="mlflow-metrics", data=log_data)

    return model


@task
def evaluate_model(model, X_test, y_test, threshold=0.3):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred_adjusted = (y_probs > threshold).astype(int)
    print("Accuracy:", accuracy_score(y_test, y_pred_adjusted))
    print("Classification Report:\n", classification_report(y_test, y_pred_adjusted))

    accuracy = accuracy_score(y_test, y_pred_adjusted)
    report = classification_report(y_test, y_pred_adjusted, output_dict=True)
    # Log the accuracy and classification report
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Classification Report:\n{report}")

    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run():
        mlflow.log_metric("test_accuracy", accuracy)
        print("✅ Evaluation metrics logged to MLflow.")
        # Envoyer les logs à Elasticsearch
        # log_data = {"metric": "accuracy", "value": 0.92}
        log_data = {
            "metric": "accuracy",
            "value": accuracy,
            "@timestamp": pd.Timestamp.now().isoformat(),
        }
        log_to_elasticsearch(index="mlflow-metrics", data=log_data)

    class_0 = list(report.keys())[
        0
    ]  # Récupère la première clé (normalement "0" ou False)
    class_1 = list(report.keys())[1]
    markdown_content = f"""
    ### 🔍 Model Evaluation Metrics
    - **Accuracy**: {accuracy:.4f}
    - **Precision (Class {class_0})**: {report[class_0]["precision"]:.4f}
    - **Recall (Class {class_0})**: {report[class_0]["recall"]:.4f}
    - **F1-score (Class {class_0})**: {report[class_0]["f1-score"]:.4f}
    - **Precision (Class {class_1})**: {report[class_1]["precision"]:.4f}
    - **Recall (Class {class_1})**: {report[class_1]["recall"]:.4f}
    - **F1-score (Class {class_1})**: {report[class_1]["f1-score"]:.4f}
    """
    create_markdown_artifact(
        key="model-metrics", markdown=markdown_content  # Remplace "_" par "-"
    )

    report_path = "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write(markdown_content)

    # Enregistrer le rapport Markdown comme artefact dans MLflow
    mlflow.log_artifact(report_path)

    return accuracy, report


@task
def save_model(model, filename="svm_model.joblib"):
    joblib.dump(model, filename)


@task
def load_model(filename="svm_model.joblib"):
    return joblib.load(filename)


@task
def load_prepared_data(filename="prepared_data.joblib"):
    return joblib.load(filename)


@task
@task
def predict(features):
    # Charger le modèle entraîné
    model = joblib.load("svm_model.joblib")

    # Charger le scaler
    scaler = joblib.load("scaler.joblib")

    # Vérifier la dimension de features
    features = np.array(features).reshape(1, -1)  # Transformer en tableau 2D

    # Transformer les features avec le scaler
    features_scaled = scaler.transform(features)

    # Prédire avec le modèle SVM
    prediction = model.predict(features_scaled)
    # typiquement celle créée lors de l'entraînement ou d'une autre prédiction.
    with mlflow.start_run(nested=True):
        # Enregistre les caractéristiques d'entrée (sous forme de liste) comme paramètre dans MLflow.
        mlflow.log_param("input_features", features.tolist())
        # Enregistre la première prédiction (valeur unique) comme une métrique dans MLflow.
        # Cela peut être utile pour analyser ou suivre les résultats de différentes prédictions.
        mlflow.log_metric("prediction", prediction[0])

    return prediction.tolist()


@task
def monitor_system_task(duration=15):
    start_time = time.time()  # Time when the monitoring starts
    while time.time() - start_time < duration:
        # Surveillance de la CPU
        cpu_usage = psutil.cpu_percent(interval=1)
        print(f"Usage CPU : {cpu_usage}%")

        # Surveillance de la RAM
        memory = psutil.virtual_memory()
        print(
            f"Usage RAM : {memory.percent}% (Total: {memory.total / (1024 ** 3):.2f} GB)"
        )

        # Surveillance de l'espace disque
        disk = psutil.disk_usage("/")
        print(
            f"Usage disque : {disk.percent}% (Total: {disk.total / (1024 ** 3):.2f} GB)"
        )

        time.sleep(5)  # Wait for 5 seconds before the next iteration


@flow
def ml_pipeline(prepare: bool = True, train: bool = True, evaluate: bool = True):

    # Démarrer la surveillance en parallèle
    monitor_system_task.submit(duration=15)
    if prepare:
        X_train, X_test, y_train, y_test, scaler = prepare_data()  # Store result
        print("Données préparées.")

    if train:
        X_train, X_test, y_train, y_test, scaler = load_prepared_data()
        model = train_model(X_train, y_train)
        save_model(model)
        print("Modèle entraîné et sauvegardé.")

    if evaluate:
        X_train, X_test, y_train, y_test, scaler = load_prepared_data()
        model = load_model()
        evaluate_model(model, X_test, y_test)
    if predict:
        X_train, X_test, y_train, y_test, scaler = (
            load_prepared_data()
        )  # Prépare les données
        model = load_model()  # Charge le modèle sauvegardé
        predict(X_test.iloc[0])  # Prédire sur un seul échantillon
