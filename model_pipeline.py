from prefect import flow, task
import pandas as pd
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import logging

# Configure logging if needed (optional, can be done globally in your script)
logging.basicConfig(level=logging.INFO)


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
):
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

    # Normalize numeric features
    scaler = StandardScaler()
    numerical_cols = training_dataset.select_dtypes(include=[np.number]).columns
    training_dataset[numerical_cols] = scaler.fit_transform(
        training_dataset[numerical_cols]
    )
    test_dataset[numerical_cols] = scaler.transform(test_dataset[numerical_cols])

    X_train = training_dataset.drop(columns=["Churn"])
    y_train = training_dataset["Churn"]
    X_test = test_dataset.drop(columns=["Churn"])
    y_test = test_dataset["Churn"]

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Save prepared data
    joblib.dump(
        (X_train_resampled, X_test, y_train_resampled, y_test, scaler),
        "prepared_data.joblib",
    )
    joblib.dump(scaler, "scaler.joblib")  # Sauvegarde du scaler
    return X_train_resampled, X_test, y_train_resampled, y_test, scaler


@task
def train_model(X_train, y_train):
    model = SVC(
        kernel="linear", probability=True, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


@task
def evaluate_model(model, X_test, y_test, threshold=0.3):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred_adjusted = (y_probs > threshold).astype(int)
    print("Accuracy:", accuracy_score(y_test, y_pred_adjusted))
    print("Classification Report:\n", classification_report(y_test, y_pred_adjusted))

    accuracy = accuracy_score(y_test, y_pred_adjusted)
    report = classification_report(y_test, y_pred_adjusted)
    # Log the accuracy and classification report
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Classification Report:\n{report}")
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

    return prediction.tolist()


@flow
def ml_pipeline(prepare: bool = True, train: bool = True, evaluate: bool = True):
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
