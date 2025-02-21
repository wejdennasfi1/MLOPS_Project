import argparse
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

def main():
    parser = argparse.ArgumentParser(description="Pipeline Machine Learning")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    args = parser.parse_args()

  

    if args.prepare:
        X_train, X_test, y_train, y_test, scaler = prepare_data()
        print("Données préparées.")

    if args.train:
        X_train, X_test, y_train, y_test, scaler = prepare_data()
        model = train_model(X_train, y_train)
        save_model(model, "svm_model.joblib")
        print("Modèle entraîné et sauvegardé.")

    if args.evaluate:
        X_train, X_test, y_train, y_test, scaler = prepare_data()
        model = load_model("svm_model.joblib")
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
