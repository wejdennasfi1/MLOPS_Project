from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.svm import SVC


# Initialize FastAPI app
app = FastAPI()

MODEL_PATH = "svm_model.joblib"

# Try loading the SVM model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading the model: {str(e)}")


class PredictionInput(BaseModel):
    features: list


@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Reshape the input features and predict using the SVM model
        input_array = np.array(input_data.features).reshape(1, -1)
        prediction = model.predict(input_array)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error during prediction: {str(e)}"
        )


class RetrainInput(BaseModel):
    kernel: str = "linear"  # Default kernel is linear
    C: float = 1.0  # Regularization parameter C
    random_state: int = 42  # Random state for reproducibility


# Function to simulate loading of training data (replace with actual data)
def load_data():
    X_train = np.random.rand(100, 9)  # Example feature set (100 samples, 9 features)
    y_train = np.random.randint(0, 2, 100)  # Random binary target variable
    return X_train, y_train


@app.post("/retrain")
def retrain(params: RetrainInput):
    try:
        global model
        # Load the training data (replace this with your actual data loading function)
        X_train, y_train = load_data()

        # Initialize and train the SVM model with the provided parameters
        model = SVC(
            kernel=params.kernel,
            C=params.C,
            random_state=params.random_state,
            class_weight="balanced",  # Optional: Helps with class imbalance
        )
        model.fit(X_train, y_train)

        # Save the newly trained model
        joblib.dump(model, MODEL_PATH)

        return {"message": "SVM Model retrained successfully", "params": params.dict()}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error during retraining: {str(e)}"
        )
