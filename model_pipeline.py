import pandas as pd
import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report,precision_recall_curve
from sklearn.model_selection import train_test_split

def compute_bounds(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def replace_outliers_with_median(df, column):
    lower_bound, upper_bound = compute_bounds(df, column)
    median_value = df[column].median()
    df[column] = df[column].where((df[column] >= lower_bound) & (df[column] <= upper_bound), median_value)
    return df

def replace_outliers_with_mean(df, column):
    lower_bound, upper_bound = compute_bounds(df, column)
    mean_value = df[column].mean()
    df[column] = df[column].where((df[column] >= lower_bound) & (df[column] <= upper_bound), mean_value)
    return df

def prepare_data(train_path='~/wejden_nasfi_4DS8_ml_project_1/Churn-bigml-80.csv', test_path='~/wejden_nasfi_4DS8_ml_project_1/Churn-bigml-20.csv'):
    training_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)
    
    for col in training_dataset.select_dtypes(include=[np.number]).columns:
        if training_dataset[col].nunique() > 2:
            training_dataset = replace_outliers_with_mean(training_dataset, col)
            test_dataset = replace_outliers_with_mean(test_dataset, col)
        else:
            training_dataset = replace_outliers_with_median(training_dataset, col)
            test_dataset = replace_outliers_with_median(test_dataset, col)
    
    label_encoders = {}
    for col in training_dataset.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        training_dataset[col] = le.fit_transform(training_dataset[col])
        test_dataset[col] = le.transform(test_dataset[col])
        label_encoders[col] = le
    
    scaler = StandardScaler()
    numerical_cols = training_dataset.select_dtypes(include=[np.number]).columns
    training_dataset[numerical_cols] = scaler.fit_transform(training_dataset[numerical_cols])
    test_dataset[numerical_cols] = scaler.transform(test_dataset[numerical_cols])
    
    X_train = training_dataset.drop(columns=['Churn'])
    y_train = training_dataset['Churn']
    X_test = test_dataset.drop(columns=['Churn'])
    y_test = test_dataset['Churn']
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test, scaler

def train_model(X_train, y_train):
    model = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, threshold=0.3):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred_adjusted = (y_probs > threshold).astype(int)
    print("Accuracy:", accuracy_score(y_test, y_pred_adjusted))
    print("Classification Report:\n", classification_report(y_test, y_pred_adjusted))

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)
