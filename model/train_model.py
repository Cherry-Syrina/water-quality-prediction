import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from data.generate_data import generate_dataset, get_feature_names

MODEL_PARAMS = {
    "n_estimators":  150,
    "max_depth":     5,
    "learning_rate": 0.1,
    "subsample":     0.85,
    "random_state":  42,
}

def preprocess(df):
    X = df[get_feature_names()].values
    y = df["Potability"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler

def train_pipeline(n_samples=3000):
    df = generate_dataset(n_samples=n_samples)
    X_train_s, X_test_s, y_train, y_test, scaler = preprocess(df)
    model = GradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    metrics = {
        "accuracy":              accuracy_score(y_test, y_pred),
        "roc_auc":               roc_auc_score(y_test, y_prob),
        "confusion_matrix":      confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, target_names=["Non-Potable", "Potable"], output_dict=True),
        "y_pred":                y_pred,
    }
    return model, scaler, metrics, X_test_s, y_test
