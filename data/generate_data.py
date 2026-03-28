import numpy as np
import pandas as pd

WHO_STANDARDS = {
    "pH":               (6.5,  8.5),
    "Dissolved_Oxygen": (6.5,  14.0),
    "Turbidity":        (0.0,  4.0),
    "Temperature":      (15.0, 27.0),
    "Nitrate":          (0.0,  10.0),
    "BOD":              (0.0,  3.0),
    "Conductivity":     (200,  600),
    "Coliform":         (0,    50),
}

def _generate_safe_samples(n):
    return pd.DataFrame({
        "pH":               np.random.normal(7.2, 0.4, n).clip(6.5, 8.5),
        "Dissolved_Oxygen": np.random.normal(8.0, 0.8, n).clip(6.5, 11.0),
        "Turbidity":        np.random.normal(2.5, 0.8, n).clip(0.1, 4.5),
        "Temperature":      np.random.normal(22.0, 2.5, n).clip(15.0, 27.0),
        "Nitrate":          np.random.normal(5.0, 1.5, n).clip(0.0, 10.0),
        "BOD":              np.random.normal(2.0, 0.6, n).clip(0.5, 3.5),
        "Conductivity":     np.random.normal(400, 60, n).clip(200, 600),
        "Coliform":         np.random.normal(30, 15, n).clip(0, 100),
        "Potability":       1,
    })

def _generate_unsafe_samples(n):
    half = n // 2
    ph_acidic   = np.random.normal(5.5, 0.5, half).clip(4.0, 6.4)
    ph_alkaline = np.random.normal(9.5, 0.5, n - half).clip(8.6, 11.0)
    ph_values   = np.concatenate([ph_acidic, ph_alkaline])
    return pd.DataFrame({
        "pH":               ph_values,
        "Dissolved_Oxygen": np.random.normal(4.5, 1.2, n).clip(0.0, 6.4),
        "Turbidity":        np.random.normal(8.5, 2.5, n).clip(4.6, 20.0),
        "Temperature":      np.random.normal(28.0, 3.0, n).clip(27.1, 38.0),
        "Nitrate":          np.random.normal(25.0, 8.0, n).clip(10.1, 50.0),
        "BOD":              np.random.normal(7.5, 2.0, n).clip(3.6, 15.0),
        "Conductivity":     np.random.normal(900, 150, n).clip(600, 1500),
        "Coliform":         np.random.normal(400, 150, n).clip(100, 1000),
        "Potability":       0,
    })

def generate_dataset(n_samples=3000, safe_ratio=0.55, random_state=42):
    np.random.seed(random_state)
    n_safe   = int(n_samples * safe_ratio)
    n_unsafe = n_samples - n_safe
    df = pd.concat([_generate_safe_samples(n_safe), _generate_unsafe_samples(n_unsafe)], ignore_index=True)
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

def get_feature_names():
    return ["pH", "Dissolved_Oxygen", "Turbidity", "Temperature", "Nitrate", "BOD", "Conductivity", "Coliform"]
