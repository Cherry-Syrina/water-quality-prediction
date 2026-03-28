import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

WHO_LIMITS = {
    "pH":               (6.5,  8.5,  "Acidity/alkalinity — WHO std: 6.5–8.5"),
    "Dissolved_Oxygen": (6.5,  14.0, "Aquatic life support — safe: >6.5 mg/L"),
    "Turbidity":        (0.0,  4.0,  "Water clarity — WHO limit: <4 NTU"),
    "Temperature":      (15.0, 27.0, "Ecosystem health — optimal: 15–27°C"),
    "Nitrate":          (0.0,  10.0, "Agricultural runoff — WHO limit: <10 mg/L"),
    "BOD":              (0.0,  3.0,  "Organic pollution — good water: <3 mg/L"),
    "Conductivity":     (200,  600,  "Dissolved salts — safe: 200–600 µS/cm"),
    "Coliform":         (0,    50,   "Bacterial contamination — safe: <50 CFU/100mL"),
}

REMEDIATION = {
    "pH":               "Adjust pH using lime (alkaline) or CO₂ injection (acidic) treatment.",
    "Dissolved_Oxygen": "Improve aeration; reduce organic load; prevent thermal stratification.",
    "Turbidity":        "Apply coagulation-flocculation; install sediment filtration systems.",
    "Temperature":      "Reduce thermal discharge; increase riparian vegetation cover.",
    "Nitrate":          "Limit fertilizer runoff; implement constructed wetland buffers.",
    "BOD":              "Upgrade wastewater treatment; reduce organic discharge upstream.",
    "Conductivity":     "Investigate industrial discharge; apply reverse osmosis if needed.",
    "Coliform":         "Chlorination or UV disinfection; identify sewage contamination source.",
}

@dataclass
class ParameterStatus:
    name:        str
    value:       float
    lower:       float
    upper:       float
    description: str
    is_safe:     bool = field(init=False)

    def __post_init__(self):
        self.is_safe = self.lower <= self.value <= self.upper

@dataclass
class PredictionResult:
    is_potable:      bool
    confidence:      float
    probabilities:   np.ndarray
    parameters:      List[ParameterStatus]
    recommendations: List[str]

def check_parameters(input_values):
    return [
        ParameterStatus(name=name, value=input_values.get(name, 0.0), lower=lo, upper=hi, description=desc)
        for name, (lo, hi, desc) in WHO_LIMITS.items()
    ]

def build_recommendations(statuses):
    return [f"[{s.name}] {REMEDIATION.get(s.name, 'Investigate further.')}" for s in statuses if not s.is_safe]

def predict(model, scaler, input_values):
    x = np.array([[input_values[f] for f in WHO_LIMITS.keys()]])
    x_scaled = scaler.transform(x)
    pred  = model.predict(x_scaled)[0]
    proba = model.predict_proba(x_scaled)[0]
    statuses = check_parameters(input_values)
    return PredictionResult(
        is_potable      = bool(pred == 1),
        confidence      = float(proba[pred] * 100),
        probabilities   = proba,
        parameters      = statuses,
        recommendations = build_recommendations(statuses),
    )

def get_risk_level(result):
    violations = sum(1 for s in result.parameters if not s.is_safe)
    if result.is_potable and violations == 0:
        return "Low Risk", "#00b894"
    elif result.is_potable and violations <= 2:
        return "Moderate Risk", "#fdcb6e"
    elif not result.is_potable and violations <= 3:
        return "High Risk", "#e17055"
    else:
        return "Critical Risk", "#d63031"
