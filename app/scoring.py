def get_risk_level(score: float) -> str:
    """Map a continuous anomaly score [0, 1] to a categorical risk level."""
    if score > 0.90:
        return "CRITICAL"
    if score > 0.75:
        return "HIGH"
    if score > 0.50:
        return "MEDIUM"
    return "LOW"
