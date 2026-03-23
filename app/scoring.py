def get_risk_level(score):

    if score > 0.9:
        return "CRITICAL"
    elif score > 0.75:
        return "HIGH"
    elif score > 0.5:
        return "MEDIUM"
    else:
        return "LOW"
