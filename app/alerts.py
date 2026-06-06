def generate_alert(score: float, risk: str) -> dict:
    """Return alert payload for the given anomaly score and risk level."""
    if risk in ("CRITICAL", "HIGH"):
        return {
            "alert": True,
            "message": (
                f"Fraudulent transaction detected — risk level: {risk}, "
                f"anomaly score: {round(score, 4)}"
            ),
        }
    return {"alert": False, "message": None}
