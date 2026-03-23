def generate_alert(score, risk):

    if risk in ["CRITICAL", "HIGH"]:
        return {
            "alert" : True,
            "message" : f"Fraud has been detected with score {score}"
        }
    return {"alert": False}
