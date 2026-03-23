from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from .model import predict_score
from .scoring import get_risk_level
from .alerts import generate_alert
from .models import Transactions
from .schemas import TransactionInput

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/detect")
def detect(transaction: TransactionInput, db: Session = Depends(get_db)):

    data = transaction.dict()
    
    score = predict_score(data)

    risk = get_risk_level(score)

    alert = generate_alert(score, risk)

    record = Transaction(
        amount = data["Amount"],
        score = score,
        risk = risk
    )

    db.add(record)
    db.commit()

    return {
        "score": score,
        "risk": risk,
        "alert": alert
    }

