from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from .model import predict_score
from .scoring import get_risk_level
from .alerts import generate_alert
from .database import SessionLocal
from .models import Transaction
from .schemas import TransactionInput

app = FastAPI(
    title="Anomaly Detection API",
    description="Real-time fraud detection using XGBoost + Isolation Forest ensemble",
    version="1.0.0",
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/detect")
def detect(transaction: TransactionInput, db: Session = Depends(get_db)) -> dict:
    try:
        data = transaction.model_dump()

        score = predict_score(data)
        risk = get_risk_level(score)
        alert = generate_alert(score, risk)

        record = Transaction(
            amount=data["Amount"],
            score=score,
            risk=risk,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "transaction_id": record.id,
            "score": round(score, 4),
            "risk": risk,
            "alert": alert,
        }
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

