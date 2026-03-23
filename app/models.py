from sqlalchemy import Column, Integer, Float, String
from .database import Base

class Transaction(Base):

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)

    amount = Column(Float)

    score = Column(Float)

    risk = Column(String)
