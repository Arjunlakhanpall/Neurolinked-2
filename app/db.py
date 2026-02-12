from datetime import datetime
import json
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import Optional
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./neuralinked.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class InferenceResult(Base):
    __tablename__ = "inference_results"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    text = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    sqi = Column(Float, nullable=True)
    meta = Column(Text, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)

def save_result(text: str, confidence: float, sqi: Optional[float], meta: dict):
    db = SessionLocal()
    try:
        obj = InferenceResult(text=text, confidence=confidence, sqi=sqi, meta=json.dumps(meta))
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj.id
    finally:
        db.close()

def list_results(limit: int = 100):
    db = SessionLocal()
    try:
        rows = db.query(InferenceResult).order_by(InferenceResult.timestamp.desc()).limit(limit).all()
        out = []
        for r in rows:
            out.append({"id": r.id, "timestamp": r.timestamp.isoformat(), "text": r.text, "confidence": r.confidence, "sqi": r.sqi, "meta": json.loads(r.meta) if r.meta else {}})
        return out
    finally:
        db.close()

