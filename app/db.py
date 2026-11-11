"""
Task 5 - Prediction history. SQLite locally, PostgreSQL in Docker Compose;
the backend is chosen by DATABASE_URL only.
"""
from datetime import UTC, datetime

from sqlalchemy import DateTime, String, create_engine, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    verdict: Mapped[str] = mapped_column(String(32))
    p_ai: Mapped[float]
    confidence: Mapped[float]
    threshold: Mapped[float]
    model_name: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(16))      # local | remote
    latency_ms: Mapped[float]


class Store:
    def __init__(self, url):
        self.engine = create_engine(url, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(self.engine, expire_on_commit=False)

    def add(self, **fields):
        with self.session.begin() as s:
            row = Prediction(**fields)
            s.add(row)
        return row

    def recent(self, limit=100):
        with self.session() as s:
            return s.scalars(select(Prediction).order_by(Prediction.id.desc())
                             .limit(limit)).all()

    def ping(self):
        try:
            with self.engine.connect() as c:
                c.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False
