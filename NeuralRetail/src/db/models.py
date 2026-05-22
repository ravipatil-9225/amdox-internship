"""
SQLAlchemy Database Models for NeuralRetail
Defines ORM schemas for Users, Customers, Predictions, and Audit Logs.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime

Base = declarative_base()


class User(Base):
    """Platform users with role-based access control."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="viewer")  # admin, analyst, viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    predictions = relationship("PredictionLog", back_populates="user")


class CustomerProfile(Base):
    """Customer profiles with RFM scores and segment assignments."""
    __tablename__ = "customer_profiles"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(50), unique=True, nullable=False, index=True)
    age = Column(Integer)
    region = Column(String(100))
    segment = Column(String(100))
    segment_id = Column(Integer)
    rfm_recency = Column(Float)
    rfm_frequency = Column(Float)
    rfm_monetary = Column(Float)
    lifetime_value = Column(Float)
    churn_probability = Column(Float)
    churn_risk_segment = Column(String(50))
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PredictionLog(Base):
    """Immutable audit log for all model predictions."""
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    prediction_type = Column(String(50), nullable=False)  # demand, churn, segment, inventory
    request_payload = Column(JSON)
    response_payload = Column(JSON)
    model_version = Column(String(100))
    mlflow_run_id = Column(String(100))
    latency_ms = Column(Float)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="predictions")


class DemandForecast(Base):
    """Stored demand forecasts for historical tracking."""
    __tablename__ = "demand_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    sku_id = Column(String(50), nullable=False, index=True)
    store_id = Column(String(50))
    forecast_date = Column(DateTime, nullable=False)
    predicted_demand = Column(Float)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    actual_demand = Column(Float, nullable=True)  # filled later for accuracy tracking
    mape = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class InventorySnapshot(Base):
    """Point-in-time inventory snapshots for trend analysis."""
    __tablename__ = "inventory_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    sku_id = Column(String(50), nullable=False, index=True)
    store_id = Column(String(50))
    current_stock = Column(Integer)
    safety_stock = Column(Integer)
    reorder_qty = Column(Integer)
    stockout_risk = Column(Float)
    abc_class = Column(String(1))
    dead_stock = Column(Boolean, default=False)
    snapshot_at = Column(DateTime, default=datetime.utcnow)


class DriftReport(Base):
    """Model drift detection audit trail."""
    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    report_type = Column(String(50))  # data_drift, model_drift, prediction_drift
    total_features = Column(Integer)
    drifted_features = Column(Integer)
    drift_share = Column(Float)
    psi_score = Column(Float, nullable=True)
    action_required = Column(Boolean, default=False)
    report_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)


class DataQualityResult(Base):
    """Data quality gate results for compliance tracking."""
    __tablename__ = "data_quality_results"

    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String(100), nullable=False)
    total_expectations = Column(Integer)
    passed_expectations = Column(Integer)
    dq_score = Column(Float)
    gate_passed = Column(Boolean)
    details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


# Database engine factory
def get_engine(database_url: str = "sqlite:///data/neuralretail.db"):
    """Create a database engine. Defaults to SQLite for local dev."""
    return create_engine(database_url, echo=False)


def get_session(database_url: str = "sqlite:///data/neuralretail.db"):
    """Create a database session."""
    engine = get_engine(database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def init_db(database_url: str = "sqlite:///data/neuralretail.db"):
    """Initialize database and create all tables."""
    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at: {database_url}")
    print(f"Tables created: {list(Base.metadata.tables.keys())}")
    return engine


if __name__ == "__main__":
    init_db()
