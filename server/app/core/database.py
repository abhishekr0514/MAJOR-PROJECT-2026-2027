from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

engine = create_async_engine(settings.DATABASE_URI, echo=False)
SessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, autocommit=False, autoflush=False
)


class Base(DeclarativeBase):
    pass


async def get_db():
    """FastAPI dependency — yields an async database session."""
    async with SessionLocal() as session:
        yield session
