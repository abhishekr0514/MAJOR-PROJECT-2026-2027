from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import engine
from app.features.users.auth_router import auth_router
from app.features.users.router import user_router
from app.features.hospitals.router import hospital_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Dispose the DB engine gracefully on shutdown.

    NOTE: Schema migrations are handled by Alembic — run
          ``uv run alembic upgrade head`` before starting the server.
    """
    yield
    await engine.dispose()


def get_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
    app.include_router(user_router, prefix="/users", tags=["User Management"])
    app.include_router(
        hospital_router, prefix="/hospitals", tags=["Hospital Management"]
    )
    return app


app = get_app()
