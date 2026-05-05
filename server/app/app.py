from fastapi import FastAPI
from app.core.config import settings
from app.features.users.auth_router import auth_router


def get_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)
    app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
    return app


app = get_app()
