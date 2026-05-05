"""Central model registry.

Import this module wherever you need all SQLAlchemy models to be registered
with Base.metadata (e.g. seed.py, tests, Alembic env.py).

The order of imports matters: tables with no FK dependencies first.
"""

# ruff: noqa: F401
from app.features.hospitals.models import Hospital
from app.features.users.models import User
