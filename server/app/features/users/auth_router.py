from fastapi import APIRouter

auth_router = APIRouter()


@auth_router.post("/login")
async def login():
    return {"message": "Login logic not implemented yet"}


@auth_router.post("/signup")
async def signup():
    return {"message": "Signup logic not implemented yet"}
