from typing import Literal
from fastapi import APIRouter, Depends
from server.dependencies import auth_service
from server.services import AuthService

router = APIRouter(tags=["Auth"])


@router.post("/login", response_model=bool, status_code=200)
async def login(email: str, s_service: AuthService = Depends(auth_service)) -> bool:
    return s_service.login(email)
