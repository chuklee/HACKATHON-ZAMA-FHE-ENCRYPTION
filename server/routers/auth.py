from fastapi import APIRouter, Depends, HTTPException
from server.dependencies import auth_service
from server.services import AuthService

router = APIRouter(tags=["Auth"])


@router.post("/login", response_model=bool)
async def login(email: str, s_service: AuthService = Depends(auth_service)) -> bool:
    if s_service.login(email):
        return True
    raise HTTPException(status_code=404, detail="User not found")
