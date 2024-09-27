from fastapi import APIRouter, Depends, HTTPException
from server.dependencies import auth_service
from server.services import AuthService

router = APIRouter(tags=["Inference"])


@router.post("/inference")
async def inference(email: str, s_service: AuthService = Depends(auth_service)) -> bool:
    if not s_service.login(email):
        raise HTTPException(status_code=404, detail="User not found")
