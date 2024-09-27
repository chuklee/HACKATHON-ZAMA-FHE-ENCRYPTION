from typing import Literal
from fastapi import APIRouter

router = APIRouter(tags=["Auth"])


@router.post("/login")
async def login(email: str) -> Literal[True]:
    return True
