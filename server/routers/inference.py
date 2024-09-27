from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from server.dependencies import inference_service
from server.services import InferenceService
from typing import List

router = APIRouter(tags=["Inference"])


@router.post("/check_face")
async def check_face(
    id: str,
    crypted_image: List[float],
    i_service: InferenceService = Depends(dependency=inference_service),
) -> dict:
    if not i_service.check_user_exists(user_id=id):
        raise HTTPException(status_code=404, detail="User not found")
    try:
        crypted_result: float = i_service.check_face(crypted_image=crypted_image)
        token: int = i_service.generate_unique_token()
        return {"token": token * crypted_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(path="/check_user_exists", response_model=bool)
async def check_user_exists(
    user_id: str, i_service: InferenceService = Depends(dependency=inference_service)
) -> bool:
    if i_service.check_user_exists(user_id=user_id):
        return True
    raise HTTPException(status_code=404, detail="User not found")


@router.post("/check_token")
async def check_token(
    token: int, i_service: InferenceService = Depends(dependency=inference_service)
) -> bool:
    return i_service.check_token(token=token)


@router.post(path="/sign-in")
async def sign_in(
    user_id: str,
    crypted_model: UploadFile = File(...),
    i_service: InferenceService = Depends(dependency=inference_service),
) -> dict:
    if not i_service.check_user_exists(user_id=user_id):
        raise HTTPException(status_code=404, detail="User not found")
    
    # Read the contents of the uploaded file
    model_content: bytes = await crypted_model.read()
    
    # Pass the file content to the register_user method
    i_service.register_user(user_id=user_id, crypted_model=model_content)
    return {"token": i_service.generate_unique_token()}
