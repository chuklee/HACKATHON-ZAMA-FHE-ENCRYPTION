import base64
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from server.dependencies import inference_service
from server.services import InferenceService
from typing import List
from pydantic import BaseModel
import time
router = APIRouter(tags=["Inference"])

i_service = InferenceService()


class CheckFaceRequest(BaseModel):
    crypted_image: str
    serialized_key: str
    user_id: str


@router.post("/check_face")
async def check_face(
    request: CheckFaceRequest,
) -> dict:
    time.sleep(1)
    if not i_service.check_user_exists(user_id=request.user_id):
        raise HTTPException(status_code=404, detail="User not found")
    try:
        # Decode both crypted_image and serialized_key from base64
        crypted_image_bytes = base64.b64decode(request.crypted_image)
        serialized_key_bytes = base64.b64decode(request.serialized_key)

        crypted_result: float = i_service.check_face(
            crypted_image=crypted_image_bytes,
            user_id=request.user_id,
            serialized_key=serialized_key_bytes,
        )
        with open(f"client/temp/{request.user_id}/crypted_result.bin", "wb") as file:
            file.write(crypted_result)
        print("Result as been saved")
        return {"result": "HAS BEEN SAVED"}
    except Exception as e:
        print(f"Error in check_face: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(path="/check_user_exists", response_model=bool)
async def check_user_exists(user_id: str) -> bool:
    if i_service.check_user_exists(user_id=user_id):
        return True
    raise HTTPException(status_code=404, detail="User not found")


class TokenRequest(BaseModel):
    token: int


@router.post("/check_token")
async def check_token(request: TokenRequest):
    
    print(f"Token: {request.token}")
    return i_service.check_token(token=request.token)


@router.post(path="/sign-in")
async def sign_in(
    user_id: str = Form(...),
    model: UploadFile = File(...),
    crypted_model: bytes = Form(...),
) -> dict:
    time.sleep(1)
    # Read the contents of the uploaded file
    model_bytes: bytes = await model.read()

    # Pass the file content to the register_user method
    i_service.register_user(
        user_id=user_id, crypted_model=crypted_model, circuit=model_bytes
    )
    return {"token": i_service.generate_unique_token()}
