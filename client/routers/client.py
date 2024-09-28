from fastapi import APIRouter, UploadFile, File, Depends, Form, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from client.dependencies import inference_service
from client.services import InferenceService
from PIL import Image
from io import BytesIO
from fastapi import Form

from PIL import Image
from io import BytesIO
import numpy as np
import os
import tempfile
import logging
from typing import List

router = APIRouter(tags=["Client"])


@router.get(path="/")
def login() -> FileResponse:
    return FileResponse(path="client/static/login.html")


@router.get(path="/index")
def static_index() -> FileResponse:
    return FileResponse(path="client/static/index.html")


@router.get(path="/createAccount")
def create_account(request: Request) -> FileResponse:
    return FileResponse(path="client/static/createAccount.html")


@router.post("/scan_image")
async def scan_image(
    image: UploadFile = File(...),
    email: str = Form(...),
    i_service: InferenceService = Depends(inference_service),
):
    user_id = email
    # Read the contents of the file
    contents: bytes = await image.read()
    image_path: str = i_service.save_image(contents=contents, user_id=user_id)
    image_embedding: np.ndarray = i_service.get_image_embedding(image_path=image_path)
    crypt_image = i_service.crypt_image(image_embedding, user_id)
    serialized_key: bytes = i_service.get_serialize_key(user_id=user_id)
    # request to the server
    response = await i_service.send_check_face_request(
        crypted_image=crypt_image, serialized_key=serialized_key, user_id=user_id
    )
    crypted_result = response["result"]
    decrypted_result = i_service.decrypt_result(crypted_result, user_id)
    SENSITIVE_DATA = await i_service.send_token(token=decrypted_result)
    if SENSITIVE_DATA["SENSITIVE_DATA"]:
        return JSONResponse(content={"message": "SENSITIVE_DATA"})
    else:
        return JSONResponse(content={"message": "Face recognized"})


@router.post("/submitAccount")
async def submit_account(
    video: UploadFile = File(...),
    email: str = Form(...),
    i_service: InferenceService = Depends(inference_service),
):
    """
    This function processes the create account form, reads the uploaded video,
    extracts some of the frames and saves them in a temporary directory.
    """
    # Log video file details
    logging.info(
        f"Received video: filename={video.filename}, content_type={video.content_type}"
    )
    logging.info(f"Received email: {email}")

    video_content: bytes = await video.read()
    i_service.save_video(video_content=video_content, user_id=email)
    frames_path: str = i_service.save_frames(user_id=email)
    model_path, W_enc = i_service.train_model(frames_path=frames_path, user_id=email)
    i_service.push_model(model_path=model_path, W_enc=W_enc, user_id=email)
    return JSONResponse(content={"message": "Account created successfully"})
