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


@router.post("/get_image")
async def get_image(
    image: UploadFile = File(...),
    inference_service: InferenceService = Depends(inference_service),
) -> JSONResponse:
    # Read the contents of the file
    contents = await image.read()

    # Get the size of the image
    size = len(contents)

    # Return a confirmation message with the size of the image
    return JSONResponse(
        content={
            "message": "Image received successfully",
            "filename": image.filename,
            "size": size,
        }
    )


@router.post("/submitAccount")
async def submit_account(
    email: str = Form(...),
    video: UploadFile = File(...),
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
    video_content: bytes = await video.read()

    video_id: str = i_service.save_video(video_content=video_content)
    frames = i_service.extract_frames(video_id=video_id)
    return JSONResponse(content={"message": "Video saved successfully"})
