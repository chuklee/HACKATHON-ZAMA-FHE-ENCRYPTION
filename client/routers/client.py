from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import FileResponse, JSONResponse
from client.dependencies import inference_service
from client.services import InferenceService

router = APIRouter(tags=["Client"])


@router.get(path="/")
def static_index() -> FileResponse:
    return FileResponse(path="app/static/index.html")


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
