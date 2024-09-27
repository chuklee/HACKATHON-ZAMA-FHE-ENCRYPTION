from fastapi import APIRouter, Form, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import os

router = APIRouter(tags=["Client"])

@router.get("/")
def static_index():
    return FileResponse("app/static/index.html")

@router.post("/submit")
async def submit_form(user_input: str = Form(...), image: UploadFile = File(...)):
    print(f"User input: {user_input}")
    
    # Save the uploaded image
    upload_dir = "app/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, image.filename)  # type: ignore
    
    with open(file_path, "wb") as buffer:
        content = await image.read()
        buffer.write(content)
    
    print(f"Image saved: {file_path}")
    
    return HTMLResponse("<h1>Form submitted successfully!</h1><p>Image uploaded and saved.</p><a href='/'>Go back</a>")
