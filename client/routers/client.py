from fastapi import APIRouter, UploadFile, File, Depends , Form
from fastapi.responses import FileResponse, JSONResponse
from client.dependencies import inference_service
from client.services import InferenceService
from PIL import Image
from io import BytesIO
from fastapi import Form

import cv2
from PIL import Image
from io import BytesIO
import numpy as np
import os
import tempfile

router = APIRouter(tags=["Client"])


@router.get(path="/")
def static_index() -> FileResponse:
    return FileResponse(path="client/static/login.html")

@router.get(path="/index")
def static_index() -> FileResponse:
    return FileResponse(path="client/static/index.html")

@router.get(path="/createAccount")
def static_index() -> FileResponse:
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
    inference_service: InferenceService = Depends(inference_service),
) -> JSONResponse:
    """
    Cette fonction traite le formulaire de création de compte, lit la vidéo téléchargée,
    extrait toutes les frames et les enregistre dans un répertoire temporaire.
    """
    # Vérifier le type de fichier vidéo
    if video.content_type not in ["video/webm", "video/mp4", "video/avi"]:
        return JSONResponse(
            status_code=400,
            content={"message": "Type de fichier vidéo non supporté."}
        )
    
    try:
        # Lire le contenu du fichier vidéo
        video_contents = await video.read()

        # Créer un fichier temporaire pour stocker la vidéo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_video:
            tmp_video.write(video_contents)
            tmp_video_path = tmp_video.name

        # Ouvrir la vidéo avec OpenCV
        vidcap = cv2.VideoCapture(tmp_video_path)
        if not vidcap.isOpened():
            return JSONResponse(
                status_code=400,
                content={"message": "Impossible d'ouvrir la vidéo."}
            )

        # Créer un répertoire temporaire pour stocker les frames
        temp_dir = tempfile.mkdtemp(prefix="video_frames_")
        frame_count = 0
        success, image = vidcap.read()

        while success:
            # Définir le chemin de la frame
            frame_filename = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(temp_dir, frame_filename)
            
            # Enregistrer la frame en tant qu'image JPEG
            cv2.imwrite(frame_path, image)
            
            # Lire la prochaine frame
            success, image = vidcap.read()
            frame_count += 1

        # Libérer les ressources
        vidcap.release()

        # Optionnel : Supprimer le fichier vidéo temporaire
        os.remove(tmp_video_path)
        
        
        print(frame_count)
        os.mkdir("frames")
        for i in range(frame_count):
           cv2.imwrite("frames/frame%d.jpg" % i, image)     # save frame as JPEG file
           success,image = vidcap.read()
           print('Read a new frame: ', success)
        

        # Vous pouvez maintenant utiliser les frames enregistrées dans `temp_dir`
        # Par exemple, vous pouvez passer ce répertoire à un service d'inférence

        return JSONResponse(
            status_code=200,
            content={
                "message": "Compte créé avec succès.",
                "email": email,
                "frames_extracted": frame_count,
                "frames_directory": temp_dir  # Attention : Exposer le chemin peut poser des risques de sécurité
            }
        )

    except Exception as e:
        # Gérer les exceptions et retourner une réponse d'erreur appropriée
        return JSONResponse(
            status_code=500,
            content={"message": "Une erreur est survenue lors du traitement de la vidéo.", "detail": str(e)}
        )