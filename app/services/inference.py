import numpy as np
from deepface import DeepFace
from PIL import Image
from io import BytesIO
from typing import Union

class InferenceService:
    def __init__(self):
        self.model = DeepFace.build_model("Facenet")
    
    def create_embeddings(self, bytes_image: bytes) -> Union[np.ndarray, None]:
        try:
            image = Image.open(BytesIO(bytes_image))
            embeddings = DeepFace.represent(image, model_name="Facenet", model=self.model)            
            return embeddings[0]['embedding']
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return None