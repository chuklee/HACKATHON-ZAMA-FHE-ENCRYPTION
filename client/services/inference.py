import numpy as np
import requests
from deepface import DeepFace
from PIL import Image
from io import BytesIO
from typing import Union, List
import os
from fastapi import HTTPException
import uuid
import cv2
from .preprocess import load_dataset, RegNet, compute_embeddings_and_labels
from sklearn.linear_model import LogisticRegression
from concrete.ml.torch.compile import compile_torch_model
import logging
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from client.config import settings
import httpx
import base64


class InferenceService:
    def __init__(self):
        self.model = None

    def save_video(self, video_content: bytes, user_id: str) -> None:
        """
        Extract frames from the video content asynchronously using FFmpeg.
        """
        try:
            video_id = str(uuid.uuid4())
            os.makedirs(f"client/temp/{user_id}", exist_ok=True)
            with open(file=f"client/temp/{user_id}/video.mp4", mode="wb") as f:
                f.write(video_content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving video: {e}")

    def save_frames(self, user_id: str) -> str:
        """
        Save 10 random frames from the video content using cv2 and return the path to the frames.
        """
        video_path = f"client/temp/{user_id}/video.mp4"
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video not found")
        os.makedirs(f"client/temp/{user_id}/images", exist_ok=True)
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        while success and count < 10:
            cv2.imwrite(
                f"client/temp/{user_id}/images/frame%d.jpg" % count, image
            )  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
        return f"client/temp/{user_id}/images"

    def train_model(self, frames_path: str, user_id: str):
        """
        Train a model on the frames and return the path to the model.
        """
        embeddings, labels = load_dataset(frames_path, cache=True)
        embeddings = (embeddings - np.mean(embeddings, axis=0)) / np.std(
            embeddings, axis=0
        )
        model = LogisticRegression(C=1 / 5)
        model.fit(embeddings, y=labels)
        nb_sample = 100
        W = model.coef_
        b = model.intercept_.reshape(-1, 1)
        X_train_rand = np.random.normal(0, 1, [nb_sample, embeddings.shape[1]])
        W_rand = np.random.normal(0, 1, [nb_sample, W.shape[1]])
        reg_net = RegNet(b)
        quantized_module = compile_torch_model(
            reg_net,  # our model
            (
                X_train_rand,
                W_rand,
            ),  # a representative input-set to be used for both quantization and compilation
            n_bits=6,
            rounding_threshold_bits={"n_bits": 6, "method": "approximate"},
        )
        save_path: str = f"client/temp/model/{user_id}"
        fhemodel_dev = FHEModelDev(save_path, quantized_module)
        fhemodel_dev.save()
        W_encrypted = self.encrypt_weights(save_path, W)
        return save_path, W_encrypted

    def encrypt_weights(self, save_path: str, W) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Load the weights from the model.
        """
        fhemodel_client = FHEModelClient(save_path, key_dir=save_path)
        input_e = fhemodel_client.quantize_encrypt_serialize(np.zeros((1, 128)), W)
        return input_e[1]

    def push_model(self, model_path: str, W_enc, user_id: str) -> None:
        """
        Push the model to the servrer
        """
        try:
            # Define the URL of the endpoint you want to send the file to
            url = settings.SERVEUR_ENDPOINT + "/sign-in"

            # Prepare the file for upload
            with open(model_path + "/server.zip", "rb") as file:
                files = {"model": (f"{user_id}_model.zip", file, "application/zip")}
                os.makedirs(f"server/users/{user_id}", exist_ok=True)
                with open(f"server/users/{user_id}/W_enc.bin", "wb") as file:
                    file.write(W_enc)
                # Additional data to send with the request
                data = {"user_id": user_id, "crypted_model": W_enc}

                # Send a POST request to the server
                response = requests.post(url, files=files, data=data)

            # Check if the request was successful
            if response.status_code == 200:
                print(f"Model for user {user_id} uploaded successfully")
            else:
                print(f"Failed to upload model. Status code: {response.status_code}")
                print(f"Response: {response.text}")

        except Exception as e:
            print(f"Error pushing model to server: {str(e)}")

    def save_image(self, contents: bytes, user_id: str) -> str:
        """
        Save the image and return the path to the image.
        """
        image_id = str(uuid.uuid4())
        os.makedirs(f"client/temp/image/{user_id}", exist_ok=True)
        with open(file=f"client/temp/image/{user_id}/{image_id}.jpg", mode="wb") as f:
            f.write(contents)
        return f"client/temp/image/{user_id}/{image_id}.jpg"

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Get the image embedding
        """
        embeddings, _ = compute_embeddings_and_labels(images=[image_path], label=0)
        return embeddings[0]

    def crypt_image(self, image_embedding: np.ndarray, user_id: str):
        """
        Crypt the image embedding
        """
        fhemodel_client = FHEModelClient(
            path_dir=f"client/temp/model/{user_id}",
            key_dir=f"client/temp/model/{user_id}",
        )
        input_e = fhemodel_client.quantize_encrypt_serialize(
            image_embedding.reshape(1, 128), np.zeros((1, 128))
        )
        return input_e[0]

    def get_serialize_key(self, user_id: str) -> bytes:
        """
        Get the serialize key
        """
        fhemodel_client = FHEModelClient(
            path_dir=f"client/temp/model/{user_id}",
            key_dir=f"client/temp/model/{user_id}",
        )
        return fhemodel_client.get_serialized_evaluation_keys()

    async def send_check_face_request(
        self, crypted_image: bytes, serialized_key: bytes, user_id: str
    ) -> dict:
        with open(
            f"server/users/{user_id}/serialized_evaluation_keys.ekl", "wb"
        ) as file:
            file.write(serialized_key)
        with open(f"server/users/{user_id}/crypted_image.bin", "wb") as file:
            file.write(crypted_image)
        server_url = (
            settings.SERVEUR_ENDPOINT + "/check_face"
        )  # Use the server URL from settings
        

        # Encode the serialized_key to base64
        serialized_key_b64 = base64.b64encode(serialized_key).decode("utf-8")
        crypted_image_b64 = base64.b64encode(crypted_image).decode("utf-8")
        data = {
            "crypted_image": crypted_image_b64,
            "serialized_key": serialized_key_b64,
            "user_id": user_id,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(server_url, json=data, timeout=200)
            response.raise_for_status()
            return response.json()

    def decrypt_result(self, result: bytes, user_id: str):
        """
        Decrypt the result
        """
        fhemodel_client = FHEModelClient(
            path_dir=f"client/temp/model/{user_id}",
            key_dir=f"client/temp/model/{user_id}",
        )
        return fhemodel_client.deserialize_decrypt_dequantize(result)[0][0]

    async def send_token(self, token):
        """
        Send the token to the server
        """
        server_url = settings.SERVEUR_ENDPOINT + "/check_token"
        data = {"token": int(token)}
        async with httpx.AsyncClient() as client:
            response = await client.post(server_url, json=data)
            response.raise_for_status()
            return response.json()
