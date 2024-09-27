import numpy as np
from deepface import DeepFace
from PIL import Image
from io import BytesIO
from typing import Union, List
import os
from fastapi import HTTPException
import uuid
import cv2
from .preprocess import load_dataset, RegNet
from sklearn.linear_model import LogisticRegression
from concrete.ml.torch.compile import compile_torch_model


class InferenceService:
    def __init__(self):
        self.model = None

    def save_video(self, video_content: bytes) -> str:
        """
        Extract frames from the video content asynchronously using FFmpeg.
        """
        try:
            video_id = str(uuid.uuid4())
            os.makedirs("temp", exist_ok=True)
            with open(file=f"temp/{video_id}.mp4", mode="wb") as f:
                f.write(video_content)
            return video_id
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving video: {e}")

    def save_frames(self, video_id: str) -> str:
        """
        Save 10 random frames from the video content using cv2 and return the path to the frames.
        """
        cap = cv2.VideoCapture(filename=f"temp/{video_id}.mp4")
        for i in range(10):
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(filename=f"temp/{video_id}/frame_{i}.jpg", img=frame)
            else:
                raise HTTPException(status_code=500, detail=f"Error saving frame {i}")
        cap.release()
        return f"temp/{video_id}/"

    def train_model(self, frames_path: str) -> str:
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
        X_rand_stack = np.hstack([X_train_rand, W_rand])
        reg_net = RegNet(b)
        quantized_module = compile_torch_model(
            reg_net,  # our model
            X_rand_stack,  # a representative input-set to be used for both quantization and compilation
            n_bits=6,
            rounding_threshold_bits={"n_bits": 6, "method": "approximate"},
        )