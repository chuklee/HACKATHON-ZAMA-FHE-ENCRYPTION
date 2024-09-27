import numpy as np
from deepface import DeepFace
from PIL import Image
from io import BytesIO
from typing import Union, List
import os
import subprocess
from fastapi import HTTPException
import uuid
import cv2


class InferenceService:
    def __init__(self):
        self.model = None
        self.ffmpeg_path = self._get_ffmpeg_path()
        self.ffprobe_path = self._get_ffprobe_path()

    def _get_ffmpeg_path(self):
        try:
            return subprocess.check_output(["which", "ffmpeg"]).decode().strip()
        except subprocess.CalledProcessError:
            return "ffmpeg"  # Default to just the command name if not found

    def _get_ffprobe_path(self):
        try:
            return subprocess.check_output(["which", "ffprobe"]).decode().strip()
        except subprocess.CalledProcessError:
            return "ffprobe"  # Default to just the command name if not found

    def create_embeddings(self, bytes_image: bytes) -> Union[np.ndarray, None]:
        try:
            image = Image.open(BytesIO(bytes_image))
            embeddings = DeepFace.represent(
                image, model_name="Facenet", model=self.model
            )
            return embeddings[0]["embedding"]
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return None

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

    def extract_frames(self, video_id: str) -> List[np.ndarray]:
        """
        Extract 10 random frames from the video content using cv2.
        """
        cap = cv2.VideoCapture(f"temp/{video_id}.mp4")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        if total_frames > 0:
            random_frame_indices = sorted(
                np.random.choice(total_frames, min(10, total_frames), replace=False)
            )

            for frame_index in random_frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

        cap.release()
        return frames
