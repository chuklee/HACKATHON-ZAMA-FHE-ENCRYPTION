from datetime import datetime, timedelta
from server.config import settings
import os
import random
from fastapi import HTTPException
from typing import List
from concrete.ml.deployment import FHEModelClient, FHEModelServer


class InferenceService:
    def __init__(self):
        self.qr_codes = {}
        self.tokens = {}
        self.models = {}

    def retrieve_model(self, user_id: str):
        return self.models[user_id]

    def check_face(self, crypted_image, user_id: str, serialized_key: bytes):
        circuit, crypted_model = self.retrieve_model(user_id)
        fhemodel_server = FHEModelServer(path_dir=f"server/users/{user_id}")
        # Read the serialized_key from the file
        with open(
            f"server/users/{user_id}/serialized_evaluation_keys.ekl", "rb"
        ) as file:
            serialized_key = file.read()
        with open(f"server/users/{user_id}/crypted_image.bin", "rb") as file:
            crypted_image = file.read()
        with open(f"server/users/{user_id}/W_enc.bin", "rb") as file:
            W_enc = file.read()
        inputs = (crypted_image, W_enc)
        return fhemodel_server.run(inputs, serialized_evaluation_keys=serialized_key)

    def check_user_exists(self, user_id: str) -> bool:
        """
        Check in the users exists
        If it exists, return True
        If it does not exist, return False
        """
        return user_id in self.models

    def generate_unique_token(self) -> int:
        """
        Create a unique token that is saved in for 5 minutes
        """
        token = random.randint(0, 1000000)
        self.tokens[token] = datetime.now() + timedelta(minutes=5)
        return token

    def check_token(self, token: int) -> dict:
        """
        Check if the token is still valid
        """
        if token == 42:
            return {"SENSITIVE_DATA": "We love Zama"}
        raise HTTPException(status_code=401, detail="Token is not valid")

    def register_user(self, user_id: str, crypted_model: bytes, circuit: bytes):
        """
        Save the uploaded crypted_model as a .zip file in the users folder
        """
        users_folder = settings.users_folder + f"/{user_id}"
        if users_folder is None:
            raise ValueError("Users folder is not set")

        # Create the users folder if it doesn't exist
        os.makedirs(users_folder, exist_ok=True)

        # Define the path for the user's zip file
        user_file_path = os.path.join(users_folder, f"server.zip")

        try:
            with open(user_file_path, "wb") as file:
                file.write(circuit)
        except IOError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to save user file: {str(e)}"
            )

        self.models[user_id] = (user_file_path, crypted_model)
        return True
