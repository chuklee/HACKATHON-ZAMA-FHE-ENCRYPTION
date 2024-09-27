from datetime import datetime, timedelta
from server.config import settings
import os
import random
from fastapi import HTTPException
from typing import List


class InferenceService:
    def __init__(self):
        self.qr_codes = {}
        self.tokens = {}

    def retrieve_model(self, pk: str):
        # TODO
        pass

    def check_face(self, crypted_image: List[float], user_id: str) -> int:
        # TODO: Implement actual face checking logic
        return 9

    def check_user_exists(self, user_id: str) -> bool:
        """
        Check in the users folder if a file named {email}.parquet exists
        If it exists, return True
        If it does not exist, return False
        """
        users_folder = settings.users_folder
        if users_folder is None:
            raise ValueError("Users folder is not set")
        user_file = os.path.join(users_folder, f"{user_id}.zip")
        print(user_file)
        return os.path.exists(user_file)

    def generate_unique_token(self) -> int:
        """
        Create a unique token that is saved in for 5 minutes
        """
        token = random.randint(0, 1000000)
        self.tokens[token] = datetime.now() + timedelta(minutes=5)
        return token

    def check_token(self, token: int) -> bool:
        """
        Check if the token is still valid
        """
        if token in self.tokens:
            return self.tokens[token] > datetime.now()
        raise HTTPException(status_code=401, detail="Token is not valid")

    def register_user(self, user_id: str, crypted_model: bytes):
        """
        Save the uploaded crypted_model as a .zip file in the users folder
        """
        users_folder = settings.users_folder
        if users_folder is None:
            raise ValueError("Users folder is not set")

        # Create the users folder if it doesn't exist
        os.makedirs(users_folder, exist_ok=True)

        # Define the path for the user's zip file
        user_file_path = os.path.join(users_folder, f"{user_id}.zip")

        try:
            with open(user_file_path, "wb") as file:
                file.write(crypted_model)
        except IOError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to save user file: {str(e)}"
            )
        return True
