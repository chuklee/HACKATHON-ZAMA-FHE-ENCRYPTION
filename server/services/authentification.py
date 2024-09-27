import os
from server.config import settings


class AuthService:

    def __init__(self):
        pass

    def login(self, email: str) -> bool:
        """
        Check in the users folder if a file named {email}.parquet exists
        If it exists, return True
        If it does not exist, return False
        """
        users_folder = settings.users_folder
        if users_folder is None:
            raise ValueError("Users folder is not set")
        user_file = os.path.join(users_folder, f"{email}.parquet")
        print(user_file)
        return os.path.exists(user_file)
