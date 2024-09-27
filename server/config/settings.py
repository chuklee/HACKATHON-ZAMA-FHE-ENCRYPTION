from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Config values for the service. Values are set automatically if a .env file is found
    in the directory root.
    """

    SERVICE_PORT: int = Field(default=8000)
    users_folder: str | None = Field(default="server/users")


settings = Settings()  # type: ignore
