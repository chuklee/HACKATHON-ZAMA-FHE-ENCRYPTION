from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Config values for the service. Values are set automatically if a .env file is found
    in the directory root.
    """

    SERVICE_PORT: str | None = Field(default="7860")

settings = Settings()  # type: ignore