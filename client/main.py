import uvicorn

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from client.routers import client_router
from client.config import settings
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Client Service",
        description="PPAI HACKATHON",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )
    app.include_router(router=client_router)
    app.mount("/static", StaticFiles(directory="client/static"), name="static")
    return app


def main() -> None:
    """Main function to set up logging and run the server."""
    app: FastAPI = create_app()

    uvicorn.run(app=app, host="0.0.0.0", port=settings.SERVICE_PORT)  # type: ignore


if __name__ == "__main__":
    main()
