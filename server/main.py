import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.routers import inference_router
from server.config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Server Service",
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
    app.include_router(router=inference_router)
    return app


def main() -> None:
    """Main function to set up logging and run the server."""
    app: FastAPI = create_app()

    uvicorn.run(app=app, host="0.0.0.0", port=settings.SERVICE_PORT)


if __name__ == "__main__":
    main()
