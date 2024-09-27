import uvicorn

from fastapi import FastAPI
from server.routers import auth_router
from server.config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Server Service",
        description="PPAI HACKATHON",
        version="1.0.0",
    )

    app.include_router(router=auth_router)
    return app


def main() -> None:
    """Main function to set up logging and run the server."""
    app: FastAPI = create_app()

    uvicorn.run(app=app, host="0.0.0.0", port=settings.SERVICE_PORT)


if __name__ == "__main__":
    main()
