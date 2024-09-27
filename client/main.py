import uvicorn

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from client.routers import client_router

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Client Service",
        description="PPAI HACKATHON",
        version="1.0.0",
    )

    app.include_router(router=client_router)
    app.mount("/static", StaticFiles(directory="client/static"), name="static")
    return app


def main() -> None:
    """Main function to set up logging and run the server."""
    app: FastAPI = create_app()

    uvicorn.run(app=app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
