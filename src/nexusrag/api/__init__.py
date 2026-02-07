"""API module for FastAPI endpoints."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from nexusrag.api.routes import router

# Frontend directory
FRONTEND_DIR = Path(__file__).parent.parent.parent.parent / "frontend"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="NexusRAG",
        description="Self-correcting retrieval for scientific literature synthesis",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(router)

    # Serve static files
    if FRONTEND_DIR.exists():
        # Mount CSS directory
        css_dir = FRONTEND_DIR / "css"
        if css_dir.exists():
            app.mount("/css", StaticFiles(directory=css_dir), name="css")

        # Mount JS directory
        js_dir = FRONTEND_DIR / "js"
        if js_dir.exists():
            app.mount("/js", StaticFiles(directory=js_dir), name="js")

        @app.get("/")
        async def serve_frontend() -> FileResponse:
            """Serve the frontend HTML."""
            return FileResponse(FRONTEND_DIR / "index.html")

    return app


# Create app instance
app = create_app()

__all__ = ["app", "create_app", "router"]
