#!/usr/bin/env python3
"""Run NexusRAG FastAPI server."""

import uvicorn

from nexusrag.config import get_settings


def main():
    """Start the FastAPI server."""
    settings = get_settings()
    uvicorn.run(
        "nexusrag.api:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=True,
        reload_dirs=["src/nexusrag", "frontend"],
    )


if __name__ == "__main__":
    main()
