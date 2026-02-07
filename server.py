#!/usr/bin/env python3
"""Run NexusRAG FastAPI server."""

import uvicorn


def main():
    """Start the FastAPI server."""
    uvicorn.run(
        "nexusrag.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src/nexusrag", "frontend"],
    )


if __name__ == "__main__":
    main()
