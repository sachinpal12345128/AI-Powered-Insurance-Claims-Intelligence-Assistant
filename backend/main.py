import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from backend.config.settings import get_settings
from backend.routers import query, ingest, analytics, health

settings = get_settings()

if settings.langchain_tracing_v2 == "true" and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

app = FastAPI(
    title="Insurance Claims Intelligence Assistant",
    description="AI-powered fraud detection and claims investigation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router,     prefix="/api/v1")
app.include_router(ingest.router,    prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")
app.include_router(health.router,    prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Insurance Claims Intelligence Assistant API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    # Only watch source code dirs. Without reload_dirs, every Chroma/BM25
    # write under ./data triggers a restart and breaks in-flight requests.
    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
        reload_dirs=["backend", "etl"],
    )
