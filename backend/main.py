import ssl
import urllib3
import urllib3.util.ssl_

# Disable SSL certificate verification globally — corporate proxy with self-signed cert.
ssl._create_default_https_context = ssl._create_unverified_context

# Patch ssl.create_default_context (used by langchain_openai and others)
_orig_cdc = ssl.create_default_context
def _patched_cdc(*args, **kwargs):
    ctx = _orig_cdc(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
ssl.create_default_context = _patched_cdc

# Patch urllib3's SSL context factory (used by Pinecone's internal HTTP pool)
def _urllib3_no_verify(*args, **kwargs):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

urllib3.util.ssl_.create_urllib3_context = _urllib3_no_verify
try:
    import urllib3.connection
    urllib3.connection.create_urllib3_context = _urllib3_no_verify
except Exception:
    pass
try:
    import urllib3.connectionpool
    urllib3.connectionpool.create_urllib3_context = _urllib3_no_verify
except Exception:
    pass

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
    # reload=False: single-process mode — no watcher subprocess.
    # This guarantees every `python -m backend.main` starts completely fresh
    # with the latest code. Previously reload=True caused stale subprocesses
    # to accumulate on Windows and serve old code after restarts.
    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )
