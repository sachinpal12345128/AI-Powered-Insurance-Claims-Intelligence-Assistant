import shutil, os, tempfile, threading
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from backend.models import IngestRequest
from etl.run_etl import run_etl
from backend.retrieval.bm25_search import invalidate_bm25_cache
from backend.retrieval.vector_store import invalidate_vector_store_cache
from backend.cache.query_cache import get_cache

router = APIRouter()

_job: dict = {"status": "idle", "result": None}
_lock = threading.Lock()
_running = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_csv(csv_path: str) -> str:
    p = Path(csv_path)
    if not p.is_absolute():
        p = (PROJECT_ROOT / csv_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return str(p)


def _run_etl_job(csv_path: str, cleanup_path: str | None = None):
    global _job, _running
    _job = {"status": "running", "result": None}
    try:
        result = run_etl(csv_path)
        _job = {"status": "done", "result": result}
        if result.success:
            invalidate_bm25_cache()
            invalidate_vector_store_cache()
            get_cache().clear()
    except Exception as e:
        _job = {"status": "error", "result": None, "error": str(e)}
    finally:
        with _lock:
            _running = False
        if cleanup_path:
            try:
                os.unlink(cleanup_path)
            except Exception:
                pass


@router.post("/ingest")
async def ingest(req: IngestRequest, bg: BackgroundTasks):
    global _running
    with _lock:
        if _running:
            raise HTTPException(status_code=409, detail="An ingestion is already running. Wait for it to finish or poll /ingest/status.")
        try:
            abs_path = _resolve_csv(req.csv_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
        _running = True
        _job["status"] = "queued"
    bg.add_task(_run_etl_job, abs_path)
    return {"status": "started", "message": "ETL running in background. Poll /ingest/status for progress."}


@router.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...), bg: BackgroundTasks = BackgroundTasks()):
    global _running
    with _lock:
        if _running:
            raise HTTPException(status_code=409, detail="An ingestion is already running.")
        _running = True
        _job["status"] = "queued"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    shutil.copyfileobj(file.file, tmp)
    tmp.close()
    bg.add_task(_run_etl_job, tmp.name, tmp.name)
    return {"status": "started", "message": "ETL running in background."}


@router.get("/ingest/status")
async def ingest_status():
    job = _job
    if job["status"] == "done" and job["result"]:
        r = job["result"]
        return {
            "status": "done",
            "success": r.success,
            "rows_loaded": r.rows_loaded,
            "chunks_created": r.chunks_created,
            "vectors_stored": r.vectors_stored,
            "embedding_model": r.embedding_model,
            "elapsed_seconds": r.elapsed_seconds,
            "error": r.error or None,
        }
    return {"status": job.get("status", "idle"), "running": _running, "error": job.get("error")}


@router.get("/ingest/debug-env")
async def ingest_debug_env():
    """Quick sanity check: confirms what .env values the backend actually loaded."""
    from backend.config.settings import get_settings
    s = get_settings()
    return {
        "INGEST_SAMPLE_N": os.getenv("INGEST_SAMPLE_N", "(not set)"),
        "CHUNK_SIZE": s.chunk_size,
        "EMBED_BATCH_SIZE": s.embed_batch_size,
        "LLM_PRIMARY": os.getenv("LLM_PRIMARY", "(not set)"),
        "OPENAI_API_KEY_set": bool(s.openai_api_key),
        "GROQ_API_KEY_set": bool(s.groq_api_key),
    }
