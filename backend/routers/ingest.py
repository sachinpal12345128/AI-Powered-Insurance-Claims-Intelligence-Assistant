import shutil, os, tempfile
from fastapi import APIRouter, HTTPException, UploadFile, File
from backend.models import IngestRequest, IngestResponse
from etl.run_etl import run_etl

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    result = run_etl(req.csv_path)
    return IngestResponse(
        success=result.success,
        rows_loaded=result.rows_loaded,
        chunks_created=result.chunks_created,
        vectors_stored=result.vectors_stored,
        embedding_model=result.embedding_model,
        elapsed_seconds=result.elapsed_seconds,
        error=result.error or None,
    )


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        result = run_etl(tmp_path)
    finally:
        os.unlink(tmp_path)
    return IngestResponse(
        success=result.success,
        rows_loaded=result.rows_loaded,
        chunks_created=result.chunks_created,
        vectors_stored=result.vectors_stored,
        embedding_model=result.embedding_model,
        elapsed_seconds=result.elapsed_seconds,
        error=result.error or None,
    )
