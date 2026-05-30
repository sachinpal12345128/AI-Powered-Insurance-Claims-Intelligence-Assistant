"""run_etl — orchestrates extract → transform → load pipeline."""
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from etl.extract import extract
from etl.transform import preprocess, chunk_texts
from etl.load import embed_and_store, build_bm25_index

logger = logging.getLogger(__name__)


@dataclass
class ETLResult:
    rows_loaded: int = 0
    chunks_created: int = 0
    vectors_stored: int = 0
    embedding_model: str = ""
    elapsed_seconds: float = 0.0
    success: bool = False
    error: str = ""


def run_etl(
    csv_path: str,
    on_step: Optional[Callable[[str, str, int], None]] = None,
) -> ETLResult:
    """
    Run full ETL pipeline.
    on_step(step_name, status, progress_pct) — optional callback.
    """
    result = ETLResult()
    t_start = time.time()

    def step(name, fn, *args, **kwargs):
        logger.info(f"[etl] {name} …")
        if on_step:
            on_step(name, "running", 0)
        try:
            out = fn(*args, **kwargs)
            if on_step:
                on_step(name, "done", 100)
            return out
        except Exception as e:
            if on_step:
                on_step(name, "error", 0)
            raise e

    try:
        df          = step("extract",   extract,        csv_path)
        result.rows_loaded = len(df)

        texts, metadatas, ids = step("preprocess", preprocess, df)

        c_texts, c_metas, c_ids = step("chunk", chunk_texts, texts, metadatas, ids)
        result.chunks_created = len(c_texts)

        embed_result = step("load", embed_and_store, c_texts, c_metas, c_ids)
        result.vectors_stored  = embed_result["total_vectors"]
        result.embedding_model = embed_result["embedding_model"]

        step("bm25", build_bm25_index, texts, ids)

        result.elapsed_seconds = round(time.time() - t_start, 2)
        result.success = True
        logger.info(f"ETL complete in {result.elapsed_seconds}s")

    except Exception as e:
        result.error = str(e)
        result.elapsed_seconds = round(time.time() - t_start, 2)
        logger.error(f"ETL failed: {e}")

    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else "./datasets/fraud_oracle.csv"
    r = run_etl(path)
    print(f"\nResult: {r}")
