"""Transform — preprocess rows into text blocks and chunk them."""
import pandas as pd
import logging
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Preprocessing ─────────────────────────────────────────────────────────────

def row_to_text(row: pd.Series) -> str:
    parts = [
        f"Claim ID: {row.get('claim_id', 'N/A')}",
        f"Policy type: {row.get('policy_type', 'N/A')}",
        f"Base policy: {row.get('base_policy', 'N/A')}",
        f"Accident area: {row.get('customer_region', 'N/A')}",
        f"Vehicle make: {row.get('vehicle_make', 'N/A')}",
        f"Vehicle category: {row.get('vehicle_category', 'N/A')}",
        f"Vehicle price: {row.get('claim_amount', 'N/A')}",
        f"Fault: {row.get('fault', 'N/A')}",
        f"Police report filed: {row.get('police_report_filed', 'N/A')}",
        f"Witness present: {row.get('witness_present', 'N/A')}",
        f"Agent type: {row.get('agent_type', 'N/A')}",
        f"Number of supplements: {row.get('number_of_supplements', 'N/A')}",
        f"Address change near claim: {row.get('address_change', 'N/A')}",
        f"Past number of claims: {row.get('customer_history', 'N/A')}",
        f"Driver rating: {row.get('driver_rating', 'N/A')}",
        f"Deductible: {row.get('deductible', 'N/A')}",
        f"Age of policy holder: {row.get('age_of_policy_holder', 'N/A')}",
        f"Age of vehicle: {row.get('age_of_vehicle', 'N/A')}",
        f"Number of cars: {row.get('number_of_cars', 'N/A')}",
        f"Incident date: {row.get('incident_date', 'N/A')}",
        f"Fraud label: {'Fraud' if row.get('fraud_label') == 1 else 'Legitimate'}",
    ]
    return ". ".join(parts) + "."


def build_metadata(row: pd.Series) -> dict:
    return {
        "claim_id": str(row.get("claim_id", "")),
        "policy_type": str(row.get("policy_type", "")),
        "base_policy": str(row.get("base_policy", "")),
        "customer_region": str(row.get("customer_region", "")),
        "fraud_label": int(row.get("fraud_label", 0)),
        "fault": str(row.get("fault", "")),
        "police_report_filed": str(row.get("police_report_filed", "")),
        "witness_present": str(row.get("witness_present", "")),
        "agent_type": str(row.get("agent_type", "")),
        "incident_date": str(row.get("incident_date", "")),
        "claim_amount": str(row.get("claim_amount", "")),
        "driver_rating": str(row.get("driver_rating", "")),
    }


def preprocess(df: pd.DataFrame) -> tuple[list[str], list[dict], list[str]]:
    texts, metadatas, ids = [], [], []
    for _, row in df.iterrows():
        texts.append(row_to_text(row))
        metadatas.append(build_metadata(row))
        ids.append(str(row.get("claim_id", f"doc-{_}")))
    logger.info(f"Preprocessed {len(texts)} records into text blocks.")
    return texts, metadatas, ids


# ── Chunking ──────────────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Lightweight recursive splitter — no torch / langchain_text_splitters dependency."""
    if len(text) <= chunk_size:
        return [text]
    seps = [". ", ", ", " "]
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            for sep in seps:
                idx = text.rfind(sep, start, end)
                if idx > start:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end])
        start = max(end - overlap, start + 1)
        if end >= n:
            break
    return chunks


def chunk_texts(
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> tuple[list[str], list[dict], list[str]]:
    chunked_texts, chunked_metas, chunked_ids = [], [], []
    for text, meta, doc_id in zip(texts, metadatas, ids):
        chunks = _split_text(text, settings.chunk_size, settings.chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunked_texts.append(chunk)
            chunked_metas.append({**meta, "chunk_index": i, "parent_id": doc_id})
            chunked_ids.append(f"{doc_id}_chunk_{i}")
    logger.info(f"Chunked {len(texts)} docs → {len(chunked_texts)} chunks")
    return chunked_texts, chunked_metas, chunked_ids
