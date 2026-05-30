"""
ContextualCompression: extracts only the relevant snippet from each retrieved chunk.
Implements LangChain's contextual compression pattern using the project LLM directly.
Falls back to returning original chunks unchanged on any error.
"""
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

_COMPRESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a precise document compressor. "
        "Given a query and a document excerpt, extract ONLY the sentences "
        "directly relevant to the query. "
        "If nothing is relevant, respond with exactly: NO_OUTPUT"
    )),
    ("human", "Query: {query}\n\nDocument:\n{document}\n\nRelevant excerpt:"),
])


def compress_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """
    Contextual compression: replace each chunk's 'text' with the query-relevant excerpt.
    Preserves all other fields (id, metadata, score).
    Falls back to originals if the LLM is unavailable.
    """
    if not chunks:
        return chunks

    try:
        from backend.config.llm import get_llm_with_fallback
        llm = get_llm_with_fallback()
        chain = _COMPRESS_PROMPT | llm | StrOutputParser()

        compressed = []
        for chunk in chunks:
            try:
                excerpt = chain.invoke({"query": query, "document": chunk["text"]}).strip()
                # Keep original if LLM says nothing is relevant or returns empty
                new_text = chunk["text"] if (not excerpt or excerpt == "NO_OUTPUT") else excerpt
                compressed.append({**chunk, "text": new_text})
            except Exception as inner_e:
                logger.debug(f"[compressor] chunk compression failed ({inner_e}); keeping original.")
                compressed.append(chunk)

        logger.info(f"[compressor] Compressed {len(chunks)} chunks for query: {query[:60]}")
        return compressed

    except Exception as e:
        logger.warning(f"[compressor] Contextual compression unavailable ({e}); using raw chunks.")
        return chunks
