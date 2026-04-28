"""
query.py — Pipeline RAG con soporte de intención.

Flujo:
  1. Clasificar intención del usuario
  2. Si la intención necesita catálogo → buscar en catalog_indexer
  3. Si hay PDFs indexados → buscar en rag_documents (PDFs)
  4. Construir prompt según intención
  5. Generar respuesta con GPT-4o
"""

from __future__ import annotations

import os
import base64

from openai import OpenAI
from index import get_embedding, get_or_create_collection
import catalog_indexer
from intent_classifier import (
    classify_intent,
    INTENTS_NEED_RAG,
    INTENT_SYSTEM_PROMPTS,
)

TOP_K = 3

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ---------------------------------------------------------------------------
# Recuperación desde PDFs (colección original)
# ---------------------------------------------------------------------------


def retrieve_from_pdfs(query: str, top_k: int = TOP_K) -> dict:
    """Recupera páginas relevantes desde los PDFs indexados."""
    collection = get_or_create_collection()
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    return results


# ---------------------------------------------------------------------------
# Generación con catálogo de productos
# ---------------------------------------------------------------------------


def generate_answer_catalog(
    query: str,
    catalog_results: list[dict],
    intent: str = "otro",
) -> tuple[str, list[str]]:
    """
    Genera respuesta usando contexto del catálogo de laptops.
    """
    if not catalog_results:
        return "No encontré productos que coincidan con tu consulta.", []

    system_prompt = INTENT_SYSTEM_PROMPTS.get(intent, INTENT_SYSTEM_PROMPTS["otro"])
    context = "\n\n---\n\n".join(r["text"] for r in catalog_results)
    sources = [r["metadata"]["nombre"] for r in catalog_results]

    response = _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Consulta del cliente: {query}\n\n"
                    f"Productos disponibles en catálogo Ktronix:\n\n{context}"
                ),
            },
        ],
        max_tokens=600,
    )
    return response.choices[0].message.content, sources


# ---------------------------------------------------------------------------
# Generación con PDFs (instrucciones, manuales)
# ---------------------------------------------------------------------------


def generate_answer_pdfs(
    query: str,
    retrieved_results: dict,
    intent: str = "otro",
) -> tuple[str, list[str]]:
    """Genera respuesta usando las páginas de PDF recuperadas."""
    if not retrieved_results["ids"][0]:
        return "No se encontraron páginas relevantes.", []

    image_content: list[dict] = []
    sources: list[str] = []

    for doc, metadata in zip(
        retrieved_results["documents"][0], retrieved_results["metadatas"][0]
    ):
        image_path = metadata["image_path"]
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
            image_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )
            image_content.append(
                {
                    "type": "text",
                    "text": f"[Página {metadata['page_number']} de {metadata['source_pdf']}]",
                }
            )
        sources.append(f"{metadata['source_pdf']}, Página {metadata['page_number']}")

    system_prompt = INTENT_SYSTEM_PROMPTS.get(intent, INTENT_SYSTEM_PROMPTS["otro"])

    if not image_content:
        context = "\n\n".join(retrieved_results["documents"][0])
        response = _get_client().chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Pregunta: {query}\n\nContexto:\n{context}"},
            ],
            max_tokens=600,
        )
        return response.choices[0].message.content, sources

    response = _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Pregunta: {query}\n\nPáginas relevantes:"},
                    *image_content,
                    {"type": "text", "text": "Responde basándote en estas páginas."},
                ],
            },
        ],
        max_tokens=600,
    )
    return response.choices[0].message.content, sources


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def answer_question(query: str, intent: str | None = None) -> dict:
    """
    Pipeline completo:
      1. Clasificar intención (si no se provee)
      2. Buscar en catálogo o PDFs según intención
      3. Generar respuesta contextualizada
    """
    # Clasificar intención si no se proveyó
    intent_result = classify_intent(query) if intent is None else {"intent": intent, "needs_rag": intent in INTENTS_NEED_RAG}
    current_intent = intent_result["intent"]
    needs_rag = intent_result.get("needs_rag", False)

    # Intentar primero con catálogo de productos
    if needs_rag and catalog_indexer.catalog_is_ready():
        catalog_results = catalog_indexer.search_catalog(query, top_k=TOP_K, intent=current_intent)
        if catalog_results:
            answer, sources = generate_answer_catalog(query, catalog_results, current_intent)
            return {
                "answer": answer,
                "sources": sources,
                "pages_found": len(catalog_results),
                "intent": current_intent,
                "source_type": "catalog",
            }

    # Fallback a PDFs si están indexados
    try:
        pdf_results = retrieve_from_pdfs(query)
        if pdf_results["ids"][0]:
            answer, sources = generate_answer_pdfs(query, pdf_results, current_intent)
            return {
                "answer": answer,
                "sources": sources,
                "pages_found": len(pdf_results["ids"][0]),
                "intent": current_intent,
                "source_type": "pdf",
            }
    except Exception:
        pass

    # Sin RAG: responder solo con LLM guiado por intención
    system_prompt = INTENT_SYSTEM_PROMPTS.get(current_intent, INTENT_SYSTEM_PROMPTS["otro"])
    response = _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        max_tokens=400,
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": [],
        "pages_found": 0,
        "intent": current_intent,
        "source_type": "llm_only",
    }


def answer_question_safe(query: str, intent: str | None = None) -> dict:
    """Pipeline con manejo de errores."""
    try:
        return answer_question(query, intent)
    except Exception as e:
        print(f"Error en RAG pipeline: {e}")
        return {
            "answer": "Lo siento, no pude procesar esa pregunta en este momento.",
            "sources": [],
            "pages_found": 0,
            "intent": "otro",
            "source_type": "error",
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        result = answer_question_safe(q)
        print(f"\nIntención: {result['intent']}")
        print(f"Respuesta:\n{result['answer']}")
        if result["sources"]:
            print(f"\nFuentes: {', '.join(result['sources'])}")
    else:
        print("Uso: python query.py <pregunta>")
