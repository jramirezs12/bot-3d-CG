"""
query.py — Pipeline de RAG multimodal: recuperación + generación con GPT-4o.
"""
import os
import base64
from openai import OpenAI
from index import get_embedding, get_or_create_collection

TOP_K = 3

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def retrieve(query: str, top_k: int = TOP_K) -> dict:
    """Recupera las páginas más relevantes para la consulta."""
    collection = get_or_create_collection()
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    return results


def generate_answer(query: str, retrieved_results: dict) -> tuple[str, list[str]]:
    """Genera una respuesta usando GPT-4o con las imágenes recuperadas."""
    if not retrieved_results["ids"][0]:
        return "No se encontraron páginas relevantes.", []

    image_content: list[dict] = []
    sources: list[str] = []

    for doc, metadata in zip(
        retrieved_results["documents"][0], retrieved_results["metadatas"][0]
    ):
        image_path = metadata["image_path"]
        if not os.path.exists(image_path):
            continue

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

    if not image_content:
        # Fallback a solo texto si no hay imágenes
        context = "\n\n".join(retrieved_results["documents"][0])
        response = _get_client().chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Responde preguntas sobre instrucciones de ensamblaje basándote en el contexto proporcionado.",
                },
                {
                    "role": "user",
                    "content": f"Pregunta: {query}\n\nContexto:\n{context}",
                },
            ],
            max_tokens=600,
        )
        return response.choices[0].message.content, sources

    response = _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Responde preguntas sobre instrucciones de ensamblaje basándote en "
                    "las páginas de instrucciones mostradas. Sé específico, menciona "
                    "números de página cuando sea útil e indica si algo no está claro "
                    "en las imágenes."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Pregunta: {query}\n\nPáginas de instrucciones relevantes:",
                    },
                    *image_content,
                    {"type": "text", "text": "Responde basándote en estas páginas."},
                ],
            },
        ],
        max_tokens=600,
    )
    return response.choices[0].message.content, sources


def answer_question(query: str) -> dict:
    """Pipeline completo: recupera y genera respuesta."""
    results = retrieve(query)
    answer, sources = generate_answer(query, results)
    pages_found = len(results["ids"][0]) if results["ids"] else 0
    return {
        "answer": answer,
        "sources": sources,
        "pages_found": pages_found,
    }


def answer_question_safe(query: str) -> dict:
    """Pipeline con manejo de errores y fallback a solo texto."""
    try:
        return answer_question(query)
    except Exception as e:
        print(f"Error en RAG pipeline: {e}")
        try:
            results = retrieve(query)
            if results["documents"][0]:
                context = results["documents"][0][0][:500]
                return {
                    "answer": f"Basado en las instrucciones: {context}...",
                    "sources": [],
                    "pages_found": 0,
                }
        except Exception:
            pass
        return {
            "answer": "Lo siento, no pude procesar esa pregunta.",
            "sources": [],
            "pages_found": 0,
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        result = answer_question_safe(q)
        print(f"\nRespuesta:\n{result['answer']}")
        if result["sources"]:
            print(f"\nFuentes: {', '.join(result['sources'])}")
    else:
        print("Uso: python query.py <pregunta>")
