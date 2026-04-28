"""
app.py — Servicio FastAPI: RAG multimodal + Clasificador de Intenciones.

Endpoints:
  GET  /health              — estado del servicio
  POST /classify-intent     — clasifica la intención de un texto
  POST /query               — pipeline RAG completo (intención → búsqueda → respuesta)
  POST /upload              — sube PDF, lo procesa e indexa
  POST /index-catalog       — indexa el catálogo de laptops en ChromaDB
  POST /rebuild-index       — re-indexa PDFs desde caché
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

import setup as setup_module
import preprocess as preprocess_module
import index as index_module
import query as query_module
import catalog_indexer
from intent_classifier import get_classifier, classify_intent

# ---------------------------------------------------------------------------
# Estado global
# ---------------------------------------------------------------------------

_index_ready = False
_catalog_ready = False
_classifier_ready = False


def _try_load_index() -> bool:
    global _index_ready
    try:
        collection = index_module.get_or_create_collection()
        _index_ready = collection.count() > 0
    except Exception:
        _index_ready = False
    return _index_ready


def _try_load_catalog() -> bool:
    global _catalog_ready
    _catalog_ready = catalog_indexer.catalog_is_ready()
    return _catalog_ready


def _try_load_classifier() -> bool:
    global _classifier_ready
    try:
        clf = get_classifier()
        _classifier_ready = clf.pipeline is not None
    except Exception:
        _classifier_ready = False
    return _classifier_ready


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(index_module.CHROMA_DIR, exist_ok=True)
    os.makedirs(preprocess_module.IMAGES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(preprocess_module.CACHE_FILE), exist_ok=True)

    _try_load_index()
    _try_load_catalog()
    _try_load_classifier()

    print(f"RAG service iniciado.")
    print(f"  PDFs indexados   : {_index_ready}")
    print(f"  Catálogo listo   : {_catalog_ready}")
    print(f"  Clasificador     : {_classifier_ready}")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ktronix RAG + Intent Service",
    description=(
        "Pipeline conversacional para e-commerce de laptops: "
        "clasificación de intenciones + RAG sobre catálogo de productos."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str
    intent: str | None = None  # Opcional: si ya se clasificó antes


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    pages_found: int
    intent: str
    source_type: str


class IntentRequest(BaseModel):
    text: str


class IntentResponse(BaseModel):
    intent: str
    confidence: float
    scores: dict[str, float]
    needs_rag: bool
    system_prompt: str


class StatusResponse(BaseModel):
    ok: bool
    index_ready: bool
    catalog_ready: bool
    classifier_ready: bool
    document_count: int
    service: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=StatusResponse)
def health():
    doc_count = 0
    try:
        col = index_module.get_or_create_collection()
        doc_count = col.count()
    except Exception:
        pass
    return StatusResponse(
        ok=True,
        index_ready=_index_ready,
        catalog_ready=_catalog_ready,
        classifier_ready=_classifier_ready,
        document_count=doc_count,
        service="ktronix-rag-intent",
    )


@app.post("/classify-intent", response_model=IntentResponse)
def classify_intent_endpoint(body: IntentRequest):
    """
    Clasifica la intención de un texto.

    Intenciones: recomendacion, especificaciones, precio, comparacion,
                 disponibilidad, soporte_tecnico, saludo, despedida, otro
    """
    if not _classifier_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "El clasificador de intenciones no está listo. "
                "Ejecuta train_intent.py primero."
            ),
        )
    result = classify_intent(body.text)
    return IntentResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query_rag(body: QueryRequest):
    """
    Pipeline completo:
      1. Clasificar intención (si no se provee)
      2. Buscar en catálogo o PDFs según intención
      3. Generar respuesta contextualizada con GPT-4o
    """
    result = query_module.answer_question_safe(body.question, body.intent)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        pages_found=result["pages_found"],
        intent=result.get("intent", "otro"),
        source_type=result.get("source_type", "unknown"),
    )


@app.post("/index-catalog")
def index_catalog():
    """Indexa o re-indexa el catálogo de laptops en ChromaDB."""
    global _catalog_ready
    try:
        count = catalog_indexer.build_catalog_index()
        _catalog_ready = True
        return {"ok": True, "products_indexed": count}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al indexar catálogo: {e}")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Recibe un PDF, lo procesa con GPT-4o y lo añade al índice."""
    global _index_ready

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF.")

    safe_name = os.path.basename(file.filename)
    dest_path = os.path.join(preprocess_module.DATA_DIR, safe_name)

    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        pages = preprocess_module.preprocess_pdfs(pdf_dir=preprocess_module.DATA_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el PDF: {e}")

    if not pages:
        raise HTTPException(status_code=422, detail="No se pudieron extraer páginas del PDF.")

    try:
        index_module.build_index(pages)
        _index_ready = True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al indexar: {e}")

    return {"ok": True, "filename": safe_name, "pages_indexed": len(pages)}


@app.post("/rebuild-index")
def rebuild_index():
    """Re-indexa todos los PDFs ya procesados (usa caché de descripciones)."""
    global _index_ready
    try:
        pages = preprocess_module.preprocess_pdfs()
        if not pages:
            raise HTTPException(status_code=422, detail="No hay páginas para indexar.")
        index_module.build_index(pages)
        _index_ready = True
        return {"ok": True, "pages_indexed": len(pages)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("RAG_PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
