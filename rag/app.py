"""
app.py — Servicio FastAPI que expone el pipeline de RAG multimodal.

Endpoints:
  GET  /health           — estado del servicio e índice
  POST /upload           — sube un PDF, lo procesa e indexa
  POST /query            — hace una pregunta al RAG pipeline
  POST /rebuild-index    — re-indexa todos los PDFs ya procesados
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ajustar sys.path para imports relativos dentro del paquete rag/
import sys
sys.path.insert(0, os.path.dirname(__file__))

import setup as setup_module
import preprocess as preprocess_module
import index as index_module
import query as query_module

# ---------------------------------------------------------------------------
# Estado global del servicio
# ---------------------------------------------------------------------------

_index_ready = False


def _try_load_index() -> bool:
    """Intenta cargar el índice existente. Devuelve True si tiene documentos."""
    global _index_ready
    try:
        collection = index_module.get_or_create_collection()
        _index_ready = collection.count() > 0
    except Exception:
        _index_ready = False
    return _index_ready


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Crear directorios necesarios
    os.makedirs(index_module.CHROMA_DIR, exist_ok=True)
    os.makedirs(preprocess_module.IMAGES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(preprocess_module.CACHE_FILE), exist_ok=True)
    _try_load_index()
    print(f"RAG service iniciado. Índice listo: {_index_ready}")
    yield


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multimodal RAG Service",
    description="Servicio de RAG multimodal para consultas sobre documentos con imágenes.",
    version="1.0.0",
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


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    pages_found: int


class StatusResponse(BaseModel):
    ok: bool
    index_ready: bool
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
        document_count=doc_count,
        service="multimodal-rag",
    )


@app.post("/query", response_model=QueryResponse)
def query_rag(body: QueryRequest):
    if not _index_ready:
        raise HTTPException(
            status_code=503,
            detail="El índice RAG no está listo. Sube un PDF primero.",
        )
    result = query_module.answer_question_safe(body.question)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        pages_found=result["pages_found"],
    )


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Recibe un PDF, lo procesa con GPT-4o y lo añade al índice."""
    global _index_ready

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF.")

    # Guardar el PDF en el directorio de datos
    safe_name = os.path.basename(file.filename)
    dest_path = os.path.join(preprocess_module.DATA_DIR, safe_name)

    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Procesar el PDF recién subido
    try:
        pages = preprocess_module.preprocess_pdfs(pdf_dir=preprocess_module.DATA_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el PDF: {e}")

    if not pages:
        raise HTTPException(status_code=422, detail="No se pudieron extraer páginas del PDF.")

    # Re-construir el índice
    try:
        index_module.build_index(pages)
        _index_ready = True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al indexar: {e}")

    return {
        "ok": True,
        "filename": safe_name,
        "pages_indexed": len(pages),
    }


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
