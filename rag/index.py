"""
index.py — Genera embeddings y los almacena en ChromaDB.
"""
import os
import json
from openai import OpenAI
import chromadb

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PAGES_FILE = os.path.join(DATA_DIR, "cache", "all_pages.json")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
COLLECTION_NAME = "rag_documents"

_client: OpenAI | None = None
_chroma: chromadb.PersistentClient | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma
    if _chroma is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma


def get_embedding(text: str) -> list[float]:
    """Obtiene el embedding de un texto con text-embedding-3-small."""
    response = _get_client().embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def build_index(pages: list[dict] | None = None) -> chromadb.Collection:
    """
    Genera embeddings para cada página y los indexa en ChromaDB.
    Si pages es None, los carga desde PAGES_FILE.
    """
    if pages is None:
        if not os.path.exists(PAGES_FILE):
            raise FileNotFoundError(
                f"No se encontró {PAGES_FILE}. "
                "Ejecuta preprocess.py primero."
            )
        with open(PAGES_FILE, "r") as f:
            pages = json.load(f)

    print(f"Indexando {len(pages)} páginas...")

    chroma = get_chroma_client()

    try:
        chroma.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for i, page in enumerate(pages):
        text_to_embed = (
            f"Fuente: {page['source_pdf']}, Página {page['page_number']}\n\n"
            f"{page['description']}"
        )
        embedding = get_embedding(text_to_embed)

        collection.add(
            ids=[f"page_{i}"],
            embeddings=[embedding],
            documents=[page["description"]],
            metadatas=[
                {
                    "source_pdf": page["source_pdf"],
                    "page_number": page["page_number"],
                    "image_path": page["image_path"],
                }
            ],
        )

        if (i + 1) % 5 == 0 or i == len(pages) - 1:
            print(f"  Indexadas {i + 1}/{len(pages)} páginas")

    print(f"\n✓ Índice creado con {collection.count()} documentos")
    return collection


def get_or_create_collection() -> chromadb.Collection:
    """Devuelve la colección existente o lanza un error si no existe."""
    chroma = get_chroma_client()
    return chroma.get_collection(name=COLLECTION_NAME)


if __name__ == "__main__":
    build_index()
