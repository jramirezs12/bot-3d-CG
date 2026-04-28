"""
catalog_indexer.py — Indexa el catálogo de laptops de Ktronix en ChromaDB.

Convierte cada producto del catálogo JSON en un documento de texto enriquecido
y lo indexa con embeddings de OpenAI para búsqueda semántica por RAG.

Uso:
    python catalog_indexer.py

También puede importarse y llamarse desde app.py.
"""

from __future__ import annotations

import json
import os

import chromadb
from openai import OpenAI

# ---------------------------------------------------------------------------
# Rutas y constantes
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog_laptops.json")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
CATALOG_COLLECTION = "laptop_catalog"

_openai_client: OpenAI | None = None
_chroma_client: chromadb.PersistentClient | None = None


# ---------------------------------------------------------------------------
# Clientes
# ---------------------------------------------------------------------------


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def _get_chroma() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client


# ---------------------------------------------------------------------------
# Conversión producto → texto
# ---------------------------------------------------------------------------


def product_to_text(product: dict) -> str:
    """
    Convierte un producto del catálogo JSON en un texto enriquecido
    optimizado para búsqueda semántica.
    """
    usos = ", ".join(product.get("usos_recomendados", []))
    puertos = ", ".join(product.get("puertos", []))

    return (
        f"Producto: {product['nombre']}\n"
        f"Marca: {product['marca']}\n"
        f"Categoría: {product['categoria']}\n"
        f"Precio: ${product['precio_cop']:,} COP\n"
        f"Stock disponible: {product['stock']} unidades\n"
        f"Procesador: {product['procesador']} ({product['nucleos']} núcleos)\n"
        f"RAM: {product['ram_gb']}GB\n"
        f"Almacenamiento: {product['almacenamiento']}\n"
        f"Pantalla: {product['pantalla']}\n"
        f"Tarjeta gráfica: {product['tarjeta_grafica']}\n"
        f"Batería: {product['bateria_horas']} horas\n"
        f"Peso: {product['peso_kg']}kg\n"
        f"Sistema operativo: {product['sistema_operativo']}\n"
        f"Puertos: {puertos}\n"
        f"WiFi: {product['wifi']}\n"
        f"Garantía: {product['garantia_meses']} meses\n"
        f"Usos recomendados: {usos}\n"
        f"Descripción: {product['descripcion']}"
    )


def get_embedding(text: str) -> list[float]:
    """Obtiene embedding con text-embedding-3-small."""
    response = _get_openai().embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Indexación
# ---------------------------------------------------------------------------


def build_catalog_index(catalog_path: str = CATALOG_PATH) -> int:
    """
    Carga el catálogo JSON e indexa todos los productos en ChromaDB.

    Returns:
        Número de productos indexados.
    """
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"No se encontró el catálogo en {catalog_path}")

    with open(catalog_path, "r", encoding="utf-8") as f:
        products = json.load(f)

    print(f"Indexando {len(products)} productos del catálogo Ktronix...")

    chroma = _get_chroma()

    # Recrear colección limpia
    try:
        chroma.delete_collection(name=CATALOG_COLLECTION)
    except Exception:
        pass

    collection = chroma.create_collection(
        name=CATALOG_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    for i, product in enumerate(products):
        text = product_to_text(product)
        embedding = get_embedding(text)

        collection.add(
            ids=[product["id"]],
            embeddings=[embedding],
            documents=[text],
            metadatas=[
                {
                    "id": product["id"],
                    "nombre": product["nombre"],
                    "marca": product["marca"],
                    "categoria": product["categoria"],
                    "precio_cop": product["precio_cop"],
                    "stock": product["stock"],
                }
            ],
        )
        print(f"  [{i+1}/{len(products)}] {product['nombre']}")

    print(f"\n✓ Catálogo indexado: {collection.count()} productos en ChromaDB")
    return collection.count()


# ---------------------------------------------------------------------------
# Consulta del catálogo
# ---------------------------------------------------------------------------


def search_catalog(
    query: str,
    top_k: int = 3,
    intent: str = "recomendacion",
) -> list[dict]:
    """
    Busca productos relevantes en el catálogo según la consulta.

    Returns:
        Lista de productos con texto y metadatos.
    """
    chroma = _get_chroma()

    try:
        collection = chroma.get_collection(name=CATALOG_COLLECTION)
    except Exception:
        return []

    if collection.count() == 0:
        return []

    query_embedding = get_embedding(query)

    # Para comparaciones, traer más resultados
    k = min(top_k * 2 if intent == "comparacion" else top_k, collection.count())

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    output: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append(
            {
                "text": doc,
                "metadata": meta,
                "similarity": round(1 - dist, 4),
            }
        )

    return output


def catalog_is_ready() -> bool:
    """Verifica si el catálogo ya está indexado."""
    try:
        chroma = _get_chroma()
        col = chroma.get_collection(name=CATALOG_COLLECTION)
        return col.count() > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    count = build_catalog_index()
    print(f"\nTotal productos indexados: {count}")

    # Prueba rápida
    print("\n── Prueba de búsqueda ──────────────────────────────────────")
    queries = [
        "laptop para gaming con buena tarjeta gráfica",
        "portátil económico para estudiantes",
        "laptop premium para diseño gráfico",
    ]
    for q in queries:
        results = search_catalog(q, top_k=2)
        print(f"\nConsulta: '{q}'")
        for r in results:
            print(f"  → {r['metadata']['nombre']} (similitud: {r['similarity']})")
