"""
preprocess.py — Convierte PDFs a imágenes y genera descripciones con GPT-4o.
"""
import os
import json
import base64
from pdf2image import convert_from_path
from openai import OpenAI

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CACHE_FILE = os.path.join(DATA_DIR, "cache", "descriptions.json")
PAGES_FILE = os.path.join(DATA_DIR, "cache", "all_pages.json")

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def convert_pdf_to_images(pdf_path: str) -> list[str]:
    """Convierte un PDF en imágenes PNG por página."""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    images = convert_from_path(pdf_path, dpi=150)
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(IMAGES_DIR, f"{pdf_name}_page_{i + 1:03d}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths


def describe_page(image_path: str) -> str:
    """Usa GPT-4o para describir una página de instrucciones."""
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    response = _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Describe esta página de instrucciones de ensamblaje en detalle. "
                    "Incluye: números de paso, piezas mostradas, herramientas necesarias, "
                    "acciones demostradas, cantidades (como '2x'), advertencias y números "
                    "de pieza si son visibles. Esta descripción ayudará a las personas a "
                    "encontrar esta página cuando tengan preguntas."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                    {"type": "text", "text": "Describe esta página de instrucciones."},
                ],
            },
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


def preprocess_pdfs(pdf_dir: str | None = None) -> list[dict]:
    """
    Procesa todos los PDFs en pdf_dir, genera descripciones y devuelve
    la lista de páginas con sus metadatos.
    """
    if pdf_dir is None:
        pdf_dir = DATA_DIR

    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

    cache: dict[str, str] = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)

    all_pages: list[dict] = []

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No se encontraron PDFs en", pdf_dir)
        return all_pages

    for pdf_file in sorted(pdf_files):
        print(f"\nProcesando {pdf_file}...")
        pdf_path = os.path.join(pdf_dir, pdf_file)

        print("  Convirtiendo a imágenes...")
        image_paths = convert_pdf_to_images(pdf_path)
        print(f"  ✓ {len(image_paths)} páginas")

        print("  Generando descripciones con GPT-4o...")
        for i, image_path in enumerate(image_paths):
            if image_path in cache:
                description = cache[image_path]
                status = "caché"
            else:
                description = describe_page(image_path)
                cache[image_path] = description
                with open(CACHE_FILE, "w") as f:
                    json.dump(cache, f, indent=2)
                status = "generada"

            all_pages.append(
                {
                    "source_pdf": pdf_file,
                    "page_number": i + 1,
                    "image_path": image_path,
                    "description": description,
                }
            )
            print(f"    Página {i + 1}/{len(image_paths)} [{status}]")

    with open(PAGES_FILE, "w") as f:
        json.dump(all_pages, f, indent=2)

    print(f"\n✓ {len(all_pages)} páginas procesadas en total")
    return all_pages


if __name__ == "__main__":
    preprocess_pdfs()
