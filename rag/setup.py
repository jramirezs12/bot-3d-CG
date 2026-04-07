"""
setup.py — Crea directorios y descarga PDFs de ejemplo (instrucciones IKEA).
"""
import os
import requests

SAMPLE_PDFS = {
    "malm_desk.pdf": "https://www.ikea.com/us/en/assembly_instructions/malm-desk-white__AA-516949-7-2.pdf",
    "malm_bed.pdf": "https://www.ikea.com/us/en/assembly_instructions/malm-bed-frame-low__AA-75286-15_pub.pdf",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CACHE_DIR = os.path.join(DATA_DIR, "cache")


def setup(extra_pdfs: dict | None = None):
    """Crea directorios y descarga PDFs de ejemplo."""
    for d in [DATA_DIR, IMAGES_DIR, CACHE_DIR]:
        os.makedirs(d, exist_ok=True)
    print("✓ Directorios creados")

    pdfs = {**SAMPLE_PDFS, **(extra_pdfs or {})}
    for filename, url in pdfs.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Descargando {filename}...")
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"  ✓ Guardado ({len(response.content) / 1024:.1f} KB)")
            except Exception as e:
                print(f"  ✗ Error al descargar {filename}: {e}")
        else:
            print(f"✓ {filename} ya existe")

    return DATA_DIR


if __name__ == "__main__":
    setup()
