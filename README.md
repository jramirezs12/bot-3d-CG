# PWA-WebAR-Bot-Cartoon

Aplicación PWA móvil con un bot cartoon 3D en WebAR (MindAR + Three.js), chat por IA (OpenAI), TTS (Web Speech API) y **RAG multimodal** para consultas sobre documentos con imágenes.

Estructura:
```
pwa-webar-bot/
├─ frontend/            # React + Vite app (WebAR + UI)
│  ├─ public/
│  │  ├─ manifest.json
│  │  ├─ target.mind      # placeholder (reemplazar por tu target real)
│  │  └─ bot.glb          # placeholder (reemplazar por tu GLB real)
│  ├─ src/
│  │  ├─ main.jsx
│  │  ├─ App.jsx
│  │  ├─ ar/ArScene.jsx
│  │  ├─ components/ChatUI.jsx
│  │  └─ styles.css
│  ├─ vite.config.js
│  ├─ index.html
│  └─ package.json
├─ backend/             # Node/Express proxy para IA (OpenAI) y RAG
│  ├─ src/
│  │  └─ server.js
│  ├─ .env.example
│  └─ package.json
├─ rag/                 # Servicio Python de RAG multimodal (FastAPI)
│  ├─ app.py            # API FastAPI (health, query, upload, rebuild-index)
│  ├─ setup.py          # Descarga PDFs de ejemplo
│  ├─ preprocess.py     # PDF → imágenes + descripciones GPT-4o
│  ├─ index.py          # Embeddings + ChromaDB
│  ├─ query.py          # Pipeline RAG: recuperación + generación
│  └─ requirements.txt
├─ .gitignore
└─ README.md
```

## Requisitos
- Node.js 18+
- Python 3.10+
- `poppler-utils` instalado en el sistema (para pdf2image):
  - Ubuntu/Debian: `sudo apt-get install poppler-utils`
  - macOS: `brew install poppler`
  - Windows: descargar de la [página oficial](https://github.com/oschwartz10612/poppler-windows) y añadir al PATH
- Un archivo `target.mind` generado con MindAR (image targets).
- Un modelo `bot.glb` (cartoon) para mostrar en AR.
- Variable de entorno `OPENAI_API_KEY` en `backend/.env` y en el entorno del servicio RAG.

## Instalación

1) Backend
```
cd backend
cp .env.example .env
# Edita .env y coloca tu OPENAI_API_KEY (y RAG_SERVICE_URL si es necesario)
npm install
npm start
# Backend en http://localhost:3001
```

2) Servicio RAG (Python)
```
cd rag
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
export OPENAI_API_KEY=...
# Opcional: descarga PDFs de ejemplo de instrucciones IKEA
python setup.py
# Inicia el servicio FastAPI en http://localhost:8000
python app.py
```

3) Frontend
```
cd ../frontend
npm install
npm run dev
# Frontend en http://localhost:5173 (proxy /api a 3001)
```

Producción (build):
```
cd frontend
npm run build
npm run preview
```

## Notas de AR (MindAR)
- Reemplaza `frontend/public/target.mind` con tu target real (generado con la herramienta de MindAR).
- Reemplaza `frontend/public/bot.glb` por tu modelo 3D en formato GLB.
- Cámara: probar en móvil con HTTPS para permisos de cámara (en dev, el navegador puede requerir localhost o flags).

## PWA
- `manifest.json` está configurado; añade iconos reales en `frontend/public/icons/`.
- Para PWA completa (offline), puedes añadir un service worker (no imprescindible para esta demo).

## Configurar API base en producción
- Por defecto el frontend usa `'/api'`. En dev funciona por el proxy de Vite.
- En producción, configura `VITE_API_BASE` para apuntar a tu backend:
  - Ejemplo: `VITE_API_BASE=https://tu-backend.tld`
  - El fetch quedará `fetch(`${import.meta.env.VITE_API_BASE || ''}/api/chat`);

## Seguridad
- No subas tu `OPENAI_API_KEY` a repos públicos.
- Usa `.env` y mantén el repo privado si es necesario.

## Módulo de Predicción (TensorFlow.js + USE)

El backend incorpora un módulo de predicción basado en [Universal Sentence Encoder (USE)](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder) para análisis de intención y sentimiento.

> **Nota:** La primera carga del modelo USE puede tardar varios segundos (descarga de pesos ~25 MB). Durante ese tiempo los endpoints `/api/analyze` y `/api/similarity` responden `503`. El endpoint `/api/chat` continúa funcionando sin análisis hasta que el modelo esté listo.

### Nuevos endpoints

#### `POST /api/analyze`
Analiza la intención y sentimiento de un texto.

Request:
```json
{ "text": "hola, ¿puedes ayudarme?" }
```
Response:
```json
{
  "intent": "greeting",
  "confidence": 0.92,
  "scores": { "greeting": 0.92, "help": 0.75, ... },
  "sentiment": "neutral",
  "score": 0
}
```

#### `POST /api/similarity`
Calcula la similitud coseno entre dos textos.

Request:
```json
{ "text1": "buenos días", "text2": "hola, ¿cómo estás?" }
```
Response:
```json
{ "similarity": 0.81, "similar": true }
```

### Instalación (con nuevas dependencias)

```
cd backend
npm install
```


- Animaciones del avatar (blend shapes, gestos cuando responde).
- STT (SpeechRecognition) para dictado de voz.
- TTS avanzado (Azure/Google/ElevenLabs) desde el backend.
- Deploy: Frontend (Vercel/Netlify), Backend (Render/Railway/Azure), RAG service (Railway/Fly.io).

## RAG Multimodal

El módulo RAG permite hacer preguntas sobre documentos PDF que contienen imágenes y diagramas (como instrucciones de ensamblaje de IKEA).

### Arquitectura

```
Usuario → Frontend (tab "RAG Docs")
        → Node.js backend (/api/rag/*)
        → Servicio FastAPI Python (puerto 8000)
             ├─ pdf2image: convierte páginas PDF a PNG
             ├─ GPT-4o: describe visualmente cada página
             ├─ text-embedding-3-small: genera embeddings
             ├─ ChromaDB: almacena y recupera por similitud
             └─ GPT-4o: genera respuesta con imágenes contextuales
```

### Endpoints del servicio RAG (Python)

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Estado del servicio e índice |
| POST | `/upload` | Sube y procesa un PDF (multipart/form-data, campo `file`) |
| POST | `/query` | Consulta RAG. Body: `{"question": "..."}` |
| POST | `/rebuild-index` | Re-indexa todos los PDFs ya procesados |

### Endpoints proxy (Node.js backend)

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/rag/health` | Proxy al health del servicio Python |
| POST | `/api/rag/query` | Proxy a `/query` |
| POST | `/api/rag/upload` | Proxy a `/upload` |

### Uso desde el frontend

1. Abre la pestaña **📄 RAG Docs** en la interfaz de chat.
2. Haz clic en **📎 Subir PDF** y selecciona un documento.
3. Espera a que GPT-4o procese y describa cada página (puede tardar ~1–2 min por PDF).
4. Una vez indexado, escribe tu pregunta en el campo de texto y pulsa **Enviar**.

### Variables de entorno

```
# backend/.env
RAG_SERVICE_URL=http://localhost:8000   # URL del servicio Python

# Entorno del servicio Python (rag/)
OPENAI_API_KEY=sk-...
RAG_PORT=8000                           # Puerto del servicio (opcional)
```

## Créditos
Plantilla y ejemplo por tu asistente. ¡Adáptalo a tu branding y necesidades!