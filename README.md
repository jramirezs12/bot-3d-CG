# PWA-WebAR-Bot-Cartoon

Aplicación PWA móvil con un bot cartoon 3D en WebAR (MindAR + Three.js), chat por IA (OpenAI) y TTS (Web Speech API).

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
├─ backend/             # Node/Express proxy para IA (OpenAI)
│  ├─ src/
│  │  └─ server.js
│  ├─ .env.example
│  └─ package.json
├─ .gitignore
└─ README.md
```

## Requisitos
- Node.js 18+
- Un archivo `target.mind` generado con MindAR (image targets).
- Un modelo `bot.glb` (cartoon) para mostrar en AR.
- Variable de entorno `OPENAI_API_KEY` en `backend/.env`.

## Instalación

1) Backend
```
cd backend
cp .env.example .env
# Edita .env y coloca tu OPENAI_API_KEY
npm install
npm start
# Backend en http://localhost:3001
```

2) Frontend
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

## Roadmap / Mejoras
- Animaciones del avatar (blend shapes, gestos cuando responde).
- STT (SpeechRecognition) para dictado de voz.
- TTS avanzado (Azure/Google/ElevenLabs) desde el backend.
- Deploy: Frontend (Vercel/Netlify), Backend (Render/Railway/Azure).

## Créditos
Plantilla y ejemplo por tu asistente. ¡Adáptalo a tu branding y necesidades!