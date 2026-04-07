import express from 'express'
import fetch from 'node-fetch'
import cors from 'cors'
import dotenv from 'dotenv'
import { loadModel, calculateSimilarity, classifyIntent, analyzeSentiment } from './prediction.js'

dotenv.config()
const app = express()
app.use(cors())
app.use(express.json({ limit: '50mb' }))

const PORT = process.env.PORT || 3001
const OPENAI_KEY = process.env.OPENAI_API_KEY
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini'
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1'
const RAG_SERVICE_URL = process.env.RAG_SERVICE_URL || 'http://localhost:8000'

if(!OPENAI_KEY){
  console.warn('ADVERTENCIA: No se encontró OPENAI_API_KEY en .env')
}

let modelReady = false
loadModel()
  .then(() => {
    modelReady = true
    console.log('Modelo USE cargado y listo.')
  })
  .catch(err => {
    console.error('Error al cargar el modelo USE:', err)
  })

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, service: 'pwa-webar-bot-backend', modelReady })
})

app.post('/api/analyze', async (req, res) => {
  if (!modelReady) return res.status(503).json({ error: 'Modelo no disponible aún, intenta más tarde.' })
  const { text } = req.body || {}
  if (!text) return res.status(400).json({ error: 'Falta text' })
  try {
    const intentResult = await classifyIntent(text)
    const sentimentResult = analyzeSentiment(text)
    res.json({ intent: intentResult.intent, confidence: intentResult.confidence, scores: intentResult.scores, ...sentimentResult })
  } catch (err) {
    console.error('Error en /api/analyze:', err)
    res.status(500).json({ error: 'Error al analizar el texto.' })
  }
})

app.post('/api/similarity', async (req, res) => {
  if (!modelReady) return res.status(503).json({ error: 'Modelo no disponible aún, intenta más tarde.' })
  const { text1, text2 } = req.body || {}
  if (!text1 || !text2) return res.status(400).json({ error: 'Faltan text1 y/o text2' })
  try {
    const similarity = await calculateSimilarity(text1, text2)
    res.json({ similarity, similar: similarity >= 0.7 })
  } catch (err) {
    console.error('Error en /api/similarity:', err)
    res.status(500).json({ error: 'Error al calcular similitud.' })
  }
})

app.post('/api/chat', async (req,res)=>{
  try{
    const { message } = req.body || {}
    if (!message) return res.status(400).json({ error: 'Falta message' })

    // Análisis opcional (no bloquea el chat si falla o el modelo no está listo)
    let analysis = null
    if (modelReady) {
      try {
        const intentResult = await classifyIntent(message)
        const sentimentResult = analyzeSentiment(message)
        analysis = { intent: intentResult.intent, confidence: intentResult.confidence, ...sentimentResult }
      } catch (analysisErr) {
        console.warn('Análisis falló, continuando sin él:', analysisErr.message)
      }
    }

    // Ajustar system prompt según análisis
    let systemContent = 'Eres un bot cartoon amable y breve. Responde en español.'
    if (analysis?.intent === 'farewell') {
      systemContent += ' El usuario se está despidiendo, responde de forma cálida y breve.'
    } else if (analysis?.sentiment === 'negative') {
      systemContent += ' El usuario parece frustrado, responde con empatía.'
    }

    // Cuerpo para Chat Completions
    const body = {
      model: OPENAI_MODEL,
      messages: [
        { role: 'system', content: systemContent },
        { role: 'user', content: message }
      ]
    }

    const r = await fetch(`${OPENAI_BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENAI_KEY}`
      },
      body: JSON.stringify(body)
    })

    if (!r.ok) {
      const text = await r.text()
      console.error('OpenAI error:', r.status, text)
      return res.status(502).json({ error: 'Error de upstream OpenAI' })
    }

    const json = await r.json()
    const reply = json?.choices?.[0]?.message?.content?.trim() || 'Lo siento, no pude obtener respuesta.'
    res.json({ reply, analysis })
  }catch(err){
    console.error(err)
    res.status(500).json({ error: 'error interno' })
  }
})

// ── RAG multimodal ──────────────────────────────────────────────────────────

/**
 * GET /api/rag/health
 * Comprueba el estado del servicio RAG Python.
 */
app.get('/api/rag/health', async (_req, res) => {
  try {
    const r = await fetch(`${RAG_SERVICE_URL}/health`, { signal: AbortSignal.timeout(5000) })
    const json = await r.json()
    res.json(json)
  } catch (err) {
    res.status(503).json({ ok: false, error: 'Servicio RAG no disponible.', detail: err.message })
  }
})

/**
 * POST /api/rag/query
 * Body: { question: string }
 * Responde con la respuesta generada por el pipeline RAG.
 */
app.post('/api/rag/query', async (req, res) => {
  const { question } = req.body || {}
  if (!question) return res.status(400).json({ error: 'Falta el campo "question".' })
  try {
    const r = await fetch(`${RAG_SERVICE_URL}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      signal: AbortSignal.timeout(60000)
    })
    if (!r.ok) {
      const text = await r.text()
      return res.status(r.status).json({ error: text })
    }
    const json = await r.json()
    res.json(json)
  } catch (err) {
    console.error('Error en /api/rag/query:', err)
    res.status(502).json({ error: 'Error al contactar el servicio RAG.', detail: err.message })
  }
})

/**
 * POST /api/rag/upload
 * Reenvía un PDF (multipart/form-data) al servicio RAG para indexación.
 */
app.post('/api/rag/upload', async (req, res) => {
  try {
    // Reenviar el body multipart directamente al servicio Python
    const contentType = req.headers['content-type'] || ''
    const r = await fetch(`${RAG_SERVICE_URL}/upload`, {
      method: 'POST',
      headers: { 'content-type': contentType },
      body: req,
      signal: AbortSignal.timeout(300000)
    })
    if (!r.ok) {
      const text = await r.text()
      return res.status(r.status).json({ error: text })
    }
    const json = await r.json()
    res.json(json)
  } catch (err) {
    console.error('Error en /api/rag/upload:', err)
    res.status(502).json({ error: 'Error al subir el archivo al servicio RAG.', detail: err.message })
  }
})

// ── Servidor ─────────────────────────────────────────────────────────────────

app.listen(PORT, ()=> console.log(`Backend escuchando en http://localhost:${PORT}`))
