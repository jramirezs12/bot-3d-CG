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

if (!OPENAI_KEY) {
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

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, service: 'pwa-webar-bot-backend', modelReady })
})

// ---------------------------------------------------------------------------
// Análisis NLP local (TensorFlow.js USE)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Chat principal — usa clasificador de intenciones del servicio Python
// ---------------------------------------------------------------------------

/**
 * Llama al clasificador de intenciones del servicio RAG Python.
 * Retorna null si el servicio no está disponible (no bloquea el chat).
 */
async function classifyIntentPython(message) {
  try {
    const r = await fetch(`${RAG_SERVICE_URL}/classify-intent`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: message }),
      signal: AbortSignal.timeout(3000),
    })
    if (!r.ok) return null
    return await r.json()
  } catch {
    return null
  }
}

app.post('/api/chat', async (req, res) => {
  try {
    const { message } = req.body || {}
    if (!message) return res.status(400).json({ error: 'Falta message' })

    // 1. Clasificar intención — primero intenta con el clasificador Python
    let intentData = await classifyIntentPython(message)

    // Fallback al clasificador USE local si Python no está disponible
    let localAnalysis = null
    if (!intentData && modelReady) {
      try {
        const intentResult = await classifyIntent(message)
        const sentimentResult = analyzeSentiment(message)
        localAnalysis = {
          intent: intentResult.intent,
          confidence: intentResult.confidence,
          sentiment: sentimentResult.sentiment,
        }
        intentData = localAnalysis
      } catch (err) {
        console.warn('Análisis local falló:', err.message)
      }
    }

    const currentIntent = intentData?.intent || 'otro'
    const needsRag = intentData?.needs_rag ?? false

    // 2. Si la intención necesita RAG, delegar al pipeline RAG
    if (needsRag) {
      try {
        const ragRes = await fetch(`${RAG_SERVICE_URL}/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: message, intent: currentIntent }),
          signal: AbortSignal.timeout(30000),
        })
        if (ragRes.ok) {
          const ragJson = await ragRes.json()
          return res.json({
            reply: ragJson.answer,
            sources: ragJson.sources || [],
            analysis: {
              intent: ragJson.intent || currentIntent,
              confidence: intentData?.confidence || 0,
              source_type: ragJson.source_type || 'rag',
              needs_rag: true,
            },
          })
        }
      } catch (ragErr) {
        console.warn('RAG no disponible, usando LLM directo:', ragErr.message)
      }
    }

    // 3. Respuesta directa con LLM (intenciones que no necesitan catálogo)
    const systemPrompt = intentData?.system_prompt || buildSystemPrompt(currentIntent, localAnalysis)

    const body = {
      model: OPENAI_MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: message },
      ],
    }

    const r = await fetch(`${OPENAI_BASE_URL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENAI_KEY}`,
      },
      body: JSON.stringify(body),
    })

    if (!r.ok) {
      const text = await r.text()
      console.error('OpenAI error:', r.status, text)
      return res.status(502).json({ error: 'Error de upstream OpenAI' })
    }

    const json = await r.json()
    const reply = json?.choices?.[0]?.message?.content?.trim() || 'Lo siento, no pude obtener respuesta.'

    res.json({
      reply,
      sources: [],
      analysis: {
        intent: currentIntent,
        confidence: intentData?.confidence || 0,
        source_type: 'llm',
        needs_rag: false,
      },
    })
  } catch (err) {
    console.error(err)
    res.status(500).json({ error: 'Error interno del servidor.' })
  }
})

/**
 * Construye un system prompt básico según la intención (fallback local).
 */
function buildSystemPrompt(intent, analysis) {
  const base = 'Eres el asistente virtual de Ktronix, tienda de tecnología colombiana. Responde en español.'
  if (intent === 'despedida') return base + ' El usuario se despide. Responde cálidamente.'
  if (intent === 'saludo') return base + ' El usuario saluda. Preséntate brevemente.'
  if (analysis?.sentiment === 'negative') return base + ' El usuario parece frustrado. Responde con empatía.'
  return base
}

// ---------------------------------------------------------------------------
// RAG multimodal (proxy al servicio Python)
// ---------------------------------------------------------------------------

app.get('/api/rag/health', async (_req, res) => {
  try {
    const r = await fetch(`${RAG_SERVICE_URL}/health`, { signal: AbortSignal.timeout(5000) })
    const json = await r.json()
    res.json(json)
  } catch (err) {
    res.status(503).json({ ok: false, error: 'Servicio RAG no disponible.', detail: err.message })
  }
})

app.post('/api/rag/query', async (req, res) => {
  const { question, intent } = req.body || {}
  if (!question) return res.status(400).json({ error: 'Falta el campo "question".' })
  try {
    const r = await fetch(`${RAG_SERVICE_URL}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, intent }),
      signal: AbortSignal.timeout(60000),
    })
    if (!r.ok) {
      const text = await r.text()
      return res.status(r.status).json({ error: text })
    }
    res.json(await r.json())
  } catch (err) {
    console.error('Error en /api/rag/query:', err)
    res.status(502).json({ error: 'Error al contactar el servicio RAG.', detail: err.message })
  }
})

app.post('/api/rag/upload', async (req, res) => {
  try {
    const contentType = req.headers['content-type'] || ''
    const chunks = []
    for await (const chunk of req) chunks.push(chunk)
    const buffer = Buffer.concat(chunks)

    const r = await fetch(`${RAG_SERVICE_URL}/upload`, {
      method: 'POST',
      headers: { 'content-type': contentType },
      body: buffer,
      signal: AbortSignal.timeout(300000),
    })
    if (!r.ok) {
      const text = await r.text()
      return res.status(r.status).json({ error: text })
    }
    res.json(await r.json())
  } catch (err) {
    console.error('Error en /api/rag/upload:', err)
    res.status(502).json({ error: 'Error al subir el archivo al servicio RAG.', detail: err.message })
  }
})

/** Indexa el catálogo de laptops */
app.post('/api/rag/index-catalog', async (_req, res) => {
  try {
    const r = await fetch(`${RAG_SERVICE_URL}/index-catalog`, {
      method: 'POST',
      signal: AbortSignal.timeout(60000),
    })
    if (!r.ok) {
      const text = await r.text()
      return res.status(r.status).json({ error: text })
    }
    res.json(await r.json())
  } catch (err) {
    res.status(502).json({ error: 'Error al indexar catálogo.', detail: err.message })
  }
})

// ---------------------------------------------------------------------------
// Servidor
// ---------------------------------------------------------------------------

app.listen(PORT, () => console.log(`Backend escuchando en http://localhost:${PORT}`))
