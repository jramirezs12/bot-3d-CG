import express from 'express'
import fetch from 'node-fetch'
import cors from 'cors'
import dotenv from 'dotenv'

dotenv.config()
const app = express()
app.use(cors())
app.use(express.json())

const PORT = process.env.PORT || 3001
const OPENAI_KEY = process.env.OPENAI_API_KEY
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini'
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1'

if(!OPENAI_KEY){
  console.warn('ADVERTENCIA: No se encontró OPENAI_API_KEY en .env')
}

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, service: 'pwa-webar-bot-backend' })
})

app.post('/api/chat', async (req,res)=>{
  try{
    const { message } = req.body || {}
    if (!message) return res.status(400).json({ error: 'Falta message' })

    // Cuerpo para Chat Completions
    const body = {
      model: OPENAI_MODEL,
      messages: [
        { role: 'system', content: 'Eres un bot cartoon amable y breve. Responde en español.' },
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
    res.json({ reply })
  }catch(err){
    console.error(err)
    res.status(500).json({ error: 'error interno' })
  }
})

app.listen(PORT, ()=> console.log(`Backend escuchando en http://localhost:${PORT}`))
