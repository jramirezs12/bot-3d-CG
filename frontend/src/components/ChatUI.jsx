import React, { useState, useRef } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || ''

export default function ChatUI(){
  const [messages, setMessages] = useState([])
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState('chat') // 'chat' | 'rag'
  const [ragReady, setRagReady] = useState(false)
  const [uploading, setUploading] = useState(false)
  const fileRef = useRef(null)

  // ── Chat normal ────────────────────────────────────────────────────────────

  async function send(){
    if(!text || loading) return
    const userMsg = { from: 'user', text }
    setMessages(m => [...m, userMsg])
    setText('')
    setLoading(true)

    try {
      if (mode === 'rag') {
        await sendRag(userMsg.text)
      } else {
        await sendChat(userMsg.text)
      }
    } finally {
      setLoading(false)
    }
  }

  async function sendChat(message){
    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ message })
      })
      const data = await res.json()
      const reply = data?.reply || 'Lo siento, no pude obtener respuesta.'
      const analysis = data?.analysis || null
      const debugSuffix = analysis
        ? ` [intent: ${analysis.intent}, sentiment: ${analysis.sentiment}]`
        : ''
      const botMsg = { from:'bot', text: reply + debugSuffix }
      setMessages(m => [...m, botMsg])

      if ('speechSynthesis' in window) {
        const utter = new SpeechSynthesisUtterance(reply)
        utter.lang = 'es-ES'
        window.speechSynthesis.cancel()
        window.speechSynthesis.speak(utter)
      }

      if (window.__AR_BOT_MODEL) {
        window.__AR_BOT_MODEL.userData.lastReply = reply
      }
    } catch (e) {
      console.error(e)
      setMessages(m => [...m, { from: 'bot', text: 'Error al contactar el servidor.' }])
    }
  }

  // ── RAG ────────────────────────────────────────────────────────────────────

  async function sendRag(question){
    try {
      const res = await fetch(`${API_BASE}/api/rag/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      })
      const data = await res.json()
      if (!res.ok) {
        setMessages(m => [...m, { from: 'bot', text: data?.error || 'Error en el servicio RAG.' }])
        return
      }
      const sourceLine = data.sources?.length
        ? `\n\n📄 Fuentes: ${data.sources.join(' · ')}`
        : ''
      setMessages(m => [...m, { from: 'bot', text: (data.answer || 'Sin respuesta.') + sourceLine, rag: true }])
    } catch (e) {
      console.error(e)
      setMessages(m => [...m, { from: 'bot', text: 'Error al consultar el servicio RAG.' }])
    }
  }

  async function uploadPdf(e){
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    setMessages(m => [...m, { from: 'bot', text: `⏳ Procesando "${file.name}"…` }])

    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API_BASE}/api/rag/upload`, {
        method: 'POST',
        body: form
      })
      const data = await res.json()
      if (!res.ok) {
        setMessages(m => [...m, { from: 'bot', text: `❌ Error: ${data?.error || 'No se pudo procesar el PDF.'}` }])
      } else {
        setRagReady(true)
        setMessages(m => [...m, { from: 'bot', text: `✅ "${file.name}" indexado (${data.pages_indexed} páginas). ¡Ahora puedes hacer preguntas!` }])
      }
    } catch (e) {
      console.error(e)
      setMessages(m => [...m, { from: 'bot', text: 'Error al subir el archivo.' }])
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  function onKeyDown(e){
    if (e.key === 'Enter') send()
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="chat-ui">
      <div className="chat-tabs">
        <button
          className={`tab-btn${mode === 'chat' ? ' active' : ''}`}
          onClick={() => setMode('chat')}
        >💬 Chat</button>
        <button
          className={`tab-btn${mode === 'rag' ? ' active' : ''}`}
          onClick={() => setMode('rag')}
        >📄 RAG Docs</button>
      </div>

      {mode === 'rag' && (
        <div className="rag-toolbar">
          <label className="upload-btn" title="Subir PDF para indexar">
            📎 Subir PDF
            <input
              ref={fileRef}
              type="file"
              accept="application/pdf"
              style={{ display: 'none' }}
              onChange={uploadPdf}
              disabled={uploading}
            />
          </label>
          <span className={`rag-status ${ragReady ? 'ready' : 'not-ready'}`}>
            {ragReady ? '● Índice listo' : '○ Sin documentos'}
          </span>
        </div>
      )}

      <div className="messages">
        {messages.map((m,i)=> (
          <div key={i} className={`msg ${m.from}${m.rag ? ' rag' : ''}`}>{m.text}</div>
        ))}
      </div>
      <div className="controls">
        <input
          value={text}
          onChange={e=>setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={
            loading ? 'Esperando respuesta…' :
            mode === 'rag' ? 'Pregunta sobre el documento…' :
            'Escribe algo…'
          }
          disabled={loading}
        />
        <button onClick={send} disabled={loading || uploading}>{loading ? '…' : 'Enviar'}</button>
      </div>
    </div>
  )
}
