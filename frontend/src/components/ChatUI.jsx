import React, { useState, useRef, useEffect } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || ''

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
const hasSpeechRecognition = !!SpeechRecognition

// ── Audio unlock — requerido en Android/iOS antes del primer speak ─────────
let audioUnlocked = false
function unlockAudio() {
  if (audioUnlocked || !('speechSynthesis' in window)) return
  try {
    const u = new SpeechSynthesisUtterance(' ')
    u.volume = 0; u.rate = 10  // rapidísimo para que no se note
    u.onend = () => { audioUnlocked = true }
    window.speechSynthesis.cancel()
    window.speechSynthesis.speak(u)
  } catch {}
}

// ── Elegir la mejor voz en español (sincrónico) ────────────────────────────
function getBestVoice() {
  const voices = window.speechSynthesis?.getVoices() || []
  const preferred = [
    'Google español de Estados Unidos', 'Google español',
    'Microsoft Dalia Online (Natural) - Spanish (Mexico)',
    'Microsoft Sabina Online (Natural) - Spanish (Mexico)',
    'Microsoft Elena Online (Natural) - Spanish (Spain)',
    'Microsoft Laura Online (Natural) - Spanish (Spain)',
    'Mónica', 'Paulina', 'Jorge',   // voces macOS/iOS en español
  ]
  for (const name of preferred) {
    const v = voices.find(v => v.name === name); if (v) return v
  }
  return voices.find(v => v.lang.startsWith('es')) || null
}

// ── Síntesis de voz ────────────────────────────────────────────────────────
function speakText(text) {
  if (!('speechSynthesis' in window) || !text) return
  window.speechSynthesis.cancel()

  const doSpeak = () => {
    const utter = new SpeechSynthesisUtterance(text)
    const voice = getBestVoice()
    if (voice) utter.voice = voice
    utter.lang   = voice?.lang || 'es-ES'
    utter.rate   = 1.05
    utter.pitch  = 1.0
    utter.volume = 1.0
    // Android: cancel + pequeño delay evita silencio
    window.speechSynthesis.cancel()
    setTimeout(() => window.speechSynthesis.speak(utter), 50)
  }

  if (window.speechSynthesis.getVoices().length > 0) {
    doSpeak()
  } else {
    // Voces aún no cargadas (Android las carga tarde)
    const onVoices = () => { window.speechSynthesis.onvoiceschanged = null; doSpeak() }
    window.speechSynthesis.onvoiceschanged = onVoices
    // Timeout fallback por si onvoiceschanged no dispara
    setTimeout(() => { if (window.speechSynthesis.onvoiceschanged === onVoices) { window.speechSynthesis.onvoiceschanged = null; doSpeak() } }, 1000)
  }
}

export default function ChatUI(){
  const [messages, setMessages] = useState([])
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState('chat')
  const [ragReady, setRagReady] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [listening, setListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [micError, setMicError] = useState('')
  const fileRef = useRef(null)
  const recognitionRef = useRef(null)

  // ── Micrófono — nueva instancia cada vez ──────────────────────────────────
  function startListening() {
    if (!hasSpeechRecognition) return
    setMicError('')

    const rec = new SpeechRecognition()
    rec.lang = 'es-ES'
    rec.continuous = false      // para automáticamente cuando detecta silencio
    rec.interimResults = true   // muestra transcripción en tiempo real
    rec.maxAlternatives = 1

    let finalText = ''

    rec.onstart = () => {
      setListening(true)
      setTranscript('')
      finalText = ''
    }

    rec.onresult = (e) => {
      let interim = ''
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript
        if (e.results[i].isFinal) finalText += t
        else interim += t
      }
      setTranscript(finalText + interim)
    }

    // Se dispara automáticamente cuando Chrome detecta silencio
    rec.onend = () => {
      setListening(false)
      setTranscript('')
      if (finalText.trim()) {
        // Auto-enviar el mensaje capturado
        setText('')
        send(finalText.trim())
      }
    }

    rec.onerror = (e) => {
      const msgs = {
        'not-allowed':  'Permiso de micrófono denegado.',
        'no-speech':    null,
        'network':      'Error de red. Usa Chrome con internet activo.',
        'audio-capture':'No se encontró micrófono.',
        'aborted':      null,
      }
      const msg = e.error in msgs ? msgs[e.error] : `Error: ${e.error}`
      if (msg) setMicError(msg)
      setListening(false)
      setTranscript('')
    }

    recognitionRef.current = rec
    try { rec.start() } catch(e) { console.error(e) }
  }

  function stopListening() {
    recognitionRef.current?.stop()
    setListening(false)
    setTranscript('')
  }

  function toggleVoice() {
    unlockAudio()
    if (listening) stopListening()
    else startListening()
  }

  // ── Chat normal ────────────────────────────────────────────────────────────

  async function send(overrideText){
    unlockAudio()
    const msg = overrideText || text
    if(!msg || loading) return
    const userMsg = { from: 'user', text: msg }
    setMessages(m => [...m, userMsg])
    setText('')
    setLoading(true)

    try {
      if (mode === 'rag') {
        await sendRag(msg)
      } else {
        await sendChat(msg)
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
      const botMsg = { from:'bot', text: reply }
      setMessages(m => [...m, botMsg])

      // Activar animación de "hablar" en el bot AR
      if (window.__AR_BOT_SPEAK) window.__AR_BOT_SPEAK(reply)

      // Síntesis de voz — selecciona la mejor voz neural disponible
      speakText(reply)
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

  // Expone inyección de mensajes externos (visión) + voz
  React.useEffect(() => {
    window.__INJECT_BOT_MSG = (text) => {
      setMessages(m => [...m, { from: 'bot', text }])
      speakText(text)                    // el bot también habla
      window.__AR_BOT_SPEAK?.(text)      // anima la boca/brazos
    }
    return () => { delete window.__INJECT_BOT_MSG }
  }, [])

  // ── Render ─────────────────────────────────────────────────────────────────
  const messagesEndRef = React.useRef(null)
  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, listening])

  return (
    <div className="chat-ui" onPointerDown={unlockAudio}>

      {/* Tabs */}
      <div className="chat-tabs">
        <button className={`tab-btn${mode === 'chat' ? ' active' : ''}`} onClick={() => setMode('chat')}>
          💬 Chat
        </button>
        <button className={`tab-btn${mode === 'rag' ? ' active' : ''}`} onClick={() => setMode('rag')}>
          📄 Documentos
        </button>
      </div>

      {/* RAG toolbar */}
      {mode === 'rag' && (
        <div className="rag-toolbar">
          <label className="upload-btn">
            📎 Subir PDF
            <input ref={fileRef} type="file" accept="application/pdf"
              style={{ display:'none' }} onChange={uploadPdf} disabled={uploading} />
          </label>
          <span className={`rag-status ${ragReady ? 'ready' : 'not-ready'}`}>
            {ragReady ? '● Índice listo' : '○ Sin documentos'}
          </span>
        </div>
      )}

      {/* Mensajes */}
      <div className="messages">
        {messages.length === 0 && !listening && (
          <div className="messages-empty">
            <div className="messages-empty-icon">💻</div>
            <div className="messages-empty-title">¿En qué te puedo ayudar?</div>
            <div className="messages-empty-sub">
              Pregunta por precios, recomendaciones,<br/>comparaciones o soporte técnico
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.from}${m.rag ? ' rag' : ''}`}>
            {m.text}
          </div>
        ))}
        {loading && (
          <div className="msg bot">
            <span className="loading-dots">
              <span/><span/><span/>
            </span>
          </div>
        )}
        {listening && (
          <div className="msg bot interim">
            🎤 {transcript ? transcript + '…' : 'Escuchando…'}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Error micrófono */}
      {micError && <div className="mic-error">⚠️ {micError}</div>}

      {/* Input */}
      <div className="controls">
        {hasSpeechRecognition && (
          <button
            className={`mic-btn${listening ? ' active' : ''}`}
            onClick={toggleVoice}
            disabled={loading}
            title={listening ? 'Detener' : 'Hablar'}
          >
            {listening ? '⏹' : '🎤'}
          </button>
        )}
        <input
          value={listening ? transcript : text}
          onChange={e => { if (!listening) setText(e.target.value) }}
          onKeyDown={e => e.key === 'Enter' && !listening && send()}
          placeholder={
            listening       ? 'Escuchando… habla ahora' :
            loading         ? 'Ktronix está pensando…'  :
            mode === 'rag'  ? 'Pregunta sobre el documento…' :
                              'Pregunta sobre laptops o precios…'
          }
          disabled={loading}
          readOnly={listening}
        />
        <button
          className="send-btn"
          onClick={() => send()}
          disabled={loading || uploading || listening}
        >
          {loading ? '···' : 'Enviar'}
        </button>
      </div>

    </div>
  )
}
