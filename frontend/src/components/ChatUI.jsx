import React, { useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || ''

export default function ChatUI(){
  const [messages, setMessages] = useState([])
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)

  async function send(){
    if(!text || loading) return
    const userMsg = { from: 'user', text }
    setMessages(m => [...m, userMsg])
    setText('')
    setLoading(true)

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ message: userMsg.text })
      })
      const data = await res.json()
      const reply = data?.reply || 'Lo siento, no pude obtener respuesta.'
      const analysis = data?.analysis || null
      const debugSuffix = analysis
        ? ` [intent: ${analysis.intent}, sentiment: ${analysis.sentiment}]`
        : ''
      const botMsg = { from:'bot', text: reply + debugSuffix }
      setMessages(m => [...m, botMsg])

      // TTS (Web Speech API)
      if ('speechSynthesis' in window) {
        const utter = new SpeechSynthesisUtterance(reply)
        utter.lang = 'es-ES'
        window.speechSynthesis.cancel()
        window.speechSynthesis.speak(utter)
      }

      // Hook visual sobre el modelo (placeholder para futuros textos/gestos)
      if (window.__AR_BOT_MODEL) {
        window.__AR_BOT_MODEL.userData.lastReply = reply
      }
    } catch (e) {
      console.error(e)
      setMessages(m => [...m, { from: 'bot', text: 'Error al contactar el servidor.' }])
    } finally {
      setLoading(false)
    }
  }

  function onKeyDown(e){
    if (e.key === 'Enter') send()
  }

  return (
    <div className="chat-ui">
      <div className="messages">
        {messages.map((m,i)=> (
          <div key={i} className={`msg ${m.from}`}>{m.text}</div>
        ))}
      </div>
      <div className="controls">
        <input
          value={text}
          onChange={e=>setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={loading ? 'Esperando respuesta...' : 'Escribe algo...'}
          disabled={loading}
        />
        <button onClick={send} disabled={loading}>{loading ? '...' : 'Enviar'}</button>
      </div>
    </div>
  )
}