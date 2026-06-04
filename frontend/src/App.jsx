import React, { useState, useRef } from 'react'
import ArScene from './ar/ArScene'
import ChatUI from './components/ChatUI'

export default function App() {
  const [facingMode, setFacingMode]   = useState('user')
  const [visionBusy, setVisionBusy]   = useState(false)
  const arRef  = useRef(null)
  const chatRef = useRef(null)   // para inyectar mensajes externos

  const flipCamera = () => setFacingMode(f => f === 'user' ? 'environment' : 'user')

  async function analyzeCamera() {
    if (visionBusy) return
    const b64 = arRef.current?.captureFrame()
    if (!b64) {
      window.__INJECT_BOT_MSG?.('No pude capturar la cámara. Asegúrate de haber dado permiso de cámara.')
      setVisionBusy(false)
      return
    }

    setVisionBusy(true)
    // Inyecta mensaje "pensando..." en el chat
    window.__INJECT_BOT_MSG?.('👁️ Analizando lo que veo…')

    try {
      const res  = await fetch('/api/vision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: b64 }),
      })
      const data = await res.json()
      const desc = data.reply || data.description || 'No pude analizar la imagen.'
      window.__INJECT_BOT_MSG?.(desc)   // inyecta mensaje + habla + anima
    } catch {
      window.__INJECT_BOT_MSG?.('Error al analizar la imagen.')
    } finally {
      setVisionBusy(false)
    }
  }

  return (
    <div className="app-root">
      <header className="header">
        <h1 className="header-title">Ktronix</h1>
        <p className="header-sub">Asistente de laptops con IA</p>
      </header>

      <div className="ar-wrapper">
        <ArScene ref={arRef} facingMode={facingMode} />

        {/* Flip cámara */}
        <button className="cam-btn cam-btn--flip" onClick={flipCamera} title="Cambiar cámara">🔄</button>

        {/* Visión */}
        <button
          className={`cam-btn cam-btn--vision${visionBusy ? ' busy' : ''}`}
          onClick={analyzeCamera}
          title="Analizar lo que ve la cámara"
          disabled={visionBusy}
        >
          {visionBusy ? '⏳' : '👁️'}
        </button>
      </div>

      <ChatUI />
    </div>
  )
}
