import React from 'react'
import ArScene from './ar/ArScene'
import ChatUI from './components/ChatUI'

export default function App(){
  return (
    <div className="app-root">
      <h1 className="title">Cartoon AR Bot — Demo</h1>
      <div className="ar-area">
        <ArScene />
      </div>
      <ChatUI />
    </div>
  )
}