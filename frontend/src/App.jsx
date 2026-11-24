import { useEffect, useRef, useState } from 'react'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

function newSessionId() {
  if (crypto.randomUUID) return crypto.randomUUID()
  return 's-' + Math.random().toString(36).slice(2) + Date.now().toString(36)
}

export default function App() {
  const [sessionId] = useState(newSessionId)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [systemPrompt, setSystemPrompt] = useState('')
  const [sending, setSending] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send() {
    const text = input.trim()
    if (!text || sending) return
    setInput('')
    setSending(true)
    setMessages((m) => [...m, { role: 'user', content: text }])
    try {
      const body = { session_id: sessionId, message: text }
      if (systemPrompt.trim()) body.system_prompt = systemPrompt.trim()
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json().catch(() => null)
      if (res.ok && data && data.reply) {
        setMessages((m) => [...m, { role: 'assistant', content: data.reply, metadata: data.metadata }])
      } else {
        const detail = data && data.detail
          ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail))
          : `Request failed (HTTP ${res.status})`
        setMessages((m) => [...m, { role: 'error', content: detail }])
      }
    } catch (err) {
      setMessages((m) => [...m, { role: 'error', content: 'Network error: could not reach the backend.' }])
    } finally {
      setSending(false)
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="app">
      <header>
        <h1>LLM Chatbot</h1>
        <span className="session">session {sessionId.slice(0, 8)}</span>
      </header>

      <div className="system-prompt-row">
        <input
          data-testid="system-prompt-input"
          type="text"
          placeholder="Optional system prompt for this session"
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
        />
      </div>

      <main className="chat">
        {messages.length === 0 && (
          <p className="empty">Send a message to start the conversation.</p>
        )}
        {messages.map((m, i) => {
          if (m.role === 'user') {
            return (
              <div key={i} className="bubble user">
                <div className="text">{m.content}</div>
              </div>
            )
          }
          if (m.role === 'error') {
            return (
              <div key={i} className="bubble error">
                <div className="text">Error: {m.content}</div>
              </div>
            )
          }
          const md = m.metadata || {}
          return (
            <div key={i} className="bubble assistant" data-testid="assistant-message">
              <div className="text">{m.content}</div>
              <div className="meta">
                <span data-testid="meta-source">{md.resolution_source}</span>
                <span data-testid="meta-model">{md.model}</span>
                <span data-testid="meta-latency">{md.latency_ms} ms</span>
                <span>
                  in <span data-testid="meta-input-tokens">{md.input_tokens}</span> tok
                </span>
                <span>
                  out <span data-testid="meta-output-tokens">{md.output_tokens}</span> tok
                </span>
              </div>
            </div>
          )
        })}
        <div ref={bottomRef} />
      </main>

      <footer className="composer">
        <textarea
          data-testid="chat-input"
          rows={2}
          placeholder="Type a message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <button data-testid="send-button" onClick={send} disabled={sending || !input.trim()}>
          {sending ? 'Sending...' : 'Send'}
        </button>
      </footer>
    </div>
  )
}
