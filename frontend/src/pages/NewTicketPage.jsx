import React, { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { submitTicket } from '../services/api.js'
import Navbar from '../components/Navbar.jsx'
import NLPPanel from '../components/NLPPanel.jsx'
import RLDecisionPanel from '../components/RLDecisionPanel.jsx'

// ── Greeting detection (short-circuit — no API call) ────────────────────────
const THANKS_RE   = /^(thanks|thank you)\b/i
const GOODBYE_RE  = /^(bye|goodbye)\b/i
const GREETING_RE = /^(hi|hello|hey|good morning|good afternoon|good evening|good night|thanks|thank you|welcome|bye|goodbye|ok|okay|great|perfect|got it|understood)\b/i

function getGreetingResponse(text) {
  if (THANKS_RE.test(text))  return "You're welcome! 😊 Let me know if you have any other issues."
  if (GOODBYE_RE.test(text)) return 'Goodbye! Have a great day. 👋'
  return "Hello! 👋 I'm NexResolve, your AI support assistant. Please describe your technical issue and I'll help route and resolve it for you."
}

// ── Bot message resolution ───────────────────────────────────────────────────
// Priority: generator_output (suggest) > clarification_question (clarify) > response (all others)
const NON_TECHNICAL_INTENTS = ['billing', 'general', 'feature_request']

function resolveBotMessage(data) {
  const strategy = data?.rl?.strategy?.toLowerCase()
  const intent   = data?.nlp?.intent_group?.toLowerCase()

  // Non-technical intent that RL labelled clarify → treat as route visually
  if (strategy === 'clarify' && NON_TECHNICAL_INTENTS.includes(intent)) {
    return `Your ${intent} request has been received and routed to the appropriate team.`
  }

  if (strategy === 'suggest') {
    const g = data?.generator_output
    return (g && g.trim()) ? g : 'A solution has been identified. Please try the suggested steps.'
  }
  if (strategy === 'clarify') {
    const q = data?.clarification_question
    return (q && q.trim()) ? q : 'Could you please provide more details about your issue?'
  }
  // route / escalate / fallback
  const r = data?.response
  return (r && r.trim()) ? r : 'Your ticket has been received and is being processed.'
}

// ── Typing indicator ─────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="flex items-end gap-2 mb-4">
      <div className="bg-slate-700 w-8 h-8 rounded-full flex items-center justify-center text-sm flex-shrink-0">
        🤖
      </div>
      <div className="bg-slate-800 border border-slate-700 rounded-2xl rounded-tl-sm px-4 py-3">
        <div className="flex gap-1 items-center h-4">
          <span className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
          <span className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
          <span className="w-2 h-2 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
        <p className="text-slate-500 text-xs italic mt-1">Analyzing your ticket… This may take up to 30 seconds</p>
      </div>
    </div>
  )
}

// ── Main component ───────────────────────────────────────────────────────────
export default function NewTicketPage() {
  const [text,             setText]             = useState('')
  const [messages,         setMessages]         = useState([])
  const [loading,          setLoading]          = useState(false)
  const [latestResult,     setLatestResult]     = useState(null)
  const [ticketId,         setTicketId]         = useState(null)   // Fix 7: session tracking
  const [conversationEnded, setConversationEnded] = useState(false) // Fix 6
  const messagesEndRef = useRef(null)
  const navigate = useNavigate()

  useEffect(() => { document.title = 'NexResolve — New Ticket' }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function handleSubmit(e) {
    e.preventDefault()
    const trimmed = text.trim()
    if (!trimmed || loading || conversationEnded) return

    setText('')

    // Greeting short-circuit (Fix 4)
    if (GREETING_RE.test(trimmed)) {
      setMessages((prev) => [
        ...prev,
        { role: 'user', content: trimmed },
        { role: 'bot',  content: getGreetingResponse(trimmed) },
      ])
      return
    }

    setMessages((prev) => [...prev, { role: 'user', content: trimmed }])
    setLoading(true)

    try {
      // Fix 7: pass ticket_id on follow-up turns
      const res  = await submitTicket(trimmed, ticketId)
      const data = res.data

      // Store ticket_id from first response
      if (!ticketId && data.ticket_id) {
        setTicketId(data.ticket_id)
      }

      setLatestResult(data)

      const botContent = resolveBotMessage(data)   // Fix 1 + Fix 3 + Fix 8
      setMessages((prev) => [...prev, { role: 'bot', content: botContent }])

      // Fix 6: lock conversation when ended
      if (data.conversation_end) {
        setConversationEnded(true)
      }
    } catch (err) {
      const errMsg = err.response?.data?.detail ?? 'Something went wrong. Please try again.'
      setMessages((prev) => [...prev, { role: 'bot', content: `Error: ${errMsg}` }])
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="flex h-screen pt-14">
      <Navbar />

      {/* ── Left: Chat ───────────────────────────────────────────────────── */}
      <div className="w-3/5 flex flex-col gradient-bg">
        {/* Breadcrumb */}
        <div className="flex items-center gap-1 px-4 py-3 text-slate-500 text-sm border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm">
          <button onClick={() => navigate('/dashboard')} className="hover:text-slate-200 transition-colors">
            ← Dashboard
          </button>
          <span className="text-slate-700">›</span>
          <span className="text-slate-400">New Ticket</span>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4">
          {messages.length === 0 && !loading && (
            <div className="flex flex-col items-center justify-center h-full gap-3">
              <div className="text-center">
                <p style={{
                  fontSize: '1.8rem',
                  fontWeight: '700',
                  letterSpacing: '0.04em',
                  color: '#e2e8f0',
                  marginBottom: '0.3rem',
                }}>
                  NexResolve
                </p>
                <p style={{
                  fontSize: '0.85rem',
                  color: '#64748b',
                  letterSpacing: '0.06em',
                  fontStyle: 'italic',
                }}>
                  Intelligent support, resolved at the next level.
                </p>
              </div>
              <p className="text-slate-600 text-sm text-center mt-4">
                Describe your issue below — our AI will analyze and route it instantly.
              </p>
            </div>
          )}

          {messages.map((msg, i) =>
            msg.role === 'user' ? (
              <div key={i} className="flex justify-end mb-4">
                <div className="bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-lg text-sm whitespace-pre-wrap">
                  {msg.content}
                </div>
              </div>
            ) : (
              <div key={i} className="flex items-end gap-2 mb-4">
                <div className="bg-slate-700 w-8 h-8 rounded-full flex items-center justify-center text-sm flex-shrink-0">
                  🤖
                </div>
                <div className="bg-slate-800 border border-slate-700 rounded-2xl rounded-tl-sm px-4 py-3 max-w-lg shadow-sm text-sm text-slate-200">
                  <div className="max-h-64 overflow-y-auto whitespace-pre-wrap">
                    {msg.content}
                  </div>
                </div>
              </div>
            )
          )}

          {loading && <TypingIndicator />}
          <div ref={messagesEndRef} />
        </div>

        {/* Fix 6: Conversation-ended banner */}
        {conversationEnded && (
          <div className="border-t border-slate-800 bg-slate-900/90 backdrop-blur-sm px-4 py-3 flex items-center justify-between">
            <p className="text-slate-400 text-sm">
              This conversation has been closed. Start a new ticket from the dashboard.
            </p>
            <button
              onClick={() => navigate('/dashboard')}
              className="ml-4 text-blue-400 text-sm hover:text-blue-300 transition-colors whitespace-nowrap"
            >
              Back to Dashboard
            </button>
          </div>
        )}

        {/* Input — disabled when conversation ended */}
        {!conversationEnded && (
          <form
            onSubmit={handleSubmit}
            className="border-t border-slate-800 p-4 bg-slate-900/90 backdrop-blur-sm flex gap-3 items-end"
          >
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading}
              rows={3}
              placeholder="Describe your issue in detail..."
              className="flex-1 resize-none bg-slate-800 border border-slate-600 text-white placeholder-slate-500 rounded-lg p-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={loading || !text.trim()}
              className="bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-5 py-2.5 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                  </svg>
                  Analyzing
                </>
              ) : (
                'Send'
              )}
            </button>
          </form>
        )}
      </div>

      {/* ── Right: Analysis Sidebar ──────────────────────────────────────── */}
      <div className="w-2/5 bg-slate-950 overflow-y-auto border-l border-slate-800">
        {!latestResult ? (
          <div className="flex items-center justify-center h-full p-8">
            <p className="text-slate-600 text-sm text-center">
              🔍 AI Analysis will appear here after you submit your ticket
            </p>
          </div>
        ) : (
          <>
            <NLPPanel
              nlp={latestResult?.nlp}
              similarity={latestResult?.rag?.top_similarity ?? 0}
              ragTopK={latestResult?.rag?.top_k ?? []}
              sla={latestResult?.sla ?? {}}
            />
            <RLDecisionPanel
              rl={latestResult?.rl}
              resolved={latestResult?.resolved}
              sla_breach={latestResult?.sla_breach}
            />
          </>
        )}
      </div>
    </div>
  )
}
