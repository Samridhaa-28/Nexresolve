import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getTicket } from '../services/api.js'
import Navbar from '../components/Navbar.jsx'
import NLPPanel from '../components/NLPPanel.jsx'
import RLDecisionPanel from '../components/RLDecisionPanel.jsx'
import { STRATEGY_COLORS } from '../components/TicketCard.jsx'

function StrategyBadge({ strategy }) {
  if (!strategy) return null
  const s = strategy.toLowerCase()
  const c = STRATEGY_COLORS[s] ?? { light: 'bg-slate-700', text: 'text-slate-300' }
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${c.light} ${c.text} ml-2`}>
      {s.toUpperCase()}
    </span>
  )
}

function formatTs(ts) {
  if (!ts) return ''
  try {
    return new Date(ts).toLocaleString('en-US', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    })
  } catch { return '' }
}

export default function TicketDetailPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [ticket, setTicket] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  // Sidebar always shows the first (most recent resolved) assistant turn's RAG
  const activeTurn = 0

  useEffect(() => {
    document.title = 'NexResolve — Ticket Detail'
    getTicket(id)
      .then((res) => setTicket(res.data))
      .catch((err) => {
        setError(err.response?.status === 404 ? 'Ticket not found.' : 'Failed to load ticket.')
      })
      .finally(() => setLoading(false))
  }, [id])

  if (loading) {
    return (
      <div className="flex h-screen pt-14 items-center justify-center bg-slate-950">
        <Navbar />
        <div className="text-slate-500 text-sm animate-pulse">Loading ticket…</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-screen pt-14 items-center justify-center bg-slate-950">
        <Navbar />
        <div className="text-center">
          <p className="text-red-400 text-sm mb-4">{error}</p>
          <button onClick={() => navigate('/dashboard')} className="text-blue-400 text-sm hover:underline">
            ← Back to Dashboard
          </button>
        </div>
      </div>
    )
  }

  // Build message list — use stored messages array (multi-turn) if present,
  // otherwise fall back to raw_text + generated_response (legacy single-turn docs)
  const storedMessages = ticket?.messages ?? []
  const hasMessages    = storedMessages.length > 0

  // Collect assistant turns in order (for sidebar navigation)
  const assistantTurns = storedMessages.filter((m) => m.role === 'assistant')

  // Determine what to show in the sidebar:
  // - Use the activeTurn's rag_top_k if available
  // - Fall back to ticket.rag_result.top_k (last pipeline result)
  const activeTurnData  = assistantTurns[activeTurn] ?? {}
  const sidebarRagTopK  = activeTurnData.rag_top_k ?? ticket?.rag_result?.top_k ?? []
  const sidebarSim      = ticket?.rag_result?.top_similarity ?? 0
  const sidebarNlp      = ticket?.nlp_result
  const sidebarRl       = ticket?.rl_decision
  const sidebarSla      = ticket?.sla_result ?? {}

  return (
    <div className="flex h-screen pt-14">
      <Navbar />

      {/* ── Left: Full conversation thread ───────────────────────────────── */}
      <div className="w-3/5 flex flex-col bg-slate-950">
        {/* Breadcrumb */}
        <div className="flex items-center gap-1 px-4 py-3 text-slate-500 text-sm border-b border-slate-800 bg-slate-900">
          <button onClick={() => navigate('/dashboard')} className="hover:text-slate-200 transition-colors">
            ← Back to Dashboard
          </button>
          {assistantTurns.length > 1 && (
            <span className="ml-auto text-slate-600 text-xs">
              {assistantTurns.length} turns
            </span>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4">
          {hasMessages ? (
            // Render full stored messages array
            storedMessages.map((msg, i) => {
              const isUser = msg.role === 'user'
              const turnIdx = isUser ? -1 : assistantTurns.indexOf(msg)

              return isUser ? (
                <div key={i} className="flex justify-end mb-4">
                  <div className="bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-lg text-sm whitespace-pre-wrap">
                    {msg.text}
                  </div>
                </div>
              ) : (
                <div key={i} className="flex items-start gap-2 mb-4">
                  <div className="bg-slate-700 w-8 h-8 rounded-full flex items-center justify-center text-sm flex-shrink-0 mt-1">
                    🤖
                  </div>
                  <div className="flex-1 max-w-lg">
                    <div className="flex items-center mb-1 flex-wrap gap-1">
                      <StrategyBadge strategy={msg.strategy} />
                      {msg.sla_snapshot && (() => {
                        const sla      = msg.sla_snapshot
                        const breached = sla.sla_breach_flag === 1
                        const color    = breached
                          ? '#ef4444'
                          : (sla.sla_remaining_norm ?? 1) < 0.3
                          ? '#f59e0b'
                          : '#22c55e'
                        return (
                          <span style={{ fontSize: '0.72rem', color, fontWeight: 500 }}>
                            {breached
                              ? '🔴 SLA Breached'
                              : `⏱ ${sla.sla_remaining_hours}h remaining`}
                          </span>
                        )
                      })()}
                    </div>
                    <div className="bg-slate-800 border border-slate-700 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm text-sm text-slate-200">
                      <div className="max-h-64 overflow-y-auto whitespace-pre-wrap">
                        {msg.text || '(no response)'}
                      </div>
                    </div>
                    <span className="text-slate-600 text-xs mt-1 block">{formatTs(msg.timestamp)}</span>
                  </div>
                </div>
              )
            })
          ) : (
            // Legacy fallback: single-turn doc without messages array
            <>
              <div className="flex justify-end mb-4">
                <div className="bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-lg text-sm whitespace-pre-wrap">
                  {ticket?.raw_text ?? '(no text)'}
                </div>
              </div>
              <div className="flex items-end gap-2 mb-4">
                <div className="bg-slate-700 w-8 h-8 rounded-full flex items-center justify-center text-sm flex-shrink-0">
                  🤖
                </div>
                <div className="bg-slate-800 border border-slate-700 rounded-2xl rounded-tl-sm px-4 py-3 max-w-lg shadow-sm text-sm text-slate-200 whitespace-pre-wrap">
                  {ticket?.generated_response ?? 'Ticket processed.'}
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* ── Right: Analysis sidebar — shows data for selected turn ───────── */}
      <div className="w-2/5 bg-slate-950 overflow-y-auto border-l border-slate-800">
        <NLPPanel
          nlp={sidebarNlp}
          similarity={sidebarSim}
          ragTopK={sidebarRagTopK}
          sla={sidebarSla}
        />
        <RLDecisionPanel
          rl={sidebarRl}
          resolved={ticket?.resolved}
          sla_breach={ticket?.sla_breach}
        />
      </div>
    </div>
  )
}
