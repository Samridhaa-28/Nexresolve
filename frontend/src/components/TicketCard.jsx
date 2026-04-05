import React from 'react'
import { useNavigate } from 'react-router-dom'

export const STRATEGY_COLORS = {
  suggest:  { bg: 'bg-blue-600',   light: 'bg-blue-900',   text: 'text-blue-300'   },
  route:    { bg: 'bg-purple-600', light: 'bg-purple-900', text: 'text-purple-300' },
  clarify:  { bg: 'bg-amber-600',  light: 'bg-amber-900',  text: 'text-amber-300'  },
  escalate: { bg: 'bg-red-600',    light: 'bg-red-900',    text: 'text-red-300'    },
}

function formatTimestamp(ts) {
  if (!ts) return ''
  try {
    return new Date(ts).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return ts
  }
}

function OutcomeBadge({ resolved, sla_breach, strategy }) {
  if (resolved === true) {
    return <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-green-900 text-green-300">Resolved</span>
  }
  if (sla_breach === true) {
    return <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-red-900 text-red-300">SLA Breach</span>
  }
  if (strategy === 'escalate') {
    return <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-amber-900 text-amber-300">Escalated</span>
  }
  return <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-slate-700 text-slate-300">Pending</span>
}

export default function TicketCard({ ticket, onDelete }) {
  const navigate = useNavigate()
  const strategy = ticket?.strategy?.toLowerCase()
  const colors = STRATEGY_COLORS[strategy] ?? { light: 'bg-slate-700', text: 'text-slate-300' }
  const text = ticket?.raw_text ?? ''

  function handleDelete(e) {
    e.stopPropagation()
    onDelete?.(ticket.ticket_id)
  }

  return (
    <div
      onClick={() => navigate(`/ticket/${ticket.ticket_id}`)}
      className="relative bg-slate-800 rounded-xl border border-slate-700 shadow-sm p-4 hover:bg-slate-700 hover:border-slate-600 cursor-pointer transition-all mb-3"
    >
      {/* Delete button */}
      <button
        onClick={handleDelete}
        className="absolute top-3 right-3 text-slate-500 hover:text-red-400 transition-colors text-sm leading-none p-1"
        title="Delete ticket"
      >
        ✕
      </button>

      <p className="text-slate-200 font-medium text-sm mb-2 pr-6 truncate" title={text}>
        {text.length > 80 ? text.slice(0, 80) + '…' : text || 'No description'}
      </p>
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <span className="text-slate-500 text-xs">{formatTimestamp(ticket?.timestamp)}</span>
        <div className="flex items-center gap-2">
          <OutcomeBadge
            resolved={ticket?.resolved}
            sla_breach={ticket?.sla_breach}
            strategy={strategy}
          />
          {strategy && (
            <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${colors.light} ${colors.text}`}>
              {strategy.toUpperCase()}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
