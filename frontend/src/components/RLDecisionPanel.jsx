import React from 'react'
import { STRATEGY_COLORS } from './TicketCard.jsx'

export default function RLDecisionPanel({ rl, resolved, sla_breach }) {
  const strategy = rl?.strategy?.toLowerCase()
  const colors = STRATEGY_COLORS[strategy]
  const bgClass = colors?.bg ?? 'bg-slate-600'

  let outcome
  if (resolved === true) {
    outcome = <span className="text-green-400 font-medium">✓ Resolved</span>
  } else if (sla_breach === true) {
    outcome = <span className="text-red-400 font-medium">✗ SLA Breach</span>
  } else {
    outcome = <span className="text-amber-400 font-medium">→ In Progress</span>
  }

  return (
    <div className="bg-slate-950 text-white p-4 border-t border-slate-800">
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
        RL Decision
      </p>

      {/* Strategy badge */}
      <div className="mb-4">
        <span className={`${bgClass} text-white font-bold px-4 py-2 rounded-lg text-sm inline-block`}>
          {rl?.strategy?.toUpperCase() ?? 'UNKNOWN'}
        </span>
      </div>

      {/* Action */}
      <div className="mb-3">
        <span className="text-slate-500 text-xs">Action: </span>
        <span className="text-white text-sm font-medium font-mono">{rl?.action ?? 'N/A'}</span>
      </div>

      {/* Strategy + action indices */}
      <div className="mb-3">
        <p className="text-slate-600 text-xs">
          Strategy idx: <span className="text-slate-400 font-mono">{rl?.strategy_idx ?? 'N/A'}</span>
          {' · '}
          Action idx: <span className="text-slate-400 font-mono">{rl?.action_idx ?? 'N/A'}</span>
        </p>
      </div>

      {/* Outcome */}
      <div className="text-sm">{outcome}</div>
    </div>
  )
}
