import React, { useState } from 'react'

function safeFixed(val, digits = 2) {
  return typeof val === 'number' ? val.toFixed(digits) : '0.00'
}

function BarRow({ label, value, colorClass, rightLabel }) {
  const pct = typeof value === 'number' ? Math.min(100, Math.max(0, value * 100)) : 0
  return (
    <div className="space-y-1">
      <div className="flex justify-between items-center">
        <span className="text-slate-400 text-xs">{label}</span>
        {rightLabel && <span className="text-slate-400 text-xs">{rightLabel}</span>}
      </div>
      <div className="bg-slate-800 rounded-full h-1.5">
        <div className={`h-1.5 rounded-full ${colorClass}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function SectionHeader({ children }) {
  return (
    <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
      {children}
    </p>
  )
}

function urgencyColor(score) {
  if (score > 0.7) return 'bg-red-500'
  if (score >= 0.4) return 'bg-amber-500'
  return 'bg-green-500'
}

function frustrationColor(level) {
  if (level > 0.6) return 'bg-red-500'
  if (level >= 0.3) return 'bg-amber-500'
  return 'bg-green-500'
}

function sentimentColor(label) {
  if (label === 'positive') return 'text-green-400'
  if (label === 'negative') return 'text-red-400'
  return 'text-slate-400'
}

// Per-item expandable retrieved solution (Fix 8B)
function RetrievedItem({ item, index }) {
  const [expanded, setExpanded] = useState(false)
  const text    = item?.solution ?? ''
  const isLong  = text.length > 150
  const display = expanded ? text : text.slice(0, 150) + (isLong ? '…' : '')

  return (
    <div className="bg-slate-800 rounded-lg p-2">
      <div className="flex justify-between items-center mb-1">
        <span className="text-slate-500 text-xs">Result {index + 1}</span>
        <span className="text-blue-400 text-xs font-mono">
          Score: {safeFixed(item?.score ?? 0)}
        </span>
      </div>
      <p className="text-slate-400 text-xs leading-relaxed">{display}</p>
      {isLong && (
        <button
          onClick={() => setExpanded((x) => !x)}
          className="text-blue-500 text-xs mt-1 hover:text-blue-400 transition-colors"
        >
          {expanded ? 'See less' : 'See more'}
        </button>
      )}
    </div>
  )
}

export default function NLPPanel({ nlp, similarity, ragTopK, sla }) {
  const n    = nlp ?? {}
  const sim  = typeof similarity === 'number' ? similarity : 0
  const topK = Array.isArray(ragTopK) ? ragTopK : []
  const s    = sla ?? {}

  const [ragExpanded, setRagExpanded] = useState(false)

  const entityMap = {
    has_version:    'version',
    has_error_type: 'error type',
    has_platform:   'platform',
    has_hardware:   'hardware',
  }
  const detected = Object.entries(entityMap).filter(([key]) => n[key] === 1)
  const missing  = Object.entries(entityMap).filter(([key]) => n[key] !== 1)

  return (
    <div className="bg-slate-950 text-white p-4 overflow-y-auto">
      <h2 className="text-sm font-semibold text-white mb-4">AI Analysis</h2>

      {/* ── Intent ───────────────────────────────────────────────── */}
      <div className="border-b border-slate-800 pb-3 mb-3">
        <SectionHeader>Intent</SectionHeader>
        <div className="flex items-center justify-between mb-1">
          <span className="text-white font-medium text-sm capitalize">{n?.intent_group ?? 'N/A'}</span>
          {n?.uncertainty_flag === 1 && (
            <span className="text-xs text-amber-400 font-medium">Uncertain</span>
          )}
        </div>
        <BarRow
          label="Confidence"
          value={n?.confidence_score ?? 0}
          colorClass="bg-blue-500"
          rightLabel={`${Math.round((n?.confidence_score ?? 0) * 100)}%`}
        />
      </div>

      {/* ── Urgency ──────────────────────────────────────────────── */}
      <div className="border-b border-slate-800 pb-3 mb-3">
        <SectionHeader>Urgency</SectionHeader>
        {n?.urgent_flag === 1 && (
          <span className="bg-red-600 text-white text-xs px-1.5 py-0.5 rounded font-medium mb-2 inline-block">
            URGENT
          </span>
        )}
        <BarRow
          label="Urgency Score"
          value={n?.urgency_score ?? 0}
          colorClass={urgencyColor(n?.urgency_score ?? 0)}
          rightLabel={safeFixed(n?.urgency_score ?? 0)}
        />
      </div>

      {/* ── Entities ─────────────────────────────────────────────── */}
      <div className="border-b border-slate-800 pb-3 mb-3">
        <SectionHeader>Entities Detected</SectionHeader>
        <div className="flex flex-wrap gap-1">
          {detected.length === 0 ? (
            <span className="text-slate-500 text-xs italic">None detected</span>
          ) : (
            detected.map(([, label]) => (
              <span key={label} className="bg-slate-700 text-slate-300 text-xs px-2 py-0.5 rounded-full">
                {label}
              </span>
            ))
          )}
        </div>
      </div>

      {/* ── Missing Fields ───────────────────────────────────────── */}
      <div className="border-b border-slate-800 pb-3 mb-3">
        <SectionHeader>Missing Fields</SectionHeader>
        <p className="text-slate-500 text-xs mb-1">{n?.missing_count ?? 0} field(s) missing</p>
        <div className="flex flex-wrap gap-1">
          {missing.length === 0 ? (
            <span className="text-slate-500 text-xs italic">All fields present</span>
          ) : (
            missing.map(([, label]) => (
              <span key={label} className="bg-red-900 text-red-300 text-xs px-2 py-0.5 rounded-full">
                {label}
              </span>
            ))
          )}
        </div>
      </div>

      {/* ── Sentiment ────────────────────────────────────────────── */}
      <div className="border-b border-slate-800 pb-3 mb-3">
        <SectionHeader>Sentiment</SectionHeader>
        <div className="flex items-center justify-between">
          <span className={`text-sm font-medium capitalize ${sentimentColor(n?.sentiment_label)}`}>
            {n?.sentiment_label ?? 'neutral'}
          </span>
          <span className="text-slate-500 text-xs">{safeFixed(n?.sentiment_score ?? 0)}</span>
        </div>
      </div>

      {/* ── Frustration ──────────────────────────────────────────── */}
      <div className="border-b border-slate-800 pb-3 mb-3">
        <SectionHeader>Frustration</SectionHeader>
        <BarRow
          label="Frustration Level"
          value={n?.frustration_level ?? 0}
          colorClass={frustrationColor(n?.frustration_level ?? 0)}
          rightLabel={safeFixed(n?.frustration_level ?? 0)}
        />
      </div>

      {/* ── RAG Similarity ───────────────────────────────────────── */}
      <div className="border-b border-slate-800 pb-3 mb-3">
        <SectionHeader>RAG Similarity</SectionHeader>
        <div className="flex items-center justify-between mb-1">
          <span className="text-slate-400 text-xs">Top Match</span>
          <span className="text-white font-mono text-sm">{safeFixed(sim)}</span>
        </div>
        <div className="bg-slate-800 rounded-full h-1">
          <div className="h-1 rounded-full bg-blue-500" style={{ width: `${Math.min(100, sim * 100)}%` }} />
        </div>
        {n?.needs_clarification === 1 && (
          <p className="text-amber-400 text-xs mt-2">⚠ Clarification may be needed</p>
        )}
      </div>

      {/* ── SLA Status ───────────────────────────────────────────── */}
      {s.sla_limit_hours !== undefined && (
        <div className="border-b border-slate-800 pb-3 mb-3">
          <SectionHeader>SLA Status</SectionHeader>
          {(() => {
            const breached      = s.sla_breach_flag === 1
            const remaining     = typeof s.sla_remaining_norm === 'number' ? s.sla_remaining_norm : 1
            const usedPct       = Math.round((1 - remaining) * 100)
            const barColor      = breached || remaining < 0.1
              ? '#ef4444'
              : remaining < 0.3
              ? '#f59e0b'
              : '#22c55e'
            return (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', margin: '4px 0', fontSize: '0.85rem' }}>
                  <span style={{ color: '#94a3b8' }}>SLA Limit</span>
                  <span style={{ fontWeight: 500, color: '#e2e8f0' }}>{s.sla_limit_hours}h</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', margin: '4px 0', fontSize: '0.85rem' }}>
                  <span style={{ color: '#94a3b8' }}>Time Remaining</span>
                  <span style={{ fontWeight: 500, color: barColor }}>
                    {breached ? 'Breached' : `${s.sla_remaining_hours}h`}
                  </span>
                </div>
                <div style={{ width: '100%', height: 6, backgroundColor: '#1e293b', borderRadius: 3, marginTop: 6, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${Math.min(usedPct, 100)}%`, backgroundColor: barColor, borderRadius: 3, transition: 'width 0.4s ease' }} />
                </div>
                <div style={{
                  display: 'inline-block', marginTop: 6, padding: '2px 8px', borderRadius: 4,
                  fontSize: '0.75rem',
                  backgroundColor: breached ? '#7f1d1d' : '#14532d',
                  color: breached ? '#fca5a5' : '#86efac',
                }}>
                  {breached ? '🔴 SLA Breached' : '✅ Within SLA'}
                </div>
              </>
            )
          })()}
        </div>
      )}

      {/* ── Retrieved Knowledge (Fix 8A + 8B) ────────────────────── */}
      <div className="pb-1">
        {sim >= 0.50 && topK.length > 0 ? (
          <>
            <button
              onClick={() => setRagExpanded((x) => !x)}
              className="w-full flex items-center justify-between text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 hover:text-slate-400 transition-colors"
            >
              <span>Retrieved Knowledge</span>
              <span>{ragExpanded ? '▲' : '▼'}</span>
            </button>
            {ragExpanded && (
              <div className="space-y-2">
                {topK.map((item, i) => (
                  <RetrievedItem key={i} item={item} index={i} />
                ))}
              </div>
            )}
          </>
        ) : (
          <p className="text-slate-600 text-xs italic">
            No confident knowledge base match found for this ticket.
            {' '}(Similarity: {safeFixed(sim)})
          </p>
        )}
      </div>
    </div>
  )
}
