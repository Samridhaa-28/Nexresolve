import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext.jsx'
import { getHistory, deleteTicket } from '../services/api.js'
import Navbar from '../components/Navbar.jsx'
import TicketCard from '../components/TicketCard.jsx'

function SkeletonCard() {
  return (
    <div className="animate-pulse bg-slate-800 rounded-xl h-24 mb-3 border border-slate-700" />
  )
}

export default function DashboardPage() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [tickets, setTickets] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    document.title = 'NexResolve — Dashboard'
    getHistory()
      .then((res) => setTickets(res.data ?? []))
      .catch(() => setError('Failed to load tickets. Please try again.'))
      .finally(() => setLoading(false))
  }, [])

  async function handleDelete(ticketId) {
    // Optimistic update — remove immediately
    setTickets((prev) => prev.filter((t) => t.ticket_id !== ticketId))
    try {
      await deleteTicket(ticketId)
    } catch {
      // If delete fails, re-fetch to restore correct state
      getHistory()
        .then((res) => setTickets(res.data ?? []))
        .catch(() => {})
    }
  }

  return (
    <div className="bg-slate-950 min-h-screen pt-14">
      <Navbar />

      <div className="max-w-3xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-2xl font-semibold text-white">
            Welcome back, {user?.username} 👋
          </h1>
          <button
            onClick={() => navigate('/ticket/new')}
            className="bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg px-4 py-2 transition-colors"
          >
            + New Ticket
          </button>
        </div>

        {/* Ticket list */}
        <h2 className="text-lg font-semibold text-slate-300 mb-4">Your Tickets</h2>

        {loading && (
          <>
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </>
        )}

        {!loading && error && (
          <p className="text-red-400 text-sm">{error}</p>
        )}

        {!loading && !error && tickets.length === 0 && (
          <div className="flex flex-col items-center justify-center py-24 text-slate-500">
            <span className="text-4xl mb-4">🎫</span>
            <p>No tickets yet. Submit your first one 🎫</p>
          </div>
        )}

        {!loading && !error && tickets.map((ticket) => (
          <TicketCard key={ticket.ticket_id} ticket={ticket} onDelete={handleDelete} />
        ))}
      </div>
    </div>
  )
}
