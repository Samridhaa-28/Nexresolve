import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { loginUser } from '../services/api.js'
import { useAuth } from '../context/AuthContext.jsx'

export default function LoginPage() {
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [error, setError]       = useState('')
  const [loading, setLoading]   = useState(false)
  const { login }  = useAuth()
  const navigate   = useNavigate()

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const res = await loginUser(email, password)
      login(res.data.access_token, res.data.username)
      navigate('/dashboard')
    } catch (err) {
      setError(err.response?.data?.detail ?? 'Login failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="gradient-bg min-h-screen flex flex-col items-center justify-center px-4">

      {/* ── Centered branding (Fix 9A) ─────────────────────────── */}
      <div className="text-center mb-8">
        <h1 style={{
          fontSize: '2.8rem',
          fontWeight: '700',
          letterSpacing: '0.04em',
          color: '#e2e8f0',
          marginBottom: '0.4rem',
          lineHeight: 1.1,
        }}>
          NexResolve
        </h1>
        <p style={{
          fontSize: '1rem',
          color: '#94a3b8',
          letterSpacing: '0.08em',
          fontStyle: 'italic',
        }}>
          Intelligent support, resolved at the next level.
        </p>
      </div>

      {/* ── Form card ─────────────────────────────────────────── */}
      <div className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl p-8 w-full max-w-md">
        <p className="text-slate-300 font-semibold text-lg mb-1">Sign in</p>
        <p className="text-slate-500 text-sm mb-6">Welcome back — enter your credentials below.</p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full bg-slate-900 border border-slate-600 text-white placeholder-slate-500 rounded-lg px-3 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
              placeholder="you@example.com"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full bg-slate-900 border border-slate-600 text-white placeholder-slate-500 rounded-lg px-3 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
              placeholder="••••••••"
            />
          </div>

          {error && <p className="text-red-400 text-sm">{error}</p>}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white rounded-lg py-2.5 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Signing in…' : 'Sign In'}
          </button>
        </form>

        <p className="text-center mt-4">
          <Link to="/register" className="text-blue-400 text-sm hover:underline">
            Don't have an account? Register
          </Link>
        </p>
      </div>
    </div>
  )
}
