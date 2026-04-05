import React from 'react'
import { useAuth } from '../context/AuthContext.jsx'
import { useNavigate } from 'react-router-dom'

export default function Navbar() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  function handleLogout() {
    logout()
    navigate('/login')
  }

  return (
    <nav className="fixed top-0 left-0 right-0 h-14 bg-slate-900 z-50 flex items-center justify-between px-6 shadow-sm">
      <span className="text-blue-400 font-bold text-xl">🎫 NexResolve</span>
      <div className="flex items-center gap-3">
        {user && (
          <>
            <span className="text-slate-300 text-sm mr-1">{user.username}</span>
            <button
              onClick={handleLogout}
              className="text-slate-400 hover:text-white text-sm transition-colors"
            >
              Logout
            </button>
          </>
        )}
      </div>
    </nav>
  )
}
