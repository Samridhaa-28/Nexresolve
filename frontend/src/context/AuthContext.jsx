import React, { createContext, useContext, useState, useEffect } from 'react'
import axios from 'axios'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const token = localStorage.getItem('nx_token')
    const username = localStorage.getItem('nx_username')

    if (token && username) {
      // Verify token is still valid
      axios
        .get('/auth/me', { headers: { Authorization: `Bearer ${token}` } })
        .then(() => {
          setUser({ token, username })
        })
        .catch(() => {
          localStorage.removeItem('nx_token')
          localStorage.removeItem('nx_username')
          setUser(null)
        })
        .finally(() => setLoading(false))
    } else {
      setLoading(false)
    }
  }, [])

  function login(token, username) {
    localStorage.setItem('nx_token', token)
    localStorage.setItem('nx_username', username)
    setUser({ token, username })
  }

  function logout() {
    localStorage.removeItem('nx_token')
    localStorage.removeItem('nx_username')
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
