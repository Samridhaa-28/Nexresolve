import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000',
})

// Attach token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('nx_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// On 401 clear auth and redirect to login
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('nx_token')
      localStorage.removeItem('nx_username')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export function registerUser(username, email, password) {
  return api.post('/auth/register', { username, email, password })
}

export function loginUser(email, password) {
  return api.post('/auth/login', { email, password })
}

export function getMe() {
  return api.get('/auth/me')
}

/**
 * Submit a ticket (or a follow-up reply to an existing ticket).
 * @param {string} text        - Message text
 * @param {string|null} ticketId - Pass on follow-up turns; omit on first turn
 */
export function submitTicket(text, ticketId = null) {
  const body = { text }
  if (ticketId) body.ticket_id = ticketId
  return api.post('/ticket/new', body)
}

export function getHistory() {
  return api.get('/history/me')
}

export function getTicket(ticketId) {
  return api.get(`/ticket/${ticketId}`)
}

export function deleteTicket(ticketId) {
  return api.delete(`/ticket/${ticketId}`)
}

export default api
