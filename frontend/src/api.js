// Centralised API calls to the FastAPI backend

const BASE = '/api'

async function get(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`)
  return res.json()
}

export const api = {
  analytics:              () => get('/analytics'),
  popular:                (n = 10) => get(`/popular?n=${n}`),
  clusters:               () => get('/clusters'),
  metrics:                () => get('/metrics'),
  recommend:              (userId, n = 10) => get(`/recommend/${userId}?n=${n}`),
  movie:                  (movieId) => get(`/movie/${movieId}`),
}
