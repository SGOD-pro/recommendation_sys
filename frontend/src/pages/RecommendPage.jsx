import { useState } from 'react'
import { api } from '../api'
import { Spinner, ErrorBox } from '../components/ui'
import { Film, Search, Star, Layers, Zap, TrendingUp, ChevronRight } from 'lucide-react'

const METHOD_COLOR = {
  hybrid:     { bg: 'bg-indigo-500/15', text: 'text-indigo-300', border: 'border-indigo-500/30' },
  popularity: { bg: 'bg-amber-500/15',  text: 'text-amber-300',  border: 'border-amber-500/30' },
}

function ScoreBar({ label, value, color }) {
  const colorMap = { cf: '#6366f1', cb: '#22d3ee', pop: '#f59e0b', final: '#10b981' }
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-slate-500 w-12 flex-shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700"
          style={{ width: `${Math.round(value * 100)}%`, background: colorMap[color] }} />
      </div>
      <span className="text-xs font-mono" style={{ color: colorMap[color] }}>{value?.toFixed(3)}</span>
    </div>
  )
}

function MovieCard({ rec, rank }) {
  const [expanded, setExpanded] = useState(false)
  const mc = METHOD_COLOR[rec.method?.split(' ')[0]] ?? METHOD_COLOR.hybrid

  const genres = rec.genres?.split('|') ?? []

  return (
    <div className="glass glass-hover rounded-2xl p-5 cursor-pointer" onClick={() => setExpanded(e => !e)}>
      <div className="flex items-start justify-between gap-3">
        {/* Rank badge */}
        <div className="w-8 h-8 rounded-xl bg-indigo-500/15 flex items-center justify-center flex-shrink-0 text-sm font-black text-indigo-300">
          {rank}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2 mb-1">
            <p className="text-sm font-semibold text-white leading-snug">{rec.title}</p>
            <span className={`badge flex-shrink-0 ${mc.bg} ${mc.text} border ${mc.border}`}>
              {rec.method?.includes('hybrid') ? '⚡ Hybrid' : '🔥 Popular'}
            </span>
          </div>

          {/* Genres */}
          <div className="flex flex-wrap gap-1 mb-2">
            {genres.map(g => (
              <span key={g} className="badge bg-white/5 text-slate-400 border border-white/10">{g}</span>
            ))}
          </div>

          {/* Score */}
          <div className="flex items-center gap-2">
            <Star size={12} className="text-amber-400" />
            <span className="text-xs text-amber-400 font-bold">{(rec.score * 100).toFixed(1)}</span>
            <span className="text-xs text-slate-600">score</span>
          </div>
        </div>

        <ChevronRight size={14} className={`text-slate-500 flex-shrink-0 transition-transform ${expanded ? 'rotate-90' : ''}`} />
      </div>

      {/* Expanded score breakdown */}
      {expanded && rec.cf_score !== undefined && (
        <div className="mt-4 pt-4 border-t border-white/5 space-y-1.5">
          <p className="text-xs text-slate-500 mb-2">Score Breakdown</p>
          <ScoreBar label="CF"    value={rec.cf_score}  color="cf" />
          <ScoreBar label="CB"    value={rec.cb_score}  color="cb" />
          <ScoreBar label="Pop"   value={rec.pop_score} color="pop" />
          <ScoreBar label="Final" value={rec.score}     color="final" />
        </div>
      )}
    </div>
  )
}

export default function RecommendPage() {
  const [userId, setUserId] = useState('')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [popular, setPopular] = useState(null)
  const [popLoading, setPopLoading] = useState(false)

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!userId.trim()) return
    setLoading(true); setError(null); setResults(null)
    try {
      const data = await api.recommend(parseInt(userId), 10)
      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const loadPopular = async () => {
    setPopLoading(true)
    try {
      const data = await api.popular(10)
      setPopular(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setPopLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="page-title mb-1">Recommendations</h1>
        <p className="text-slate-500 text-sm">Enter a User ID (1–610) to get personalised movie recommendations</p>
      </div>

      {/* Search form */}
      <form onSubmit={handleSearch} className="glass rounded-2xl p-6">
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="flex-1 relative">
            <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="number" min="1" max="610" value={userId}
              onChange={e => setUserId(e.target.value)}
              placeholder="Enter User ID (1 – 610)"
              className="w-full pl-10 pr-4 py-3 rounded-xl bg-white/5 border border-white/10
                         text-white placeholder-slate-600 text-sm
                         focus:outline-none focus:border-indigo-500/50 focus:bg-white/8
                         transition-all duration-200"
            />
          </div>
          <button type="submit" disabled={loading}
            className="px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50
                       text-white text-sm font-semibold transition-all duration-200
                       flex items-center gap-2 justify-center">
            <Zap size={14} />
            {loading ? 'Getting Recs…' : 'Recommend'}
          </button>
          <button type="button" onClick={loadPopular} disabled={popLoading}
            className="px-6 py-3 rounded-xl glass border border-white/10 hover:border-indigo-500/30
                       text-slate-300 text-sm font-medium transition-all duration-200
                       flex items-center gap-2 justify-center">
            <TrendingUp size={14} />
            Popular
          </button>
        </div>

        {/* Quick-pick IDs */}
        <div className="flex flex-wrap gap-2 mt-4">
          <span className="text-xs text-slate-600">Quick:</span>
          {[1, 15, 100, 200, 414, 599, 610].map(id => (
            <button key={id} type="button" onClick={() => setUserId(String(id))}
              className={`text-xs px-3 py-1 rounded-lg transition-colors
                ${String(id) === userId
                  ? 'bg-indigo-500/30 text-indigo-300 border border-indigo-500/40'
                  : 'bg-white/5 text-slate-400 border border-white/8 hover:border-indigo-500/30 hover:text-indigo-300'
                }`}>
              {id}
            </button>
          ))}
        </div>
      </form>

      {/* Results */}
      {loading && <Spinner />}
      {error && <ErrorBox message={error} />}

      {results && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Film size={18} className="text-indigo-400" />
              Top {results.count} for User {results.user_id}
            </h2>
            <span className="badge bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
              {results.recommendations[0]?.method?.includes('hybrid') ? '⚡ Hybrid' : '🔥 Popularity'}
            </span>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {results.recommendations.map((rec, i) => (
              <MovieCard key={rec.movieId} rec={rec} rank={i + 1} />
            ))}
          </div>
        </div>
      )}

      {/* Popular movies */}
      {popLoading && <Spinner />}
      {popular && !results && (
        <div className="space-y-4">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <TrendingUp size={18} className="text-amber-400" />
            Most Popular Movies
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {popular.movies.map((m, i) => (
              <div key={m.movieId} className="glass glass-hover rounded-2xl p-5">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-xl bg-amber-500/15 flex items-center justify-center flex-shrink-0 text-sm font-black text-amber-300">
                    {i + 1}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-white mb-1 truncate">{m.title}</p>
                    <div className="flex flex-wrap gap-1 mb-2">
                      {m.genres?.split('|').map(g => (
                        <span key={g} className="badge bg-white/5 text-slate-400 border border-white/10">{g}</span>
                      ))}
                    </div>
                    <div className="flex items-center gap-3 text-xs text-slate-500">
                      <span className="flex items-center gap-1"><Star size={11} className="text-amber-400" />{m.avg_rating?.toFixed(2)}</span>
                      <span>{m.num_ratings?.toLocaleString()} ratings</span>
                      <span className="text-indigo-400 font-mono">score {m.popularity_score?.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Info cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[
          { icon: Zap,      label: 'Hybrid Engine', desc: '0.6×CF + 0.3×CB + 0.1×Popularity gives personalised scores', color: 'text-indigo-400' },
          { icon: Layers,   label: 'Cold-Start',    desc: 'Users with no history get popularity-based recommendations', color: 'text-amber-400' },
          { icon: Film,     label: 'Click to Expand', desc: 'Click any movie card to see the CF / CB / Popularity score breakdown', color: 'text-cyan-400' },
        ].map(({ icon: Icon, label, desc, color }) => (
          <div key={label} className="glass rounded-2xl p-5">
            <Icon size={16} className={`${color} mb-2`} />
            <p className="text-sm font-semibold text-white mb-1">{label}</p>
            <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
