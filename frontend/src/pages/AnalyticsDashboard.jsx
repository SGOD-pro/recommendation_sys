import { useEffect, useState } from 'react'
import { api } from '../api'
import { Spinner, ErrorBox, StatCard, ChartCard } from '../components/ui'
import { Users, Film, Star, TrendingUp } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Legend
} from 'recharts'

const COLORS = ['#6366f1','#22d3ee','#10b981','#f59e0b','#f43f5e','#8b5cf6','#ec4899','#14b8a6','#f97316','#84cc16']

const TIP = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass-dark rounded-xl px-3 py-2 text-xs border border-white/10">
      <p className="text-slate-300 font-medium mb-1">{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color }}>{p.name}: <b>{typeof p.value === 'number' ? p.value.toLocaleString() : p.value}</b></p>
      ))}
    </div>
  )
}

export default function AnalyticsDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.analytics()
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Spinner />
  if (error) return <ErrorBox message={error} />

  const { dataset_stats, user_stats, movie_stats, genre_stats, rating_distribution } = data

  // Rating distribution chart data
  const ratingData = rating_distribution.labels.map((l, i) => ({
    rating: l,
    count: rating_distribution.values[i],
  }))

  // Genre chart data
  const genreData = genre_stats.most_popular_genres.map(g => ({
    name: g.genres.length > 10 ? g.genres.slice(0, 10) : g.genres,
    ratings: g.total_ratings,
    avgRating: g.avg_rating,
  }))

  // Most rated movies
  const mostRated = movie_stats.most_rated_movies.map(m => ({
    name: m.title.replace(/\s*\(\d{4}\)/, '').slice(0, 22),
    ratings: m.num_ratings,
    avg: m.avg_rating,
  }))

  // Highest rated
  const highestRated = movie_stats.highest_rated_movies.map(m => ({
    name: m.title.replace(/\s*\(\d{4}\)/, '').slice(0, 22),
    avg: m.avg_rating,
    ratings: m.num_ratings,
  }))

  return (
    <div className="space-y-8">
      <div>
        <h1 className="page-title mb-1">Analytics Dashboard</h1>
        <p className="text-slate-500 text-sm">MovieLens 100K · Dataset statistics & insights</p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={Users}     label="Total Users"   value={dataset_stats.total_users.toLocaleString()}   color="indigo" />
        <StatCard icon={Film}      label="Total Movies"  value={dataset_stats.total_movies.toLocaleString()}  color="cyan" />
        <StatCard icon={Star}      label="Total Ratings" value={dataset_stats.total_ratings.toLocaleString()} color="amber" />
        <StatCard icon={TrendingUp} label="Avg Rating"   value={dataset_stats.avg_rating}                    color="emerald"
                  sub={`Range ${dataset_stats.rating_range.min}–${dataset_stats.rating_range.max}`} />
      </div>

      {/* Rating distribution */}
      <ChartCard title="Rating Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={ratingData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="rating" tick={{ fill: '#64748b', fontSize: 12 }} />
            <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
            <Tooltip content={<TIP />} />
            <Bar dataKey="count" name="Ratings" radius={[6,6,0,0]}>
              {ratingData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Genre distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard title="Genre Popularity (by rating count)">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={genreData} layout="vertical" margin={{ top: 0, right: 16, left: 60, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} />
              <YAxis dataKey="name" type="category" tick={{ fill: '#94a3b8', fontSize: 11 }} width={60} />
              <Tooltip content={<TIP />} />
              <Bar dataKey="ratings" name="Ratings" radius={[0,4,4,0]}>
                {genreData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Genre Distribution (pie)">
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={genreData} dataKey="ratings" nameKey="name" cx="50%" cy="50%"
                   outerRadius={90} innerRadius={40} paddingAngle={2} label={({ name }) => name}>
                {genreData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip content={<TIP />} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Movie tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard title="Most Rated Movies">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 border-b border-white/5">
                <th className="text-left py-2 font-medium">#</th>
                <th className="text-left py-2 font-medium">Title</th>
                <th className="text-right py-2 font-medium">Ratings</th>
                <th className="text-right py-2 font-medium">Avg</th>
              </tr>
            </thead>
            <tbody>
              {movie_stats.most_rated_movies.map((m, i) => (
                <tr key={i} className="border-b border-white/3 hover:bg-white/3 transition-colors">
                  <td className="py-2 text-slate-500">{i + 1}</td>
                  <td className="py-2 text-slate-200">{m.title.replace(/\s*\(\d{4}\)/, '')}</td>
                  <td className="py-2 text-right text-indigo-300 font-medium">{m.num_ratings.toLocaleString()}</td>
                  <td className="py-2 text-right text-emerald-400">{m.avg_rating}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </ChartCard>

        <ChartCard title="Highest Rated Movies (≥50 ratings)">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 border-b border-white/5">
                <th className="text-left py-2 font-medium">#</th>
                <th className="text-left py-2 font-medium">Title</th>
                <th className="text-right py-2 font-medium">Avg ★</th>
                <th className="text-right py-2 font-medium">Count</th>
              </tr>
            </thead>
            <tbody>
              {movie_stats.highest_rated_movies.map((m, i) => (
                <tr key={i} className="border-b border-white/3 hover:bg-white/3 transition-colors">
                  <td className="py-2 text-slate-500">{i + 1}</td>
                  <td className="py-2 text-slate-200">{m.title.replace(/\s*\(\d{4}\)/, '')}</td>
                  <td className="py-2 text-right text-amber-400 font-bold">{m.avg_rating}</td>
                  <td className="py-2 text-right text-slate-400">{m.num_ratings.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </ChartCard>
      </div>

      {/* Most active users */}
      <ChartCard title="Most Active Users (by rating count)">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={user_stats.most_active_users.map(u => ({
            user: `User ${u.userId}`,
            ratings: u.total_ratings,
            avg: u.average_rating,
          }))} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="user" tick={{ fill: '#64748b', fontSize: 11 }} />
            <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
            <Tooltip content={<TIP />} />
            <Bar dataKey="ratings" name="# Ratings" fill="#6366f1" radius={[4,4,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  )
}
