import { useEffect, useState } from 'react'
import { api } from '../api'
import { Spinner, ErrorBox, StatCard, ChartCard } from '../components/ui'
import { Users, Layers, Activity, TrendingUp } from 'lucide-react'
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, RadarChart,
  Radar, PolarGrid, PolarAngleAxis
} from 'recharts'

const COLORS  = ['#6366f1','#22d3ee','#10b981','#f59e0b','#f43f5e']
const CLUSTER_ICONS = ['🎬','🚀','🎭','😄','💥']

const TIP = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass rounded-xl px-3 py-2 text-xs border border-white/10">
      <p className="text-slate-300 font-medium mb-1">{label}</p>
      {payload.map((p, i) => <p key={i} style={{ color: p.color }}>{p.name}: <b>{p.value?.toLocaleString?.() ?? p.value}</b></p>)}
    </div>
  )
}

export default function EngagementDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.clusters()
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Spinner />
  if (error) return <ErrorBox message={error} />

  const { clusters, total_users, cluster_distribution, activity_distribution } = data

  const clusterPie = clusters.map((c, i) => ({
    name: c.name, value: c.size, color: COLORS[i], pct: c.percentage,
  }))

  const activityBar = (activity_distribution?.labels ?? ['Low','Medium','High','Very High'])
    .map((l, i) => ({ level: l, users: activity_distribution?.values?.[i] ?? 0 }))

  const radarData = clusters.map(c => ({
    cluster: c.name.split(' ')[0],
    avgRating: c.avg_rating,
    activity: Math.min(c.avg_num_ratings / 20, 5),
    size: c.size / (total_users / 5),
  }))

  return (
    <div className="space-y-8">
      <div>
        <h1 className="page-title mb-1">User Engagement</h1>
        <p className="text-slate-500 text-sm">KMeans user segmentation · Activity patterns</p>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={Users}     label="Total Users"    value={total_users.toLocaleString()} color="indigo" />
        <StatCard icon={Layers}    label="Clusters"       value={clusters.length}              color="cyan" />
        <StatCard icon={Activity}  label="Avg Ratings/User" value={Math.round(clusters.reduce((s, c) => s + c.avg_num_ratings * c.size, 0) / total_users)} color="emerald" />
        <StatCard icon={TrendingUp} label="Avg Rating"    value={(clusters.reduce((s, c) => s + c.avg_rating * c.size, 0) / total_users).toFixed(2)} color="amber" />
      </div>

      {/* Cluster cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        {clusters.map((c, i) => (
          <div key={c.cluster_id} className="glass glass-hover rounded-2xl p-5">
            <div className="text-2xl mb-2">{CLUSTER_ICONS[i]}</div>
            <p className="text-sm font-bold text-white mb-1">{c.name}</p>
            <p className="text-xs text-slate-500 mb-3">{c.top_genre} fans</p>
            <div className="space-y-1.5">
              <Row label="Users"      val={`${c.size.toLocaleString()} (${c.percentage}%)`} color={COLORS[i]} />
              <Row label="Avg rating" val={c.avg_rating}                                     color={COLORS[i]} />
              <Row label="Avg count"  val={`${c.avg_num_ratings} ratings`}                   color={COLORS[i]} />
            </div>
            {/* Mini progress bar */}
            <div className="mt-3 h-1 rounded-full bg-white/5">
              <div className="h-1 rounded-full transition-all duration-700"
                style={{ width: `${c.percentage}%`, background: COLORS[i] }} />
            </div>
          </div>
        ))}
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cluster pie */}
        <ChartCard title="Cluster Distribution">
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={clusterPie} dataKey="value" nameKey="name"
                   cx="50%" cy="50%" outerRadius={90} innerRadius={45} paddingAngle={3}
                   label={({ name, pct }) => `${name.split(' ')[0]} ${pct}%`}
                   labelLine={false}>
                {clusterPie.map((c, i) => <Cell key={i} fill={c.color} />)}
              </Pie>
              <Tooltip content={<TIP />} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Activity distribution */}
        <ChartCard title="Activity Level Distribution">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={activityBar} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="level" tick={{ fill: '#64748b', fontSize: 12 }} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip content={<TIP />} />
              <Bar dataKey="users" name="Users" radius={[6,6,0,0]}>
                {activityBar.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Cluster comparison bar */}
      <ChartCard title="Cluster Comparison – Avg Rating vs Avg Activity">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={clusters.map(c => ({
            name: c.name.split(' ')[0],
            'Avg Rating': c.avg_rating,
            'Avg Ratings/User': Math.round(c.avg_num_ratings),
          }))} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 12 }} />
            <YAxis yAxisId="left"  orientation="left"  tick={{ fill: '#6366f1', fontSize: 11 }} />
            <YAxis yAxisId="right" orientation="right" tick={{ fill: '#22d3ee', fontSize: 11 }} />
            <Tooltip content={<TIP />} />
            <Bar yAxisId="left"  dataKey="Avg Rating"       fill="#6366f1" radius={[4,4,0,0]} />
            <Bar yAxisId="right" dataKey="Avg Ratings/User" fill="#22d3ee" radius={[4,4,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  )
}

function Row({ label, val, color }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-slate-500">{label}</span>
      <span className="text-xs font-medium" style={{ color }}>{val}</span>
    </div>
  )
}
