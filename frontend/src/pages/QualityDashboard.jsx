import { useEffect, useState } from 'react'
import { api } from '../api'
import { Spinner, ErrorBox, StatCard, ChartCard } from '../components/ui'
import { Target, TrendingUp, Award, Users } from 'lucide-react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell
} from 'recharts'

const TIP = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass-dark rounded-xl px-3 py-2 text-xs border border-white/10">
      <p className="text-slate-300 font-medium mb-1">{label}</p>
      {payload.map((p, i) => <p key={i} style={{ color: p.color }}>{p.name}: <b>{p.value}</b></p>)}
    </div>
  )
}

function MetricGauge({ label, value, color, max = 1, description }) {
  const pct = Math.min((value / max) * 100, 100)
  const colorMap = { indigo: '#6366f1', cyan: '#22d3ee', emerald: '#10b981', amber: '#f59e0b' }
  const hex = colorMap[color] ?? '#6366f1'

  return (
    <div className="glass glass-hover rounded-2xl p-6 text-center">
      {/* Circular gauge */}
      <div className="relative w-24 h-24 mx-auto mb-4">
        <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
          <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="10" />
          <circle cx="50" cy="50" r="40" fill="none" stroke={hex} strokeWidth="10"
            strokeDasharray={`${2 * Math.PI * 40}`}
            strokeDashoffset={`${2 * Math.PI * 40 * (1 - pct / 100)}`}
            strokeLinecap="round" style={{ transition: 'stroke-dashoffset 1.2s ease' }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xl font-black" style={{ color: hex }}>{(value * 100).toFixed(1)}%</span>
        </div>
      </div>
      <p className="text-sm font-bold text-white mb-1">{label}</p>
      <p className="text-xs text-slate-500 leading-relaxed">{description}</p>
    </div>
  )
}

export default function QualityDashboard() {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.metrics()
      .then(setMetrics)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Spinner />
  if (error) return <ErrorBox message={error} />

  const { precision_at_k, recall_at_k, ndcg_at_k, k, n_users_evaluated } = metrics

  const radarData = [
    { metric: `Precision@${k}`, value: precision_at_k },
    { metric: `Recall@${k}`,    value: recall_at_k },
    { metric: `NDCG@${k}`,      value: ndcg_at_k },
  ]

  const barData = [
    { name: `Precision@${k}`, value: +(precision_at_k * 100).toFixed(2), fill: '#6366f1' },
    { name: `Recall@${k}`,    value: +(recall_at_k    * 100).toFixed(2), fill: '#22d3ee' },
    { name: `NDCG@${k}`,      value: +(ndcg_at_k      * 100).toFixed(2), fill: '#10b981' },
  ]

  return (
    <div className="space-y-8">
      <div>
        <h1 className="page-title mb-1">Recommendation Quality</h1>
        <p className="text-slate-500 text-sm">
          Evaluated on {n_users_evaluated} users · K = {k} · Holdout test set
        </p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={Target}     label={`Precision@${k}`} value={`${(precision_at_k*100).toFixed(1)}%`} color="indigo"
                  sub="Relevant in top-K" />
        <StatCard icon={TrendingUp} label={`Recall@${k}`}    value={`${(recall_at_k*100).toFixed(1)}%`}    color="cyan"
                  sub="Coverage of relevant" />
        <StatCard icon={Award}      label={`NDCG@${k}`}      value={`${(ndcg_at_k*100).toFixed(1)}%`}      color="emerald"
                  sub="Ranked quality" />
        <StatCard icon={Users}      label="Users Evaluated"  value={n_users_evaluated}                      color="amber"
                  sub={`K = ${k} recommendations`} />
      </div>

      {/* Gauges */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        <MetricGauge label={`Precision@${k}`} value={precision_at_k} color="indigo"
          description="Fraction of recommended movies that were actually relevant to the user." />
        <MetricGauge label={`Recall@${k}`}    value={recall_at_k}    color="cyan"
          description="Fraction of relevant movies that were successfully recommended in top-K." />
        <MetricGauge label={`NDCG@${k}`}      value={ndcg_at_k}      color="emerald"
          description="Discounted Cumulative Gain – rewards ranking relevant items higher." />
      </div>

      {/* Bar chart */}
      <ChartCard title="Metric Comparison (%)">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={barData} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 12 }} />
            <YAxis domain={[0, Math.max(...barData.map(d => d.value)) * 1.3]}
                   tick={{ fill: '#64748b', fontSize: 11 }} unit="%" />
            <Tooltip content={<TIP />} />
            {barData.map((d, i) => (
              <Bar key={d.name} dataKey="value" name={d.name} fill={d.fill} radius={[6,6,0,0]} hide={i > 0} />
            ))}
            <Bar dataKey="value" name="Score" radius={[6,6,0,0]}>
              {barData.map((d, i) => <Cell key={i} fill={d.fill} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Radar */}
      <ChartCard title="Quality Radar">
        <ResponsiveContainer width="100%" height={260}>
          <RadarChart data={radarData} cx="50%" cy="50%" outerRadius={90}>
            <PolarGrid stroke="rgba(255,255,255,0.08)" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <Radar dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.25} strokeWidth={2} />
          </RadarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Metric explanation */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[
          { title: `Precision@${k}`, formula: `|Relevant ∩ Recommended| / K`, color: '#6366f1',
            desc: 'Measures exactness. High precision = fewer irrelevant recommendations.' },
          { title: `Recall@${k}`,    formula: `|Relevant ∩ Recommended| / |Relevant|`, color: '#22d3ee',
            desc: 'Measures completeness. High recall = few relevant items missed.' },
          { title: `NDCG@${k}`,      formula: `DCG@K / IDCG@K`, color: '#10b981',
            desc: 'Ranking-aware metric. Items at the top of the list receive higher weights.' },
        ].map(({ title, formula, color, desc }) => (
          <div key={title} className="glass rounded-2xl p-5">
            <p className="text-sm font-bold text-white mb-1">{title}</p>
            <code className="text-xs font-mono px-2 py-1 rounded-lg bg-black/30 block mb-2" style={{ color }}>
              {formula}
            </code>
            <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
