// Shared UI helpers

export function Spinner() {
  return (
    <div className="flex items-center justify-center h-40">
      <div className="w-10 h-10 border-2 border-indigo-400/30 border-t-indigo-400 rounded-full animate-spin" />
    </div>
  )
}

export function ErrorBox({ message }) {
  return (
    <div className="glass rounded-2xl p-6 border border-red-500/30 text-red-400 text-sm text-center">
      ⚠ {message}
    </div>
  )
}

const ICON_BG = {
  indigo:  'text-indigo-400 bg-indigo-500/10',
  cyan:    'text-cyan-400 bg-cyan-500/10',
  emerald: 'text-emerald-400 bg-emerald-500/10',
  amber:   'text-amber-400 bg-amber-500/10',
  rose:    'text-rose-400 bg-rose-500/10',
}

export function StatCard({ icon: Icon, label, value, sub, color = 'indigo' }) {
  const cls = ICON_BG[color] ?? ICON_BG.indigo
  return (
    <div className="stat-card animate-fade-in-up">
      <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${cls}`}>
        <Icon size={18} />
      </div>
      <p className="text-2xl font-bold text-white mt-2">{value}</p>
      <p className="text-sm font-medium text-slate-300">{label}</p>
      {sub && <p className="text-xs text-slate-500">{sub}</p>}
    </div>
  )
}

export function ChartCard({ title, children, className = '' }) {
  return (
    <div className={`chart-card animate-fade-in-up ${className}`}>
      <p className="section-title">{title}</p>
      {children}
    </div>
  )
}
