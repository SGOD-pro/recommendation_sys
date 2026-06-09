import { NavLink, Outlet } from 'react-router-dom'
import {
  Home, BarChart3, Users, Star, Film, Menu, X, Cpu
} from 'lucide-react'
import { useState } from 'react'

const NAV = [
  { to: '/',           label: 'Home',        icon: Home },
  { to: '/analytics',  label: 'Analytics',   icon: BarChart3 },
  { to: '/engagement', label: 'Engagement',  icon: Users },
  { to: '/quality',    label: 'Quality',     icon: Star },
  { to: '/recommend',  label: 'Recommend',   icon: Film },
]

export default function Layout() {
  const [open, setOpen] = useState(false)

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className={`fixed inset-y-0 left-0 z-50 w-64 glass border-r border-white/5 flex flex-col transition-transform duration-300 ${open ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0`}>
        {/* Logo */}
        <div className="flex items-center gap-3 p-6 border-b border-white/5">
          <div className="w-9 h-9 rounded-xl bg-indigo-500/20 flex items-center justify-center animate-pulse-glow">
            <Cpu size={18} className="text-indigo-400" />
          </div>
          <div>
            <p className="font-bold text-sm text-white leading-tight">Hybrid AI</p>
            <p className="text-xs text-slate-500">Recommender</p>
          </div>
        </div>

        {/* Nav links */}
        <nav className="flex-1 p-4 space-y-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200
                 ${isActive
                   ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                   : 'text-slate-400 hover:text-white hover:bg-white/5'
                 }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-white/5 text-xs text-slate-600 text-center">
          MovieLens 100K · SVD + TF-IDF
        </div>
      </aside>

      {/* Mobile topbar */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-40 glass border-b border-white/5 flex items-center justify-between px-4 h-14">
        <div className="flex items-center gap-2">
          <Cpu size={16} className="text-indigo-400" />
          <span className="font-bold text-sm">Hybrid AI Rec</span>
        </div>
        <button onClick={() => setOpen(!open)} className="p-2 text-slate-400 hover:text-white">
          {open ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {/* Backdrop */}
      {open && <div className="fixed inset-0 z-40 bg-black/60 lg:hidden" onClick={() => setOpen(false)} />}

      {/* Main content */}
      <main className="flex-1 lg:ml-64 pt-14 lg:pt-0 min-h-screen">
        <div className="p-6 max-w-7xl mx-auto">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
