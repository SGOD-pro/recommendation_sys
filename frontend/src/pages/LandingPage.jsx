import { Cpu, Code2, Mail, User, Brain, Layers, Database, BarChart3 } from 'lucide-react'

const MODULES = [
  { num: '01', title: 'Data Processing',       desc: 'Merge & clean MovieLens 100K dataset' },
  { num: '02', title: 'Interaction Analysis',  desc: 'Genre, user & movie statistics' },
  { num: '03', title: 'Train / Val / Test',    desc: '3-way user-wise stratified split' },
  { num: '04', title: 'Popularity Baseline',   desc: 'Cold-start score = avg_rating × log(n)' },
  { num: '05', title: 'Collaborative Filter',  desc: 'SVD matrix factorization (Surprise)' },
  { num: '06', title: 'Content-Based Filter',  desc: 'TF-IDF + cosine similarity on genres' },
  { num: '07', title: 'Hybrid Engine',         desc: '0.6×CF + 0.3×CB + 0.1×Popularity' },
  { num: '08', title: 'Tie-Breaking',          desc: 'avg_rating → num_ratings tiebreak' },
  { num: '09', title: 'Evaluation',            desc: 'Precision@10 · Recall@10 · NDCG@10' },
  { num: '10', title: 'User Clustering',       desc: 'KMeans segmentation (5 clusters)' },
  { num: '11', title: 'FastAPI Backend',       desc: '6 REST endpoints, lifespan startup' },
  { num: '12', title: 'Dashboard',             desc: 'React + TailwindCSS v4 + Recharts' },
]

export default function LandingPage() {
  return (
    <div className="space-y-12 animate-fade-in-up">
      {/* ── Hero ─────────────────────────────────── */}
      <section className="relative rounded-3xl overflow-hidden p-10 text-center"
        style={{ background: 'linear-gradient(135deg,#0f0f2e 0%,#1a1a4e 50%,#0d1a3e 100%)' }}>
        {/* Glow orbs */}
        <div className="absolute -top-20 -left-20 w-64 h-64 bg-indigo-600/20 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-20 -right-20 w-64 h-64 bg-cyan-600/15 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10">
          {/* Icon badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full glass border border-indigo-500/30 text-xs text-indigo-300 font-medium mb-6">
            <Cpu size={12} />
            AI Internship Project · MovieLens 100K
          </div>

          <h1 className="text-4xl md:text-5xl font-black tracking-tight mb-4">
            <span className="gradient-text">Hybrid AI</span><br />
            <span className="text-white">Recommendation System</span>
          </h1>

          <p className="text-slate-400 max-w-xl mx-auto text-base leading-relaxed mb-8">
            Collaborative Filtering (SVD) · Content-Based (TF-IDF) · Popularity Baseline
            combined in a weighted hybrid engine for personalised movie recommendations.
          </p>

          {/* CTA chips */}
          <div className="flex flex-wrap items-center justify-center gap-3">
            {['Precision@10', 'Recall@10', 'NDCG@10', '610 Users', '9.7K Movies', '100K Ratings'].map(tag => (
              <span key={tag} className="badge bg-white/5 text-slate-300 border border-white/10">{tag}</span>
            ))}
          </div>
        </div>
      </section>

      {/* ── Student Card ─────────────────────────── */}
      <section className="gradient-border glass rounded-3xl p-8 max-w-lg mx-auto text-center">
        <div className="w-16 h-16 rounded-2xl bg-indigo-500/20 flex items-center justify-center mx-auto mb-4 animate-pulse-glow">
          <User size={28} className="text-indigo-300" />
        </div>
        <h2 className="text-xl font-bold text-white mb-1">Internship Submission</h2>
        <p className="text-slate-500 text-sm mb-6">AI / ML Track</p>

        <div className="space-y-3 text-left">
          <InfoRow icon={User}   label="Full Name"      value="[Your Full Name]" />
          <InfoRow icon={Mail}   label="Registered Email ID" value="[Your Registered Email ID]" />
          <InfoRow icon={Code2}  label="Project Topic"  value="AI-Based Recommendation System" />
        </div>
      </section>

      {/* ── Tech Stack ───────────────────────────── */}
      <section>
        <h2 className="text-xl font-bold text-white mb-6 text-center">Technology Stack</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { icon: Database, label: 'Data',    items: ['Pandas', 'NumPy', 'MovieLens CSV'] },
            { icon: Brain,    label: 'ML',       items: ['Surprise SVD', 'TF-IDF', 'KMeans'] },
            { icon: Layers,   label: 'Backend',  items: ['FastAPI', 'Uvicorn', 'Python 3.14'] },
            { icon: BarChart3, label: 'Frontend', items: ['React', 'Tailwind v4', 'Recharts'] },
          ].map(({ icon: Icon, label, items }) => (
            <div key={label} className="glass glass-hover rounded-2xl p-5">
              <div className="flex items-center gap-2 mb-3">
                <Icon size={16} className="text-indigo-400" />
                <span className="text-sm font-semibold text-slate-200">{label}</span>
              </div>
              <ul className="space-y-1">
                {items.map(i => (
                  <li key={i} className="text-xs text-slate-500 flex items-center gap-1.5">
                    <span className="w-1 h-1 rounded-full bg-indigo-400 flex-shrink-0" />
                    {i}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {/* ── Modules Grid ─────────────────────────── */}
      <section>
        <h2 className="text-xl font-bold text-white mb-6 text-center">System Modules</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {MODULES.map(({ num, title, desc }) => (
            <div key={num} className="glass glass-hover rounded-2xl p-4">
              <span className="text-xs font-mono text-indigo-400/60 font-bold">Module {num}</span>
              <p className="text-sm font-semibold text-white mt-1 mb-1">{title}</p>
              <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Formula ──────────────────────────────── */}
      <section className="glass rounded-3xl p-8 text-center">
        <h2 className="text-lg font-bold text-white mb-4">Hybrid Scoring Formula</h2>
        <div className="font-mono text-sm text-slate-200 bg-black/30 rounded-2xl px-6 py-4 inline-block">
          <span className="text-indigo-300">Final Score</span>
          {' = '}
          <span className="text-emerald-400">0.6</span> × CF
          {' + '}
          <span className="text-cyan-400">0.3</span> × Content
          {' + '}
          <span className="text-amber-400">0.1</span> × Popularity
        </div>
        <p className="text-xs text-slate-500 mt-4">
          Tie-break → avg_rating → num_ratings
        </p>
      </section>
    </div>
  )
}

function InfoRow({ icon: Icon, label, value }) {
  return (
    <div className="flex items-center gap-3 p-3 rounded-xl bg-white/3">
      <Icon size={14} className="text-indigo-400 flex-shrink-0" />
      <div>
        <p className="text-xs text-slate-500">{label}</p>
        <p className="text-sm font-medium text-white">{value}</p>
      </div>
    </div>
  )
}
