import { NavLink, useLocation } from 'react-router-dom'
import { LayoutDashboard, Search, BarChart2, Database, Cpu, Shield } from 'lucide-react'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/investigate', icon: Search, label: 'Investigate' },
  { to: '/analytics', icon: BarChart2, label: 'Analytics' },
]
const sysItems = [
  { to: '/ingestion', icon: Database, label: 'Data ingestion' },
  { to: '/flow', icon: Cpu, label: 'System flow' },
]

const PAGE_TITLES = {
  '/': 'Dashboard',
  '/investigate': 'Claims investigation',
  '/analytics': 'Analytics',
  '/ingestion': 'Data ingestion',
  '/flow': 'System flow',
  '/claim': 'Claim detail',
}

export default function Layout({ children }) {
  const { pathname } = useLocation()
  const title = PAGE_TITLES[pathname] ||
    Object.entries(PAGE_TITLES).find(([k]) => k !== '/' && pathname.startsWith(k))?.[1] ||
    PAGE_TITLES['/']

  return (
    <div className="root-shell">
      <header className="global-header">
        <div className="global-header-inner">
          <div>
            <div className="global-header-title">AI-Powered Insurance Claims Intelligence Assistant</div>
            <div className="global-header-sub">Insurance Operations · Fraud Intelligence</div>
          </div>
        </div>
      </header>
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div style={{ display: 'flex', alignItems: 'center', gap: 9, marginBottom: 4 }}>
            <div style={{ width: 30, height: 30, borderRadius: 8, background: 'var(--blue-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Shield size={16} color="var(--blue-text)" />
            </div>
            <div>
              <div className="sidebar-logo-title">Claims AI</div>
              <div className="sidebar-logo-sub">Intelligence Assistant</div>
            </div>
          </div>
        </div>

        <div className="sidebar-section">Main</div>
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
            <Icon size={16} />{label}
          </NavLink>
        ))}

        <div className="sidebar-section">System</div>
        {sysItems.map(({ to, icon: Icon, label }) => (
          <NavLink key={to} to={to} className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
            <Icon size={16} />{label}
          </NavLink>
        ))}

        <div className="sidebar-bottom">
          <div style={{ fontSize: 12, color: 'var(--text2)' }}><span className="model-dot" />GPT-4o-mini</div>
          <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 3 }}>Pinecone · LangGraph</div>
        </div>
      </aside>

      <div className="main-area">
        <header className="topbar">
          <span style={{ flex: 1, fontSize: 14, fontWeight: 600 }}>{title}</span>
          <span className="badge badge-red">12 high risk</span>
          <span className="badge badge-orange">38 pending</span>
        </header>
        <main className="page-content">{children}</main>
      </div>
    </div>
    </div>
  )
}
