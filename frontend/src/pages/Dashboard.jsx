import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { analyticsAPI, healthAPI } from '../api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const RISK_BADGE = (score) => {
  const s = parseFloat(score)
  if (s >= 0.75) return <span className="badge badge-red">{s.toFixed(2)}</span>
  if (s >= 0.5) return <span className="badge badge-orange">{s.toFixed(2)}</span>
  return <span className="badge badge-green">{s.toFixed(2)}</span>
}

const MOCK_CLAIMS = [
  { id: 'POL-9821', type: 'Sport – Collision', region: 'Urban', risk: 0.91, status: 'Review' },
  { id: 'POL-4402', type: 'Sedan – Liability', region: 'Urban', risk: 0.87, status: 'Review' },
  { id: 'POL-7715', type: 'Sport – Liability', region: 'Rural', risk: 0.74, status: 'Pending' },
  { id: 'POL-3308', type: 'Utility – Collision', region: 'Urban', risk: 0.69, status: 'Pending' },
  { id: 'POL-6190', type: 'Sedan – Collision', region: 'Urban', risk: 0.22, status: 'Cleared' },
]

export default function Dashboard() {
  const [analytics, setAnalytics] = useState(null)
  const [health, setHealth] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    analyticsAPI().then(setAnalytics).catch(() => {})
    healthAPI().then(setHealth).catch(() => {})
  }, [])

  const policyData = analytics
    ? Object.entries(analytics.by_policy_type || {}).slice(0, 6).map(([k, v]) => ({ name: k.slice(0, 14), value: v }))
    : []

  return (
    <div>
      {health && !health.chroma_ok && (
        <div style={{ background: 'var(--orange-bg)', border: '1px solid #ffd8a8', borderRadius: 8, padding: '10px 14px', marginBottom: 14, fontSize: 12, color: 'var(--orange-text)' }}>
          ⚠ Vector store not ready. Run data ingestion first.
        </div>
      )}

      <div className="grid-4 mb-4">
        <div className="stat-card"><div className="stat-label">Total claims</div><div className="stat-value">{analytics ? analytics.total_claims.toLocaleString() : '—'}</div><div className="stat-sub">Full dataset</div></div>
        <div className="stat-card"><div className="stat-label">Fraud detected</div><div className="stat-value c-red">{analytics ? analytics.fraud_count.toLocaleString() : '—'}</div><div className="stat-sub">{analytics ? `${(analytics.fraud_rate * 100).toFixed(1)}% of claims` : ''}</div></div>
        <div className="stat-card"><div className="stat-label">High-risk flagged</div><div className="stat-value c-orange">12</div><div className="stat-sub">Awaiting review</div></div>
        <div className="stat-card"><div className="stat-label">System status</div><div className="stat-value c-green" style={{ fontSize: 16, marginTop: 4 }}>{health ? (health.chroma_ok ? '✓ Ready' : '✗ Setup needed') : '—'}</div><div className="stat-sub">{health ? health.llm_model : ''}</div></div>
      </div>

      <div className="grid-2 mb-4">
        <div className="card">
          <div className="card-title">Recent high-risk claims</div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Claim ID</th><th>Type</th><th>Region</th><th>Risk</th><th>Status</th></tr></thead>
              <tbody>
                {MOCK_CLAIMS.map(c => (
                  <tr key={c.id} onClick={() => navigate(`/claim/${c.id}`)}>
                    <td className="text-blue">{c.id}</td>
                    <td>{c.type}</td>
                    <td>{c.region}</td>
                    <td>{RISK_BADGE(c.risk)}</td>
                    <td><span className={`badge ${c.status === 'Review' ? 'badge-orange' : c.status === 'Cleared' ? 'badge-green' : 'badge-gray'}`}>{c.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <div className="card-title">Fraud by policy type</div>
          {policyData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={policyData} layout="vertical" margin={{ left: 10, right: 20 }}>
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={90} />
                <Tooltip formatter={(v) => [v, 'Fraud cases']} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {policyData.map((_, i) => <Cell key={i} fill={i < 2 ? '#e03131' : i < 4 ? '#e8590c' : '#2f9e44'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ color: 'var(--text3)', fontSize: 12, textAlign: 'center', padding: '40px 0' }}>No data — run ingestion first</div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-title">Top fraud signals</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 16, fontSize: 12 }}>
          <div><div style={{ fontWeight: 600, color: 'var(--red)', marginBottom: 4 }}>No police report</div><div className="text-muted">67% of fraudulent claims filed without a police report</div></div>
          <div><div style={{ fontWeight: 600, color: 'var(--orange)', marginBottom: 4 }}>External agent</div><div className="text-muted">58% of fraud cases involved external agent type</div></div>
          <div><div style={{ fontWeight: 600, color: 'var(--orange)', marginBottom: 4 }}>Multiple supplements</div><div className="text-muted">Claims with 3+ supplements have 2.4× higher fraud rate</div></div>
        </div>
      </div>
    </div>
  )
}
