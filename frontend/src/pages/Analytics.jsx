import { useEffect, useState } from 'react'
import { analyticsAPI } from '../api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts'

const COLORS = ['#e03131', '#e8590c', '#2f9e44', '#228be6', '#7048e8', '#0ca678']

const SIGNALS = [
  { indicator: 'No police report', rate: '67%', signal: 'High', cls: 'badge-red' },
  { indicator: 'External agent', rate: '58%', signal: 'High', cls: 'badge-red' },
  { indicator: '3+ supplements', rate: '51%', signal: 'Medium', cls: 'badge-orange' },
  { indicator: 'Address change', rate: '44%', signal: 'Medium', cls: 'badge-orange' },
  { indicator: 'No witness', rate: '38%', signal: 'Medium', cls: 'badge-orange' },
  { indicator: 'New policy <30d', rate: '29%', signal: 'Low', cls: 'badge-gray' },
]

export default function Analytics() {
  const [data, setData] = useState(null)

  useEffect(() => { analyticsAPI().then(setData).catch(() => {}) }, [])

  const regionData = data
    ? Object.entries(data.by_region || {}).map(([k, v]) => ({ name: k, fraud: v }))
    : [{ name: 'Urban', fraud: 663 }, { name: 'Rural', fraud: 260 }]

  const policyData = data
    ? Object.entries(data.by_policy_type || {}).slice(0, 6).map(([k, v]) => ({ name: k.slice(0, 18), value: v }))
    : []

  return (
    <div>
      <div className="grid-4 mb-4">
        <div className="stat-card"><div className="stat-label">Total claims</div><div className="stat-value">{data ? data.total_claims.toLocaleString() : '15,419'}</div></div>
        <div className="stat-card"><div className="stat-label">Fraud cases</div><div className="stat-value c-red">{data ? data.fraud_count : '923'}</div></div>
        <div className="stat-card"><div className="stat-label">Fraud rate</div><div className="stat-value c-orange">{data ? `${(data.fraud_rate * 100).toFixed(1)}%` : '6.0%'}</div></div>
        <div className="stat-card"><div className="stat-label">Legitimate claims</div><div className="stat-value c-green">{data ? (data.total_claims - data.fraud_count).toLocaleString() : '14,496'}</div></div>
      </div>

      <div className="grid-2 mb-4">
        <div className="card">
          <div className="card-title">Fraud by accident area</div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={regionData}>
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => [v, 'Fraud cases']} />
              <Bar dataKey="fraud" fill="#e03131" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title">Fraud distribution by policy type</div>
          {policyData.length > 0
            ? <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie data={policyData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, percent }) => `${name.slice(0,10)} ${(percent*100).toFixed(0)}%`} labelLine={false} fontSize={9}>
                    {policyData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            : <div style={{ color: 'var(--text3)', fontSize: 12, textAlign: 'center', padding: '50px 0' }}>Run ingestion to load data</div>
          }
        </div>
      </div>

      <div className="card">
        <div className="card-title">Top fraud indicators</div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Indicator</th><th>Fraud rate</th><th>Signal strength</th></tr></thead>
            <tbody>
              {SIGNALS.map(s => (
                <tr key={s.indicator}>
                  <td>{s.indicator}</td>
                  <td style={{ fontWeight: 600 }}>{s.rate}</td>
                  <td><span className={`badge ${s.cls}`}>{s.signal}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
