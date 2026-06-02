import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, Zap } from 'lucide-react'
import { queryAPI } from '../api'
import usePersistedState from '../hooks/usePersistedState'

const CHIPS = [
  'Multiple claims same vehicle',
  'No police report urban area',
  'High repair cost outliers',
  'Address change before claim',
  'Third-party fault pattern',
]

const riskBadge = (score) => {
  if (score >= 0.75) return <span className="badge badge-red">Risk {score?.toFixed(2)}</span>
  if (score >= 0.5)  return <span className="badge badge-orange">Risk {score?.toFixed(2)}</span>
  return <span className="badge badge-green">Risk {score?.toFixed(2)}</span>
}

export default function Investigate() {
  const [query, setQuery]   = usePersistedState('inv.query', '')
  const [result, setResult] = usePersistedState('inv.result', null)
  const [error, setError]   = usePersistedState('inv.error', '')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const submit = async (q = query) => {
    if (!q.trim()) return
    setLoading(true); setError(''); setResult(null)
    try {
      const data = await queryAPI(q)
      setResult(data)
    } catch (e) {
      setError(e.message || 'Query failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="card mb-4">
        <div style={{ fontSize: 12, color: 'var(--text2)', marginBottom: 8 }}>Ask about insurance claims in natural language</div>
        <div className="input-row mb-3">
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && submit()}
            placeholder="e.g. Show claims with multiple supplements and no police report in urban areas"
          />
          <button className="btn btn-primary" onClick={() => submit()} disabled={loading} style={{ whiteSpace: 'nowrap' }}>
            {loading ? <span className="spinner" /> : <Search size={14} />}
            {loading ? 'Analysing...' : 'Analyse'}
          </button>
        </div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {CHIPS.map(c => (
            <span key={c} onClick={() => { setQuery(c); submit(c) }}
              style={{ padding: '3px 10px', borderRadius: 20, border: '1px solid var(--border2)', fontSize: 11, color: 'var(--text2)', cursor: 'pointer', background: 'var(--surface2)' }}>
              {c}
            </span>
          ))}
        </div>
      </div>

      {error && <div style={{ background: 'var(--red-bg)', border: '1px solid #ffc9c9', borderRadius: 8, padding: '10px 14px', marginBottom: 14, fontSize: 12, color: 'var(--red-text)' }}>{error}</div>}

      {result && (
        <div style={{ display: 'flex', gap: 14 }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 12, color: 'var(--text2)', marginBottom: 10 }}>
              {result.matched_claims?.length || 0} results - hybrid search - reranked
              {result.cache_hit && <span className="badge badge-green" style={{ marginLeft: 8 }}><Zap size={10} /> cache hit</span>}
            </div>

            <div style={{ background: 'var(--blue-bg)', borderLeft: '3px solid var(--blue)', borderRadius: 8, padding: '12px 14px', marginBottom: 14, fontSize: 12, lineHeight: 1.6 }}>
              <div style={{ fontWeight: 600, marginBottom: 4, color: 'var(--blue-text)' }}>AI Summary</div>
              {result.answer}
            </div>

            {result.matched_claims?.map((claim, i) => {
              const meta = claim.metadata || {}
              const risk = result.risk_score
              return (
                <div key={i} onClick={() => navigate(`/claim/${meta.claim_id || claim.id}`)}
                  style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, padding: '13px 16px', marginBottom: 10, cursor: 'pointer' }}>
                  <div className="flex items-center justify-between mb-3">
                    <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--blue-text)' }}>{meta.claim_id || claim.id} - {meta.policy_type || ''}</span>
                    <span style={{ fontSize: 11, color: 'var(--text3)' }}>score {(claim.rerank_score ?? claim.score ?? 0).toFixed(3)}</span>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.5, marginBottom: 8 }}>{claim.text?.slice(0, 220)}...</div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {i === 0 && risk != null && riskBadge(risk)}
                    {meta.customer_region && <span className="badge badge-gray">{meta.customer_region}</span>}
                    {meta.agent_type && <span className="badge badge-gray">{meta.agent_type} agent</span>}
                    {meta.police_report_filed === 'No' && <span className="badge badge-orange">No police report</span>}
                    {meta.fraud_label === 1 && <span className="badge badge-red">Fraud label</span>}
                  </div>
                </div>
              )
            })}
          </div>

          <div style={{ width: 210, flexShrink: 0 }}>
            <div className="card mb-3">
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 10 }}>Fraud signals</div>
              {result.fraud_signals?.length > 0
                ? result.fraud_signals.slice(0, 4).map((s, i) => (
                    <div key={i} style={{ marginBottom: 8 }}>
                      <div style={{ fontSize: 11, fontWeight: 500 }}>{s.signal || s}</div>
                      {s.severity && <span className={`badge ${s.severity === 'high' ? 'badge-red' : s.severity === 'medium' ? 'badge-orange' : 'badge-gray'}`} style={{ marginTop: 2 }}>{s.severity}</span>}
                    </div>
                  ))
                : <div style={{ fontSize: 11, color: 'var(--text3)' }}>No signals detected</div>
              }
            </div>
            <div className="card mb-3">
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>Overall risk</div>
              <div style={{ fontSize: 28, fontWeight: 700, color: result.risk_score >= 0.75 ? 'var(--red)' : result.risk_score >= 0.5 ? 'var(--orange)' : 'var(--green)' }}>
                {result.risk_score != null ? result.risk_score.toFixed(2) : 'n/a'}
              </div>
              <div className="risk-bar">
                <div className={result.risk_score >= 0.75 ? 'risk-fill-high' : result.risk_score >= 0.5 ? 'risk-fill-med' : 'risk-fill-low'} style={{ width: `${(result.risk_score || 0) * 100}%` }} />
              </div>
              <div style={{ fontSize: 11, color: 'var(--text2)', marginTop: 6 }}>
                {result.escalated ? 'Escalated to senior review' : result.fraud_cluster !== 'none' ? `Cluster: ${result.fraud_cluster}` : ''}
              </div>
            </div>
            {result.model_used && (
              <div style={{ fontSize: 11, color: 'var(--text3)', textAlign: 'center' }}>
                via {result.model_used}
                {result.cache_hit && ' - cached'}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
