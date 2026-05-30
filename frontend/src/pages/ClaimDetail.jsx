import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Database, AlertCircle, FileCheck, Lightbulb } from 'lucide-react'
import { queryAPI } from '../api'

const AGENTS = [
  { key: 'retrieval', icon: Database, label: 'Retrieval agent', color: 'var(--blue-text)' },
  { key: 'fraud', icon: AlertCircle, label: 'Fraud analysis agent', color: 'var(--red)' },
  { key: 'policy', icon: FileCheck, label: 'Policy validation agent', color: 'var(--orange)' },
  { key: 'recommendation', icon: Lightbulb, label: 'Recommendation agent', color: 'var(--green)' },
]

export default function ClaimDetail() {
  const { claimId } = useParams()
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Run a targeted query for this specific claim
    queryAPI(`Analyse claim ${claimId} for fraud risk and provide investigation recommendations`)
      .then(setData)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [claimId])

  if (loading) return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200 }}>
      <span className="spinner" style={{ width: 24, height: 24 }} />
    </div>
  )

  const topClaim = data?.matched_claims?.[0]
  const meta = topClaim?.metadata || {}
  const risk = data?.risk_score || 0

  const agentOutputs = {
    retrieval: `Found ${data?.matched_claims?.length || 0} similar claims. Top match score: ${topClaim?.rerank_score?.toFixed(3) || '—'}.`,
    fraud: data?.fraud_signals?.length
      ? `${data.fraud_signals.length} fraud signals detected. ${data.fraud_signals.slice(0,2).map(s=>s.signal||s).join(', ')}.`
      : 'No fraud signals detected.',
    policy: data?.policy_compliance === 'violations_found'
      ? `${data.policy_violations?.length} policy violation(s): ${data.policy_violations?.map(v=>v.rule||v).join(', ')}.`
      : 'No policy violations found.',
    recommendation: data?.recommendations?.[0] || 'Standard review recommended.',
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16, cursor: 'pointer', color: 'var(--text2)', fontSize: 12 }}
        onClick={() => navigate(-1)}>
        <ArrowLeft size={14} /> Back to results
      </div>

      <div className="grid-2 mb-4">
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
            <span style={{ fontSize: 14, fontWeight: 600 }}>{claimId}</span>
            <span className={`badge ${risk >= 0.75 ? 'badge-red' : risk >= 0.5 ? 'badge-orange' : 'badge-green'}`}>
              {risk >= 0.75 ? '⚠ High risk' : risk >= 0.5 ? 'Medium risk' : 'Low risk'}
            </span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 12px', fontSize: 12 }}>
            {[
              ['Policy type', meta.policy_type],
              ['Base policy', meta.base_policy],
              ['Region', meta.customer_region],
              ['Fault', meta.fault],
              ['Vehicle price', meta.claim_amount],
              ['Police report', meta.police_report_filed],
              ['Witness present', meta.witness_present],
              ['Agent type', meta.agent_type],
              ['Incident date', meta.incident_date],
              ['Fraud label', meta.fraud_label == 1 ? 'Flagged' : 'Clean'],
            ].map(([label, val]) => (
              <div key={label}>
                <span style={{ color: 'var(--text2)' }}>{label}: </span>
                <span style={{ color: (label === 'Police report' && val === 'No') || (label === 'Fraud label' && val === 'Flagged') ? 'var(--red)' : 'var(--text)' }}>
                  {val || '—'}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div className="card mb-3" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 11, color: 'var(--text2)', marginBottom: 8 }}>Fraud risk score</div>
            <div style={{ fontSize: 40, fontWeight: 700, color: risk >= 0.75 ? 'var(--red)' : risk >= 0.5 ? 'var(--orange)' : 'var(--green)' }}>
              {risk.toFixed(2)}
            </div>
            <div className="risk-bar" style={{ marginTop: 10 }}>
              <div className={risk >= 0.75 ? 'risk-fill-high' : risk >= 0.5 ? 'risk-fill-med' : 'risk-fill-low'} style={{ width: `${risk * 100}%` }} />
            </div>
            <div style={{ fontSize: 11, color: 'var(--text2)', marginTop: 8 }}>
              {risk >= 0.75 ? 'Investigate immediately' : risk >= 0.5 ? 'Review recommended' : 'Low priority'}
            </div>
          </div>

          <div className="card">
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Policy compliance</div>
            <div style={{ fontSize: 14, fontWeight: 600, color: data?.policy_compliance === 'violations_found' ? 'var(--orange)' : 'var(--green)' }}>
              {data?.policy_violations?.length || 0} violations
            </div>
            {data?.policy_violations?.slice(0, 2).map((v, i) => (
              <div key={i} style={{ fontSize: 11, color: 'var(--text2)', marginTop: 4 }}>{v.description || v.rule || v}</div>
            ))}
          </div>
        </div>
      </div>

      <div className="agent-grid mb-4">
        {AGENTS.map(({ key, icon: Icon, label, color }) => (
          <div key={key} className="agent-card">
            <div className="agent-name"><Icon size={14} style={{ color }} />{label}</div>
            <div className="agent-output">{agentOutputs[key]}</div>
          </div>
        ))}
      </div>

      <div className="card mb-4">
        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: 'var(--blue-text)' }}>✦ AI investigation summary</div>
        <div style={{ fontSize: 13, lineHeight: 1.7 }}>{data?.answer || '—'}</div>
      </div>

      {data?.recommendations?.length > 0 && (
        <div className="card">
          <div className="card-title">Recommended investigation actions</div>
          <ol style={{ paddingLeft: 18 }}>
            {data.recommendations.map((r, i) => (
              <li key={i} style={{ fontSize: 12, marginBottom: 8, lineHeight: 1.5 }}>{r}</li>
            ))}
          </ol>
        </div>
      )}
    </div>
  )
}
