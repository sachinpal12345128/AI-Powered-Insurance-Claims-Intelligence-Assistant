import { useState, useRef } from 'react'
import { FolderOpen, Play, CheckCircle, XCircle, Loader } from 'lucide-react'
import { ingestAPI, ingestUploadAPI, ingestStatusAPI } from '../api'

const DEFAULT_PATH = './data/source_data/fraud_oracle.csv'

const STEPS = [
  { key: 'load',       icon: 'CSV', label: 'Load CSV',                detail: 'fraud_oracle.csv - Pandas' },
  { key: 'preprocess', icon: 'PRE', label: 'Preprocess & normalize',  detail: 'Rename cols, synthesize dates, build text blocks' },
  { key: 'chunk',      icon: 'CHK', label: 'Chunking',                detail: 'Recursive splitter - 512 tok - overlap 64' },
  { key: 'embed',      icon: 'EMB', label: 'Generate embeddings',     detail: 'OpenAI text-embedding-3-small or sklearn fallback' },
  { key: 'bm25',       icon: 'BM',  label: 'Build BM25 index',        detail: 'rank_bm25 keyword index' },
  { key: 'store',      icon: 'DB',  label: 'Store in ChromaDB',       detail: 'Persist vectors + metadata' },
]

const STEP_DURATIONS = [1200, 1500, 1000, 8000, 900, 1200]

function addLog(setLogs, msg, cls = 'log-default') {
  const ts = new Date().toTimeString().slice(0, 8)
  setLogs(prev => [...prev, { ts, msg, cls }])
}

export default function Ingestion() {
  const [csvPath, setCsvPath] = useState(DEFAULT_PATH)
  const [uploadFile, setUploadFile] = useState(null)
  const [stepStatus, setStepStatus] = useState({})
  const [logs, setLogs] = useState([{ ts: '--', msg: 'Ready. Click Start ingestion.', cls: 'log-info' }])
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const fileRef = useRef()

  const simulateProgress = async () => {
    for (let i = 0; i < STEPS.length; i++) {
      const key = STEPS[i].key
      setStepStatus(s => ({ ...s, [key]: 'running' }))
      addLog(setLogs, `${STEPS[i].label}...`, 'log-info')
      await new Promise(r => setTimeout(r, STEP_DURATIONS[i]))
    }
  }

  const startIngest = async () => {
    setRunning(true); setError(''); setResult(null)
    setStepStatus({}); setLogs([])
    addLog(setLogs, 'Starting ingestion pipeline...', 'log-info')

    try {
      uploadFile ? await ingestUploadAPI(uploadFile) : await ingestAPI(csvPath)
    } catch (e) {
      const msg = e.message || 'Ingest call failed'
      addLog(setLogs, `Error: ${msg}`, 'log-err')
      setError(msg); setRunning(false); return
    }

    addLog(setLogs, 'ETL running in background - polling for status...', 'log-info')
    simulateProgress()

    const poll = setInterval(async () => {
      try {
        const status = await ingestStatusAPI()
        if (status.status === 'done') {
          clearInterval(poll)
          STEPS.forEach(s => setStepStatus(prev => ({ ...prev, [s.key]: 'done' })))
          addLog(setLogs, `Ingestion complete. ${status.vectors_stored?.toLocaleString()} vectors stored.`, 'log-ok')
          setResult(status); setRunning(false)
        } else if (status.status === 'error') {
          clearInterval(poll)
          addLog(setLogs, `Error: ${status.error}`, 'log-err')
          setError(status.error); setRunning(false)
        }
      } catch (_) { /* ignore poll errors */ }
    }, 3000)
  }

  return (
    <div>
      <div className="grid-2">
        <div>
          <div className="card mb-4">
            <div className="card-title"><FolderOpen size={15} /> Source file</div>

            <div style={{ border: '1px solid var(--border)', borderRadius: 8, padding: 10, marginBottom: 12, fontSize: 12 }}>
              <div style={{ color: 'var(--text2)', marginBottom: 4 }}>AI-Powered-Insurance-Claims/</div>
              <div style={{ paddingLeft: 16 }}>
                <div style={{ color: 'var(--text2)', marginBottom: 4 }}>data/</div>
                <div style={{ paddingLeft: 16, background: 'var(--blue-bg)', borderRadius: 6, padding: '5px 8px', color: 'var(--blue-text)', fontWeight: 500 }}>
                  source_data/fraud_oracle.csv <span className="badge badge-green" style={{ marginLeft: 6 }}>15,419 rows</span>
                </div>
              </div>
            </div>

            <div onClick={() => fileRef.current?.click()}
              style={{
                border: `1.5px dashed ${uploadFile ? 'var(--green)' : 'var(--border2)'}`,
                borderRadius: 10, padding: '20px', textAlign: 'center', cursor: 'pointer',
                background: uploadFile ? 'var(--green-bg)' : 'var(--surface2)',
                marginBottom: 12,
              }}>
              <div style={{ fontSize: 24, marginBottom: 6 }}>{uploadFile ? 'OK' : 'UP'}</div>
              <div style={{ fontSize: 12, fontWeight: 500, color: uploadFile ? 'var(--green-text)' : 'var(--text2)' }}>
                {uploadFile ? uploadFile.name : 'Click to upload a different CSV file'}
              </div>
              <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 3 }}>or use the default path below</div>
            </div>
            <input type="file" ref={fileRef} accept=".csv" style={{ display: 'none' }}
              onChange={e => { if (e.target.files[0]) setUploadFile(e.target.files[0]) }} />

            <div style={{ marginBottom: 12 }}>
              <label style={{ fontSize: 11, color: 'var(--text2)', display: 'block', marginBottom: 4 }}>CSV path (server-side)</label>
              <input value={csvPath} onChange={e => setCsvPath(e.target.value)} disabled={!!uploadFile} />
              {uploadFile && <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 3 }}>Path overridden by uploaded file</div>}
            </div>

            {uploadFile && (
              <button className="btn btn-ghost" style={{ width: '100%', marginBottom: 10 }} onClick={() => setUploadFile(null)}>
                Clear uploaded file
              </button>
            )}

            <button className="btn btn-primary" style={{ width: '100%' }} onClick={startIngest} disabled={running}>
              {running ? <><span className="spinner" /> Ingesting...</> : <><Play size={14} /> Start ingestion</>}
            </button>
          </div>

          <div className="card">
            <div className="card-title">Ingestion config</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {[['Embedding model', 'OpenAI text-embedding-3-small'], ['Chunk size', '512'], ['Chunk overlap', '64'], ['Collection', 'insurance_claims'], ['Batch size', '256'], ['Metadata', 'All fields']].map(([lbl, val]) => (
                <div key={lbl}>
                  <label style={{ fontSize: 11, color: 'var(--text2)', display: 'block', marginBottom: 3 }}>{lbl}</label>
                  <input defaultValue={val} style={{ fontSize: 12 }} readOnly />
                </div>
              ))}
            </div>
          </div>
        </div>

        <div>
          <div className="card mb-4">
            <div className="card-title" style={{ justifyContent: 'space-between' }}>
              Data preview
              <span className="badge badge-gray">first 4 rows</span>
            </div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>PolicyNo</th><th>PolicyType</th><th>AccidentArea</th><th>Fault</th><th>Fraud</th></tr></thead>
                <tbody>
                  {[['POL-1','Sport - Liability','Urban','Policy Holder',0],['POL-2','Sport - Collision','Urban','Policy Holder',0],['POL-3','Sport - Collision','Urban','Policy Holder',0],['POL-4','Sedan - Liability','Rural','Third Party',1]].map(([id,type,area,fault,fraud]) => (
                    <tr key={id}><td>{id}</td><td>{type}</td><td>{area}</td><td>{fault}</td>
                      <td><span className={`badge ${fraud ? 'badge-red' : 'badge-green'}`}>{fraud ? 'Fraud' : 'Clean'}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
              <span style={{ fontSize: 13, fontWeight: 600 }}>Live ingestion flow</span>
              <span className={`badge ${running ? 'badge-orange' : result ? 'badge-green' : 'badge-gray'}`}>
                {running ? 'Running' : result ? 'Complete' : 'Idle'}
              </span>
            </div>

            {STEPS.map((step, i) => {
              const status = stepStatus[step.key]
              return (
                <div key={step.key} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '9px 0', borderBottom: i < STEPS.length - 1 ? '1px solid var(--border)' : 'none' }}>
                  <div style={{ width: 32, height: 32, borderRadius: 8, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: status === 'done' ? 'var(--green-bg)' : status === 'running' ? 'var(--orange-bg)' : status === 'error' ? 'var(--red-bg)' : 'var(--surface2)' }}>
                    {status === 'done' ? <CheckCircle size={16} color="var(--green)" />
                      : status === 'running' ? <Loader size={16} color="var(--orange)" style={{ animation: 'spin 1s linear infinite' }} />
                      : status === 'error' ? <XCircle size={16} color="var(--red)" />
                      : <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text2)' }}>{step.icon}</span>}
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 2 }}>{step.label}</div>
                    <div style={{ fontSize: 11, color: 'var(--text2)' }}>{step.detail}</div>
                  </div>
                  <span style={{ fontSize: 11, fontWeight: 500,
                    color: status === 'done' ? 'var(--green)' : status === 'running' ? 'var(--orange)' : status === 'error' ? 'var(--red)' : 'var(--text3)' }}>
                    {status === 'done' ? 'Done' : status === 'running' ? 'Running...' : status === 'error' ? 'Error' : '--'}
                  </span>
                </div>
              )
            })}

            <div className="log-box mt-3">
              {logs.map((l, i) => (
                <div key={i} className={l.cls}>[{l.ts}] {l.msg}</div>
              ))}
            </div>

            {result && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8, marginTop: 14 }}>
                {[['Rows loaded', result.rows_loaded?.toLocaleString()], ['Chunks', result.chunks_created?.toLocaleString()], ['Vectors', result.vectors_stored?.toLocaleString()], ['Time', `${result.elapsed_seconds}s`]].map(([l, v]) => (
                  <div key={l} style={{ background: 'var(--surface2)', borderRadius: 8, padding: '10px', textAlign: 'center' }}>
                    <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--green)' }}>{v}</div>
                    <div style={{ fontSize: 10, color: 'var(--text2)', marginTop: 2 }}>{l}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
