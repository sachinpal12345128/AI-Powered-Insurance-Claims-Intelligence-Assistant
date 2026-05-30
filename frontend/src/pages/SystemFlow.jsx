import { useState } from 'react'

const DETAILS = {
  csv: 'fraud_oracle.csv · 15,419 rows · 33 columns. FraudFound_P is fraud label (6% positive rate). PolicyNumber used as claim_id proxy.',
  pandas: 'Pandas preprocessing: column renaming, incident_date synthesis from Month+Year, row-to-text conversion for embedding.',
  chunk: 'RecursiveCharacterTextSplitter: chunk_size=512, overlap=64. Each claim row → structured text block → split into chunks with metadata inherited.',
  embed: 'OpenAI text-embedding-3-small (1536-dim) via org gateway. Fallback → sentence-transformers all-MiniLM-L6-v2 (local, 384-dim, free). Model name stored as metadata.',
  chroma: 'ChromaDB local persistent store. Collection: insurance_claims. Cosine similarity space. Supports metadata filtering on policy_type, region, fraud_label.',
  bm25: 'BM25Okapi (rank_bm25) keyword index over full claim texts. Serialized to disk. Loaded once at startup.',
  user: 'Claims analyst enters natural language query from React UI (e.g. "show fraud claims with no police report in urban areas").',
  fastapi: 'FastAPI async REST endpoints: POST /query, POST /ingest, GET /claim/{id}, GET /analytics, GET /health. Auto OpenAPI docs at /docs.',
  guardin: 'Presidio PII detection → anonymize names/emails/phones. Custom injection pattern check. Returns 400 if blocked.',
  cache: 'In-memory QueryCache: MD5 exact match + cosine near-duplicate (threshold 0.92). TTL 10 min. Cache hit skips LLM entirely.',
  compress: 'LangChain ContextualCompression reduces retrieved chunk tokens by ~40% before passing to LLM context window.',
  orch: 'LangGraph StateGraph orchestrator. 4 agent nodes with conditional A2A edge. GPT-4o-mini primary → Groq llama-3.1-70b fallback → HuggingFace fallback. tenacity exponential backoff (max 3 retries).',
  ret_agent: 'ChromaDB semantic + BM25 keyword search → weighted hybrid fusion (0.6 semantic + 0.4 BM25) → CrossEncoder rerank (Cohere primary → local fallback).',
  fraud_agent: 'Analyzes fraud signals: no police report (+0.25), external agent (+0.20), no witness (+0.12), etc. Triggers A2A escalation to recommendation agent if risk > 0.75.',
  policy_agent: 'Checks filing window, required documents, coverage match, policy active status. Returns violations list with severity.',
  rec_agent: 'Synthesizes all agent outputs into investigation report. 3–5 specific actions. Receives A2A escalation from fraud agent.',
  deepeval: 'DeepEval: FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric, GEval (LLM-as-judge). Scores logged per query.',
  guardout: 'Checks risk score bounds (0–1), removes hallucinated claim IDs not in retrieved set, validates non-empty answer.',
  langsmith: 'LangSmith traces every LangGraph run: agent invocations, tool calls, token usage, latency per node, model fallback events. Enabled via LANGCHAIN_TRACING_V2=true.',
  response: 'Structured JSON: answer, risk_score, fraud_signals, policy_violations, recommendations, matched_claims, model_used, escalated.',
  react: 'React + Vite frontend. Pages: Dashboard, Investigate, Claim Detail, Analytics, Ingestion, System Flow. Connected to FastAPI via proxy.',
}

const Node = ({ id, label, sublabel, cls, onClick }) => (
  <div className="flow-node" onClick={() => onClick(id)}>
    <div className={`flow-box ${cls}`}>{label}</div>
    {sublabel && <div className="flow-label">{sublabel}</div>}
  </div>
)
const Arr = () => <div className="flow-arrow">→</div>
const ArrDown = () => <div style={{ display: 'flex', justifyContent: 'center', color: 'var(--text3)', fontSize: 14, padding: '4px 0' }}>↓</div>
const Plus = () => <div className="flow-arrow" style={{ fontWeight: 600 }}>+</div>

export default function SystemFlow() {
  const [detail, setDetail] = useState('Click any component to see details.')

  const click = (id) => setDetail(DETAILS[id] || 'Click any component to see details.')

  return (
    <div>
      {/* Section 1 */}
      <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8 }}>Pipeline 1 — Data ingestion (offline)</div>
      <div className="card mb-3">
        <div className="flow-pipe-center" style={{ gap: 3 }}>
          <Node id="csv" label="📊 fraud_oracle.csv" sublabel="15,419 rows" cls="flow-box-gray" onClick={click} />
          <Arr /><Node id="pandas" label="⚙ Pandas" sublabel="Preprocess" cls="flow-box-gray" onClick={click} />
          <Arr /><Node id="chunk" label="✂ Chunking" sublabel="512 tok · RC splitter" cls="flow-box-blue" onClick={click} />
          <Arr /><Node id="embed" label="🔢 Embedding" sublabel="text-embed-3-small" cls="flow-box-blue" onClick={click} />
          <Arr /><Node id="chroma" label="🗄 ChromaDB" sublabel="Vector store" cls="flow-box-green" onClick={click} />
          <Plus /><Node id="bm25" label="🔍 BM25 Index" sublabel="rank_bm25" cls="flow-box-green" onClick={click} />
        </div>
        <div className="flow-fallback">
          <span>⚡</span>
          <span><strong>Embedding fallback:</strong> OpenAI unavailable → sentence-transformers all-MiniLM-L6-v2 (local, free). Model stored as metadata.</span>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'center', margin: '4px 0' }}>
        <div style={{ width: 2, height: 20, background: 'var(--border2)', borderRadius: 1 }} />
      </div>

      {/* Section 2 */}
      <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8 }}>Pipeline 2 — Query (online · real-time)</div>
      <div className="card mb-3">
        {/* Row 1 */}
        <div className="flow-pipe-center" style={{ gap: 3, marginBottom: 8 }}>
          <Node id="user" label="👤 User query" sublabel="React UI" cls="flow-box-gray" onClick={click} />
          <Arr /><Node id="fastapi" label="⚡ FastAPI" sublabel="REST endpoint" cls="flow-box-gray" onClick={click} />
          <Arr /><Node id="guardin" label="🛡 Input guardrails" sublabel="Presidio · injection" cls="flow-box-orange" onClick={click} />
          <Arr /><Node id="cache" label="⚡ Cache check" sublabel="In-memory · hash" cls="flow-box-green" onClick={click} />
          <Arr /><Node id="compress" label="🗜 Prompt compression" sublabel="ContextualCompression" cls="flow-box-blue" onClick={click} />
        </div>

        <ArrDown />

        {/* Orchestrator */}
        <div className="flow-section" style={{ cursor: 'pointer' }} onClick={() => click('orch')}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text2)' }}>🕸 LangGraph orchestrator</span>
            <div style={{ display: 'flex', gap: 6 }}>
              <span className="badge badge-blue">GPT-4o-mini</span>
              <span className="badge badge-orange">→ Groq fallback</span>
              <span className="badge badge-gray">→ HuggingFace</span>
            </div>
          </div>
          <div style={{ fontSize: 10, color: 'var(--text3)', marginBottom: 10, background: 'var(--surface2)', borderRadius: 6, padding: '4px 8px', display: 'inline-block' }}>
            tenacity · exponential backoff · max 3 retries before model switch
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8, marginBottom: 10 }}>
            {[
              { id: 'ret_agent', icon: '🗃', label: 'Retrieval agent', desc: 'ChromaDB + BM25 → hybrid → CrossEncoder rerank' },
              { id: 'fraud_agent', icon: '🚨', label: 'Fraud analysis agent', desc: 'Signal detection · risk scoring · A2A escalation' },
              { id: 'policy_agent', icon: '📋', label: 'Policy validation', desc: 'Compliance check · violation detection' },
              { id: 'rec_agent', icon: '💡', label: 'Recommendation agent', desc: 'Investigation guidance · explainable output' },
            ].map(a => (
              <div key={a.id} onClick={e => { e.stopPropagation(); click(a.id) }}
                style={{ background: 'var(--surface2)', borderRadius: 8, padding: '10px 12px', cursor: 'pointer', border: '1px solid var(--border)' }}>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>{a.icon} {a.label}</div>
                <div style={{ fontSize: 10, color: 'var(--text2)', lineHeight: 1.4 }}>{a.desc}</div>
              </div>
            ))}
          </div>
          <div style={{ fontSize: 10, color: 'var(--text3)', background: 'var(--surface2)', borderRadius: 6, padding: '4px 8px', display: 'inline-block' }}>
            ⇄ A2A escalation: Fraud agent → Recommendation agent when risk score &gt; 0.75
          </div>
        </div>

        <ArrDown />

        {/* Row 3 */}
        <div className="flow-pipe-center" style={{ gap: 3, marginTop: 8 }}>
          <Node id="deepeval" label="📊 DeepEval" sublabel="Faithfulness · GEval" cls="flow-box-purple" onClick={click} />
          <Arr /><Node id="guardout" label="🛡 Output guardrails" sublabel="Hallucination · bounds" cls="flow-box-orange" onClick={click} />
          <Arr /><Node id="langsmith" label="📡 LangSmith" sublabel="Full trace" cls="flow-box-teal" onClick={click} />
          <Arr /><Node id="response" label="📤 Response" sublabel="JSON · risk + recs" cls="flow-box-green" onClick={click} />
          <Arr /><Node id="react" label="🖥 React UI" sublabel="Dashboard · Investigate" cls="flow-box-gray" onClick={click} />
        </div>
      </div>

      {/* Detail panel */}
      <div className="flow-detail">
        <span style={{ marginRight: 6 }}>ℹ</span>{detail}
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 14, marginTop: 12, flexWrap: 'wrap' }}>
        {[['flow-box-blue','LangChain / LangGraph'],['flow-box-green','Storage / retrieval'],['flow-box-orange','Safety / guardrails'],['flow-box-purple','Evaluation'],['flow-box-teal','Observability'],['flow-box-gray','I/O layer']].map(([cls, lbl]) => (
          <div key={lbl} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text2)' }}>
            <span className={`flow-box ${cls}`} style={{ padding: '2px 8px', fontSize: 10 }}>{lbl}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
