import axios from 'axios'

// Per-endpoint timeouts:
//   query / analytics / claim : 120s  (full RAG pipeline can take a while)
//   health / ingest-trigger / ingest-status : 10s  (instant endpoints)
//   ingest/upload : 600s (large file upload can take a few minutes)
const api       = axios.create({ baseURL: '/api/v1', timeout: 120000 })
const apiQuick  = axios.create({ baseURL: '/api/v1', timeout: 10000  })
const apiUpload = axios.create({ baseURL: '/api/v1', timeout: 600000 })

function describeError(e) {
  if (e.response) {
    const d = e.response.data
    if (typeof d === 'string') return `${e.response.status}: ${d}`
    if (d?.detail) return typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail)
    if (d?.error)  return d.error
    return `${e.response.status} ${e.response.statusText || ''}`.trim()
  }
  if (e.code === 'ECONNABORTED') return 'Request timed out. The backend may be processing a long query.'
  if (e.code === 'ERR_NETWORK')  return 'Cannot reach backend at http://localhost:8000 - is it running?'
  return e.message || 'Unknown error'
}

async function call(client, method, url, body) {
  try {
    const r = await (method === 'get' ? client.get(url) : client.post(url, body))
    return r.data
  } catch (e) {
    const err = new Error(describeError(e))
    err.original = e
    throw err
  }
}

export const queryAPI        = (query, filters) => call(api,      'post', '/query',         { query, filters })
export const ingestAPI       = (csv_path)       => call(apiQuick, 'post', '/ingest',        { csv_path })
export const ingestStatusAPI = ()               => call(apiQuick, 'get',  '/ingest/status')
export const getClaimAPI     = (id)             => call(apiQuick, 'get',  `/claim/${id}`)
export const analyticsAPI    = ()               => call(apiQuick, 'get',  '/analytics')
export const healthAPI       = ()               => call(apiQuick, 'get',  '/health')

export const ingestUploadAPI = (file) => {
  const form = new FormData()
  form.append('file', file)
  return apiUpload.post('/ingest/upload', form, { headers: { 'Content-Type': 'multipart/form-data' } })
                  .then(r => r.data)
                  .catch(e => { throw new Error(describeError(e)) })
}
