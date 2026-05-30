import axios from 'axios'

const api = axios.create({ baseURL: '/api/v1', timeout: 120000 })

export const queryAPI      = (query, filters)  => api.post('/query',   { query, filters }).then(r => r.data)
export const ingestAPI     = (csv_path)         => api.post('/ingest',  { csv_path }).then(r => r.data)
export const ingestUploadAPI = (file) => {
  const form = new FormData()
  form.append('file', file)
  return api.post('/ingest/upload', form, { headers: { 'Content-Type': 'multipart/form-data' } }).then(r => r.data)
}
export const getClaimAPI   = (claimId)          => api.get(`/claim/${claimId}`).then(r => r.data)
export const analyticsAPI  = ()                 => api.get('/analytics').then(r => r.data)
export const healthAPI     = ()                 => api.get('/health').then(r => r.data)
