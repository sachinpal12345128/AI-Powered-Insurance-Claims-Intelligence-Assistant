import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Investigate from './pages/Investigate'
import ClaimDetail from './pages/ClaimDetail'
import Analytics from './pages/Analytics'
import Ingestion from './pages/Ingestion'
import SystemFlow from './pages/SystemFlow'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/investigate" element={<Investigate />} />
        <Route path="/claim/:claimId" element={<ClaimDetail />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/ingestion" element={<Ingestion />} />
        <Route path="/flow" element={<SystemFlow />} />
      </Routes>
    </Layout>
  )
}
