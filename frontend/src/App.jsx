import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import LandingPage from './pages/LandingPage'
import AnalyticsDashboard from './pages/AnalyticsDashboard'
import EngagementDashboard from './pages/EngagementDashboard'
import QualityDashboard from './pages/QualityDashboard'
import RecommendPage from './pages/RecommendPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index            element={<LandingPage />} />
          <Route path="analytics" element={<AnalyticsDashboard />} />
          <Route path="engagement" element={<EngagementDashboard />} />
          <Route path="quality"   element={<QualityDashboard />} />
          <Route path="recommend" element={<RecommendPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
