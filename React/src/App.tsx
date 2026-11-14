import { Route, Routes } from 'react-router-dom'
import { getHealth, getMeta } from './api/client'
import { Footer, Header } from './components/Layout'
import { useAsync } from './lib/useAsync'
import About from './pages/About'
import Home from './pages/Home'
import Results from './pages/Results'

export default function App() {
  const meta = useAsync(() => getMeta(), [])
  const health = useAsync(() => getHealth(), [])
  return (
    <div className="app">
      <Header apiStatus={health.status} />
      <main className="container main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/result" element={<Results />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </main>
      <Footer rate={meta.data?.units.usd_to_bdt} apiStatus={health.status} />
    </div>
  )
}
