import { Route, Routes } from 'react-router-dom'
<<<<<<< HEAD
import { getHealth, getMeta } from './api/client'
=======
import { getMeta } from './api/client'
>>>>>>> parent of 630a375 (clean ui)
import { Footer, Header } from './components/Layout'
import { useAsync } from './lib/useAsync'
import About from './pages/About'
import Home from './pages/Home'
import Results from './pages/Results'

export default function App() {
  const meta = useAsync(() => getMeta(), [])
<<<<<<< HEAD
  const health = useAsync(() => getHealth(), [])
  return (
    <div className="app">
      <Header apiStatus={health.status} />
=======
  return (
    <div className="app">
      <Header />
>>>>>>> parent of 630a375 (clean ui)
      <main className="container main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/result" element={<Results />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </main>
<<<<<<< HEAD
      <Footer rate={meta.data?.units.usd_to_bdt} apiStatus={health.status} />
=======
      <Footer rate={meta.data?.units.usd_to_bdt} />
>>>>>>> parent of 630a375 (clean ui)
    </div>
  )
}
