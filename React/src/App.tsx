import { useEffect } from 'react'
import { Route, Routes, useLocation } from 'react-router-dom'
import { getHealth } from './api/client'
import { Footer, Header } from './components/Layout'
import { startKit } from './lib/kit'
import { useAsync } from './lib/useAsync'
import About from './pages/About'
import Home from './pages/Home'
import Results from './pages/Results'

export default function App() {
  const { pathname } = useLocation()
  const health = useAsync(() => getHealth(), [])

  // the design kit scans the page once, after React has drawn it (lib/kit.ts)
  useEffect(() => {
    void startKit()
    // a link from another page (/#plan) arrives before the section exists: jump once it is drawn
    const id = decodeURIComponent(window.location.hash.slice(1))
    if (id) requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView())
  }, [])

  // an in-app jump (the form to its result) starts at the top of the new page
  useEffect(() => {
    if (!window.location.hash) window.scrollTo(0, 0)
  }, [pathname])

  return (
    <>
      <Header apiStatus={health.status} />
      <main id="top">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/result" element={<Results />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </main>
      <Footer apiStatus={health.status} />
    </>
  )
}
