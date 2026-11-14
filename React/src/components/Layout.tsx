import { Link, NavLink } from 'react-router-dom'
import { useI18n } from '../i18n/I18nProvider'
<<<<<<< HEAD

type ApiStatus = 'loading' | 'ok' | 'error'

function ApiStatusPill({ status }: { status: ApiStatus }) {
  const { t } = useI18n()
  const label = status === 'ok' ? t('status.connected') : status === 'error' ? t('status.unavailable') : t('status.checking')
  return <span className={`api-status ${status}`} title={label}><span aria-hidden>●</span> {label}</span>
}

export function Header({ apiStatus = 'loading' }: { apiStatus?: ApiStatus }) {
=======
import { fmtInt } from '../lib/format'

export function Header() {
>>>>>>> parent of 630a375 (clean ui)
  const { t, lang, setLang } = useI18n()
  return (
    <header className="header">
      <div className="container header-inner">
        <Link to="/" className="brand">
          <img src="/favicon.svg" alt="" width={36} height={36} />
          <span>
            <strong>{t('app.name')}</strong>
            <small>{t('app.tagline')}</small>
          </span>
        </Link>
<<<<<<< HEAD
        <span className="header-actions">
          <ApiStatusPill status={apiStatus} />
          <span className="lang-toggle" role="group" aria-label={t('lang.label')}>
            <button className={lang === 'bn' ? 'on' : ''} onClick={() => setLang('bn')} lang="bn">
              বাংলা
            </button>
            <button className={lang === 'en' ? 'on' : ''} onClick={() => setLang('en')} lang="en">
              English
            </button>
          </span>
        </span>
=======
        <div className="lang-toggle" role="group" aria-label={t('lang.label')}>
          <button className={lang === 'bn' ? 'on' : ''} onClick={() => setLang('bn')} lang="bn">
            বাংলা
          </button>
          <button className={lang === 'en' ? 'on' : ''} onClick={() => setLang('en')} lang="en">
            English
          </button>
        </div>
>>>>>>> parent of 630a375 (clean ui)
      </div>
      <nav className="container nav">
        <NavLink to="/" end>
          {t('nav.home')}
        </NavLink>
        <NavLink to="/about">{t('nav.about')}</NavLink>
      </nav>
    </header>
  )
}

<<<<<<< HEAD
export function Footer({ rate, apiStatus = 'loading' }: { rate?: number; apiStatus?: ApiStatus }) {
=======
export function Footer({ rate }: { rate?: number }) {
>>>>>>> parent of 630a375 (clean ui)
  const { t, lang } = useI18n()
  return (
    <footer className="footer">
      <div className="container">
        <p className="advisory">⚠️ {t('advisory')}</p>
        {rate ? <p className="muted small">{t('footer.units', { rate: fmtInt(rate, lang) })}</p> : null}
<<<<<<< HEAD
        <p className="muted small"><ApiStatusPill status={apiStatus} /></p>
=======
>>>>>>> parent of 630a375 (clean ui)
        <p className="muted small">{t('footer.made')}</p>
      </div>
    </footer>
  )
}
