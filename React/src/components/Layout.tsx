import { Link, NavLink } from 'react-router-dom'
import { useI18n } from '../i18n/I18nProvider'
import { fmtInt } from '../lib/format'

export function Header() {
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
        <div className="lang-toggle" role="group" aria-label={t('lang.label')}>
          <button className={lang === 'bn' ? 'on' : ''} onClick={() => setLang('bn')} lang="bn">
            বাংলা
          </button>
          <button className={lang === 'en' ? 'on' : ''} onClick={() => setLang('en')} lang="en">
            English
          </button>
        </div>
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

export function Footer({ rate }: { rate?: number }) {
  const { t, lang } = useI18n()
  return (
    <footer className="footer">
      <div className="container">
        <p className="advisory">⚠️ {t('advisory')}</p>
        {rate ? <p className="muted small">{t('footer.units', { rate: fmtInt(rate, lang) })}</p> : null}
        <p className="muted small">{t('footer.made')}</p>
      </div>
    </footer>
  )
}
