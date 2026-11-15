import { useLocation } from 'react-router-dom'
import { useI18n } from '../i18n/I18nProvider'

/* Links that lead to the home page's sections are plain anchors: on the home page they jump in place,
   elsewhere they load the home page, where the design kit starts once (lib/kit.ts). */
function useHomeHref() {
  const { pathname } = useLocation()
  return (id: string) => (pathname === '/' ? `#${id}` : `/#${id}`)
}

/** The example plan is a real plan page: Rajshahi with the usual answers. */
const EXAMPLE = '/result?d=Rajshahi&p=balanced'

const Arrow = () => (
  <svg className="m-arrow" viewBox="0 0 12 12" aria-hidden="true">
    <path className="m-arrow__stem" d="M1 6h8" />
    <path className="m-arrow__head" d="M4.5 2.5 8 6l-3.5 3.5" />
  </svg>
)
export { Arrow }

type ApiStatus = 'loading' | 'ok' | 'error'

/** Whether the FastAPI backend answered /api/health: a coloured dot only; the words are in its tooltip. */
function ApiStatusPill({ status }: { status: ApiStatus }) {
  const { t } = useI18n()
  const label = status === 'ok' ? t('status.connected') : status === 'error' ? t('status.unavailable') : t('status.checking')
  return <span className={`api-status ${status}`} title={label} role="img" aria-label={label}><span aria-hidden="true">●</span></span>
}

function LangToggle() {
  const { t, lang, setLang } = useI18n()
  return (
    <div className="lang-toggle" role="group" aria-label={t('lang.label')}>
      <button type="button" aria-pressed={lang === 'bn'} onClick={() => setLang('bn')} lang="bn">বাংলা</button>
      <button type="button" aria-pressed={lang === 'en'} onClick={() => setLang('en')} lang="en">English</button>
    </div>
  )
}

function Logo({ href }: { href: string }) {
  const { t } = useI18n()
  return (
    <a className="nav__logo" href={href} aria-label={t('app.name')}>
      <span className="nav__logo-icon" aria-hidden="true">
        <svg viewBox="0 0 28 28" width="28" height="28" className="logo-svg">
          <circle cx="14" cy="14" r="13" fill="#136248" />
          <ellipse cx="14" cy="14" rx="11" ry="5" fill="none" stroke="#99dfbe" strokeWidth="1.2" transform="rotate(-25 14 14)" strokeDasharray="2,2" opacity="0.85" />
          <circle cx="21" cy="9" r="1.8" fill="#ffffff" />
          <path d="M14 20 C14 15 11 13 8 13 C8 17 11 20 14 20 Z" fill="#60d394" />
          <path d="M14 20 C14 13 18 10 21 11 C21 15 17 19 14 20 Z" fill="#ffffff" />
          <path d="M14 20 L14 12" stroke="#05333a" strokeWidth="1.2" strokeLinecap="round" />
        </svg>
      </span>
      <span className="nav__logo-text">{t('app.name')}</span>
    </a>
  )
}

export function Header({ apiStatus = 'loading' }: { apiStatus?: ApiStatus }) {
  const { t } = useI18n()
  const { pathname } = useLocation()
  const href = useHomeHref()
  return (
    <header className="nav">
      <div className="wrap nav__bar">
        <a className="skip-link" href="#top">{t('nav.skip')}</a>
        <Logo href={pathname === '/' ? '#top' : '/'} />
        <nav className="nav__links" aria-label={t('nav.main')}>
          <div className="nav__item"><a className="nav__link" href={href('plan')}>{t('nav.plan')}</a></div>
          <div className="nav__item"><a className="nav__link" href={href('data')}>{t('nav.data')}</a></div>
          <div className="nav__item"><a className="nav__link" href={href('faq')}>{t('nav.faq')}</a></div>
          <div className="nav__item">
            <a className="nav__link" href="/about" aria-current={pathname === '/about' ? 'page' : undefined}>{t('nav.about')}</a>
          </div>
        </nav>
        <div className="nav__actions">
          <ApiStatusPill status={apiStatus} />
          <LangToggle />
          <a className="btn btn--primary" href={href('plan')}>{t('cta.main')}</a>
        </div>
        <button className="nav__toggle" type="button" aria-expanded="false" aria-label={t('nav.menu')}>
          <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><path d="M2 5h14M2 13h14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
        </button>
      </div>
    </header>
  )
}

export function Footer({ apiStatus = 'loading' }: { apiStatus?: ApiStatus }) {
  const { t } = useI18n()
  const href = useHomeHref()
  const { pathname } = useLocation()
  return (
    <footer className="footer closing footer--statement" id="contact" aria-labelledby="close-h">
      <div className="wrap">
        <div className="closing__cta close--type">
          <div className="section-head">
            <h2 id="close-h">{t('close.h2')}</h2>
            <p className="t-small muted">{t('close.p')}</p>
          </div>
          <a className="btn btn--primary btn--xl btn--wide" href={href('plan')}>{t('cta.main')}<Arrow /></a>
        </div>
        <div className="closing__links">
          <div className="grid">
            <div className="span-4 stack-4">
              <Logo href={pathname === '/' ? '#top' : '/'} />
              <p className="footer__lede">{t('footer.lede')}</p>
            </div>
            <div className="span-3 start-7 footer__col">
              <h3>{t('nav.plan')}</h3>
              <ul>
                <li><a href={EXAMPLE}>{t('menu.example')}</a></li>
                <li><a href={href('how')}>{t('menu.g2')}</a></li>
              </ul>
            </div>
            <div className="span-3 start-10 footer__col">
              <h3>{t('footer.site')}</h3>
              <ul>
                <li><a href={href('data')}>{t('menu.data')}</a></li>
                <li><a href={href('faq')}>{t('nav.faq')}</a></li>
                <li><a href="/about">{t('nav.about')}</a></li>
              </ul>
            </div>
          </div>
          <div className="footer__legal">
            <span>© 2026 {t('app.name')} · {t('footer.made')}</span>
            <ApiStatusPill status={apiStatus} />
          </div>
        </div>
      </div>
    </footer>
  )
}
