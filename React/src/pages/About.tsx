import { Link } from 'react-router-dom'
import { useI18n } from '../i18n/I18nProvider'
import type { StringKey } from '../i18n/strings'

const DATASETS: { name: string; nameKey?: StringKey; desc: StringKey; icon: string }[] = [
  { name: 'NASA POWER', desc: 'about.ds.power', icon: '☀️' },
  { name: 'NASA SMAP', desc: 'about.ds.smap', icon: '💧' },
  { name: 'NASA GRACE / GRACE-FO', desc: 'about.ds.grace', icon: '🛰️' },
  { name: 'NASA MODIS & ECOSTRESS', desc: 'about.ds.modis', icon: '🌿' },
  { name: '', nameKey: 'about.ds.statsName', desc: 'about.ds.stats', icon: '📊' },
]

const TERMS: StringKey[] = ['gl.gw', 'gl.wet', 'gl.n', 'gl.bare', 'gl.irr']

export default function About() {
  const { t } = useI18n()
  return (
    <div className="about">
      <section className="hero">
        <h1>{t('about.title')}</h1>
        <p>{t('about.intro')}</p>
      </section>

      <section className="card">
        <h2>{t('about.stepsTitle')}</h2>
        <ol className="steps">
          {(['about.step1', 'about.step2', 'about.step3', 'about.step4'] as StringKey[]).map((k) => (
            <li key={k}>{t(k)}</li>
          ))}
        </ol>
      </section>

      <section className="card">
        <h2>{t('about.dataTitle')}</h2>
        <ul className="datasets">
          {DATASETS.map((d) => (
            <li key={d.desc}>
              <span className="ds-icon" aria-hidden>{d.icon}</span>
              <span>
                <strong>{d.nameKey ? t(d.nameKey) : d.name}</strong>
                <span className="muted">{t(d.desc)}</span>
              </span>
            </li>
          ))}
        </ul>
      </section>

      <section className="card">
        <h2>{t('about.glossaryTitle')}</h2>
        <dl className="glossary">
          {TERMS.map((k) => (
            <div key={k}>
              <dt>{t(k)}</dt>
              <dd>{t(`${k}.d` as StringKey)}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="card warn-card">
        <h2>⚠️ {t('about.limitsTitle')}</h2>
        <ul>
          <li>{t('about.limit1')}</li>
          <li>{t('about.limit2')}</li>
          <li>{t('about.limit3')}</li>
        </ul>
      </section>

      <p className="center"><Link className="primary big" to="/">{t('about.cta')} →</Link></p>
    </div>
  )
}
