import { Arrow } from '../components/Layout'
import { useI18n } from '../i18n/I18nProvider'
import type { StringKey } from '../i18n/strings'
import { NASA_PRODUCTS } from '../lib/nasaProducts'

const TERMS: StringKey[] = ['gl.gw', 'gl.wet', 'gl.n', 'gl.bare', 'gl.irr']
const STEPS: StringKey[] = ['about.step1', 'about.step2', 'about.step3', 'about.step4']

const NASA_ITEMS = [
  { prod: NASA_PRODUCTS.power, desc: 'about.ds.power' as StringKey },
  { prod: NASA_PRODUCTS.smap, desc: 'about.ds.smap' as StringKey },
  { prod: NASA_PRODUCTS.grace, desc: 'about.ds.grace' as StringKey },
  { prod: NASA_PRODUCTS.modis, desc: 'about.ds.modis' as StringKey },
  { prod: NASA_PRODUCTS.earthdata, desc: 'about.ds.stats' as StringKey },
]

/** An inner page in the kit's system: the page's wrap, type and rows, one column of reading. */
export default function About() {
  const { t } = useI18n()
  return (
    <div className="page">
      <div className="wrap">
        <header className="page__head">
          <div className="section-tag-row">
            <span className="badge-pill">🛰️ Earth Observation & Living Canopy</span>
          </div>
          <h1>{t('about.title')}</h1>
          <p className="t-lead measure">{t('about.intro')}</p>
          <div className="btn-row"><a className="btn btn--primary" href="/#plan">{t('cta.main')}<Arrow /></a></div>
        </header>

        <section className="page__block grid grid--top" aria-labelledby="steps-h">
          <h2 id="steps-h" className="span-5">{t('about.stepsTitle')}</h2>
          <ol className="rows fs-steps span-6 start-7">
            {STEPS.map((k, i) => (
              <li key={k}><span className="step-num">{i + 1}</span><p>{t(k)}</p></li>
            ))}
          </ol>
        </section>

        <section className="page__block grid grid--top" aria-labelledby="nasa-data-h">
          <h2 id="nasa-data-h" className="span-5">{t('about.dataTitle')}</h2>
          <ul className="rows span-6 start-7">
            {NASA_ITEMS.map((item) => (
              <li key={item.prod.id} className="dataset-item">
                <div className="dataset-title-row">
                  <span className="rows__name">{item.prod.name}</span>
                  <a
                    href={item.prod.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="link-ext-pill"
                    title={`${t('link.nasaProduct')}: ${item.prod.name} (${t('link.external')})`}
                  >
                    NASA Product ↗
                  </a>
                </div>
                <p>{t(item.desc)}</p>
                <span className="muted small">{item.prod.fullName} · {item.prod.period}</span>
              </li>
            ))}
          </ul>
        </section>

        <section className="page__block grid grid--top" aria-labelledby="terms-h">
          <h2 id="terms-h" className="span-5">{t('about.glossaryTitle')}</h2>
          <div className="faq span-6 start-7">
            {TERMS.map((k) => (
              <div className="faq__item" key={k}>
                <h3>{t(k)}</h3>
                <p>{t(`${k}.d` as StringKey)}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="page__block grid grid--top" aria-labelledby="limits-h">
          <h2 id="limits-h" className="span-5">{t('about.limitsTitle')}</h2>
          <ul className="rows span-6 start-7">
            {(['about.limit1', 'about.limit2', 'about.limit3'] as StringKey[]).map((k) => (
              <li key={k}><p>{t(k)}</p></li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  )
}
