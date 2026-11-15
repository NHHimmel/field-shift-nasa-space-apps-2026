import { Arrow } from '../components/Layout'
import { useI18n } from '../i18n/I18nProvider'
import type { StringKey } from '../i18n/strings'

const TERMS: StringKey[] = ['gl.gw', 'gl.wet', 'gl.n', 'gl.bare', 'gl.irr']
const STEPS: StringKey[] = ['about.step1', 'about.step2', 'about.step3', 'about.step4']

/** An inner page in the kit's system: the page's wrap, type and rows, one column of reading. */
export default function About() {
  const { t } = useI18n()
  return (
    <div className="page">
      <div className="wrap">
        <header className="page__head">
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
