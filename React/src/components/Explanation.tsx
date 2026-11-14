import { getExplanation } from '../api/client'
import type { RecommendRequest, StrategyKey } from '../api/types'
import { useI18n } from '../i18n/I18nProvider'
import type { StringKey } from '../i18n/strings'
import { useAsync } from '../lib/useAsync'

const LIST_SECTIONS: { key: 'nasa_evidence' | 'soil_benefits' | 'expert_checks'; title: StringKey; icon: string }[] = [
  { key: 'nasa_evidence', title: 'exp.nasa', icon: '🛰️' },
  { key: 'soil_benefits', title: 'exp.soil', icon: '🌱' },
  { key: 'expert_checks', title: 'exp.checks', icon: '👩‍🌾' },
]

export function Explanation({ req, strategy }: { req: RecommendRequest; strategy: StrategyKey }) {
  const { t, lang } = useI18n()
  const exp = useAsync((signal) => getExplanation(req, strategy, lang, signal), [JSON.stringify(req), strategy, lang])

  const fresh = exp.data && exp.data.strategy === strategy && exp.data.lang === lang
  const sections = fresh ? exp.data!.sections : undefined

  return (
    <section className="card explanation" aria-busy={exp.status === 'loading'}>
      <h2>💬 {t('exp.title')}</h2>
      {exp.status === 'error' && !sections ? (
        <p className="error">{t('exp.error')} <button className="link" onClick={exp.retry}>{t('res.retry')}</button></p>
      ) : !sections ? (
        <div className="skeleton-group" aria-label={t('exp.loading')}>
          <p className="muted">{t('exp.loading')}</p>
          <div className="skeleton" /><div className="skeleton short" /><div className="skeleton" />
        </div>
      ) : (
        <>
          <p className="exp-headline">{sections.headline}</p>
          <h3>{t('exp.why')}</h3>
          <p>{sections.why}</p>
          {LIST_SECTIONS.map((s) =>
            sections[s.key].length ? (
              <div key={s.key}>
                <h3><span aria-hidden>{s.icon}</span> {t(s.title)}</h3>
                <ul>{sections[s.key].map((line, i) => <li key={i}>{line}</li>)}</ul>
              </div>
            ) : null,
          )}
          <h3><span aria-hidden>📊</span> {t('exp.confidence')}</h3>
          <p>{sections.confidence}</p>
          <p className="muted small source-note">
            {exp.data!.source === 'template' ? t('exp.sourceTemplate') : t('exp.sourceAi')}
          </p>
        </>
      )}
    </section>
  )
}
