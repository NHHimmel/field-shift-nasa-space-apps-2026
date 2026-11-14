import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { getMeta, getRecommendation } from '../api/client'
import { PRIORITIES, type StrategyKey } from '../api/types'
import { Explanation } from '../components/Explanation'
import { EvidencePanel } from '../components/EvidencePanel'
import { CompareList, CropCalendar, RangeBar, ScenarioPanel, StressChart } from '../components/Insights'
import { SeasonTiles, StatTiles } from '../components/PlanSummary'
import { useI18n } from '../i18n/I18nProvider'
import { makeNames } from '../lib/names'
import { requestFromParams } from '../lib/query'
import { useAsync } from '../lib/useAsync'

export default function Results() {
  const { t, lang } = useI18n()
  const [params] = useSearchParams()
  const query = params.toString()
  const req = useMemo(() => requestFromParams(params), [query]) // eslint-disable-line react-hooks/exhaustive-deps
  const meta = useAsync(() => getMeta(), [])
  const rec = useAsync(
    (signal) => (req ? getRecommendation(req, signal) : Promise.reject(new Error('missing'))),
    [query],
  )
  const names = makeNames(meta.data, lang)
  const [selected, setSelected] = useState<StrategyKey | null>(null)
  const [copied, setCopied] = useState(false)
  const [slow, setSlow] = useState(false)

  useEffect(() => setSelected(null), [query])
  useEffect(() => {
    if (rec.status !== 'loading') return setSlow(false)
    const id = window.setTimeout(() => setSlow(true), 3000)
    return () => window.clearTimeout(id)
  }, [rec.status])

  if (!req) {
    return (
      <div className="card center">
        <p>{t('res.missing')}</p>
        <Link className="primary" to="/">{t('res.start')}</Link>
      </div>
    )
  }

  const data = rec.data
  if (!data) {
    return rec.status === 'error' ? (
      <div className="card center">
        <p className="error">{t('res.error')}</p>
        {rec.error instanceof Error && rec.error.message ? <p className="muted small">{rec.error.message}</p> : null}
        <button className="primary" onClick={rec.retry}>{t('res.retry')}</button>
      </div>
    ) : (
      <div className="card center" aria-busy="true">
        <div className="spinner" aria-hidden />
        <p>{t('res.loading')}</p>
        {slow ? <p className="muted small">{t('res.loadingSlow')}</p> : null}
      </div>
    )
  }

  const requestedStrategy: StrategyKey = selected ?? data.recommended_strategy
  const card = data.strategies.find((s) => s.key === requestedStrategy) ?? data.strategies[0]
  if (!card) {
    return <div className="card center"><p className="error">{t('res.error')}</p></div>
  }
  const strategy = card.key
  const tabs = [data.recommended_strategy, ...PRIORITIES.filter((p) => p !== data.recommended_strategy)]

  async function share() {
    const url = window.location.href
    try {
      if (navigator.share) {
        await navigator.share({ title: t('app.name'), url })
        return
      }
      await navigator.clipboard.writeText(url)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 2000)
    } catch {
      /* user cancelled or clipboard blocked */
    }
  }

  return (
    <div className="results">
      <div className="toolbar no-print">
        <Link to={`/?${query}`} className="link">← {t('res.change')}</Link>
        <span className="toolbar-actions">
          <button className="ghost" onClick={share}>🔗 {copied ? t('res.copied') : t('res.share')}</button>
          <button className="ghost" onClick={() => window.print()}>🖨️ {t('res.print')}</button>
        </span>
      </div>

      <div className="summary-chips">
        <span className="chip static">📍 {names.district(data.district.key)}</span>
        <span className="chip static">
          {data.inputs.irrigation ? t('res.irrYes') : t('res.irrNo')}
          {data.inputs.irrigation_was_default ? t('res.irrDefault') : ''}
        </span>
        <span className="chip static">{t('res.soil', { level: t(`soil.${req.soil_fertility}`) })}</span>
      </div>

      <div className="tabs no-print" role="tablist">
        {tabs.map((k) => (
          <button key={k} role="tab" aria-selected={k === strategy} className={k === strategy ? 'on' : ''} onClick={() => setSelected(k)}>
            {t(`strategy.${k}`)}
            {k === data.recommended_strategy ? <span className="badge">⭐ {t('res.forYou')}</span> : null}
          </button>
        ))}
      </div>

      <section className="card plan">
        <h2>{strategy === 'baseline' ? t('res.currentTitle') : t('res.planTitle')}</h2>
        <SeasonTiles card={card} names={names} />
        <StatTiles card={card} />
      </section>

      <Explanation req={req} strategy={strategy} />
      <EvidencePanel rec={data} />
      <RangeBar card={card} years={data.data_info.climate_years} />
      <CropCalendar card={card} names={names} />
      <ScenarioPanel rec={data} strategy={strategy} names={names} />
      <StressChart rec={data} />
      <CompareList rec={data} selected={strategy} onSelect={(k) => { setSelected(k); window.scrollTo({ top: 0, behavior: 'smooth' }) }} names={names} />
    </div>
  )
}
