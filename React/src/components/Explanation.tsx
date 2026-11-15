import type { ReactNode } from 'react'
import { getExplanation } from '../api/client'
import type { Recommendation, RecommendRequest, StrategyCard, StrategyKey } from '../api/types'
import { useI18n } from '../i18n/I18nProvider'
import type { StringKey } from '../i18n/strings'
import { fmtInt, fmtMm, fmtPct, fmtTaka } from '../lib/format'
import { useAsync } from '../lib/useAsync'

const LIST_SECTIONS: { key: 'nasa_evidence' | 'soil_benefits' | 'expert_checks'; title: StringKey; icon: string }[] = [
  { key: 'nasa_evidence', title: 'exp.nasa', icon: '🛰️' },
  { key: 'soil_benefits', title: 'exp.soil', icon: '🌱' },
  { key: 'expert_checks', title: 'exp.checks', icon: '👩‍🌾' },
]

/** Highlights satellite metrics, numbers, and product names in explanation lines */
function formatEvidenceLine(line: string): ReactNode {
  // Regex matches NASA products or numeric measurements in EN and BN
  const pattern = /(NASA POWER|NASA SMAP|NASA GRACE|NASA MODIS|ECOSTRESS|নাসার POWER|নাসার SMAP|নাসার GRACE|[০-৯0-9]+(?:\.[০-৯0-9]+)?\s*(?:মিমি|mm|কেজি|kg|%|টাকা|taka|দিন|days|°C|°সে|বছর|years))/g
  const parts = line.split(pattern)

  if (parts.length === 1) return line

  return parts.map((part, i) => {
    if (!part) return null
    if (pattern.test(part)) {
      return (
        <span key={i} className="evidence-val-chip">
          {part}
        </span>
      )
    }
    return part
  })
}

interface ExplanationProps {
  req: RecommendRequest
  strategy: StrategyKey
  card?: StrategyCard
  rec?: Recommendation
}

export function Explanation({ req, strategy, card, rec }: ExplanationProps) {
  const { t, lang } = useI18n()
  const exp = useAsync((signal) => getExplanation(req, strategy, lang, signal), [JSON.stringify(req), strategy, lang])

  const fresh = exp.data && exp.data.strategy === strategy && exp.data.lang === lang
  const sections = fresh ? exp.data!.sections : undefined

  // Active strategy card metrics for the evidence values panel
  const activeCard = card ?? rec?.strategies.find((s) => s.key === strategy)
  const seasons = activeCard?.seasons
  const m = activeCard?.metrics

  const gwPctl = seasons?.[0]?.groundwater_pctl ?? rec?.district.groundwater_pctl ?? 0
  const rainRabi = seasons?.[0]?.rain_mm ?? 0
  const dryDays = seasons?.[0]?.dry_spell_days ?? 0
  const wetnessRabi = seasons?.[0]?.rootzone_wetness_pct ?? 0
  const heatDays = seasons?.[1]?.heat_stress_days ?? 0
  const isBase = strategy === 'baseline'

  const profitDiff = m?.profit_change_bdt_bigha ?? 0
  const waterSaved = m?.water_saved_mm ?? 0

  return (
    <section className="card explanation" aria-busy={exp.status === 'loading'} aria-labelledby="why-heading">
      <div className="explanation-header">
        <h2 id="why-heading">💬 {t('exp.title')}</h2>
        {strategy !== 'baseline' && (
          <span className="badge-strategy-tag">{t(`strategy.${strategy}`)}</span>
        )}
      </div>

      {/* “Why this recommendation?” Evidence Tied to Actual Values Grid */}
      {activeCard && seasons && seasons.length >= 3 ? (
        <div className="evidence-values-card">
          <div className="evidence-values-header">
            <h3>⚖️ {t('exp.evidenceTitle')}</h3>
            <p className="muted small">{t('exp.evidenceSubtitle')}</p>
          </div>

          <div className="evidence-values-grid">
            {/* Tile 1: Groundwater */}
            <div className="evidence-val-box">
              <div className="val-box-head">
                <span className="val-icon" aria-hidden>💧</span>
                <span className="val-title">{t('exp.val.gw')}</span>
                <span className="val-source">NASA GRACE</span>
              </div>
              <div className="val-box-num">{fmtPct(gwPctl, lang)}</div>
              <p className="val-box-why">{t('exp.val.gwWhy')}</p>
            </div>

            {/* Tile 2: Winter Rain & Dry Days */}
            <div className="evidence-val-box">
              <div className="val-box-head">
                <span className="val-icon" aria-hidden>🌧️</span>
                <span className="val-title">{t('exp.val.rain')}</span>
                <span className="val-source">NASA POWER</span>
              </div>
              <div className="val-box-num">
                {fmtMm(rainRabi, lang)} · {fmtInt(dryDays, lang)} {lang === 'bn' ? 'দিন শুষ্ক' : 'dry days'}
              </div>
              <p className="val-box-why">{t('exp.val.rainWhy')}</p>
            </div>

            {/* Tile 3: Root-Zone Moisture */}
            <div className="evidence-val-box">
              <div className="val-box-head">
                <span className="val-icon" aria-hidden>🌱</span>
                <span className="val-title">{t('exp.val.wet')}</span>
                <span className="val-source">NASA SMAP</span>
              </div>
              <div className="val-box-num">{fmtPct(wetnessRabi, lang)} ({t('season.rabi')})</div>
              <p className="val-box-why">{t('exp.val.wetWhy')}</p>
            </div>

            {/* Tile 4: Summer Heat Stress */}
            <div className="evidence-val-box">
              <div className="val-box-head">
                <span className="val-icon" aria-hidden>🌡️</span>
                <span className="val-title">{t('exp.val.heat')}</span>
                <span className="val-source">NASA POWER</span>
              </div>
              <div className="val-box-num">
                {fmtInt(heatDays, lang)} {lang === 'bn' ? 'দিন > ৩৫°সে' : 'days > 35°C'}
              </div>
              <p className="val-box-why">{t('exp.val.heatWhy')}</p>
            </div>

            {/* Tile 5: Net Economic & Water Impact */}
            {!isBase && m ? (
              <div className="evidence-val-box highlight">
                <div className="val-box-head">
                  <span className="val-icon" aria-hidden>💰</span>
                  <span className="val-title">{t('exp.val.economy')}</span>
                  <span className="val-source">Field Shift Model</span>
                </div>
                <div className="val-box-num">
                  {profitDiff > 0 ? '+' : ''}{fmtTaka(profitDiff, lang)} · {fmtMm(Math.abs(waterSaved), lang)} {waterSaved >= 0 ? (lang === 'bn' ? 'সাশ্রয়' : 'saved') : (lang === 'bn' ? 'বেশি' : 'more')}
                </div>
                <p className="val-box-why">{t('exp.val.economyWhy')}</p>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}

      {exp.status === 'error' && !sections ? (
        <p className="error">{t('exp.error')} <button className="link" onClick={exp.retry}>{t('res.retry')}</button></p>
      ) : !sections ? (
        <div className="skeleton-group" aria-label={t('exp.loading')}>
          <p className="muted">{t('exp.loading')}</p>
          <div className="skeleton" /><div className="skeleton short" /><div className="skeleton" />
        </div>
      ) : (
        <div className="explanation-body">
          <p className="exp-headline">{sections.headline}</p>

          <div className="exp-narrative-section">
            <h3>{t('exp.why')}</h3>
            <p>{formatEvidenceLine(sections.why)}</p>
          </div>

          {LIST_SECTIONS.map((s) =>
            sections[s.key].length ? (
              <div key={s.key} className="exp-list-section">
                <h3><span aria-hidden>{s.icon}</span> {t(s.title)}</h3>
                <ul className="exp-bullet-list">
                  {sections[s.key].map((line, i) => (
                    <li key={i}>{formatEvidenceLine(line)}</li>
                  ))}
                </ul>
              </div>
            ) : null,
          )}

          <div className="exp-confidence-section">
            <h3><span aria-hidden>📊</span> {t('exp.confidence')}</h3>
            <p>{formatEvidenceLine(sections.confidence)}</p>
          </div>

          <p className="muted small source-note">
            {exp.data!.source === 'template' ? t('exp.sourceTemplate') : t('exp.sourceAi')}
          </p>
        </div>
      )}
    </section>
  )
}
