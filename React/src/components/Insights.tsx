import { useState } from 'react'
import type { Recommendation, ScenarioId, StrategyCard, StrategyKey } from '../api/types'
import { useI18n } from '../i18n/I18nProvider'
import { fmtMm, fmtMonthDay, fmtSigned, fmtTaka, fmtWindow, fmtYear } from '../lib/format'
import { cropIcon, type Names } from '../lib/names'

/* ---------------- Profit range over 20 years of NASA weather ---------------- */

export function RangeBar({ card, years }: { card: StrategyCard; years: [number, number] }) {
  const { t, lang } = useI18n()
  const r = card.range_20yr
  const lo = r.profit_bad_year_bdt_bigha
  const hi = r.profit_good_year_bdt_bigha
  const span = Math.max(hi - lo, 1)
  const pos = (v: number) => `${Math.min(100, Math.max(0, ((v - lo) / span) * 100))}%`

  return (
    <section className="card">
      <h2>📈 {t('range.title')}</h2>
      <div className="range-track" aria-hidden>
        <span className="range-fill" />
        <span className="range-dot bad" style={{ left: pos(lo) }} />
        <span className="range-dot normal" style={{ left: pos(r.profit_typical_bdt_bigha) }} />
        <span className="range-dot good" style={{ left: pos(hi) }} />
      </div>
      <div className="range-labels">
        <div><span className="muted">{t('range.bad')}</span><strong>{fmtTaka(lo, lang)}</strong></div>
        <div><span className="muted">{t('range.normal')}</span><strong>{fmtTaka(r.profit_typical_bdt_bigha, lang)}</strong></div>
        <div><span className="muted">{t('range.good')}</span><strong>{fmtTaka(hi, lang)}</strong></div>
      </div>
      <p className="muted small">
        {t('stat.perBigha')} · {t('range.note', { from: fmtYear(years[0], lang), to: fmtYear(years[1], lang) })}
      </p>
    </section>
  )
}

/* ---------------- Year calendar: growing vs bare days per season ---------------- */

export function CropCalendar({ card, names }: { card: StrategyCard; names: Names }) {
  const { t, lang } = useI18n()
  return (
    <section className="card">
      <h2>🗓️ {t('cal.title')}</h2>
      <div className="legend">
        <span><i className="swatch grow" /> {t('cal.growing')}</span>
        <span><i className="swatch bare" /> {t('cal.bare')}</span>
      </div>
      <ul className="calendar">
        {card.seasons.map((s) => {
          const pct = Math.min(100, (s.active_days / Math.max(s.season_days, 1)) * 100)
          return (
            <li key={s.season}>
              <div className="cal-head">
                <strong>{t(`season.${s.season}`)}</strong>
                <span className="muted small">{t(`season.${s.season}.sub`)}</span>
              </div>
              <div className="cal-bar" aria-hidden>
                <span className="cal-grow" style={{ width: `${pct}%` }}>
                  {cropIcon(s.crop, s.category)} {names.crop(s.crop)}
                </span>
              </div>
              <div className="cal-meta small">
                <span>{t('cal.days', { active: fmtYear(s.active_days, lang), bare: fmtYear(s.bare_days, lang) })}</span>
                {s.crop !== 'Fallow' ? (
                  <span>
                    {t('cal.sow')}: {fmtWindow(s.sowing_window, lang)} · {t('cal.harvest')}: {fmtWindow(s.harvest_window, lang)}
                  </span>
                ) : null}
              </div>
            </li>
          )
        })}
      </ul>
    </section>
  )
}

/* ---------------- What-if weather scenarios ---------------- */

const SCENARIO_ICONS: Record<ScenarioId, string> = { normal: '🌤️', drought: '🏜️', heatwave: '🔥', wet_monsoon: '🌧️' }

export function ScenarioPanel({ rec, strategy, names }: { rec: Recommendation; strategy: StrategyKey; names: Names }) {
  const { t, lang } = useI18n()
  const [active, setActive] = useState<ScenarioId>('drought')
  const normal = rec.scenarios.find((s) => s.id === 'normal')
  const sc = rec.scenarios.find((s) => s.id === active) ?? normal
  if (!sc || !normal) return null

  const row = sc.strategies.find((r) => r.key === strategy)
  const normalRow = normal.strategies.find((r) => r.key === strategy)
  const baseRow = sc.strategies.find((r) => r.key === 'baseline')
  if (!row || !normalRow) return null

  const planChanged = row.crops.join('|') !== normalRow.crops.join('|')
  const dProfit = row.metrics.profit_bdt_bigha - normalRow.metrics.profit_bdt_bigha
  const dWater = Math.round((row.metrics.water_mm - normalRow.metrics.water_mm) * 10) / 10

  return (
    <section className="card">
      <h2>⛅ {t('sc.title')}</h2>
      <div className="scenario-buttons" role="tablist">
        {rec.scenarios.map((s) => (
          <button
            key={s.id}
            role="tab"
            aria-selected={s.id === sc.id}
            className={s.id === sc.id ? 'on' : ''}
            onClick={() => setActive(s.id)}
          >
            <span aria-hidden>{SCENARIO_ICONS[s.id]}</span> {t(`sc.${s.id}`)}
          </button>
        ))}
      </div>
      {sc.id !== 'normal' ? (
        <p className="muted small">
          {t('sc.params', {
            rain: fmtSigned(sc.rain_pct, lang, '%'),
            temp: fmtSigned(sc.temp_c, lang, '°C'),
            gw: fmtSigned(sc.gw_pct, lang, '%'),
          })}
        </p>
      ) : null}

      <div className="scenario-result">
        {planChanged ? (
          <>
            <p><strong>{t('sc.newPlan')}</strong></p>
            <p className="crop-chain">{row.crops.map((c) => names.crop(c)).join(' → ')}</p>
          </>
        ) : (
          <p>{t('sc.samePlan')} <span className="crop-chain">{row.crops.map((c) => names.crop(c)).join(' → ')}</span></p>
        )}
        <div className="mini-stats">
          <div>
            <span className="muted">{t('sc.profit')}</span>
            <strong>{fmtTaka(row.metrics.profit_bdt_bigha, lang)}</strong>
            {sc.id !== 'normal' ? (
              <span className={`small ${dProfit >= 0 ? 'good' : 'bad'}`}>
                {t('sc.vsNormal', { v: `${dProfit > 0 ? '+' : dProfit < 0 ? '−' : ''}${fmtTaka(Math.abs(dProfit), lang)}` })}
              </span>
            ) : null}
          </div>
          <div>
            <span className="muted">{t('sc.water')}</span>
            <strong>{fmtMm(row.metrics.water_mm, lang)}</strong>
            {sc.id !== 'normal' ? (
              <span className={`small ${dWater <= 0 ? 'good' : 'bad'}`}>
                {t('sc.vsNormal', { v: `${dWater > 0 ? '+' : dWater < 0 ? '−' : ''}${fmtMm(Math.abs(dWater), lang)}` })}
              </span>
            ) : null}
          </div>
        </div>
        {baseRow && strategy !== 'baseline' ? (
          <p className="muted small">{t('sc.current', { v: fmtTaka(baseRow.metrics.profit_bdt_bigha, lang) })}</p>
        ) : null}
      </div>
    </section>
  )
}

/* ---------------- Peak climate-stress windows ---------------- */

export function StressChart({ rec }: { rec: Recommendation }) {
  const { t, lang } = useI18n()
  const sw = rec.stress_windows
  return (
    <section className="card">
      <h2>🌡️ {t('stress.title')}</h2>
      <p className="muted small">{t('stress.desc')}</p>
      <div className="stress-grid">
        {sw.seasons.map((s) => {
          const max = Math.max(...s.profile, 1e-9)
          return (
            <figure key={s.season} className="stress">
              <figcaption>
                <strong>{t(`season.${s.season}`)}</strong>
                <span className="small">
                  {t('stress.peak', { from: fmtMonthDay(s.peak_start, lang), to: fmtMonthDay(s.peak_end, lang) })}
                </span>
              </figcaption>
              <div className="bars" aria-hidden>
                {s.profile.map((w, i) => (
                  <span
                    key={i}
                    className={i === s.peak_index ? 'peak' : ''}
                    style={{ height: `${Math.max(4, (w / max) * 100)}%` }}
                  />
                ))}
              </div>
              <div className="bars-axis small muted" aria-hidden>
                <span>{t(`season.${s.season}.sub`).split('·')[1]?.trim()}</span>
              </div>
            </figure>
          )
        })}
      </div>
      <p className="muted small">
        {sw.source === 'trained_attention' ? t('stress.sourceModel') : t('stress.sourceProxy')}
      </p>
    </section>
  )
}

/* ---------------- Compare all plans ---------------- */

export function CompareList({ rec, selected, onSelect, names }: {
  rec: Recommendation; selected: StrategyKey; onSelect: (k: StrategyKey) => void; names: Names
}) {
  const { t, lang } = useI18n()
  return (
    <section className="card">
      <h2>⚖️ {t('cmp.title')}</h2>
      <ul className="compare">
        {rec.strategies.map((s) => (
          <li key={s.key}>
            <button className={`compare-row ${s.key === selected ? 'on' : ''}`} onClick={() => onSelect(s.key)}>
              <span className="compare-name">
                {t(`strategy.${s.key}`)}
                {s.key === rec.recommended_strategy ? <span className="badge">⭐ {t('res.forYou')}</span> : null}
              </span>
              <span className="crop-chain small">{s.crops.map((c) => names.crop(c)).join(' → ')}</span>
              <span className="compare-metrics">
                <span>{fmtTaka(s.metrics.profit_bdt_bigha, lang)}</span>
                <span>💧 {fmtMm(s.metrics.water_mm, lang)}</span>
                <span>🌱 N {fmtSigned(s.metrics.soil_n_change_kg_bigha, lang, lang === 'bn' ? ' কেজি' : ' kg')}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}
