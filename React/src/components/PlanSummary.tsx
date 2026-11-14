import type { StrategyCard } from '../api/types'
import { useI18n } from '../i18n/I18nProvider'
import { fmtKg, fmtMm, fmtTaka } from '../lib/format'
import { cropIcon, type Names } from '../lib/names'

export function SeasonTiles({ card, names }: { card: StrategyCard; names: Names }) {
  const { t } = useI18n()
  return (
    <ol className="season-tiles">
      {card.seasons.map((s, i) => (
        <li key={s.season} className={`season-tile season-${s.season} ${s.crop === 'Fallow' ? 'fallow' : ''}`}>
          <span className="season-name">{t(`season.${s.season}`)}</span>
          <span className="season-sub">{t(`season.${s.season}.sub`)}</span>
          <span className="crop-icon" aria-hidden>{cropIcon(s.crop, s.category)}</span>
          <span className="crop-name">{names.crop(s.crop)}</span>
          {i < card.seasons.length - 1 ? <span className="arrow" aria-hidden>→</span> : null}
        </li>
      ))}
    </ol>
  )
}

type Tone = 'good' | 'bad' | 'neutral'

function Stat({ icon, label, value, unit, delta, tone }: {
  icon: string; label: string; value: string; unit?: string; delta?: string; tone: Tone
}) {
  const arrow = tone === 'good' ? '▲' : tone === 'bad' ? '▼' : '●'
  return (
    <div className="stat">
      <div className="stat-label"><span aria-hidden>{icon}</span> {label}</div>
      <div className="stat-value">{value}{unit ? <small> {unit}</small> : null}</div>
      {delta ? <div className={`stat-delta ${tone}`}><span aria-hidden>{arrow}</span> {delta}</div> : null}
    </div>
  )
}

export function StatTiles({ card }: { card: StrategyCard }) {
  const { t, lang } = useI18n()
  const m = card.metrics
  const isBase = card.key === 'baseline'

  const profitTone: Tone = m.profit_change_bdt_bigha > 0 ? 'good' : m.profit_change_bdt_bigha < 0 ? 'bad' : 'neutral'
  const profitDelta = isBase
    ? undefined
    : m.profit_change_bdt_bigha === 0
      ? t('delta.same')
      : t(m.profit_change_bdt_bigha > 0 ? 'delta.more' : 'delta.less', { v: fmtTaka(Math.abs(m.profit_change_bdt_bigha), lang) })

  const waterTone: Tone = m.water_saved_mm > 0 ? 'good' : m.water_saved_mm < 0 ? 'bad' : 'neutral'
  const waterDelta = isBase
    ? undefined
    : m.water_saved_mm === 0
      ? t('delta.same')
      : t(m.water_saved_mm > 0 ? 'water.saved' : 'water.more', { v: fmtMm(Math.abs(m.water_saved_mm), lang) })

  const n = m.soil_n_change_kg_bigha
  const soilValue = n === 0 ? t('soil.flat') : t(n > 0 ? 'soil.gain' : 'soil.loss', { v: fmtKg(Math.abs(n), lang) })
  const vs = m.soil_n_change_vs_baseline_kg_bigha
  const soilTone: Tone = isBase ? (n >= 0 ? 'good' : 'bad') : vs > 0 ? 'good' : vs < 0 ? 'bad' : 'neutral'
  const soilDelta = isBase
    ? undefined
    : vs === 0
      ? t('delta.same')
      : t(vs > 0 ? 'soil.better' : 'soil.worse', { v: fmtKg(Math.abs(vs), lang) })

  return (
    <div className="stats">
      <Stat icon="৳" label={t('stat.profit')} value={fmtTaka(m.profit_bdt_bigha, lang)} unit={t('stat.perBigha')} delta={profitDelta} tone={profitTone} />
      <Stat icon="💧" label={t('stat.water')} value={fmtMm(m.water_mm, lang)} delta={waterDelta} tone={waterTone} />
      <Stat icon="🌱" label={t('stat.soil')} value={soilValue} delta={soilDelta} tone={soilTone} />
    </div>
  )
}
