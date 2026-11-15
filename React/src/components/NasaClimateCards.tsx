import type { DistrictInfo, Recommendation, StrategyCard } from '../api/types'
import { useI18n } from '../i18n/I18nProvider'
import { fmtInt, fmtMm, fmtNum, fmtPct } from '../lib/format'
import { NASA_PRODUCTS } from '../lib/nasaProducts'

interface NasaClimateCardsProps {
  card: StrategyCard
  district: DistrictInfo
  dataInfo: Recommendation['data_info']
}

export function NasaClimateCards({ card, district, dataInfo }: NasaClimateCardsProps) {
  const { t, lang } = useI18n()
  const seasons = card.seasons

  // Rainfall stats
  const totalRain = seasons.reduce((acc, s) => acc + s.rain_mm, 0)
  const dryDaysRabi = seasons[0]?.dry_spell_days ?? 0

  // Soil moisture stats
  const rabiWetness = seasons[0]?.rootzone_wetness_pct ?? 0
  const isWetnessLow = rabiWetness < 30

  // Temperature stats
  const avgTemp = seasons.reduce((acc, s) => acc + s.mean_temp_c, 0) / Math.max(seasons.length, 1)
  const summerHeatDays = seasons[1]?.heat_stress_days ?? 0

  // Groundwater stats
  const gwPctl = seasons[0]?.groundwater_pctl ?? district.groundwater_pctl
  const isGwLow = gwPctl < 30
  const isGwMed = gwPctl >= 30 && gwPctl <= 60

  const periodLabel = `${dataInfo.climate_years[0]}–${dataInfo.climate_years[1]}`

  return (
    <section className="card nasa-climate-section" aria-labelledby="climate-heading">
      <div className="nasa-climate-header">
        <div>
          <h2 id="climate-heading">🛰️ {t('climate.sectionTitle')}</h2>
          <p className="muted small">{t('climate.sectionSubtitle')}</p>
        </div>
        <span className="nasa-period-badge">
          📅 {periodLabel} · {lang === 'bn' ? '২০ বছরের ভিত্তি' : '20-Yr Baseline'}
        </span>
      </div>

      <div className="nasa-climate-grid">
        {/* Card 1: Rainfall */}
        <div className="climate-card">
          <div className="climate-card-top">
            <div className="climate-card-title-group">
              <span className="climate-icon" aria-hidden>🌧️</span>
              <div>
                <h3 className="climate-card-name">{t('climate.rain.title')}</h3>
                <span className="climate-source-tag">NASA POWER · {periodLabel}</span>
              </div>
            </div>
            <a
              href={NASA_PRODUCTS.power.url}
              target="_blank"
              rel="noopener noreferrer"
              className="climate-product-link"
              title={`${t('link.nasaProduct')}: NASA POWER (${t('link.external')})`}
            >
              NASA POWER ↗
            </a>
          </div>

          <div className="climate-metric-hero">
            <span className="hero-num">{fmtMm(totalRain, lang)}</span>
            <span className="hero-label">{t('climate.rain.annual')}</span>
          </div>

          <div className="climate-season-breakdown">
            <div className="breakdown-item">
              <span className="season-tag">{t('season.rabi')}</span>
              <strong>{fmtMm(seasons[0]?.rain_mm ?? 0, lang)}</strong>
            </div>
            <div className="breakdown-item">
              <span className="season-tag">{t('season.kharif1')}</span>
              <strong>{fmtMm(seasons[1]?.rain_mm ?? 0, lang)}</strong>
            </div>
            <div className="breakdown-item">
              <span className="season-tag">{t('season.kharif2')}</span>
              <strong>{fmtMm(seasons[2]?.rain_mm ?? 0, lang)}</strong>
            </div>
          </div>

          <div className="climate-card-alert warning">
            <span aria-hidden>☀️</span>
            <span>
              {t('climate.rain.dryDays')}: <strong>{fmtInt(dryDaysRabi, lang)} {lang === 'bn' ? 'দিন' : 'days'}</strong>
            </span>
          </div>
        </div>

        {/* Card 2: Soil Moisture */}
        <div className="climate-card">
          <div className="climate-card-top">
            <div className="climate-card-title-group">
              <span className="climate-icon" aria-hidden>🌱</span>
              <div>
                <h3 className="climate-card-name">{t('climate.smap.title')}</h3>
                <span className="climate-source-tag">NASA SMAP L4 · 2015–{dataInfo.latest_year}</span>
              </div>
            </div>
            <a
              href={NASA_PRODUCTS.smap.url}
              target="_blank"
              rel="noopener noreferrer"
              className="climate-product-link"
              title={`${t('link.nasaProduct')}: NASA SMAP (${t('link.external')})`}
            >
              NASA SMAP ↗
            </a>
          </div>

          <div className="climate-metric-hero">
            <span className="hero-num">{fmtPct(rabiWetness, lang)}</span>
            <span className="hero-label">{t('climate.smap.rabi')}</span>
          </div>

          <div className="climate-bars-group">
            <div className="climate-bar-row">
              <span className="bar-label">{t('season.rabi')}</span>
              <div className="bar-track">
                <div
                  className="bar-fill soil"
                  style={{ width: `${Math.min(100, Math.max(5, rabiWetness))}%` }}
                />
              </div>
              <span className="bar-val">{fmtPct(rabiWetness, lang)}</span>
            </div>
            <div className="climate-bar-row">
              <span className="bar-label">{t('season.kharif1')}</span>
              <div className="bar-track">
                <div
                  className="bar-fill soil"
                  style={{ width: `${Math.min(100, Math.max(5, seasons[1]?.rootzone_wetness_pct ?? 0))}%` }}
                />
              </div>
              <span className="bar-val">{fmtPct(seasons[1]?.rootzone_wetness_pct ?? 0, lang)}</span>
            </div>
            <div className="climate-bar-row">
              <span className="bar-label">{t('season.kharif2')}</span>
              <div className="bar-track">
                <div
                  className="bar-fill soil"
                  style={{ width: `${Math.min(100, Math.max(5, seasons[2]?.rootzone_wetness_pct ?? 0))}%` }}
                />
              </div>
              <span className="bar-val">{fmtPct(seasons[2]?.rootzone_wetness_pct ?? 0, lang)}</span>
            </div>
          </div>

          <div className={`climate-card-alert ${isWetnessLow ? 'warning' : 'ok'}`}>
            <span aria-hidden>{isWetnessLow ? '⚠️' : '✅'}</span>
            <span>
              {isWetnessLow ? t('climate.smap.deficit') : t('climate.smap.adequate')} · {t('climate.smap.depth')}
            </span>
          </div>
        </div>

        {/* Card 3: Temperature */}
        <div className="climate-card">
          <div className="climate-card-top">
            <div className="climate-card-title-group">
              <span className="climate-icon" aria-hidden>🌡️</span>
              <div>
                <h3 className="climate-card-name">{t('climate.temp.title')}</h3>
                <span className="climate-source-tag">NASA POWER · {periodLabel}</span>
              </div>
            </div>
            <a
              href={NASA_PRODUCTS.power.url}
              target="_blank"
              rel="noopener noreferrer"
              className="climate-product-link"
              title={`${t('link.nasaProduct')}: NASA POWER (${t('link.external')})`}
            >
              NASA POWER ↗
            </a>
          </div>

          <div className="climate-metric-hero">
            <span className="hero-num">{fmtNum(avgTemp, lang)}°C</span>
            <span className="hero-label">{t('climate.temp.mean')}</span>
          </div>

          <div className="climate-season-breakdown">
            <div className="breakdown-item">
              <span className="season-tag">{t('season.rabi')}</span>
              <strong>{fmtNum(seasons[0]?.mean_temp_c ?? 0, lang)}°C</strong>
            </div>
            <div className="breakdown-item">
              <span className="season-tag">{t('season.kharif1')}</span>
              <strong>{fmtNum(seasons[1]?.mean_temp_c ?? 0, lang)}°C</strong>
            </div>
            <div className="breakdown-item">
              <span className="season-tag">{t('season.kharif2')}</span>
              <strong>{fmtNum(seasons[2]?.mean_temp_c ?? 0, lang)}°C</strong>
            </div>
          </div>

          <div className={`climate-card-alert ${summerHeatDays > 7 ? 'warning' : 'ok'}`}>
            <span aria-hidden>🔥</span>
            <span>
              {t('climate.temp.heatDays')}: <strong>{fmtInt(summerHeatDays, lang)} {lang === 'bn' ? 'দিন' : 'days'}</strong>
            </span>
          </div>
        </div>

        {/* Card 4: Groundwater */}
        <div className="climate-card">
          <div className="climate-card-top">
            <div className="climate-card-title-group">
              <span className="climate-icon" aria-hidden>💧</span>
              <div>
                <h3 className="climate-card-name">{t('climate.grace.title')}</h3>
                <span className="climate-source-tag">NASA GRACE / GRACE-FO · {periodLabel}</span>
              </div>
            </div>
            <a
              href={NASA_PRODUCTS.grace.url}
              target="_blank"
              rel="noopener noreferrer"
              className="climate-product-link"
              title={`${t('link.nasaProduct')}: NASA GRACE (${t('link.external')})`}
            >
              NASA GRACE ↗
            </a>
          </div>

          <div className="climate-metric-hero">
            <span className="hero-num">{fmtPct(gwPctl, lang)}</span>
            <span className="hero-label">{t('climate.grace.pctl')}</span>
          </div>

          <div className="climate-grace-status-wrapper">
            <div className="grace-scale-bar" aria-hidden>
              <div className="grace-scale-zones">
                <span className="zone-low" />
                <span className="zone-med" />
                <span className="zone-high" />
              </div>
              <div
                className="grace-indicator-dot"
                style={{ left: `${Math.min(100, Math.max(0, gwPctl))}%` }}
              />
            </div>
            <div className="grace-scale-labels muted small">
              <span>0% ({lang === 'bn' ? 'সংকট' : 'Low'})</span>
              <span>30%</span>
              <span>100% ({lang === 'bn' ? 'পর্যাপ্ত' : 'Adequate'})</span>
            </div>
          </div>

          <div className={`climate-card-alert ${isGwLow ? 'danger' : isGwMed ? 'warning' : 'ok'}`}>
            <span aria-hidden>{isGwLow ? '⚠️' : '✅'}</span>
            <span>
              {isGwLow
                ? t('climate.grace.low')
                : isGwMed
                  ? t('climate.grace.med')
                  : t('climate.grace.high')}
            </span>
          </div>
        </div>
      </div>
    </section>
  )
}
