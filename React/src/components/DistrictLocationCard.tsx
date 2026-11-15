import type { DistrictInfo } from '../api/types'
import { useI18n } from '../i18n/I18nProvider'
import { fmtPct } from '../lib/format'
import type { Names } from '../lib/names'
import { BangladeshMap } from './BangladeshMap'

export function DistrictLocationCard({
  district,
  names,
}: {
  district: DistrictInfo
  names: Names
}) {
  const { t, lang } = useI18n()
  const districtName = names.district(district.key)
  const divisionName = lang === 'bn' ? district.division_bn : district.division_en
  const isGwLow = district.groundwater_pctl < 30

  const mapsUrl = `https://www.google.com/maps?q=${district.latitude},${district.longitude}`
  const worldviewUrl = `https://worldview.earthdata.nasa.gov/?v=${(district.longitude - 0.6).toFixed(2)},${(district.latitude - 0.5).toFixed(2)},${(district.longitude + 0.6).toFixed(2)},${(district.latitude + 0.5).toFixed(2)}`

  return (
    <section className="card district-location-card" aria-labelledby="loc-heading">
      <div className="district-card-header">
        <h2 id="loc-heading">📍 {t('loc.title')}</h2>
        <span className="district-badge-division">{divisionName}</span>
      </div>

      <div className="district-card-grid">
        <div className="district-map-pane">
          <BangladeshMap
            latitude={district.latitude}
            longitude={district.longitude}
            districtName={districtName}
            divisionName={divisionName}
            lang={lang}
          />
        </div>

        <div className="district-info-pane">
          <div className="district-info-title-group">
            <h3 className="district-main-name">{districtName}</h3>
            <p className="district-division-sub">{divisionName} {lang === 'bn' ? 'বিভাগ' : 'Division'}</p>
          </div>

          <dl className="district-props">
            <div className="district-prop">
              <dt className="muted small">🌾 {t('loc.aez')}</dt>
              <dd className="prop-val-highlight">{district.agro_ecological_zone}</dd>
            </div>

            <div className="district-prop">
              <dt className="muted small">🧭 {t('loc.coords')}</dt>
              <dd>
                <strong>{district.latitude.toFixed(2)}° N, {district.longitude.toFixed(2)}° E</strong>
              </dd>
            </div>

            <div className="district-prop">
              <dt className="muted small">💧 {t('loc.gwBaseline')}</dt>
              <dd className="prop-gw-row">
                <strong>{fmtPct(district.groundwater_pctl, lang)}</strong>
                {isGwLow ? (
                  <span className="badge-gw-status low">{t('climate.grace.low')}</span>
                ) : (
                  <span className="badge-gw-status ok">{t('climate.grace.med')}</span>
                )}
              </dd>
            </div>

            <div className="district-prop">
              <dt className="muted small">🚜 {t('loc.irrigation')}</dt>
              <dd>
                {district.default_irrigation ? (
                  <span className="irr-tag available">💧 {t('loc.irrAvailable')}</span>
                ) : (
                  <span className="irr-tag rainfed">🌧️ {t('loc.irrRainfed')}</span>
                )}
              </dd>
            </div>
          </dl>

          <div className="district-links-row no-print">
            <a
              href={mapsUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn--secondary btn--sm map-ext-link"
            >
              🗺️ {t('loc.openMap')} <span aria-hidden>↗</span>
            </a>
            <a
              href={worldviewUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn--secondary btn--sm map-ext-link"
            >
              🛰️ {t('loc.nasaWorldview')} <span aria-hidden>↗</span>
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}
