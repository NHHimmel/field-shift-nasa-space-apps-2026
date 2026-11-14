import type { Recommendation } from '../api/types'
import { useI18n } from '../i18n/I18nProvider'
import type { StringKey } from '../i18n/strings'

const SOURCE_ROLE_KEYS: Record<string, StringKey> = {
  'NASA POWER': 'evidence.power',
  'NASA SMAP': 'evidence.smap',
  'NASA GRACE / GRACE-FO': 'evidence.grace',
  'NASA MODIS / ECOSTRESS': 'evidence.modis',
}

export function EvidencePanel({ rec }: { rec: Recommendation }) {
  const { t, lang } = useI18n()
  const info = rec.data_info.inference
  const method = info?.recommendation_method === 'trained_neural_policy'
    ? t('res.neural')
    : t('res.search')
  const stress = info?.stress_window_source ?? rec.stress_windows.source
  const sources = rec.data_info.nasa_sources ?? []

  return (
    <details className="card evidence-panel">
      <summary>🛰️ {t('res.provenanceTitle')}</summary>

      <div className="evidence-grid">
        <div className="evidence-item">
          <span className="muted small">{t('res.method')}</span>
          <strong>{method}</strong>
          {!info?.neural_policy_used ? <span className="muted small">{t('res.provenanceNote')}</span> : null}
        </div>
        <div className="evidence-item">
          <span className="muted small">{t('res.stressSource')}</span>
          <strong>{stress === 'trained_attention' ? t('res.trainedAttention') : t('res.physicsProxy')}</strong>
        </div>
        <div className="evidence-item">
          <span className="muted small">{t('res.dataPeriod')}</span>
          <strong>{rec.data_info.climate_years[0]}–{rec.data_info.climate_years[1]}</strong>
        </div>
        <div className="evidence-item">
          <span className="muted small">{t('res.latest')}</span>
          <strong>{rec.data_info.latest_year}</strong>
        </div>
        <div className="evidence-item evidence-wide">
          <span className="muted small">{t('res.location')}</span>
          <strong>
            {rec.district.name_en} · {rec.district.latitude.toFixed(2)}°, {rec.district.longitude.toFixed(2)}°
          </strong>
        </div>
      </div>

      {sources.length ? (
        <div className="evidence-sources">
          <h3>{t('res.dataSources')}</h3>
          <ul>
            {sources.map((source) => {
              const roleKey = SOURCE_ROLE_KEYS[source.name]
              const statusLabel = source.status === 'active' ? t('res.active') : source.status === 'derived' ? t('res.derived') : t('res.fallback')
              return (
                <li key={source.name}>
                  <span>
                    <strong>{source.name}</strong>
                    <span className="muted">{roleKey ? t(roleKey) : source.role}</span>
                  </span>
                  <span className={`source-status ${source.status}`}>{statusLabel}</span>
                </li>
              )
            })}
          </ul>
        </div>
      ) : null}

      {info?.policy_status ? (
        <details className="evidence-tech">
          <summary>{lang === 'bn' ? 'প্রযুক্তিগত মডেল তথ্য' : 'Technical model details'}</summary>
          <p className="muted small">{info.policy_status}</p>
          {info.checkpoint_name ? <p className="muted small">Checkpoint: {info.checkpoint_name}</p> : null}
        </details>
      ) : null}
    </details>
  )
}
