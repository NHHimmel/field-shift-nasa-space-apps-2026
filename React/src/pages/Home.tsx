import { useMemo, useState, type FormEvent } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { getMeta } from '../api/client'
import { PRIORITIES, SEASONS, type Priority, type SeasonKey, type SoilLevel } from '../api/types'
import { ChoiceGroup } from '../components/ChoiceGroup'
import { useI18n } from '../i18n/I18nProvider'
import { fmtPct } from '../lib/format'
import { makeNames } from '../lib/names'
import { paramsFromRequest, requestFromParams } from '../lib/query'
import { useAsync } from '../lib/useAsync'

type Irr = 'yes' | 'no' | 'unknown'
const PRIORITY_ICONS: Record<Priority, string> = {
  balanced: '⚖️',
  water_conservation: '💧',
  soil_health: '🌱',
  high_value_vegetable: '🥬',
}

export default function Home() {
  const { t, lang } = useI18n()
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const initial = useMemo(() => requestFromParams(params), [params])
  const meta = useAsync(() => getMeta(), [])
  const names = makeNames(meta.data, lang)

  const [district, setDistrict] = useState(initial?.district ?? '')
  const [irr, setIrr] = useState<Irr>(initial?.irrigation === true ? 'yes' : initial?.irrigation === false ? 'no' : 'unknown')
  const [priority, setPriority] = useState<Priority>(initial?.priority ?? 'balanced')
  const [soil, setSoil] = useState<SoilLevel>(initial?.soil_fertility ?? 'unknown')
  const [current, setCurrent] = useState<Record<SeasonKey, string> | null>(initial?.current_crops ?? null)
  const [excluded, setExcluded] = useState<string[]>(initial?.excluded_crops ?? [])
  const [showError, setShowError] = useState(false)

  const groups = useMemo(() => {
    const byDiv = new Map<string, { label: string; items: { key: string; label: string }[] }>()
    for (const d of meta.data?.districts ?? []) {
      const g = byDiv.get(d.division_en) ?? { label: lang === 'bn' ? d.division_bn : d.division_en, items: [] }
      g.items.push({ key: d.key, label: lang === 'bn' ? d.name_bn : d.name_en })
      byDiv.set(d.division_en, g)
    }
    const collator = new Intl.Collator(lang === 'bn' ? 'bn' : 'en')
    return [...byDiv.values()]
      .map((g) => ({ ...g, items: g.items.sort((a, b) => collator.compare(a.label, b.label)) }))
      .sort((a, b) => collator.compare(a.label, b.label))
  }, [meta.data, lang])

  const selected = district ? names.districtInfo(district) : undefined
  const avoidable = useMemo(() => {
    const all = new Set<string>()
    for (const s of SEASONS) for (const c of meta.data?.season_crops[s] ?? []) if (c !== 'Fallow') all.add(c)
    return [...all]
  }, [meta.data])

  function onSubmit(e: FormEvent) {
    e.preventDefault()
    if (!district) {
      setShowError(true)
      return
    }
    const p = paramsFromRequest({
      district,
      irrigation: irr === 'yes' ? true : irr === 'no' ? false : null,
      priority,
      soil_fertility: soil,
      current_crops: current,
      excluded_crops: excluded,
    })
    navigate(`/result?${p.toString()}`)
  }

  const currentValue = (s: SeasonKey) => current?.[s] ?? meta.data?.default_current_crops[s] ?? ''

  function loadDemo() {
    const demo = meta.data?.districts.find((d) => d.key.toLowerCase() === 'rajshahi') ?? meta.data?.districts[0]
    if (!demo) return
    setDistrict(demo.key)
    setIrr(demo.default_irrigation ? 'yes' : 'no')
    setPriority('balanced')
    setSoil('medium')
    setCurrent(null)
    setExcluded([])
    setShowError(false)
  }

  return (
    <div className="home">
      <section className="hero">
        <h1>{t('home.title')}</h1>
        <p>{t('home.subtitle')}</p>
      </section>

      <form className="card form" onSubmit={onSubmit} noValidate>
        <label className="field">
          <span className="field-label"><span className="step">1</span> {t('home.q1')}</span>
          <select
            value={district}
            onChange={(e) => { setDistrict(e.target.value); setShowError(false) }}
            disabled={!meta.data}
            aria-invalid={showError && !district}
          >
            <option value="">{meta.data ? t('home.q1.placeholder') : t('home.loadingMeta')}</option>
            {groups.map((g) => (
              <optgroup key={g.label} label={g.label}>
                {g.items.map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
              </optgroup>
            ))}
          </select>
          {showError && !district ? <span className="error small">{t('home.needDistrict')}</span> : null}
          {meta.data ? <button type="button" className="link demo-link" onClick={loadDemo}>▶ {t('home.demo')}</button> : null}
          {meta.status === 'error' ? (
            <span className="error small">{t('res.error')} <button type="button" className="link" onClick={meta.retry}>{t('res.retry')}</button></span>
          ) : null}
        </label>

        {selected && selected.groundwater_pctl < 30 ? (
          <p className="note warn">
            💧 {t('home.gwLow', { district: names.district(selected.key), pct: fmtPct(selected.groundwater_pctl, lang) })}
          </p>
        ) : null}

        <ChoiceGroup<Irr>
          name="irrigation"
          legend={<><span className="step">2</span> {t('home.q2')}</>}
          value={irr}
          onChange={setIrr}
          choices={[
            { value: 'yes', label: t('irr.yes'), icon: '🚿' },
            { value: 'no', label: t('irr.no'), icon: '🚫' },
            { value: 'unknown', label: t('irr.unknown'), icon: '🤔' },
          ]}
        />

        <ChoiceGroup<Priority>
          name="priority"
          variant="cards"
          legend={<><span className="step">3</span> {t('home.q3')}</>}
          value={priority}
          onChange={setPriority}
          choices={PRIORITIES.map((p) => ({
            value: p,
            label: t(`strategy.${p}`),
            desc: t(`strategy.${p}.desc`),
            icon: PRIORITY_ICONS[p],
          }))}
        />

        <details className="more" open={Boolean(initial?.current_crops || initial?.excluded_crops.length || (initial && initial.soil_fertility !== 'unknown'))}>
          <summary>{t('home.more')}</summary>

          <ChoiceGroup<SoilLevel>
            name="soil"
            legend={t('home.soil')}
            value={soil}
            onChange={setSoil}
            choices={(['low', 'medium', 'high', 'unknown'] as SoilLevel[]).map((s) => ({ value: s, label: t(`soil.${s}`) }))}
          />

          <fieldset className="current">
            <legend>{t('home.current')}</legend>
            {SEASONS.map((s) => (
              <label key={s} className="field inline">
                <span>{t(`season.${s}`)} <small className="muted">{t(`season.${s}.sub`)}</small></span>
                <select
                  value={currentValue(s)}
                  onChange={(e) => {
                    const base = meta.data!.default_current_crops
                    setCurrent({ ...(current ?? base), [s]: e.target.value })
                  }}
                  disabled={!meta.data}
                >
                  {(meta.data?.season_crops[s] ?? []).map((c) => <option key={c} value={c}>{names.crop(c)}</option>)}
                </select>
              </label>
            ))}
          </fieldset>

          <fieldset>
            <legend>{t('home.avoid')}</legend>
            <div className="chips">
              {avoidable.map((c) => {
                const on = excluded.includes(c)
                return (
                  <button
                    type="button"
                    key={c}
                    className={`chip ${on ? 'on' : ''}`}
                    aria-pressed={on}
                    onClick={() => setExcluded(on ? excluded.filter((x) => x !== c) : [...excluded, c])}
                  >
                    {on ? '✕ ' : ''}{names.crop(c)}
                  </button>
                )
              })}
            </div>
          </fieldset>
        </details>

        <button type="submit" className="primary big">{t('home.submit')} →</button>
      </form>

      <ol className="how">
        <li><span aria-hidden>📍</span>{t('home.how1')}</li>
        <li><span aria-hidden>🛰️</span>{t('home.how2')}</li>
        <li><span aria-hidden>🌾</span>{t('home.how3')}</li>
      </ol>
    </div>
  )
}
