import { useMemo, useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { SEASONS, type Meta, type RecommendRequest, type SeasonKey, type SoilLevel } from '../api/types'
import { useI18n } from '../i18n/I18nProvider'
import type { Names } from '../lib/names'
import { paramsFromRequest } from '../lib/query'
import { ChoiceGroup } from './ChoiceGroup'

/** The optional answers (soil, the crops grown now, crops to leave out), asked once a plan is on
    screen, so the home page's form stays three answers long. Submitting reloads the plan's link. */
export function RefineForm({ req, meta, names }: { req: RecommendRequest; meta: Meta | undefined; names: Names }) {
  const { t } = useI18n()
  const navigate = useNavigate()
  const [soil, setSoil] = useState<SoilLevel>(req.soil_fertility)
  const [current, setCurrent] = useState<Record<SeasonKey, string> | null>(req.current_crops)
  const [excluded, setExcluded] = useState<string[]>(req.excluded_crops)

  const avoidable = useMemo(() => {
    const all = new Set<string>()
    for (const s of SEASONS) for (const c of meta?.season_crops[s] ?? []) if (c !== 'Fallow') all.add(c)
    return [...all]
  }, [meta])

  const currentValue = (s: SeasonKey) => current?.[s] ?? 'auto'

  function onSubmit(e: FormEvent) {
    e.preventDefault()
    const next = { ...req, soil_fertility: soil, current_crops: current, excluded_crops: excluded }
    navigate(`/result?${paramsFromRequest(next).toString()}`)
  }

  return (
    <section className="card refine no-print" aria-labelledby="refine-h">
      <h2 id="refine-h">{t('refine.title')}</h2>
      <p>{t('refine.p')}</p>
      <form className="fs-form fs-form--flat" onSubmit={onSubmit}>
        <ChoiceGroup<SoilLevel>
          name="soil"
          legend={t('home.soil')}
          value={soil}
          onChange={setSoil}
          choices={(['low', 'medium', 'high', 'unknown'] as SoilLevel[]).map((s) => ({ value: s, label: t(`soil.${s}`) }))}
        />
        <fieldset className="fs-current">
          <legend>{t('home.current')}</legend>
          {SEASONS.map((s) => (
            <div key={s} className="fs-field fs-field--row">
              <label htmlFor={`f-cur-${s}`}>{t(`season.${s}`)} <span className="muted">{t(`season.${s}.sub`)}</span></label>
              <select
                id={`f-cur-${s}`}
                value={currentValue(s)}
                onChange={(e) =>
                  setCurrent({
                    rabi: current?.rabi ?? 'auto',
                    kharif1: current?.kharif1 ?? 'auto',
                    kharif2: current?.kharif2 ?? 'auto',
                    [s]: e.target.value,
                  })
                }
                disabled={!meta}
              >
                <option value="auto">{t('refine.unplanted')}</option>
                {(meta?.season_crops[s] ?? []).map((c) => <option key={c} value={c}>{names.crop(c)}</option>)}
              </select>
            </div>
          ))}
        </fieldset>
        <fieldset>
          <legend>{t('home.avoid')}</legend>
          <div className="fs-chips">
            {avoidable.map((c) => {
              const on = excluded.includes(c)
              return (
                <button
                  type="button"
                  key={c}
                  className="fs-chip"
                  aria-pressed={on}
                  onClick={() => setExcluded(on ? excluded.filter((x) => x !== c) : [...excluded, c])}
                >
                  {names.crop(c)}
                </button>
              )
            })}
          </div>
        </fieldset>
        <div className="btn-row">
          <button className="btn btn--primary" type="submit">{t('refine.submit')}</button>
        </div>
      </form>
    </section>
  )
}
