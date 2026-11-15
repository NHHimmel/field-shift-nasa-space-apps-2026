import { useMemo, useState, type FormEvent } from 'react'
import { useSearchParams } from 'react-router-dom'
import { getMeta } from '../api/client'
import { PRIORITIES, type Priority } from '../api/types'
import { ChoiceGroup } from '../components/ChoiceGroup'
import { useI18n } from '../i18n/I18nProvider'
import type { StringKey } from '../i18n/strings'
import { fmtInt, fmtPct } from '../lib/format'
import { makeNames } from '../lib/names'
import { requestFromParams } from '../lib/query'
import { useAsync } from '../lib/useAsync'

/* A plain home page: the headline beside the three-question form, then how it works, the NASA data
   and the questions. The optional answers (soil, current crops, crops to avoid) are on the plan page. */

/** The irrigation answer as the plan's link writes it: 1, 0, or u (not sure). */
type Irr = '1' | '0' | 'u'

const DATASETS: { name: string; nameKey?: StringKey; desc: StringKey }[] = [
  { name: 'NASA POWER', desc: 'about.ds.power' },
  { name: 'NASA SMAP', desc: 'about.ds.smap' },
  { name: 'NASA GRACE / GRACE-FO', desc: 'about.ds.grace' },
  { name: 'NASA MODIS & ECOSTRESS', desc: 'about.ds.modis' },
  { name: '', nameKey: 'about.ds.statsName', desc: 'about.ds.stats' },
]

const STEPS: StringKey[] = ['about.step1', 'about.step2', 'about.step3', 'about.step4']

export default function Home() {
  const { t, lang } = useI18n()
  const [params] = useSearchParams()
  const initial = useMemo(() => requestFromParams(params), [params])
  const meta = useAsync(() => getMeta(), [])
  const names = makeNames(meta.data, lang)

  const [district, setDistrict] = useState(initial?.district ?? '')
  const [irr, setIrr] = useState<Irr>(initial?.irrigation === true ? '1' : initial?.irrigation === false ? '0' : 'u')
  const [priority, setPriority] = useState<Priority>(initial?.priority ?? 'balanced')
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
  const rate = meta.data?.units.usd_to_bdt

  // The form is a real GET to the plan's page (/result?d=..&irr=..&p=..); only a missing district stops it.
  function onSubmit(e: FormEvent) {
    if (!district) {
      e.preventDefault()
      setShowError(true)
      document.getElementById('f-district')?.focus()
    }
  }

  // Fills the form with a ready example (Rajshahi) so a first-time visitor can try it in one click.
  function loadDemo() {
    const demo = meta.data?.districts.find((d) => d.key.toLowerCase() === 'rajshahi') ?? meta.data?.districts[0]
    if (!demo) return
    setDistrict(demo.key)
    setIrr(demo.default_irrigation ? '1' : '0')
    setPriority('balanced')
    setShowError(false)
  }

  return (
    <>
      <section className="hero hero--split fs-home-hero" aria-labelledby="hero-h">
        <div className="wrap">
          <div className="hero-text grid hero__grid">
            <div className="hero__text span-5">
              <h1 id="hero-h">{t('hero.h1')}</h1>
              <p className="hero__lead">{t('hero.lead')}</p>
              <p className="t-small fs-hero-note">{t('plan.next')}</p>
            </div>

            <form id="plan" className="fs-form span-6 start-7" action="/result" method="get" onSubmit={onSubmit} noValidate>
              {lang === 'en' ? <input type="hidden" name="lang" value="en" /> : null}
              <div className="fs-field">
                <label htmlFor="f-district">{t('home.q1')}</label>
                <select
                  id="f-district"
                  name="d"
                  value={district}
                  onChange={(e) => { setDistrict(e.target.value); setShowError(false) }}
                  disabled={!meta.data}
                  required
                  aria-invalid={showError && !district ? true : undefined}
                  aria-describedby="f-district-err"
                >
                  <option value="">{meta.data ? t('home.q1.placeholder') : t('home.loadingMeta')}</option>
                  {groups.map((g) => (
                    <optgroup key={g.label} label={g.label}>
                      {g.items.map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
                    </optgroup>
                  ))}
                </select>
                <p className="field__error" id="f-district-err" hidden={!(showError && !district)}>{t('home.needDistrict')}</p>
                {meta.data ? <button type="button" className="link demo-link" onClick={loadDemo}>{t('home.demo')}</button> : null}
                {meta.status === 'error' ? (
                  <p className="field__error">{t('res.error')} <button type="button" className="link" onClick={meta.retry}>{t('res.retry')}</button></p>
                ) : null}
                {selected && selected.groundwater_pctl < 30 ? (
                  <p className="t-small fs-note">{t('home.gwLow', { district: names.district(selected.key), pct: fmtPct(selected.groundwater_pctl, lang) })}</p>
                ) : null}
              </div>

              <ChoiceGroup<Irr>
                name="irr"
                legend={t('home.q2')}
                value={irr}
                onChange={setIrr}
                choices={[
                  { value: '1', label: t('irr.yes') },
                  { value: '0', label: t('irr.no') },
                  { value: 'u', label: t('irr.unknown') },
                ]}
              />

              <ChoiceGroup<Priority>
                name="p"
                variant="cards"
                legend={t('home.q3')}
                value={priority}
                onChange={setPriority}
                choices={PRIORITIES.map((p) => ({ value: p, label: t(`strategy.${p}`), desc: t(`strategy.${p}.desc`) }))}
              />

              <div className="btn-row">
                <button className="btn btn--primary" type="submit">{t('cta.main')}</button>
              </div>
            </form>
          </div>
        </div>
      </section>

      <section className="fs-block" id="how" aria-labelledby="how-h">
        <div className="wrap">
          <h2 id="how-h">{t('menu.g2')}</h2>
          <ol className="fs-steps-row">
            {STEPS.map((k, i) => (
              <li key={k}>
                <span className="step-num">{fmtInt(i + 1, lang)}</span>
                <p>{t(k)}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="fs-block" id="data" aria-labelledby="data-h">
        <div className="wrap grid grid--top">
          <div className="span-5 stack-4">
            <h2 id="data-h">{t('data.h2')}</h2>
            <p>{t('data.p')}</p>
          </div>
          <ul className="rows span-6 start-7">
            {DATASETS.map((d) => (
              <li key={d.desc}>
                <span className="rows__name">{d.nameKey ? t(d.nameKey) : d.name}</span>
                <p>{t(d.desc)}</p>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="fs-block" id="faq" aria-labelledby="faq-h">
        <div className="wrap grid grid--top">
          <div className="span-5 stack-4">
            <h2 id="faq-h">{t('faq.h2')}</h2>
            <p>{t('faq.p')}</p>
          </div>
          <div className="faq span-6 start-7">
            {([1, 2, 4, 5, 6] as const).map((i) => (
              <div className="faq__item" key={i}>
                <h3>{t(`faq.q${i}` as StringKey)}</h3>
                <p>{t(`faq.a${i}` as StringKey, { rate: rate ? fmtInt(rate, lang) : '—' })}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  )
}
