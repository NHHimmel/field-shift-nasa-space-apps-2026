import type { Lang } from '../api/types'

// bn-BD gives Bangla digits + lakh grouping (১২,৩৪,৫৬৭); en-IN gives the same grouping in ASCII.
const locale = (lang: Lang) => (lang === 'bn' ? 'bn-BD' : 'en-IN')

const cache = new Map<string, Intl.NumberFormat>()
function nf(lang: Lang, maxFrac: number, grouping = true): Intl.NumberFormat {
  const key = `${lang}|${maxFrac}|${grouping}`
  let f = cache.get(key)
  if (!f) {
    f = new Intl.NumberFormat(locale(lang), { maximumFractionDigits: maxFrac, useGrouping: grouping })
    cache.set(key, f)
  }
  return f
}

/** Up to 1 decimal place — the backend already rounded, so this never hides digits. */
export const fmtNum = (v: number, lang: Lang, maxFrac = 1) => nf(lang, maxFrac).format(v)
export const fmtInt = (v: number, lang: Lang) => nf(lang, 0).format(v)
export const fmtYear = (v: number, lang: Lang) => nf(lang, 0, false).format(v)

export const fmtTaka = (v: number, lang: Lang) => `${v < 0 ? '−' : ''}৳${fmtInt(Math.abs(v), lang)}`
export const fmtMm = (v: number, lang: Lang) => `${fmtNum(v, lang)} ${lang === 'bn' ? 'মিমি' : 'mm'}`
export const fmtKg = (v: number, lang: Lang) => `${fmtNum(v, lang)} ${lang === 'bn' ? 'কেজি' : 'kg'}`
export const fmtPct = (v: number, lang: Lang) => `${fmtNum(v, lang)}%`
export const fmtSigned = (v: number, lang: Lang, unit = '') =>
  `${v > 0 ? '+' : v < 0 ? '−' : ''}${fmtNum(Math.abs(v), lang)}${unit}`

const MONTHS: Record<Lang, string[]> = {
  en: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
  bn: ['জানুয়ারি', 'ফেব্রুয়ারি', 'মার্চ', 'এপ্রিল', 'মে', 'জুন', 'জুলাই', 'আগস্ট', 'সেপ্টেম্বর', 'অক্টোবর', 'নভেম্বর', 'ডিসেম্বর'],
}
const BN_DIGITS = '০১২৩৪৫৬৭৮৯'

export function toLangDigits(s: string, lang: Lang): string {
  return lang === 'bn' ? s.replace(/[0-9]/g, (d) => BN_DIGITS[Number(d)]) : s
}

/** "Nov 01 - Nov 20" / "Feb-Mar" -> Bangla month names and digits. */
export function fmtWindow(window: string, lang: Lang): string {
  if (!window || window.trim().toLowerCase() === 'none') return '—'
  if (lang !== 'bn') return window
  const replaced = window.replace(/\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b/g, (m) => {
    const idx = MONTHS.en.indexOf(m.slice(0, 3))
    return idx >= 0 ? MONTHS.bn[idx] : m
  })
  return toLangDigits(replaced, lang)
}

/** "03-04" -> "4 Mar" / "৪ মার্চ" */
export function fmtMonthDay(mmdd: string, lang: Lang): string {
  const [m, d] = mmdd.split('-').map(Number)
  return `${fmtInt(d, lang)} ${MONTHS[lang][m - 1]}`
}
