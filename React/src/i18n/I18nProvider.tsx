import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import type { Lang } from '../api/types'
import { STRINGS, type StringKey } from './strings'

const STORAGE_KEY = 'fieldshift.lang'

interface I18n {
  lang: Lang
  setLang: (l: Lang) => void
  t: (key: StringKey, vars?: Record<string, string | number>) => string
}

const Ctx = createContext<I18n | null>(null)

function initialLang(): Lang {
  // ?lang=en or ?lang=bn in the link wins, so a page can be shared (or checked) in either language
  const asked = new URLSearchParams(window.location.search).get('lang')
  if (asked === 'bn' || asked === 'en') return asked
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY)
    if (saved === 'bn' || saved === 'en') return saved
  } catch {
    /* storage blocked — fall through */
  }
  return 'bn'
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang)

  const setLang = useCallback((l: Lang) => {
    setLangState(l)
    try {
      window.localStorage.setItem(STORAGE_KEY, l)
    } catch {
      /* ignore */
    }
  }, [])

  useEffect(() => {
    document.documentElement.lang = lang
  }, [lang])

  const t = useCallback(
    (key: StringKey, vars?: Record<string, string | number>) => {
      let s = STRINGS[lang][key] ?? STRINGS.en[key] ?? key
      if (vars) {
        for (const [k, v] of Object.entries(vars)) s = s.split(`{${k}}`).join(String(v))
      }
      return s
    },
    [lang],
  )

  const value = useMemo(() => ({ lang, setLang, t }), [lang, setLang, t])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useI18n(): I18n {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useI18n must be used inside I18nProvider')
  return ctx
}
