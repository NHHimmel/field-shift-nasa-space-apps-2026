import { useEffect, useState, type DependencyList } from 'react'

export interface AsyncState<T> {
  status: 'loading' | 'ok' | 'error'
  data?: T
  error?: unknown
  retry: () => void
}

/** Runs an async loader whenever `deps` change; keeps the previous data visible while reloading. */
export function useAsync<T>(loader: (signal: AbortSignal) => Promise<T>, deps: DependencyList): AsyncState<T> {
  const [nonce, setNonce] = useState(0)
  const [state, setState] = useState<Omit<AsyncState<T>, 'retry'>>({ status: 'loading' })

  useEffect(() => {
    const ac = new AbortController()
    setState((s) => ({ status: 'loading', data: s.data }))
    loader(ac.signal)
      .then((data) => {
        if (!ac.signal.aborted) setState({ status: 'ok', data })
      })
      .catch((error) => {
        if (!ac.signal.aborted) setState((s) => ({ status: 'error', error, data: s.data }))
      })
    return () => ac.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  return { ...state, retry: () => setNonce((n) => n + 1) }
}
