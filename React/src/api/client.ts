import type { Explanation, Lang, Meta, Recommendation, RecommendRequest, StrategyKey } from './types'

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) },
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

let metaPromise: Promise<Meta> | null = null

/** District / crop lists. Fetched once per page load. */
export function getMeta(): Promise<Meta> {
  if (!metaPromise) {
    metaPromise = request<Meta>('/api/meta').catch((err) => {
      metaPromise = null
      throw err
    })
  }
  return metaPromise
}

export function getRecommendation(req: RecommendRequest, signal?: AbortSignal): Promise<Recommendation> {
  return request<Recommendation>('/api/recommend', { method: 'POST', body: JSON.stringify(req), signal })
}

export function getExplanation(
  req: RecommendRequest,
  strategy: StrategyKey,
  lang: Lang,
  signal?: AbortSignal,
): Promise<Explanation> {
  return request<Explanation>('/api/explain', {
    method: 'POST',
    body: JSON.stringify({ ...req, strategy, lang }),
    signal,
  })
}
