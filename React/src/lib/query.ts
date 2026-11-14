import { PRIORITIES, type Priority, type RecommendRequest, type SoilLevel } from '../api/types'

// The farmer's answers live in the URL, so a result can be bookmarked or shared (no login needed).
//   /result?d=Rajshahi&irr=1&p=water_conservation&soil=low&cur=Boro|Fallow|Aman&ex=Tobacco

const SOILS: SoilLevel[] = ['low', 'medium', 'high', 'unknown']

export function requestFromParams(p: URLSearchParams): RecommendRequest | null {
  const district = p.get('d')
  if (!district) return null
  const irr = p.get('irr')
  const priority = p.get('p') as Priority | null
  const soil = p.get('soil') as SoilLevel | null
  const cur = p.get('cur')?.split('|')
  const ex = p.get('ex')

  return {
    district,
    irrigation: irr === '1' ? true : irr === '0' ? false : null,
    priority: priority && PRIORITIES.includes(priority) ? priority : 'balanced',
    soil_fertility: soil && SOILS.includes(soil) ? soil : 'unknown',
    current_crops: cur && cur.length === 3 ? { rabi: cur[0], kharif1: cur[1], kharif2: cur[2] } : null,
    excluded_crops: ex ? ex.split('|').filter(Boolean) : [],
  }
}

export function paramsFromRequest(req: RecommendRequest): URLSearchParams {
  const p = new URLSearchParams()
  p.set('d', req.district)
  if (req.irrigation !== null) p.set('irr', req.irrigation ? '1' : '0')
  p.set('p', req.priority)
  if (req.soil_fertility !== 'unknown') p.set('soil', req.soil_fertility)
  if (req.current_crops) {
    p.set('cur', [req.current_crops.rabi, req.current_crops.kharif1, req.current_crops.kharif2].join('|'))
  }
  if (req.excluded_crops.length) p.set('ex', req.excluded_crops.join('|'))
  return p
}
