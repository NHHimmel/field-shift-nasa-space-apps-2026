import type { CropInfo, DistrictInfo, Lang, Meta } from '../api/types'

export interface Names {
  crop: (key: string) => string
  district: (key: string) => string
  cropInfo: (key: string) => CropInfo | undefined
  districtInfo: (key: string) => DistrictInfo | undefined
}

export function makeNames(meta: Meta | undefined, lang: Lang): Names {
  const crops = new Map((meta?.crops ?? []).map((c) => [c.key, c]))
  const districts = new Map((meta?.districts ?? []).map((d) => [d.key, d]))
  return {
    crop: (key) => {
      const c = crops.get(key)
      return c ? (lang === 'bn' ? c.name_bn : c.name_en) : key
    },
    district: (key) => {
      const d = districts.get(key)
      return d ? (lang === 'bn' ? d.name_bn : d.name_en) : key
    },
    cropInfo: (key) => crops.get(key),
    districtInfo: (key) => districts.get(key),
  }
}

const NAME_ICONS: Record<string, string> = {
  Potato: '🥔', 'Sweet Potato': '🍠', Onion: '🧅', Garlic: '🧄', Carrot: '🥕', Chili: '🌶️',
  Cucumber: '🥒', Pumpkin: '🎃', Cabbage: '🥬', Cauliflower: '🥦', Ginger: '🫚',
  'Rape & Mustard': '🌼', Sesame: '🌼', Groundnut: '🥜', Jute: '🌿', Tobacco: '🍂',
  'Maize 1': '🌽', 'Maize 2': '🌽', Fallow: '🟫',
}
const CATEGORY_ICONS: Record<string, string> = {
  Cereal: '🌾', Pulse_Legume: '🫘', Oilseed: '🌼', Fiber: '🌿', Vegetable_Tuber_Spice: '🥬', Fallow: '🟫',
}

export function cropIcon(key: string, category?: string): string {
  return NAME_ICONS[key] ?? (category ? CATEGORY_ICONS[category] : undefined) ?? '🌱'
}
