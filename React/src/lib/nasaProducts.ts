export interface NasaProduct {
  id: string
  name: string
  fullName: string
  roleEn: string
  roleBn: string
  period: string
  status: 'active' | 'derived' | 'fallback'
  url: string
}

export const NASA_PRODUCTS: Record<string, NasaProduct> = {
  power: {
    id: 'power',
    name: 'NASA POWER',
    fullName: 'Prediction Of Worldwide Energy Resources',
    roleEn: 'Daily rainfall, solar radiation, temperature, and humidity history for agroclimatology',
    roleBn: 'কৃষি-আবহাওয়ার জন্য প্রতিদিনের বৃষ্টিপাত, সৌরবিকিরণ, তাপমাত্রা ও আর্দ্রতার ইতিহাস',
    period: '2004–2024 (20-year baseline)',
    status: 'active',
    url: 'https://power.larc.nasa.gov/',
  },
  smap: {
    id: 'smap',
    name: 'NASA SMAP',
    fullName: 'Soil Moisture Active Passive (L4 Root-Zone)',
    roleEn: 'High-resolution surface and root-zone (0–100 cm) soil moisture assimilation',
    roleBn: 'উচ্চ রেজোলিউশনে মাটির উপরিভাগ ও শিকড় স্তরের (০–১০০ সেমি) আর্দ্রতার তথ্য',
    period: '2015–2024',
    status: 'derived',
    url: 'https://smap.jpl.nasa.gov/',
  },
  grace: {
    id: 'grace',
    name: 'NASA GRACE / GRACE-FO',
    fullName: 'Gravity Recovery and Climate Experiment Follow-On',
    roleEn: 'Terrestrial water storage anomalies and regional groundwater percentile drought indicators',
    roleBn: 'মহাকর্ষ পরিমাপের মাধ্যমে ভূগর্ভস্থ পানির সঞ্চয় ও আঞ্চলিক খরার পার্সেন্টাইল',
    period: '2004–2024',
    status: 'derived',
    url: 'https://grace.jpl.nasa.gov/',
  },
  modis: {
    id: 'modis',
    name: 'NASA MODIS / ECOSTRESS',
    fullName: 'Moderate Resolution Imaging Spectroradiometer & ECOSTRESS',
    roleEn: 'Evapotranspiration (ET) demand and thermal vegetation stress for crop water consumption',
    roleBn: 'ফসলের পানি খরচের বাষ্পীভবন চাহিদা ও উদ্ভিদের তাপীয় চাপ পর্যবেক্ষণ',
    period: '2004–2024',
    status: 'derived',
    url: 'https://modis.gsfc.nasa.gov/',
  },
  earthdata: {
    id: 'earthdata',
    name: 'NASA Earthdata',
    fullName: 'NASA Earth Observing System Data and Information System',
    roleEn: 'Comprehensive open-access NASA Earth observation catalog and APIs',
    roleBn: 'নাসার পৃথিবী পর্যবেক্ষণ ডেটাসেট ও স্যাটেলাইট আর্কাইভ',
    period: 'Continuous',
    status: 'active',
    url: 'https://www.earthdata.nasa.gov/',
  },
}

export function getNasaProduct(name: string): NasaProduct | undefined {
  const norm = name.toUpperCase()
  if (norm.includes('POWER')) return NASA_PRODUCTS.power
  if (norm.includes('SMAP')) return NASA_PRODUCTS.smap
  if (norm.includes('GRACE')) return NASA_PRODUCTS.grace
  if (norm.includes('MODIS') || norm.includes('ECOSTRESS')) return NASA_PRODUCTS.modis
  return undefined
}
