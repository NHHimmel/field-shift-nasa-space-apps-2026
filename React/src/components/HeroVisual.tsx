// Dynamic visual graphic for the hero: NASA satellite telemetry scanning over agricultural field plots.
import { useI18n } from '../i18n/I18nProvider'

export function HeroVisual() {
  const { lang } = useI18n()

  return (
    <div className="hero-telemetry-card" aria-hidden="true">
      {/* Decorative ambient background mesh */}
      <div className="hero-telemetry-canvas">
        <svg
          viewBox="0 0 460 220"
          className="telemetry-svg"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <linearGradient id="scan-beam" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#1a6344" stopOpacity="0.35" />
              <stop offset="50%" stopColor="#084c57" stopOpacity="0.15" />
              <stop offset="100%" stopColor="#084c57" stopOpacity="0" />
            </linearGradient>

            <linearGradient id="field-green" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#dbe8df" />
              <stop offset="100%" stopColor="#c7d9cd" />
            </linearGradient>

            <linearGradient id="field-teal" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#d5e3e6" />
              <stop offset="100%" stopColor="#bfd3d8" />
            </linearGradient>

            <linearGradient id="field-warm" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#faecd8" />
              <stop offset="100%" stopColor="#ecd3b5" />
            </linearGradient>

            <pattern id="dot-grid" width="16" height="16" patternUnits="userSpaceOnUse">
              <circle cx="2" cy="2" r="1" fill="#084c57" opacity="0.12" />
            </pattern>
          </defs>

          {/* Coordinate grid background */}
          <rect width="460" height="220" fill="url(#dot-grid)" />

          {/* Elevation contour wave curves */}
          <g className="contour-waves" stroke="#1a6344" strokeWidth="0.8" opacity="0.25" fill="none">
            <path d="M -20,180 Q 80,130 180,160 T 380,140 T 480,170" />
            <path d="M -20,150 Q 100,100 220,130 T 400,110 T 480,140" />
            <path d="M -20,120 Q 120,70 240,100 T 420,80 T 480,110" />
          </g>

          {/* Agricultural Field Plots in isometric / 2.5D perspective */}
          <g className="field-plots" transform="translate(40, 110)">
            {/* Field Plot 1: Rice / Wheat */}
            <polygon points="0,50 90,10 180,45 90,85" fill="url(#field-green)" stroke="#1a6344" strokeWidth="1" />
            <line x1="30" y1="40" x2="120" y2="75" stroke="#1a6344" strokeWidth="0.8" opacity="0.4" strokeDasharray="3,3" />
            <line x1="60" y1="25" x2="150" y2="60" stroke="#1a6344" strokeWidth="0.8" opacity="0.4" strokeDasharray="3,3" />

            {/* Field Plot 2: Mustard / Legumes */}
            <polygon points="100,6 190,-30 280,5 190,41" fill="url(#field-warm)" stroke="#c85a17" strokeWidth="1" />
            <line x1="130" y1="-8" x2="220" y2="23" stroke="#c85a17" strokeWidth="0.8" opacity="0.4" strokeDasharray="3,3" />
            <line x1="160" y1="-20" x2="250" y2="11" stroke="#c85a17" strokeWidth="0.8" opacity="0.4" strokeDasharray="3,3" />

            {/* Field Plot 3: Summer Kharif / Haor */}
            <polygon points="195,43 285,7 375,42 285,78" fill="url(#field-teal)" stroke="#084c57" strokeWidth="1" />
            <line x1="225" y1="31" x2="315" y2="66" stroke="#084c57" strokeWidth="0.8" opacity="0.4" strokeDasharray="3,3" />
          </g>

          {/* Radar Scanning Beam */}
          <polygon points="380,25 60,165 310,185" fill="url(#scan-beam)" className="satellite-beam" />

          {/* NASA Satellite in Orbit */}
          <g className="satellite-node" transform="translate(380, 25)">
            {/* Orbital path arc */}
            <path d="M -300,-15 Q -100,5 60,35" stroke="#084c57" strokeWidth="1.2" strokeDasharray="4,4" fill="none" opacity="0.45" />

            {/* Satellite Body */}
            <rect x="-8" y="-6" width="16" height="12" rx="2" fill="#084c57" stroke="#fff" strokeWidth="1.2" />
            {/* Solar Panels */}
            <rect x="-24" y="-4" width="14" height="8" rx="1" fill="#18794e" stroke="#fff" strokeWidth="0.8" />
            <rect x="10" y="-4" width="14" height="8" rx="1" fill="#18794e" stroke="#fff" strokeWidth="0.8" />
            {/* Signal transmission pulse */}
            <circle cx="0" cy="0" r="10" fill="none" stroke="#c85a17" strokeWidth="1" className="sat-signal" />
            <circle cx="0" cy="0" r="18" fill="none" stroke="#c85a17" strokeWidth="0.8" className="sat-signal-outer" />
          </g>

          {/* Telemetry Target Pins on the fields */}
          <g className="field-target" transform="translate(130, 150)">
            <circle r="4" fill="#1a6344" />
            <circle r="10" fill="none" stroke="#1a6344" strokeWidth="1" opacity="0.6" className="target-pulse" />
          </g>

          <g className="field-target" transform="translate(235, 125)">
            <circle r="4" fill="#c85a17" />
            <circle r="10" fill="none" stroke="#c85a17" strokeWidth="1" opacity="0.6" className="target-pulse" />
          </g>
        </svg>
      </div>

      {/* Floating Telemetry Stats Bar */}
      <div className="hero-telemetry-stats">
        <div className="telemetry-pill">
          <span className="pill-dot green" />
          <span className="pill-title">4 NASA Satellites</span>
          <span className="pill-sub">POWER · SMAP · GRACE · MODIS</span>
        </div>
        <div className="telemetry-pill">
          <span className="pill-dot teal" />
          <span className="pill-title">64 Districts</span>
          <span className="pill-sub">{lang === 'bn' ? '৩০টি কৃষি-পরিবেশ অঞ্চল' : '30 Agro-Zones'}</span>
        </div>
        <div className="telemetry-pill">
          <span className="pill-dot orange" />
          <span className="pill-title">3 Seasons</span>
          <span className="pill-sub">Rabi · Kharif-1 · Kharif-2</span>
        </div>
      </div>
    </div>
  )
}
