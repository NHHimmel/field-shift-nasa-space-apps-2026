import type { ReactNode } from 'react'

export interface Choice<T extends string> {
  value: T
  label: ReactNode
  desc?: ReactNode
  icon?: string
}

interface Props<T extends string> {
  name: string
  legend: ReactNode
  choices: Choice<T>[]
  value: T
  onChange: (v: T) => void
  variant?: 'pills' | 'cards'
}

/** Big, thumb-friendly radio buttons. */
export function ChoiceGroup<T extends string>({ name, legend, choices, value, onChange, variant = 'pills' }: Props<T>) {
  return (
    <fieldset className={`choices choices-${variant}`}>
      <legend>{legend}</legend>
      <div className="choices-row">
        {choices.map((c) => (
          <label key={c.value} className={`choice ${value === c.value ? 'on' : ''}`}>
            <input
              type="radio"
              name={name}
              value={c.value}
              checked={value === c.value}
              onChange={() => onChange(c.value)}
            />
            {c.icon ? <span className="choice-icon" aria-hidden>{c.icon}</span> : null}
            <span className="choice-text">
              <span className="choice-label">{c.label}</span>
              {c.desc ? <span className="choice-desc">{c.desc}</span> : null}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  )
}
