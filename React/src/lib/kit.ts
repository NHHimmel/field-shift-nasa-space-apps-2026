/* The design kit (public/assets, from the tasteskill design system) scans the page once when its scripts
   start: the nav sheet, the hero's entrance, the effect ground, the moment, the diagram. React draws the
   page after load, so the kit is started from here, once per page load, after the first render.
   Its init is not safe to run twice, so links that lead back to the home page are full page loads. */

const asset = (path: string) => new URL(`/assets/${path}`, window.location.origin).href
const load = (path: string) => import(/* @vite-ignore */ asset(path))

let started: Promise<void> | null = null

export function startKit(): Promise<void> {
  if (!started) {
    started = (async () => {
      // system.js first (nav, tabs, effect grounds, moments), then the hover/entrance layer and the hero helpers
      await load('system.js')
      await Promise.all([load('motion/motion.js'), load('layout/layout.js')])
    })().catch((e) => console.warn('kit: could not start', e))
  }
  return started
}
