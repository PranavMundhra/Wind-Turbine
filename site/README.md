# Demo site

A static page with four sections: **Reproduce** (how to use this repository), **Present** (an animated,
plain-language story of every experiment, then an interactive results explorer, with a full-screen mode for talks),
**CARE maths** (every formula behind the score, with calculators) and **Test a turbine** (replays a month of real
Wind Farm C readings and runs the criticality counter, with an alarm). No build step and no dependencies.

Every number comes from `data.js`, which `build_data.py` generates from the committed files in `results/paper/`
(Table I, `bootstrap_vs_D.csv`, and for every seed each run's `results.json` and `per_event.csv`, plus `privacy.json`).
The CARE constants (beta, quantile, criticality threshold, weights) are read from `configs/base.yaml` and
`src/windturbine/scoring/`, and `tests/test_site_data.py` checks that the formula on the page reproduces every
published score.

```bash
python site/build_data.py           # rewrite site/data.js after results/paper/ changes
python site/build_data.py --check   # what CI runs; fails if data.js is stale
```

Open `site/index.html` in a browser to preview it; it also works from disk. On pushes to `main` that touch
`site/` or `results/paper/`, the `pages` workflow publishes it to GitHub Pages (Settings, Pages, Source:
GitHub Actions).

## Files

- `index.html`: the page. `assets/style.css`: theme tokens (light and dark) and the scene animations.
- `assets/story.js`: the animated walkthrough, one scene per experiment. Scenes are SVG drawn from `data.js`; the
  motion is CSS, and with reduced-motion settings the scenes show their finished state. The FedProx and privacy
  scenes have sliders that move the illustration and update the result from the real runs.
- `assets/app.js`: the results explorer, the reproduction guide and the full-screen presenter.
- `assets/maths.js`: the CARE maths tab. Formulas are MathML; the calculators apply the same rules as `src/windturbine/scoring/`
  to made-up examples, and the real-data widgets read the committed runs.
- `assets/simulator.js` and `sim_data.js`: the Test a turbine tab. `sim_data.js` holds five 30-day windows of real
  readings from turbine 35 (events 48 and 67), built by `build_sim_data.py` on a machine with the raw CARE to Compare
  files, which are not in this repository. No trained model runs: each reading is flagged by a stand-in score
  (one sensor against its usual range at the same wind speed and season, with the 99% rule), and the page applies
  the CARE criticality counter. `tests/test_site_data.py` replays the counter and checks each window's label.
