# Demo site

A static page with two sections: **Reproduce** (how to use this repository) and **Present** (an interactive
view of the results, with a full-screen mode for talks). No build step and no dependencies.

Every number comes from `data.js`, which `build_data.py` generates from the committed files in `results/paper/`
(Table I, `bootstrap_vs_D.csv`, each run's `per_event.csv` and `privacy.json`).

```bash
python site/build_data.py           # rewrite site/data.js after results/paper/ changes
python site/build_data.py --check   # what CI runs; fails if data.js is stale
```

Open `site/index.html` in a browser to preview it; it also works from disk. On pushes to `main` that touch
`site/` or `results/paper/`, the `pages` workflow publishes it to GitHub Pages (Settings, Pages, Source:
GitHub Actions).
