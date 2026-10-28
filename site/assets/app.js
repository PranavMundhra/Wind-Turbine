/* Reads window.WT_DATA (site/data.js, built from results/paper by site/build_data.py). No numbers are hardcoded here. */
(function () {
  'use strict';
  var D = window.WT_DATA, X = D.experiments, G = D.groups, M = D.meta, REF = M.reference;
  var REPO = 'https://github.com/PranavMundhra/Wind-Turbine';
  var METRIC = { care: 'CARE', coverage: 'Coverage', earliness: 'Earliness', reliability: 'Reliability', accuracy: 'Accuracy' };
  var MINUS = '−';

  function el(id) { return document.getElementById(id); }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
  function f3(v) { return v.toFixed(3); }
  function sgn(v, d) { d = d === undefined ? 3 : d; return (v < 0 ? MINUS : '+') + Math.abs(v).toFixed(d); }
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : many); }

  var FED = G.arms.filter(function (id) { return id !== REF && id !== 'central_sequential'; }).concat(G.mu, G.priv);
  var refCare = X[REF].metrics.care;
  var maxFed = Math.max.apply(null, FED.map(function (id) { return X[id].metrics.care; }));
  var noBeat = maxFed <= refCare;
  var nSeeds = Math.max.apply(null, Object.keys(X).map(function (id) { return X[id].seeds.length; }));
  var seedLabel = nSeeds === 1 ? 'seed ' + X[REF].seeds[0] : 'mean of ' + nSeeds + ' seeds';
  var withCi = Object.keys(X).filter(function (id) { return X[id].ci; });
  var excl = withCi.filter(function (id) { return X[id].ci.excludes_zero; }).length;

  function dn(id) {
    var g = X[id].group;
    if (g === 'mu') return 'FedProx ' + X[id].label;
    if (g === 'priv') return 'Private ' + X[id].label;
    return X[id].label;
  }
  function cat(id) { return id === REF ? 'ref' : X[id].group === 'priv' ? 'priv' : 'fed'; }

  /* ---------- header, hero, tiles, copy ---------- */
  function renderStatic() {
    el('result-line').innerHTML = (noBeat
      ? '<b>No federated setting scored higher than local training</b> (CARE ' + f3(refCare) + ')'
      : '<b>Best federated CARE ' + f3(maxFed) + '</b> against ' + f3(refCare) + ' for local training') +
      ' on Wind Farm C, ' + seedLabel + '.';
    el('hero-lead').textContent = M.n_turbines + ' turbines, each a federated client, trained a shared fault detector and were scored on ' +
      M.n_events + ' events (' + M.n_anomaly + ' with a fault).' + (noBeat ? ' No federated setting scored higher than each turbine training alone.' : '');
    el('hero-chip').textContent = (nSeeds === 1 ? 'Single seed (' + X[REF].seeds[0] + '). ' : 'Mean of ' + nSeeds + ' seeds. ') +
      (excl === 0 ? 'Every 95% interval against local training includes zero.'
        : excl + ' of ' + withCi.length + ' intervals against local training exclude zero.');

    var bestMu = G.mu.slice().sort(function (a, b) { return X[b].metrics.care - X[a].metrics.care; })[0];
    var noisy = G.priv.filter(function (id) { return X[id].epsilon !== null; });
    var noisyMean = noisy.reduce(function (s, id) { return s + X[id].metrics.care; }, 0) / noisy.length;
    var f1 = X.F1;
    var tiles = [
      ['ref', 'Local training', f3(refCare), plural(X[REF].detected, 'fault', 'faults') + ' detected of ' + M.n_anomaly],
      ['fed', 'FedAvg', f3(f1.metrics.care), sgn(f1.ci.diff) + ' vs local'],
      ['fed', 'Best FedProx', f3(X[bestMu].metrics.care), X[bestMu].label.replace('μ', 'Proximal weight μ')],
      ['priv', 'With privacy noise', '≈' + noisyMean.toFixed(2), 'The models barely train']
    ];
    el('tiles').innerHTML = tiles.map(function (t) {
      return '<div class="card tile"><div class="k"><i class="c-' + t[0] + '"></i>' + esc(t[1]) + '</div><div class="v">' + esc(t[2]) + '</div><div class="c">' + esc(t[3]) + '</div></div>';
    }).join('');

    el('strip').textContent = (nSeeds === 1 ? 'One seed (' + X[REF].seeds[0] + ') per experiment. ' : 'Means over ' + nSeeds + ' seeds. ') +
      'The 95% intervals resample the ' + M.n_turbines + ' turbines, not the ' + M.n_events + ' events, because events from one turbine are not independent.';

    var near = FED.filter(function (id) { return id === 'F1' || (X[id].group === 'mu' && /^F3_mu(0|0\.001|0\.01|0\.1)$/.test(id)); });
    var within = Math.max.apply(null, near.map(function (id) { return Math.abs(X[id].ci.diff); }));
    var strongMu = G.mu.filter(function (id) { return /^F3_mu(1|10)$/.test(id); });
    el('take-yes').innerHTML = [
      noBeat ? 'No federated setting scored higher than local training (CARE ' + f3(refCare) + ').' : 'Best federated CARE was ' + f3(maxFed) + ' against ' + f3(refCare) + ' for local training.',
      'FedAvg and FedProx with μ ≤ 0.1 came within ' + within.toFixed(3) + ' CARE of local training.',
      'Stronger proximal regularisation scored lower: ' + strongMu.map(function (id) { return X[id].label + ' gave ' + f3(X[id].metrics.care); }).join(', ') + '.',
      'With differential-privacy noise the models barely trained, so these runs give no usable relation between privacy and CARE.'
    ].map(function (t) { return '<li>' + esc(t) + '</li>'; }).join('');
    el('take-no').innerHTML = [
      (nSeeds === 1 ? 'These are single-seed results, so small differences cannot be ranked.' : 'Small differences still need paired tests.'),
      (excl === 0 ? 'Every 95% interval against local training includes zero (all ' + withCi.length + ').' : excl + ' of ' + withCi.length + ' intervals exclude zero.'),
      'The ' + M.n_events + ' events come from ' + M.n_turbines + ' turbines, so they are not independent samples.',
      'Local training was run on Kaggle with TensorFlow and the federated runs on PyTorch, so that comparison also includes a software difference.'
    ].map(function (t) { return '<li>' + esc(t) + '</li>'; }).join('');

    el('foot').innerHTML = '<span>Built from <span class="mono">results/paper/</span> by <span class="mono">python site/build_data.py</span>. Data fingerprint <span class="mono">' + M.fingerprint + '</span>.</span>' +
      '<span>Code under the MIT licence. The CARE to Compare data keep their CC BY-SA 4.0 licence.</span>';
  }

  /* ---------- explorer ---------- */
  var state = { tab: 'arms', metric: 'care', sel: { arms: 'F1', mu: 'F3_mu0.1', priv: 'F4_sigma1' }, evA: REF, evB: 'F1', evF: 'all' };

  function valueOf(id, m) { return m === 'diff' ? (id === REF ? 0 : X[id].ci.diff) : X[id].metrics[m]; }
  function niceStep(span) { var c = [.005, .01, .02, .05, .1, .2]; for (var i = 0; i < c.length; i++) if (span / c[i] <= 6) return c[i]; return .2; }

  function renderChart() {
    var ids = G[state.tab], m = state.metric, diff = m === 'diff';
    var refV = diff ? 0 : X[REF].metrics[m];
    var vals = [refV];
    ids.forEach(function (id) {
      vals.push(valueOf(id, m));
      if (diff && id !== REF && X[id].ci) vals.push(X[id].ci.lo, X[id].ci.hi);
    });
    var mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals);
    var pad = (mx - mn) * .15 || .02;
    var step = niceStep(mx - mn + 2 * pad);
    var lo = Math.floor((mn - pad) / step) * step, hi = Math.ceil((mx + pad) / step) * step;
    if (!diff) { lo = Math.max(lo, 0); hi = Math.min(hi, 1); }
    function pct(v) { return ((v - lo) / (hi - lo) * 100).toFixed(2); }
    var ticks = [];
    for (var i = 0, n = Math.round((hi - lo) / step); i <= n; i++) ticks.push(lo + i * step);
    var tf = function (t) { var s = t.toFixed(step < .01 ? 3 : 2); return diff ? (Math.abs(t) < 1e-9 ? '0' : sgn(t, step < .01 ? 3 : 2)) : s; };

    el('legend').innerHTML = (diff
      ? '<span><i class="dash"></i>Same as local training</span><span><i class="wh"></i>95% interval (turbines resampled)</span>'
      : '<span><i class="dash"></i>Local training (D) ' + f3(refV) + '</span>') +
      '<span><i class="c-ref"></i>Reference</span><span><i class="c-fed"></i>Federated</span>' +
      (state.tab === 'priv' ? '<span><i class="c-priv"></i>Private (DP noise)</span>' : '');

    var html = '<div class="grid-ov">' + ticks.map(function (t) { return '<i style="left:' + pct(t) + '%"></i>'; }).join('') +
      '<i class="ref" style="left:' + pct(refV) + '%"></i></div>';
    ids.forEach(function (id) {
      var v = valueOf(id, m), whisk = '';
      if (diff && id !== REF && X[id].ci) {
        whisk = '<i class="whisk" style="left:' + pct(X[id].ci.lo) + '%;width:' + (pct(X[id].ci.hi) - pct(X[id].ci.lo)).toFixed(2) + '%"></i>';
      }
      html += '<button type="button" class="row" data-id="' + esc(id) + '" aria-pressed="' + (id === state.sel[state.tab]) + '">' +
        '<span class="lab">' + esc(X[id].label) + '</span>' +
        '<span class="track">' + whisk + '<i class="dot c-' + cat(id) + '" style="left:' + pct(v) + '%"></i></span>' +
        '<span class="val">' + (diff ? (id === REF ? 'ref' : sgn(v)) : f3(v)) + '</span></button>';
    });
    html += '<div class="axis">' + ticks.map(function (t) { return '<span style="left:' + pct(t) + '%">' + tf(t) + '</span>'; }).join('') + '</div>';
    el('plot').innerHTML = html;

    var sr = el('slider-row');
    if (state.tab === 'mu' || state.tab === 'priv') {
      sr.hidden = false;
      var k = ids.indexOf(state.sel[state.tab]);
      el('slider').max = ids.length - 1; el('slider').value = k;
      el('slider-label').textContent = (state.tab === 'mu' ? 'FedProx weight' : 'Noise multiplier') + ': ' + X[ids[k]].label.replace(/^[^=]*= /, '') + ' (drag to change)';
    } else sr.hidden = true;
    renderDetail();
  }

  function renderDetail() {
    var id = state.sel[state.tab], e = X[id], c = e.metrics.care, ci = e.ci, tag;
    if (id === REF) tag = '<span class="tag ref">Reference</span>';
    else if (!ci.excludes_zero) tag = '<span class="tag ref">Not distinguishable from local</span>';
    else tag = '<span class="tag ' + (ci.diff < 0 ? 'bad' : 'good') + '">' + (ci.diff < 0 ? 'Lower than local' : 'Higher than local') + '</span>';
    var meters = ['coverage', 'earliness', 'reliability', 'accuracy'].map(function (k) {
      return '<div class="meter"><span>' + METRIC[k] + '</span><div class="bar-t"><div class="bar-f" style="width:' + (e.metrics[k] * 100).toFixed(1) + '%"></div></div><b>' + f3(e.metrics[k]) + '</b></div>';
    }).join('');
    var sd = e.sd && e.sd.care !== null ? ' ± ' + f3(e.sd.care) : '';
    el('detail').innerHTML =
      '<h3>' + esc(dn(id)) + '</h3>' +
      '<div class="row2"><div class="big">' + f3(c) + '</div><span class="delta">CARE' + sd + (id === REF ? '' : ' &nbsp;' + sgn(ci.diff) + ' vs local') + '</span></div>' +
      '<div>' + tag + '</div>' +
      (id === REF ? '' : '<p class="small">95% interval for the difference: ' + sgn(ci.lo) + ' to ' + sgn(ci.hi) + ' (' + ci.draws + ' draws, turbines resampled).</p>') +
      '<div class="meters">' + meters + '</div>' +
      '<div class="facts"><span><b>' + e.detected + '</b> of ' + M.n_anomaly + ' faults found</span><span><b>' + e.false_alarms + '</b> of ' + M.n_normal + ' false alarms</span></div>' +
      (e.epsilon !== null ? '<p class="small">Privacy budget ε = ' + e.epsilon.toFixed(2) + ' (largest across turbines).</p>' : '') +
      (e.note ? '<p class="small">' + esc(e.note) + '</p>' : '') +
      '<p class="path"><a href="' + REPO + '/tree/main/' + e.path + '" target="_blank" rel="noopener">' + esc(e.path) + '</a> &middot; ' + (e.seeds.length === 1 ? 'seed ' + e.seeds[0] : e.seeds.length + ' seeds') + '</p>';
  }

  /* events tab */
  var ALL_IDS = [].concat(G.arms, G.mu, G.priv);
  function fillSelect(sel, val) {
    sel.innerHTML = ALL_IDS.map(function (id) { return '<option value="' + esc(id) + '"' + (id === val ? ' selected' : '') + '>' + esc(dn(id)) + '</option>'; }).join('');
  }
  function outcome(ev, det) {
    return ev.anomaly ? (det ? ['Detected', 'good'] : ['Missed', 'bad']) : (det ? ['False alarm', 'bad'] : ['Correct', 'good']);
  }
  function renderEvents() {
    var a = new Set(X[state.evA].detected_events), b = new Set(X[state.evB].detected_events);
    var changed = 0, aF = 0, bF = 0, aA = 0, bA = 0;
    D.events.forEach(function (ev) {
      var da = a.has(ev.id), db = b.has(ev.id);
      if (da !== db) changed++;
      if (ev.anomaly) { aF += da; bF += db; } else { aA += da; bA += db; }
    });
    el('ev-summary').textContent = changed + ' of ' + M.n_events + ' events change outcome. Faults found: ' + aF + ' vs ' + bF + ' of ' + M.n_anomaly +
      '. False alarms: ' + aA + ' vs ' + bA + ' of ' + M.n_normal + '.';
    var rows = D.events.filter(function (ev) {
      var diff = a.has(ev.id) !== b.has(ev.id);
      return state.evF === 'all' || (state.evF === 'changed' && diff) || (state.evF === 'fault' && ev.anomaly) || (state.evF === 'normal' && !ev.anomaly);
    });
    var head = '<thead><tr><th>Event</th><th>Turbine</th><th>Type</th><th>' + esc(dn(state.evA)) + '</th><th>' + esc(dn(state.evB)) + '</th></tr></thead>';
    var body = rows.map(function (ev) {
      var oa = outcome(ev, a.has(ev.id)), ob = outcome(ev, b.has(ev.id));
      return '<tr class="' + (oa[0] !== ob[0] ? 'changed' : '') + '"><td class="n">' + ev.id + '</td><td class="n">' + ev.asset + '</td><td>' + (ev.anomaly ? 'Fault' : 'Normal') +
        '</td><td><span class="out ' + oa[1] + '">' + oa[0] + '</span></td><td><span class="out ' + ob[1] + '">' + ob[0] + '</span></td></tr>';
    }).join('');
    el('ev-table').innerHTML = head + '<tbody>' + (body || '<tr><td colspan="5">No events match this filter.</td></tr>') + '</tbody>';
  }

  function renderExplorer() {
    var ev = state.tab === 'events';
    el('ex-chart').hidden = ev; el('ex-events').hidden = !ev; el('metric-seg').hidden = ev;
    [].forEach.call(el('ex-tabs').children, function (c) { c.setAttribute('aria-selected', c.dataset.tab === state.tab); });
    [].forEach.call(el('metric-seg').children, function (c) { c.setAttribute('aria-pressed', c.dataset.m === state.metric); });
    if (ev) renderEvents(); else renderChart();
  }

  el('plot').addEventListener('click', function (e) {
    var b = e.target.closest('.row'); if (!b) return;
    state.sel[state.tab] = b.dataset.id; renderChart();
  });
  el('slider').addEventListener('input', function (e) { state.sel[state.tab] = G[state.tab][+e.target.value]; renderChart(); });
  el('ex-tabs').addEventListener('click', function (e) {
    var b = e.target.closest('button'); if (!b) return;
    state.tab = b.dataset.tab; renderExplorer();
  });
  el('metric-seg').addEventListener('click', function (e) {
    var b = e.target.closest('button'); if (!b) return;
    state.metric = b.dataset.m; renderExplorer();
  });
  fillSelect(el('ev-a'), state.evA); fillSelect(el('ev-b'), state.evB);
  el('ev-a').addEventListener('change', function (e) { state.evA = e.target.value; renderEvents(); });
  el('ev-b').addEventListener('change', function (e) { state.evB = e.target.value; renderEvents(); });
  el('ev-filter').addEventListener('click', function (e) {
    var b = e.target.closest('button'); if (!b) return;
    state.evF = b.dataset.f;
    [].forEach.call(el('ev-filter').children, function (c) { c.setAttribute('aria-pressed', c === b); });
    renderEvents();
  });

  /* ---------- reproduce ---------- */
  var CLONE = 'git clone https://github.com/PranavMundhra/Wind-Turbine.git && cd Wind-Turbine';
  var TRACKS = {
    a: { title: 'Rebuild the tables', sub: 'Needs Python 3.10 or later. Nothing is downloaded and no GPU is used.', steps: [
      { h: 'Get the code', p: 'Clone the repository and move into it.', c: CLONE },
      { h: 'Install', p: 'The extra plotting package is for Fig. 4.', c: 'pip install -e ".[dev]" matplotlib' },
      { h: 'Rebuild Table I and the bootstrap intervals', p: 'Reads the committed runs under results/paper and rewrites the summary tables.', c: 'python -m windturbine summarise --runs results/paper --out results/paper', see: 'results/paper/table_I.md, table_I.csv and bootstrap_vs_D.csv' },
      { h: 'Draw Fig. 4', p: 'Validation curves and CARE for the runs.', c: 'python -m windturbine figures', see: 'results/paper/figures/fig4_validation_and_care.pdf' },
      { h: 'Check against the committed table', p: 'This is the check the project’s CI runs after every push.', c: 'git diff --exit-code -- results/paper/table_I.md', see: 'no output. Your table matches the committed one.' },
      { h: 'Prefer a guided version?', p: 'The walkthrough notebook shows the same tables and figure with explanations, and is stored with its outputs.', link: [REPO + '/blob/main/notebooks/reproduce.ipynb', 'notebooks/reproduce.ipynb'] }
    ] },
    b: { title: 'Re-run the experiments', sub: 'Plan for several hours of GPU time. On one RTX 5070 Ti with three runs in parallel, a single run took 0.9 to 2.1 hours.', steps: [
      { h: 'Install PyTorch with CUDA', p: 'Pick the CUDA build for your GPU first. Keras 3 runs on the PyTorch backend, which also works on Windows.', link: ['https://pytorch.org', 'pytorch.org'] },
      { h: 'Install the package', p: 'Optionally run the tests, which take about 3 minutes on CPU.', c: ['pip install -e ".[train,dev]"', 'python -m pytest -q'] },
      { h: 'Get the data', p: 'CARE to Compare is published under CC BY-SA 4.0 and is not redistributed here. Use version 6; version 2 has timestamps that do not match its event files.', link: ['https://doi.org/10.5281/zenodo.15846963', 'doi:10.5281/zenodo.15846963'] },
      { h: 'Build the leak-free clients', p: '--raw is the folder that contains “Wind Farm C”. One partition is written per turbine.', c: 'python -m windturbine build-clients --raw /path/to/CARE_To_Compare --out /path/to/clients', see: 'the SHA-256 of dataset_manifest.json. Every experiment pins the expected hash and stops before training if it differs.' },
      { h: 'Smoke test the environment', p: 'Two rounds of everything, a few minutes. Catches problems before a long batch.', c: 'python -m windturbine run all --seeds 0 --clients /path/to/clients --out results/smoke --rounds 2 --set max_epochs=2' },
      { h: 'Run the experiments', p: 'Runs go to results/runs/<experiment>/s<seed>/, which is git-ignored. "all" uses the corrected comparator central; name central_sequential to reproduce the paper’s.', c: ['python -m windturbine list', 'python -m windturbine run all --seeds 42 --clients /path/to/clients --jobs 3'] },
      { h: 'Build Table I and Fig. 4', p: 'summarise copies the small files of each finished run into results/paper and rebuilds the tables from them.', c: ['python -m windturbine summarise --runs results/runs --out results/paper', 'python -m windturbine figures'] }
    ] },
    c: { title: 'Add more seeds', sub: 'A seed changes only the model initialisation and the minibatch order; the client partitions are fixed by the export. One seed of every experiment is about 20 run-hours, roughly 7 hours of wall time with --jobs 3.', steps: [
      { h: 'Finish track B first', p: 'Seeds reuse the same install and the same client export.' },
      { h: 'Run the priority subset', p: 'The rows the paper’s comparisons rest on, roughly 2 hours of wall time per seed. Runs that already exist are skipped, so an interrupted batch resumes when you repeat the command.', c: 'python -m windturbine run D central F1 F2 F3_mu0.1 --seeds 42 43 44 45 46 --clients /path/to/clients --jobs 3' },
      { h: 'Run the remaining rows', p: 'Optional, for the full table at every seed.', c: 'python -m windturbine run all --seeds 43 44 45 46 --clients /path/to/clients --jobs 3' },
      { h: 'Rebuild the tables', p: 'With several seeds, table_I.md reports mean ± SD, and bootstrap_vs_D.csv averages each experiment over its seeds before taking the difference from D.', c: 'python -m windturbine summarise --runs results/runs --out results/paper' }
    ] }
  };
  function renderTrack(k) {
    var t = TRACKS[k];
    el('steps-title').textContent = t.title; el('steps-sub').textContent = t.sub;
    el('steps').innerHTML = t.steps.map(function (s) {
      var cmds = s.c ? [].concat(s.c) : [];
      return '<li class="card step"><h3>' + esc(s.h) + '</h3><p>' + esc(s.p) + '</p>' +
        cmds.map(function (c) { return '<div class="code"><pre><code>' + esc(c) + '</code></pre><button class="copy" type="button">Copy</button></div>'; }).join('') +
        (s.link ? '<p class="see"><a href="' + s.link[0] + '" target="_blank" rel="noopener">' + esc(s.link[1]) + '</a></p>' : '') +
        (s.see ? '<p class="see"><b>You should see</b> ' + esc(s.see) + '</p>' : '') + '</li>';
    }).join('');
    [].forEach.call(document.querySelectorAll('.track-btn'), function (b) { b.setAttribute('aria-pressed', b.dataset.track === k); });
  }
  document.querySelector('.tracks').addEventListener('click', function (e) {
    var b = e.target.closest('.track-btn'); if (b) renderTrack(b.dataset.track);
  });
  el('steps').addEventListener('click', function (e) {
    var b = e.target.closest('.copy'); if (!b) return;
    var code = b.parentNode.querySelector('code');
    function done() { b.textContent = 'Copied'; setTimeout(function () { b.textContent = 'Copy'; }, 1500); }
    function fallback() {
      var r = document.createRange(); r.selectNodeContents(code);
      var s = window.getSelection(); s.removeAllRanges(); s.addRange(r); b.textContent = 'Press Ctrl+C';
    }
    try { navigator.clipboard.writeText(code.textContent).then(done, fallback); } catch (err) { fallback(); }
  });

  function link(path) {
    return '<a href="' + REPO + (/\/$/.test(path) ? '/tree/main/' : '/blob/main/') + path + '" target="_blank" rel="noopener">' + esc(path) + '</a>';
  }
  var MAP = [
    ['Exploratory analysis behind the feature choice (Sec. III)', ['notebooks/01_eda.ipynb'], []],
    ['Leakage audit (Sec. III, Fig. 1)', ['src/windturbine/audit/', 'notebooks/02_leakage_audit.ipynb'], ['results/leakage_audit/']],
    ['Leak-free clients (Sec. IV)', ['src/windturbine/data/clients.py'], ['configs/clients.yaml']],
    ['Autoencoder and training step', ['src/windturbine/models/autoencoder.py'], ['configs/base.yaml']],
    ['Local training (D)', ['src/windturbine/local.py'], ['configs/paper/D.yaml']],
    ['FedAvg (F1) and the centralised comparator', ['src/windturbine/federated/fedavg.py'], ['configs/paper/F1.yaml', 'configs/paper/central.yaml']],
    ['FedPer (F2), FedProx (F3), private (F4)', ['src/windturbine/federated/strategies.py'], ['configs/paper/']],
    ['CARE score and thresholds', ['src/windturbine/scoring/'], []],
    ['Table I, bootstrap and Fig. 4', ['src/windturbine/summary.py', 'notebooks/reproduce.ipynb'], ['results/paper/']],
    ['Command line', ['src/windturbine/__main__.py', 'src/windturbine/experiments.py'], []]
  ];
  el('map').innerHTML = '<thead><tr><th>Paper</th><th>Code</th><th>Config or results</th></tr></thead><tbody>' + MAP.map(function (r) {
    return '<tr><td>' + esc(r[0]) + '</td><td>' + r[1].map(link).join('<br>') + '</td><td>' + (r[2].length ? r[2].map(link).join('<br>') : '<span class="none">&ndash;</span>') + '</td></tr>';
  }).join('') + '</tbody>';

  el('notes').innerHTML = [
    ['D and F1 to F4 were trained in different environments', 'D was trained on Kaggle with TensorFlow, and its results were copied from notebook output at six decimals. F1 to F4 were trained on an RTX 5070 Ti with Keras 3 on PyTorch, on a local rebuild of the client export with identical per-turbine row counts. New D runs use the PyTorch training step, so multi-seed results do not have this difference.'],
    ['The paper’s centralised comparator drifted', 'In central_sequential each epoch trains one turbine at a time in random order, so the model moves toward the last turbine. Its validation error was lowest at epoch 2 and rose by epoch 100. The corrected comparator, central, shuffles minibatches across all turbines and is the one to use for new runs.'],
    ['All committed runs use one seed', 'Add seeds (track C) before drawing conclusions about differences of a few thousandths of CARE.']
  ].map(function (n) { return '<div class="card note"><b>' + esc(n[0]) + '</b><span>' + esc(n[1]) + '</span></div>'; }).join('');

  el('faq').innerHTML = [
    ['I have no GPU. What can I do?', 'Track A needs no GPU and no download. The test suite also runs on CPU in about 3 minutes, because its training tests use tiny synthetic data.'],
    ['The run stops before training with a dataset hash message', 'Every experiment pins the hash of the client export in configs/paper/_dataset.yaml. To use another export on purpose, add --dataset-sha <hash> and --dataset-reason "<why>" to the run command; the reason is recorded in each run’s manifest.'],
    ['An overnight batch was interrupted', 'Repeat the same command. Runs that already exist are skipped, so the batch resumes where it stopped.'],
    ['My D result does not match Table I exactly', 'The committed seed-42 D result came from Kaggle notebook output at six decimals. A new D run uses the PyTorch training step and will differ slightly.'],
    ['Does training work on Windows?', 'Yes. Keras 3 runs on the PyTorch backend, which the package selects automatically, so the GPU is used on Windows as well as Linux. requirements-lock.txt lists the exact versions of the seed-42 runs.']
  ].map(function (q) { return '<details><summary>' + esc(q[0]) + '</summary><p>' + esc(q[1]) + '</p></details>'; }).join('');

  /* ---------- views and theme ---------- */
  function show(view) {
    ['present', 'reproduce'].forEach(function (v) {
      el('view-' + v).hidden = v !== view;
      el('tab-' + v).setAttribute('aria-selected', v === view);
    });
    el('present-btn').hidden = view !== 'present';
  }
  [].forEach.call(document.querySelectorAll('.tab'), function (b) {
    b.addEventListener('click', function () { location.hash = b.dataset.view; });
  });
  function route() { show(location.hash === '#reproduce' ? 'reproduce' : 'present'); }
  window.addEventListener('hashchange', route);

  function isDark() {
    var t = document.documentElement.getAttribute('data-theme');
    return t ? t === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function themeLabel() { el('theme-btn').textContent = isDark() ? 'Light mode' : 'Dark mode'; }
  try { var saved = localStorage.getItem('wt-theme'); if (saved) document.documentElement.setAttribute('data-theme', saved); } catch (e) { /* storage unavailable */ }
  el('theme-btn').addEventListener('click', function () {
    var next = isDark() ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('wt-theme', next); } catch (e) { /* storage unavailable */ }
    themeLabel();
  });
  if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', themeLabel);

  /* ---------- full-screen presenting ---------- */
  var SLIDES = [{ id: 's-hero' }];
  for (var sc = 0; sc < window.WT_STORY.count; sc++) SLIDES.push({ id: 's-story', scene: sc });
  SLIDES.push({ id: 's-explorer', tab: 'arms' }, { id: 's-explorer', tab: 'mu' }, { id: 's-explorer', tab: 'priv' },
    { id: 's-explorer', tab: 'events' }, { id: 's-take' });
  var pres = { on: false, i: 0, fs: false };
  el('pb-dots').innerHTML = SLIDES.map(function () { return '<i></i>'; }).join('');
  function applySlide() {
    var s = SLIDES[pres.i];
    [].forEach.call(document.querySelectorAll('.slide'), function (n) { n.classList.toggle('active', n.id === s.id); });
    if (s.tab) { state.tab = s.tab; renderExplorer(); }
    if (s.scene !== undefined) window.WT_STORY.show(s.scene);
    el('pb-replay').hidden = s.scene === undefined;
    [].forEach.call(el('pb-dots').children, function (d, k) { d.classList.toggle('on', k === pres.i); });
    el('pb-prev').disabled = pres.i === 0; el('pb-next').disabled = pres.i === SLIDES.length - 1;
    window.scrollTo(0, 0);
  }
  function go(d) { var n = pres.i + d; if (n >= 0 && n < SLIDES.length) { pres.i = n; applySlide(); } }
  function startPresent() {
    pres.on = true; pres.i = 0;
    document.documentElement.classList.add('presenting');
    el('present-bar').hidden = false;
    applySlide();
    var r = document.documentElement.requestFullscreen;
    if (r) {
      try { Promise.resolve(r.call(document.documentElement)).then(function () { pres.fs = true; }, function () { /* stay in page */ }); } catch (e) { /* stay in page */ }
    }
  }
  function endPresent() {
    if (!pres.on) return;
    pres.on = false;
    document.documentElement.classList.remove('presenting');
    el('present-bar').hidden = true;
    [].forEach.call(document.querySelectorAll('.slide'), function (n) { n.classList.remove('active'); });
    if (pres.fs && document.fullscreenElement) { try { document.exitFullscreen(); } catch (e) { /* ignore */ } }
    pres.fs = false;
  }
  el('present-btn').addEventListener('click', startPresent);
  el('pb-next').addEventListener('click', function () { go(1); });
  el('pb-prev').addEventListener('click', function () { go(-1); });
  el('pb-replay').addEventListener('click', function () { window.WT_STORY.replay(); });
  el('pb-exit').addEventListener('click', endPresent);
  document.addEventListener('fullscreenchange', function () { if (pres.on && pres.fs && !document.fullscreenElement) endPresent(); });
  document.addEventListener('keydown', function (e) {
    if (!pres.on) return;
    var t = e.target && e.target.tagName;
    if (e.key === 'Escape') { endPresent(); return; }
    if ((e.key === 'r' || e.key === 'R') && SLIDES[pres.i].scene !== undefined) { window.WT_STORY.replay(); return; }
    if (t === 'INPUT' || t === 'SELECT' || t === 'TEXTAREA') return;
    if (['ArrowRight', 'ArrowDown', 'PageDown', ' '].indexOf(e.key) >= 0) { e.preventDefault(); go(1); }
    else if (['ArrowLeft', 'ArrowUp', 'PageUp'].indexOf(e.key) >= 0) { e.preventDefault(); go(-1); }
  });

  renderStatic(); renderExplorer(); renderTrack('a'); themeLabel(); route();
})();
