/* The CARE maths tab: formulas as MathML, plus calculators that apply the same rules as src/windturbine/scoring/.
   Constants (beta, quantile, criticality threshold, weights) and all real scores come from window.WT_DATA. The small
   simulations use made-up series that only illustrate the mechanics. */
(function () {
  'use strict';
  const D = window.WT_DATA, X = D.experiments, G = D.groups, M = D.meta, K = M.care, REF = M.reference;
  const BETA = K.beta, Q = K.score_quantile, THETA = K.criticality_threshold, W = K.weights;
  const WSUM = W.reduce((a, b) => a + b, 0);
  const REPO = 'https://github.com/PranavMundhra/Wind-Turbine/blob/main/';
  const MINUS = '−';
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const f3 = (v) => v.toFixed(3);
  const sgn = (v, d = 3) => (v < 0 ? MINUS : '+') + Math.abs(v).toFixed(d);
  const rng = (seed) => { let s = seed; return () => { s = (s * 16807) % 2147483647; return s / 2147483647; }; };
  const gauss = (r) => Math.sqrt(-2 * Math.log(r() || 1e-9)) * Math.cos(2 * Math.PI * r());
  const fbeta = (tp, fp, fn, beta) => { const b2 = beta * beta, den = (1 + b2) * tp + b2 * fn + fp; return den > 0 ? (1 + b2) * tp / den : 0; };
  const careOf = (c, e, r, a) => (W[0] * c + W[1] * e + W[2] * r + W[3] * a) / WSUM;
  const allIds = [].concat(G.arms, G.mu, G.priv);
  const dn = (id) => (X[id].group === 'mu' ? 'FedProx ' : X[id].group === 'priv' ? 'Private ' : '') + X[id].label;

  /* ---------- MathML helpers ---------- */
  const mi = (s) => '<mi>' + s + '</mi>', mn = (s) => '<mn>' + s + '</mn>', mo = (s) => '<mo>' + s + '</mo>', mt = (s) => '<mtext>' + s.replace(/ /g, '\u00a0') + '</mtext>';
  const row = (a) => '<mrow>' + a.join('') + '</mrow>';
  const sub = (a, b) => '<msub>' + a + b + '</msub>', sup = (a, b) => '<msup>' + a + b + '</msup>';
  const frac = (a, b) => '<mfrac>' + a + b + '</mfrac>';
  const hat = (x) => '<mover accent="true">' + x + '<mo>^</mo></mover>';
  const par = (a) => row([mo('('), row(a), mo(')')]);
  const cases = (rows) => row([mo('{'), '<mtable columnalign="left left" columnspacing="1.2em">' + rows.map((r) => '<mtr><mtd>' + r[0] + '</mtd><mtd>' + r[1] + '</mtd></mtr>').join('') + '</mtable>']);
  const block = (a) => '<div class="formula"><math display="block">' + row(a) + '</math></div>';

  const bSq = (b) => b.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
  const F_BETA = (tp, fp, fn, b) => row([sub(mi('F'), mi('β')), mo('='), frac(
    row([par([mn('1'), mo('+'), sup(mi('β'), mn('2'))]), mo('⁢'), mi(tp)]),
    row([par([mn('1'), mo('+'), sup(mi('β'), mn('2'))]), mo('⁢'), mi(tp), mo('+'), sup(mi('β'), mn('2')), mo('⁢'), mi(fn), mo('+'), mi(fp)]))]);

  const F = {
    score: block([sub(mi('e'), mi('t')), mo('='), sub(row([mo('‖'), sub(mi('x'), mi('t')), mo('−'), sub(hat(mi('x')), mi('t')), mo('‖')]), mn('2'))]),
    tau: block([mi('τ'), mo('='), sub(mi('Q'), mn(String(Q))), par([row([mo('{'), sub(mi('e'), mi('s')), mo(':'), mi('s'), mo('∈'), mt('held-out healthy data'), mo('}')])])]),
    flag: block([sub(mi('p'), mi('t')), mo('='), cases([[mn('1'), row([mt('if '), sub(mi('e'), mi('t')), mo('>'), mi('τ')])], [mn('0'), mt('otherwise')]])]),
    counter: block([sub(mi('c'), mi('t')), mo('='), cases([
      [sub(mi('c'), row([mi('t'), mo('−'), mn('1')])), mt('abnormal status')],
      [row([sub(mi('c'), row([mi('t'), mo('−'), mn('1')])), mo('+'), mn('1')]), row([mt('normal, '), sub(mi('p'), mi('t')), mo('='), mn('1')])],
      [row([mi('max'), par([sub(mi('c'), row([mi('t'), mo('−'), mn('1')])), mo('−'), mn('1'), mo(','), mn('0')])]), row([mt('normal, '), sub(mi('p'), mi('t')), mo('='), mn('0')])]])]),
    detected: block([mt('detected'), mo('='), sub(mn('1'), row([mo('['), mi('max'), mo('⁡'), sub(mi('c'), mi('t')), mo('≥'), mi('θ'), mo(']')])), mo(','), mspace(), mi('θ'), mo('='), mn(String(THETA))]),
    fbeta: block([F_BETA('TP', 'FP', 'FN')]),
    rel: block([mi('R'), mo('='), sub(mi('F'), mn(String(BETA))), par([mt('TP'), mo(','), mt('FP'), mo(','), mt('FN')]), mt('  counted over events')]),
    cov: block([sub(mi('C'), mt('event')), mo('='), sub(mi('F'), mn(String(BETA))), par([mt('TP'), mo(','), mt('FP'), mo(','), mt('FN')])]),
    covdef: '<div class="formula"><math display="block"><mtable columnalign="left" rowspacing="0.4em">' +
      [['TP', row([mo('∑'), sub(mi('p'), mi('t')), sub(mi('g'), mi('t'))])],
        ['FP', row([mo('∑'), sub(mi('p'), mi('t')), par([mn('1'), mo('−'), sub(mi('g'), mi('t'))])])],
        ['FN', row([mo('∑'), par([mn('1'), mo('−'), sub(mi('p'), mi('t'))]), sub(mi('g'), mi('t'))])]]
        .map((r) => '<mtr><mtd>' + mt(r[0]) + '</mtd><mtd>' + mo('=') + '</mtd><mtd>' + r[1] + '</mtd></mtr>').join('') + '</mtable></math></div>',
    wgt: block([mi('w'), par([mi('u')]), mo('='), cases([[mn('1'), row([mi('u'), mo('≤'), frac(mn('1'), mn('2'))])], [row([mn('1'), mo('−'), frac(row([mi('u'), mo('−'), frac(mn('1'), mn('2'))]), frac(mn('1'), mn('2'))), mo('='), mn('2'), mo('−'), mn('2'), mi('u')]), row([mi('u'), mo('>'), frac(mn('1'), mn('2'))])]]), mspace(), mi('u'), mo('='), frac(row([mi('t'), mo('−'), mt('start')]), row([mt('end'), mo('−'), mt('start')]))]),
    earl: block([sub(mi('E'), mt('event')), mo('='), frac(row([mo('∑'), mi('w'), par([sub(mi('u'), mi('t'))]), sub(mi('p'), mi('t'))]), row([mo('∑'), mi('w'), par([sub(mi('u'), mi('t'))])]))]),
    acc: block([sub(mi('A'), mt('event')), mo('='), frac(mt('TN'), row([mt('TN'), mo('+'), mt('FP')])), mo(','), mspace(), mt('over steps of a healthy event')]),
    mean: '<div class="formula"><math display="block"><mtable columnalign="left" rowspacing="0.5em">' +
      [['C', 'C', 'fault', 'f'], ['E', 'E', 'fault', 'f'], ['A', 'A', 'healthy', 'h']]
        .map((r) => '<mtr><mtd>' + mi(r[0]) + '</mtd><mtd>' + mo('=') + '</mtd><mtd>' + frac(mn('1'), sub(mi('n'), mi(r[3]))) + sub(mo('∑'), mt(r[2] + ' events')) + sub(mi(r[1]), mt('event')) + '</mtd></mtr>').join('') + '</mtable></math></div>',
    care: block([mt('CARE'), mo('='), frac(row([mi('C'), mo('+'), mi('E'), mo('+'), mi('R'), mo('+'), mn(String(W[3])), mo('⁢'), mi('A')]), mn(String(WSUM)))]),
    cases: block([mt('CARE'), mo('='), cases([
      [frac(row([mi('C'), mo('+'), mi('E'), mo('+'), mi('R'), mo('+'), mn(String(W[3])), mo('⁢'), mi('A')]), mn(String(WSUM))), row([mt('otherwise')])],
      [mi('A'), row([mt('if '), mi('A'), mo('<'), mn('0.5')])],
      [mn('0'), mt('if no event is detected')]])]),
    sd: block([mi('s'), mo('='), msqrt(frac(row([mo('∑'), sup(par([sub(mi('CARE'), mi('k')), mo('−'), mover(mi('CARE'))]), mn('2'))]), row([mi('S'), mo('−'), mn('1')])))]),
    delta: block([mi('Δ'), mo('='), mover(mt('CARE')), par([mi('a')]), mo('−'), mover(mt('CARE')), par([mt('local training')])])
  };
  function mspace() { return '<mspace width="1em"/>'; }
  function msqrt(a) { return '<msqrt>' + a + '</msqrt>'; }
  function mover(a) { return '<mover accent="true">' + a + '<mo>¯</mo></mover>'; }

  const link = (path, label) => '<a href="' + REPO + path + '" target="_blank" rel="noopener">' + (label || path) + '</a>';
  const sec = (id, n, title, text, widget) =>
    '<section class="card m-sec" id="' + id + '"><div class="m-head"><span class="m-n">' + n + '</span><h2>' + title + '</h2></div><div class="m-grid"><div class="m-text">' + text + '</div><div class="m-widget">' + widget + '</div></div></section>';

  /* ---------- page text ---------- */
  const PIPE = [
    ['m-score', 'Score', 'how far each reading is from what the model expects'],
    ['m-score', 'Flag', 'compare the score with a per-turbine threshold'],
    ['m-counter', 'Decide', 'a counter turns flags into event decisions'],
    ['m-reliability', 'Grade', 'Coverage, Earliness, Reliability, Accuracy'],
    ['m-combine', 'Combine', 'one weighted number between 0 and 1'],
    ['m-stats', 'Compare', 'seeds, spread and bootstrap intervals']
  ];

  $('m-lead').innerHTML = 'CARE turns a model’s reading-by-reading alarms into one number between 0 and 1. It rewards catching faults early and persistently while staying quiet on healthy turbines. ' +
    'Every formula on this page is the one implemented in ' + link('src/windturbine/scoring/', 'src/windturbine/scoring/') + ', and the numbers that parameterise it (β = ' + BETA + ', q = ' + Q + ', θ = ' + THETA + ', weights ' + W.join(', ') + ') are read from the repository’s config.';

  $('m-toc').innerHTML = PIPE.filter((p, i, a) => a.findIndex((q) => q[0] === p[0]) === i).map((p) => '<button type="button" class="ghost" data-go="' + p[0] + '">' + p[1] + '</button>').join('');
  $('m-flow').innerHTML = PIPE.map((p, i) => '<button type="button" class="flow-step" data-go="' + p[0] + '"><span class="flow-n">' + (i + 1) + '</span><b>' + p[1] + '</b><span>' + p[2] + '</span></button>').join('');
  const goTo = (e) => { const b = e.target.closest('[data-go]'); if (b) $(b.dataset.go).scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' }); };
  $('m-toc').addEventListener('click', goTo); $('m-flow').addEventListener('click', goTo);

  const verifyRows = allIds.map((id) => {
    const m = X[id].metrics, f = careOf(m.coverage, m.earliness, m.reliability, m.accuracy);
    return '<tr><td>' + esc(dn(id)) + '</td><td class="n">' + X[id].seeds.length + '</td><td class="n">' + f3(m.coverage) + '</td><td class="n">' + f3(m.earliness) + '</td><td class="n">' + f3(m.reliability) + '</td><td class="n">' + f3(m.accuracy) +
      '</td><td class="n"><b>' + f3(f) + '</b></td><td class="n">' + f3(m.care) + '</td><td class="n">' + (Math.abs(f - m.care) * 1e6).toFixed(1) + '</td></tr>';
  }).join('');

  $('m-sections').innerHTML =
    sec('m-score', 1, 'Score every reading, then flag it',
      '<p>A normal-behaviour autoencoder is trained on healthy data from one turbine. For each reading <i>x<sub>t</sub></i> it tries to rebuild the reading, and the size of the miss is the anomaly score <i>e<sub>t</sub></i>.</p>' + F.score +
      '<p>Each turbine gets its own threshold <i>τ</i>: the <i>q</i>-quantile (here <i>q</i> = ' + Q + ') of the scores on held-out healthy data. By construction only about 1 − <i>q</i> of healthy readings land above it.</p>' + F.tau +
      '<p>A reading is flagged when its score is above the threshold.</p>' + F.flag +
      '<p class="m-code">In the code: ' + link('src/windturbine/scoring/threshold.py') + '</p>',
      '<div class="w-box"><p class="w-title">Move the quantile and watch the flags</p><svg id="w1-svg" class="m-svg" viewBox="0 0 520 230" role="img" aria-label="Scores over time with a threshold line"></svg>' +
      '<div class="w-ctl"><label for="w1-q">Quantile <i>q</i>: <b id="w1-qv"></b></label><input type="range" id="w1-q" min="0.8" max="0.999" step="0.001"></div><p class="w-out" id="w1-out"></p>' +
      '<p class="w-note">Made-up scores, for illustration. Lower <i>q</i> catches the fault sooner but flags more healthy readings.</p></div>') +

    sec('m-counter', 2, 'From flags to an event decision',
      '<p>One flag is not an alarm. A counter <i>c<sub>t</sub></i> goes up by one for every flagged reading and down by one (never below zero) for every unflagged one. Readings taken while the turbine is not in normal operating status leave it unchanged.</p>' + F.counter +
      '<p>An event counts as detected if the counter ever reaches <i>θ</i> = ' + THETA + '. The benchmark paper sets it to 72, about 12 hours of consecutive flags at 10-minute resolution. Reaching it needs a sustained excess of flags, so isolated blips never trigger an event.</p>' + F.detected +
      '<p class="m-code">In the code: ' + link('src/windturbine/scoring/care.py', 'criticality() and event_detected()') + '</p>',
      '<div class="w-box"><p class="w-title">Which flag patterns trigger an event?</p><div class="seg" id="w2-presets" role="group" aria-label="Flag pattern"></div>' +
      '<svg id="w2-svg" class="m-svg" viewBox="0 0 520 250" role="img" aria-label="Flags and the running counter"></svg>' +
      '<div class="w-ctl"><label for="w2-t">Threshold θ: <b id="w2-tv"></b></label><input type="range" id="w2-t" min="10" max="120" step="1"></div>' +
      '<div class="w-row"><button type="button" class="ghost" id="w2-roll">New random pattern</button></div><p class="w-out" id="w2-out"></p>' +
      '<p class="w-note">Made-up flags, all in normal status. Real events run much longer.</p></div>') +

    sec('m-reliability', 3, 'Reliability: were the event decisions right?',
      '<p>Once every event is marked detected or not, count the faults that were caught (TP), the healthy events that raised an alarm (FP) and the faults that were missed (FN). Reliability is the F<sub>β</sub> score of those event counts with β = ' + BETA + '.</p>' + F.fbeta + F.rel +
      '<p>In the denominator a false alarm counts 1 and a missed fault counts β² = ' + (BETA * BETA) + ', so a β below 1 leans toward precision. The calculator shows what one more false alarm, or one more miss, does to the score. A miss also removes a hit from the numerator, so the two effects can be close.</p>' +
      '<p class="m-code">In the code: ' + link('src/windturbine/scoring/care.py', 'fbeta_from_counts()') + '</p>',
      '<div class="w-box"><p class="w-title">Try the counts from a real run</p><div class="w-sel"><div class="field"><label for="w3-exp">Experiment</label><select id="w3-exp"></select></div><div class="field"><label for="w3-seed">Seed</label><select id="w3-seed"></select></div></div>' +
      '<div class="w-ctl"><label for="w3-tp">Faults caught, TP (of ' + M.n_anomaly + '): <b id="w3-tpv"></b></label><input type="range" id="w3-tp" min="0" max="' + M.n_anomaly + '" step="1"></div>' +
      '<div class="w-ctl"><label for="w3-fp">Healthy events with an alarm, FP (of ' + M.n_normal + '): <b id="w3-fpv"></b></label><input type="range" id="w3-fp" min="0" max="' + M.n_normal + '" step="1"></div>' +
      '<div class="w-ctl"><label for="w3-b">β: <b id="w3-bv"></b></label><input type="range" id="w3-b" min="0.25" max="2" step="0.05"></div><div class="w-out" id="w3-out"></div></div>') +

    sec('m-inside', 4, 'Inside the events: Coverage, Earliness, Accuracy',
      '<p>For a fault event, only readings in normal status count. Let <i>g<sub>t</sub></i> be 1 inside the labelled fault window and 0 outside it. A flagged reading inside the labelled fault window is a hit, a flagged reading outside it is a false positive, and an unflagged reading inside it is a miss. <b>Coverage</b> is the same F<sub>β</sub> score as above, now over readings.</p>' + F.cov + F.covdef +
      '<p><b>Earliness</b> uses the same flags but weights the window: weight 1 through the first half, then falling in a straight line to 0 at the end. Raising the alarm early counts for more.</p>' + F.wgt + F.earl +
      '<p><b>Accuracy</b> looks at healthy events only: the share of readings left unflagged.</p>' + F.acc +
      '<p>Each score is then averaged over its events, every event weighing the same. Here <i>n<sub>f</sub></i> = ' + M.n_anomaly + ' fault events and <i>n<sub>h</sub></i> = ' + M.n_normal + ' healthy events.</p>' + F.mean +
      '<p class="m-code">In the code: ' + link('src/windturbine/scoring/evaluation.py', 'evaluate_care()') + ' and ' + link('src/windturbine/scoring/care.py', 'earliness_ws()') + ' (the weight function of the benchmark paper).</p>',
      '<div class="w-box"><p class="w-title">Paint the flags and read off the scores</p><div class="seg" id="w4-presets" role="group" aria-label="Preset"></div>' +
      '<p class="w-lane">A fault event <span class="w-sub">— click or drag across readings to flag them</span></p><div class="w4-wts" id="w4-wts" aria-hidden="true"></div><div class="cells" id="w4-fault"></div>' +
      '<p class="w-lane">A healthy event</p><div class="cells" id="w4-healthy"></div><div class="w-out" id="w4-out"></div>' +
      '<p class="w-note">The shaded readings are the fault window; the bars above them are the Earliness weights. Made-up event, for illustration.</p></div>') +

    sec('m-combine', 5, 'Putting the four together',
      '<p>CARE is a weighted mean of the four scores. Accuracy counts double.</p>' + F.care +
      '<p>Two rules override it. If Accuracy is below 0.5, CARE is just Accuracy. If no event is detected at all, CARE is 0.</p>' + F.cases +
      '<p>The page’s data confirm that this formula reproduces every published score. The table below recomputes CARE from each experiment’s four component means.</p>' +
      '<p class="m-code">In the code: ' + link('src/windturbine/scoring/evaluation.py', 'aggregate_event_metrics()') + '</p>',
      '<div class="w-box"><p class="w-title">Build a CARE score</p><div class="field"><label for="w5-exp">Start from</label><select id="w5-exp"></select></div>' +
      ['c:Coverage', 'e:Earliness', 'r:Reliability', 'a:Accuracy'].map((s) => { const [k, n] = s.split(':'); return '<div class="w-ctl"><label for="w5-' + k + '">' + n + ': <b id="w5-' + k + 'v"></b></label><input type="range" id="w5-' + k + '" min="0" max="1" step="0.001"></div>'; }).join('') +
      '<label class="w-check"><input type="checkbox" id="w5-none"> No event reaches the threshold</label><div class="w-out" id="w5-out"></div></div>') +
    '<div class="card m-sec" id="m-verify"><div class="m-head"><span class="m-n">✓</span><h2>The formula against every published row</h2></div>' +
      '<p class="sub">Component means from Table I, CARE recomputed with the formula above, and the published CARE. The last column is the difference in millionths, which is rounding.</p>' +
      '<div class="table-scroll"><table class="map m-verify"><thead><tr><th>Experiment</th><th>Seeds</th><th>C</th><th>E</th><th>R</th><th>A</th><th>Formula</th><th>Published</th><th>Diff ×10⁻⁶</th></tr></thead><tbody>' + verifyRows + '</tbody></table></div></div>' +

    sec('m-stats', 6, 'Comparing runs: seeds, spread and intervals',
      '<p>Training has randomness, so the main rows are run at several seeds. The table reports the mean CARE over the <i>S</i> seeds and its sample standard deviation (divisor <i>S</i> − 1).</p>' + F.sd +
      '<p>To ask whether federated learning differs from local training, the difference of the two means is computed, and its 95% interval comes from a bootstrap over <b>turbines</b>, not events: the ' + M.n_events + ' events come from ' + M.n_turbines + ' turbines, so events of one turbine are not independent.</p>' + F.delta +
      '<ol class="m-steps"><li>Draw ' + M.n_turbines + ' turbines with replacement. Keep every event of each drawn turbine.</li><li>Recompute CARE for every seed of both experiments on that same draw, and average over seeds.</li><li>Take the difference. Repeat 2000 times.</li><li>The 2.5th and 97.5th percentiles of the differences are the interval.</li></ol>' +
      '<p class="m-code">In the code: ' + link('src/windturbine/summary.py', 'bootstrap_difference()') + '</p>',
      '<div class="w-box"><p class="w-title">One resample of the turbines</p><div class="turb-grid" id="w6-grid"></div><div class="w-row"><button type="button" class="ghost" id="w6-draw">Draw a resample</button></div><p class="w-out" id="w6-out"></p>' +
      '<hr class="w-hr"><p class="w-title">The published result</p><div class="field"><label for="w6-exp">Experiment compared with local training</label><select id="w6-exp"></select></div><div class="w-out" id="w6-ci"></div></div>');


  /* ---------- how this matches the benchmark ---------- */
  $('m-sections').insertAdjacentHTML('beforeend',
    '<section class="card m-sec" id="m-bench"><div class="m-head"><span class="m-n">\u2248</span><h2>How this matches the benchmark</h2></div><div class="m-text">' +
    '<p>The definitions follow the CARE to Compare paper (G\u00fcck, Roelofs and Faulstich, <i>Data</i> 2024, <a href="https://arxiv.org/abs/2404.10320" target="_blank" rel="noopener">arXiv:2404.10320</a>), and the committed code implements them:</p>' +
    '<ul class="m-list"><li>Coverage and Reliability are F-scores with \u03b2 = ' + BETA + ', which weights precision more than recall.</li>' +
    '<li>Earliness gives detections in the first half of the event window weight 1, then lets the weight fall linearly to 0.</li>' +
    '<li>Accuracy is measured on healthy events only, over readings with a normal status. Readings with an abnormal status are ignored.</li>' +
    '<li>The counter threshold is ' + THETA + ', about 12 hours of consecutive detections at 10-minute resolution.</li>' +
    '<li>The four scores are combined with weights ' + W.join(', ') + '. CARE is 0 when no event is detected, and equals Accuracy when Accuracy is below 0.5.</li></ul>' +
    '<p class="m-code">The code is in ' + link('src/windturbine/scoring/', 'src/windturbine/scoring/') + '.</p></div></section>');

  /* ---------- widget 1: threshold ---------- */
  (function () {
    const r = rng(11), n = 3000, val = [];
    for (let i = 0; i < n; i++) val.push(Math.exp(0.32 * gauss(r)));
    val.sort((a, b) => a - b);
    const H = 90, FLT = 50, r2 = rng(23), series = [];
    for (let i = 0; i < H + FLT; i++) series.push(Math.exp(0.32 * gauss(r2)) * (i < H ? 1 : 1 + (i - H) / FLT * 1.7));
    const ymax = Math.max.apply(null, series) * 1.08;
    const quant = (q) => { const pos = (n - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos); return val[lo] + (val[hi] - val[lo]) * (pos - lo); };
    const x = (i) => 40 + i * (470 / (H + FLT - 1)), y = (e) => 196 - e / ymax * 182;
    const slider = $('w1-q'); slider.value = Q;
    function draw() {
      const q = +slider.value, tau = quant(q);
      let flagH = 0, flagF = 0, dots = '';
      series.forEach((e, i) => {
        const on = e > tau; if (on) { if (i < H) flagH++; else flagF++; }
        dots += '<circle cx="' + x(i).toFixed(1) + '" cy="' + y(e).toFixed(1) + '" r="' + (on ? 3.4 : 2.2) + '" class="' + (on ? 'pt-on' : 'pt') + '"/>';
      });
      $('w1-svg').innerHTML = '<rect class="fault-zone" x="' + x(H - 0.5).toFixed(1) + '" y="10" width="' + (510 - x(H - 0.5)).toFixed(1) + '" height="186" rx="6"/>' +
        '<line class="axis" x1="40" y1="196" x2="510" y2="196"/>' +
        '<polyline class="ln sense" points="' + series.map((e, i) => x(i).toFixed(1) + ',' + y(e).toFixed(1)).join(' ') + '"/>' + dots +
        '<line class="tau-line" x1="40" y1="' + y(tau).toFixed(1) + '" x2="510" y2="' + y(tau).toFixed(1) + '"/><text x="44" y="' + (y(tau) - 6).toFixed(1) + '" class="note">τ = ' + tau.toFixed(2) + '</text>' +
        '<text x="40" y="218" class="note">healthy</text><text x="' + x(H + 2).toFixed(1) + '" y="218" class="note">fault develops</text>';
      $('w1-qv').textContent = q.toFixed(3);
      $('w1-out').innerHTML = 'Healthy readings flagged: <b>' + flagH + ' of ' + H + '</b> · fault readings flagged: <b>' + flagF + ' of ' + FLT + '</b>';
    }
    slider.addEventListener('input', draw); draw();
  })();

  /* ---------- widget 2: the counter ---------- */
  (function () {
    const N = 200, PRESETS = [
      ['Isolated blips', () => 0.05], ['Slow drift', (t) => 0.03 + 0.92 * t / (N - 1)], ['Sustained fault', (t) => (t < 70 ? 0.03 : 0.93)], ['Half the time', () => 0.6]];
    let pick = 2, roll = 1;
    $('w2-presets').innerHTML = PRESETS.map((p, i) => '<button type="button" data-i="' + i + '" aria-pressed="' + (i === pick) + '">' + p[0] + '</button>').join('');
    const slider = $('w2-t'); slider.value = THETA;
    function draw() {
      const r = rng(97 * (pick + 1) + 31 * roll), flags = [], c = [];
      for (let t = 0; t < N; t++) flags.push(r() < PRESETS[pick][1](t) ? 1 : 0);
      let cur = 0, peak = 0, first = -1;
      const theta = +slider.value;
      flags.forEach((f, t) => { cur = f ? cur + 1 : Math.max(cur - 1, 0); c.push(cur); if (cur > peak) peak = cur; if (first < 0 && cur >= theta) first = t; });
      const x = (t) => 20 + t * (480 / (N - 1)), ymax = Math.max(peak, theta) * 1.12 || 1, y = (v) => 236 - v / ymax * 170;
      let bars = '';
      flags.forEach((f, t) => { bars += f ? '<rect x="' + (x(t) - 0.9).toFixed(1) + '" y="8" width="1.8" height="28" class="fl-on"/>' : '<rect x="' + (x(t) - 0.9).toFixed(1) + '" y="30" width="1.8" height="6" class="fl-off"/>'; });
      $('w2-svg').innerHTML = '<text x="20" y="52" class="note">flags</text>' + bars + '<line class="axis" x1="20" y1="236" x2="500" y2="236"/>' +
        '<polyline class="ln ' + (first >= 0 ? 'expect' : 'sense') + '" points="' + c.map((v, t) => x(t).toFixed(1) + ',' + y(v).toFixed(1)).join(' ') + '"/>' +
        '<line class="tau-line" x1="20" y1="' + y(theta).toFixed(1) + '" x2="500" y2="' + y(theta).toFixed(1) + '"/><text x="24" y="' + (y(theta) - 6).toFixed(1) + '" class="note">θ = ' + theta + '</text>' +
        (first >= 0 ? '<line class="ref-line" x1="' + x(first).toFixed(1) + '" y1="60" x2="' + x(first).toFixed(1) + '" y2="236"/><text x="' + Math.min(x(first) + 6, 410).toFixed(1) + '" y="76" class="note alarm">detected</text>' : '') +
        '<text x="20" y="248" class="note">counter c</text>';
      $('w2-tv').textContent = theta;
      $('w2-out').innerHTML = first >= 0 ? 'Peak counter <b>' + peak + '</b> ≥ θ = ' + theta + ': <b>event detected</b> at reading ' + (first + 1) + '.'
        : 'Peak counter <b>' + peak + '</b> &lt; θ = ' + theta + ': <b>not detected</b>, however many flags there were.';
    }
    $('w2-presets').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; pick = +b.dataset.i; [].forEach.call($('w2-presets').children, (c) => c.setAttribute('aria-pressed', c === b)); draw(); });
    $('w2-roll').addEventListener('click', () => { roll++; draw(); });
    slider.addEventListener('input', draw); draw();
  })();

  /* ---------- widget 3: reliability ---------- */
  (function () {
    const exp = $('w3-exp'), seed = $('w3-seed'), tp = $('w3-tp'), fp = $('w3-fp'), b = $('w3-b');
    exp.innerHTML = allIds.map((id) => '<option value="' + esc(id) + '"' + (id === 'F1' ? ' selected' : '') + '>' + esc(dn(id)) + '</option>').join('');
    let run = null;
    function loadRun() {
      const e = X[exp.value];
      seed.innerHTML = e.seeds.map((s) => '<option value="' + s + '">' + s + '</option>').join('');
      pickSeed();
    }
    function pickSeed() {
      run = X[exp.value].per_seed.find((p) => p.seed === +seed.value) || X[exp.value].per_seed[0];
      tp.value = run.tp; fp.value = run.fp; b.value = BETA; draw();
    }
    function draw() {
      const TP = +tp.value, FP = +fp.value, FN = M.n_anomaly - TP, beta = +b.value, b2 = beta * beta;
      const R = fbeta(TP, FP, FN, beta), den = (1 + b2) * TP + b2 * FN + FP;
      $('w3-tpv').textContent = TP; $('w3-fpv').textContent = FP; $('w3-bv').textContent = beta.toFixed(2);
      const plusFp = FP < M.n_normal ? fbeta(TP, FP + 1, FN, beta) - R : null, oneMiss = TP > 0 ? fbeta(TP - 1, FP, FN + 1, beta) - R : null;
      const same = TP === run.tp && FP === run.fp && Math.abs(beta - BETA) < 1e-9;
      $('w3-out').innerHTML =
        '<p>FN = ' + M.n_anomaly + ' − TP = <b>' + FN + '</b></p>' +
        '<p class="calc">R = (1 + ' + bSq(beta) + ') · ' + TP + ' / ((1 + ' + bSq(beta) + ') · ' + TP + ' + ' + bSq(beta) + ' · ' + FN + ' + ' + FP + ') = ' + ((1 + b2) * TP).toFixed(2) + ' / ' + den.toFixed(2) + ' = <b>' + R.toFixed(3) + '</b></p>' +
        (same ? '<p class="ok">Matches the Reliability stored for this run: ' + f3(run.reliability) + '.</p>' : '<p class="w-note">The run’s stored Reliability is ' + f3(run.reliability) + ' (TP ' + run.tp + ', FP ' + run.fp + ', β = ' + BETA + ').</p>') +
        '<div class="mini"><div><span>One more false alarm</span><b>' + (plusFp === null ? 'n/a' : sgn(plusFp)) + '</b></div><div><span>One more missed fault</span><b>' + (oneMiss === null ? 'n/a' : sgn(oneMiss)) + '</b></div></div>' +
        '<p class="w-note">At the same counts: F<sub>0.5</sub> = ' + f3(fbeta(TP, FP, FN, 0.5)) + ', F<sub>1</sub> = ' + f3(fbeta(TP, FP, FN, 1)) + ', F<sub>2</sub> = ' + f3(fbeta(TP, FP, FN, 2)) + '.</p>';
    }
    exp.addEventListener('change', loadRun); seed.addEventListener('change', pickSeed);
    [tp, fp, b].forEach((s) => s.addEventListener('input', draw)); loadRun();
  })();

  /* ---------- widget 4: inside events ---------- */
  (function () {
    const PRE = 4, WIN = 24, HEALTHY = 28;
    const flags = { fault: new Array(PRE + WIN).fill(0), healthy: new Array(HEALTHY).fill(0) };
    const weight = (k) => { const u = k / (WIN - 1); return u <= 0.5 ? 1 : Math.max(0, 1 - (u - 0.5) / 0.5); };
    const PRESETS = [
      ['Early and sustained', () => { fill('fault', (i) => i >= PRE && i < PRE + 18); fill('healthy', () => 0); }],
      ['Late alarm', () => { fill('fault', (i) => i >= PRE + 15); fill('healthy', () => 0); }],
      ['Early blip, then silence', () => { fill('fault', (i) => i >= PRE && i < PRE + 4); fill('healthy', () => 0); }],
      ['Flags everything', () => { fill('fault', () => 1); fill('healthy', () => 1); }],
      ['Flags nothing', () => { fill('fault', () => 0); fill('healthy', () => 0); }]];
    function fill(lane, fn) { flags[lane] = flags[lane].map((_, i) => (fn(i) ? 1 : 0)); }
    $('w4-presets').innerHTML = PRESETS.map((p, i) => '<button type="button" data-i="' + i + '">' + p[0] + '</button>').join('');
    $('w4-wts').innerHTML = '<span class="spc" style="grid-column:span ' + PRE + '"></span>' + Array.from({ length: WIN }, (_, k) => '<span class="wb"><i style="height:' + (weight(k) * 100).toFixed(0) + '%"></i></span>').join('');
    document.querySelectorAll('#view-maths .cells, #view-maths .w4-wts').forEach((n) => n.style.setProperty('--cols', PRE + WIN));
    function cells(lane, n, isWin) {
      $('w4-' + lane).innerHTML = Array.from({ length: n }, (_, i) => '<button type="button" class="cell' + (isWin(i) ? ' win' : '') + '" data-lane="' + lane + '" data-i="' + i + '" aria-pressed="false" aria-label="' + lane + ' reading ' + (i + 1) + '"></button>').join('');
    }
    cells('fault', PRE + WIN, (i) => i >= PRE); cells('healthy', HEALTHY, () => false);
    function draw() {
      [].forEach.call(document.querySelectorAll('#view-maths .cell'), (c) => { const on = !!flags[c.dataset.lane][+c.dataset.i]; c.classList.toggle('on', on); c.setAttribute('aria-pressed', on); });
      let tp = 0, fp = 0, ws = 0, wsum = 0;
      flags.fault.forEach((f, i) => { if (i >= PRE) { const w = weight(i - PRE); wsum += w; if (f) { tp++; ws += w; } } else if (f) fp++; });
      const fn = WIN - tp, C = fbeta(tp, fp, fn, BETA), E = wsum > 0 ? ws / wsum : 0;
      const fpH = flags.healthy.reduce((a, b) => a + b, 0), tn = HEALTHY - fpH, A = tn + fpH > 0 ? tn / (tn + fpH) : NaN;
      $('w4-out').innerHTML =
        '<div class="trio"><div><span>Coverage</span><b>' + f3(C) + '</b><small>TP ' + tp + ' · FP ' + fp + ' · FN ' + fn + '</small></div>' +
        '<div><span>Earliness</span><b>' + f3(E) + '</b><small>weighted flags ' + ws.toFixed(1) + ' of ' + wsum.toFixed(1) + '</small></div>' +
        '<div><span>Accuracy</span><b>' + f3(A) + '</b><small>TN ' + tn + ' · FP ' + fpH + '</small></div></div>';
    }
    $('w4-presets').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) { PRESETS[+b.dataset.i][1](); draw(); } });
    // click or drag to paint; a keyboard activation (detail 0) toggles one reading
    let paint = null;
    const cellAt = (e) => { const n = document.elementFromPoint(e.clientX, e.clientY); return n && n.classList && n.classList.contains('cell') ? n : null; };
    const set = (c, v) => { flags[c.dataset.lane][+c.dataset.i] = v; draw(); };
    const area = $('view-maths');
    area.addEventListener('pointerdown', (e) => { const c = e.target.closest && e.target.closest('.cell'); if (!c) return; paint = { lane: c.dataset.lane, v: flags[c.dataset.lane][+c.dataset.i] ? 0 : 1 }; set(c, paint.v); e.preventDefault(); });
    area.addEventListener('pointermove', (e) => { if (!paint) return; const c = cellAt(e); if (c && c.dataset.lane === paint.lane) set(c, paint.v); });
    window.addEventListener('pointerup', () => { paint = null; });
    area.addEventListener('click', (e) => { const c = e.target.closest && e.target.closest('.cell'); if (c && e.detail === 0) set(c, flags[c.dataset.lane][+c.dataset.i] ? 0 : 1); });
    PRESETS[0][1](); draw();
  })();

  /* ---------- widget 5: combine ---------- */
  (function () {
    const exp = $('w5-exp'), keys = ['c', 'e', 'r', 'a'], names = { c: 'coverage', e: 'earliness', r: 'reliability', a: 'accuracy' };
    exp.innerHTML = allIds.map((id) => '<option value="' + esc(id) + '"' + (id === REF ? ' selected' : '') + '>' + esc(dn(id)) + '</option>').join('');
    const load = () => { keys.forEach((k) => { $('w5-' + k).value = X[exp.value].metrics[names[k]]; }); $('w5-none').checked = false; draw(); };
    function draw() {
      const v = {}; keys.forEach((k) => { v[k] = +$('w5-' + k).value; $('w5-' + k + 'v').textContent = f3(v[k]); });
      const none = $('w5-none').checked, parts = [v.c * W[0], v.e * W[1], v.r * W[2], v.a * W[3]].map((p) => p / WSUM), plain = parts.reduce((a, b) => a + b, 0);
      const care = none ? 0 : v.a < 0.5 ? v.a : plain, rule = none ? 'No event detected, so CARE = 0.' : v.a < 0.5 ? 'Accuracy is below 0.5, so CARE = Accuracy.' : null;
      const published = X[exp.value].metrics.care, same = keys.every((k) => Math.abs(v[k] - X[exp.value].metrics[names[k]]) < 5e-4) && !none;
      const cls = ['c', 'e', 'r', 'a'];
      $('w5-out').innerHTML =
        '<div class="stack" role="img" aria-label="Contribution of each component to CARE">' + parts.map((p, i) => '<i class="seg-' + cls[i] + '" style="width:' + (p * 100).toFixed(2) + '%" title="' + ['Coverage', 'Earliness', 'Reliability', 'Accuracy × ' + W[3]][i] + '"></i>').join('') + '</div>' +
        '<div class="stack-key"><span><i class="seg-c"></i>C</span><span><i class="seg-e"></i>E</span><span><i class="seg-r"></i>R</span><span><i class="seg-a"></i>' + W[3] + 'A</span></div>' +
        '<p class="calc">(' + f3(v.c) + ' + ' + f3(v.e) + ' + ' + f3(v.r) + ' + ' + W[3] + ' · ' + f3(v.a) + ') / ' + WSUM + ' = ' + f3(plain) + '</p>' +
        '<div class="big-out"><span>CARE</span><b>' + f3(care) + '</b></div>' + (rule ? '<p class="warn-line">' + rule + '</p>' : '') +
        (same ? '<p class="ok">Matches the published CARE for this experiment: ' + f3(published) + '.</p>' : '<p class="w-note">Published CARE for ' + esc(dn(exp.value)) + ': ' + f3(published) + '.</p>');
    }
    exp.addEventListener('change', load); keys.forEach((k) => $('w5-' + k).addEventListener('input', draw)); $('w5-none').addEventListener('change', draw); load();
  })();

  /* ---------- widget 6: bootstrap ---------- */
  (function () {
    const turbines = Array.from(new Set(D.events.map((e) => e.asset))).sort((a, b) => a - b);
    $('w6-grid').innerHTML = turbines.map((t) => '<div class="turb" data-t="' + t + '"><span>' + t + '</span><b></b></div>').join('');
    function draw(counts) {
      [].forEach.call($('w6-grid').children, (n) => { const c = counts ? counts[+n.dataset.t] : 1; n.dataset.c = c; n.querySelector('b').textContent = counts ? (c ? '×' + c : '') : ''; });
      if (!counts) { $('w6-out').textContent = 'Each square is one turbine. A resample draws ' + turbines.length + ' of them with replacement.'; return; }
      const once = turbines.filter((t) => counts[t] === 1).length, many = turbines.filter((t) => counts[t] > 1).length, none = turbines.filter((t) => !counts[t]).length;
      $('w6-out').innerHTML = 'Drawn once: <b>' + once + '</b> · more than once: <b>' + many + '</b> · not at all: <b>' + none + '</b>. Every event of a drawn turbine is included, as many times as the turbine was drawn.';
    }
    $('w6-draw').addEventListener('click', () => { const counts = {}; turbines.forEach((t) => { counts[t] = 0; }); for (let i = 0; i < turbines.length; i++) counts[turbines[Math.floor(Math.random() * turbines.length)]]++; draw(counts); });
    draw(null);
    const exp = $('w6-exp');
    exp.innerHTML = allIds.filter((id) => id !== REF).map((id) => '<option value="' + esc(id) + '"' + (id === 'F1' ? ' selected' : '') + '>' + esc(dn(id)) + '</option>').join('');
    function show() {
      const e = X[exp.value], d = X[REF], ci = e.ci, mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
      const sd = (v) => { const m = mean(v); return Math.sqrt(v.reduce((a, b) => a + (b - m) * (b - m), 0) / (v.length - 1)); };
      const vals = (x) => x.per_seed.map((p) => p.care);
      const line = (x) => x.seeds.length > 1 ? 'mean ' + f3(mean(vals(x))) + ', SD ' + f3(sd(vals(x))) + ' over seeds ' + x.seeds.join(', ') : 'seed ' + x.seeds[0] + ' only: ' + f3(vals(x)[0]);
      $('w6-ci').innerHTML = '<p>' + esc(dn(exp.value)) + ': ' + line(e) + '<br>' + esc(dn(REF)) + ': ' + line(d) + '</p>' +
        '<p class="calc">Δ = ' + f3(e.metrics.care) + ' − ' + f3(d.metrics.care) + ' = <b>' + sgn(ci.diff) + '</b></p>' +
        '<p>95% interval from ' + ci.draws + ' draws: <b>' + sgn(ci.lo) + ' to ' + sgn(ci.hi) + '</b>. ' + (ci.excludes_zero ? 'It excludes zero.' : 'It includes zero, so the difference cannot be told apart from no difference.') + '</p>' +
        (e.seeds.length === 1 ? '<p class="w-note">This experiment has one seed. The local-training side is averaged over ' + d.seeds.length + ' seeds before the difference is taken.</p>' : '');
    }
    exp.addEventListener('change', show); show();
  })();

  window.WT_MATHS = { ready: true };
})();
