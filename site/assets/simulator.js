/* The Test a turbine tab: a fixed, made-up stand-in for the detection steps, applied to one reading typed in by hand.
   No trained model runs here (the weights are not in the repository). The input channels, their units and the rules it
   mimics (status gate, q-quantile threshold, counter θ) come from the pipeline; the tolerances are illustrative. */
(function () {
  'use strict';
  const K = window.WT_DATA.meta.care, Q = K.score_quantile, THETA = K.criticality_threshold;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const MINUS = '−';
  const num = (v) => (v < 0 ? MINUS : '') + Math.abs(v);
  const pct = (x) => num(+(x * 100).toFixed(1)) + '%';
  const pts = (d) => { const v = Math.round(d * 100); return (v > 0 ? '+' : '') + num(v) + ' pts'; };

  // Pipeline facts, checked by tests/test_site_data.py.
  const N_INPUTS = 243, N_ANGLES = 5;   // the model input schema in results/paper/*/manifest.json
  const NORMAL_STATUS = [0, 2];         // normal_status_ids in src/windturbine/config.py
  // Illustrative choices. The power curve is the theoretical one in notebooks/01_eda.ipynb, and the usual reactive power
  // is the mean and spread of reactive_power_122_avg over the 368,614 normal-status readings that notebook summarises.
  const CURVE = { cutIn: 3, rated: 13, cutOut: 25 };
  const REACTIVE = { usual: 0.025, spread: 0.036 };
  const POWER_SPREAD = 0.08;            // made up: 8 points of rated power
  // If both scaled gaps follow a standard bell curve on healthy readings, the score exceeds TAU on a share 1 − q of them:
  // the same q-quantile rule as the real per-turbine thresholds.
  const TAU = Math.sqrt(-2 * Math.log(1 - Q));

  const STATUS = [[0, 'Normal operation'], [1, 'Derated or curtailed'], [2, 'Idling'], [3, 'Service mode'],
    [4, 'Fault or downtime'], [5, 'Other (test, ice and similar)']];   // status_type_id labels from notebooks/01_eda.ipynb
  const statusName = (id) => STATUS.find((s) => s[0] === id)[1];
  const FIELDS = [
    { id: 'wind', label: 'Wind speed', unit: 'm/s', col: 'wind_speed_236_avg', min: 0, max: 40, help: 'Average over the 10 minutes.' },
    { id: 'power', label: 'Power output', unit: '% of rated', col: 'power_5_avg', min: -20, max: 120,
      help: 'Share of the turbine’s rated power, for example 35. The dataset stores it as 0 to 1.' },
    { id: 'reactive', label: 'Reactive power', unit: '% of rated', col: 'reactive_power_122_avg', min: -60, max: 60,
      help: 'Electrical power that does no useful work. It usually stays within a few percent of rated.' }
  ];
  // Made-up readings. `expect` is what the rule gives; the self-check at the bottom keeps the labels honest.
  const PRESETS = [
    { group: 'normal', name: 'Light breeze', status: 0, wind: 5, power: 1, reactive: 1, expect: 'normal' },
    { group: 'normal', name: 'Steady wind', status: 0, wind: 10, power: 36, reactive: 3, expect: 'normal' },
    { group: 'normal', name: 'Full power', status: 0, wind: 16, power: 99, reactive: 4, expect: 'normal' },
    { group: 'normal', name: 'Calm, idling', status: 2, wind: 2, power: 0, reactive: 0, expect: 'normal' },
    { group: 'odd', name: 'Low output for the wind', status: 0, wind: 11, power: 20, reactive: 2, expect: 'flagged' },
    { group: 'odd', name: 'Stopped in good wind', status: 0, wind: 10.5, power: 0, reactive: 0, expect: 'flagged' },
    { group: 'odd', name: 'High reactive power', status: 0, wind: 10, power: 35, reactive: 18, expect: 'flagged' },
    { group: 'odd', name: 'Low output during service', status: 3, wind: 11, power: 5, reactive: 1, expect: 'skipped' }
  ];

  /* ---------- the rule ---------- */
  const expectedPower = (v) => (v < CURVE.cutIn || v > CURVE.cutOut ? 0 : v >= CURVE.rated ? 1
    : Math.pow((v - CURVE.cutIn) / (CURVE.rated - CURVE.cutIn), 3));
  // r: status id, wind in m/s, power and reactive as fractions of rated (as the dataset stores them)
  function simulate(r) {
    const expected = expectedPower(r.wind);
    const gp = (r.power - expected) / POWER_SPREAD, gq = (r.reactive - REACTIVE.usual) / REACTIVE.spread;
    const score = Math.hypot(gp, gq);
    return { expected, gp, gq, score, kind: NORMAL_STATUS.indexOf(r.status) < 0 ? 'skipped' : score > TAU ? 'flagged' : 'normal' };
  }

  /* ---------- form ---------- */
  const group = (g, label) => '<div class="t-group"><span class="t-glabel">' + label + '</span><div class="seg" role="group" aria-label="' + label + ' examples">' +
    PRESETS.map((p, i) => (p.group === g ? '<button type="button" data-i="' + i + '" aria-pressed="false">' + esc(p.name) + '</button>' : '')).join('') + '</div></div>';
  const field = (f) => '<div class="field t-field"><label for="t-' + f.id + '">' + f.label + ' <span class="t-unit">(' + f.unit + ')</span></label>' +
    '<input type="number" id="t-' + f.id + '" name="' + f.id + '" min="' + f.min + '" max="' + f.max + '" step="any" required aria-describedby="t-' + f.id + '-help t-' + f.id + '-err">' +
    '<span class="t-help" id="t-' + f.id + '-help">' + f.help + ' Column <span class="mono">' + f.col + '</span>.</span><span class="t-err" id="t-' + f.id + '-err"></span></div>';
  $('t-form').innerHTML =
    '<div class="t-ex"><p class="w-title">Start from an example</p>' + group('normal', 'Normal') + group('odd', 'Unusual') + '</div>' +
    '<p class="w-title">Or enter one reading</p>' +
    '<div class="field t-field"><label for="t-status">Operating status</label><select id="t-status" name="status" aria-describedby="t-status-help">' +
    STATUS.map((s) => '<option value="' + s[0] + '">' + s[1] + (NORMAL_STATUS.indexOf(s[0]) < 0 ? ' (not scored)' : '') + '</option>').join('') + '</select>' +
    '<span class="t-help" id="t-status-help">What the turbine’s controller reports. Column <span class="mono">status_type_id</span>.</span></div>' +
    FIELDS.map(field).join('') +
    '<div class="w-row"><button type="submit" class="t-go">Test turbine</button><button type="reset" class="ghost">Reset</button></div>';

  function setError(f, msg) {
    $('t-' + f.id).setAttribute('aria-invalid', msg ? 'true' : 'false');
    $('t-' + f.id + '-err').textContent = msg;
  }
  // Validates the form; returns the reading in dataset units, or null after flagging and focusing the first bad field.
  function read() {
    const r = { status: +$('t-status').value };
    let first = null;
    FIELDS.forEach((f) => {
      const input = $('t-' + f.id), v = input.valueAsNumber;
      const range = f.unit === 'm/s' ? num(f.min) + ' to ' + num(f.max) + ' m/s' : num(f.min) + '% to ' + num(f.max) + '% of rated';
      const msg = input.validity.badInput ? 'Enter a number, for example 8.5.'
        : input.value.trim() === '' ? 'Enter the ' + f.label.toLowerCase() + ' in ' + f.unit + '.'
          : !(v >= f.min && v <= f.max) ? f.label + ' must be from ' + range + '.' : '';
      setError(f, msg);
      if (msg && !first) first = input;
      r[f.id] = f.unit === 'm/s' ? v : v / 100;
    });
    if (first) { first.focus(); return null; }
    return r;
  }

  /* ---------- result ---------- */
  const DISCLAIMER = '<p class="small">Simulated with a fixed, made-up rule, not the trained models. It has no measured accuracy and cannot tell whether or when a fault will happen.</p>';
  let shown = false;
  const placeholder = (msg) => { shown = false; $('t-result').classList.remove('stale'); $('t-result').innerHTML = '<h3>Result</h3><p class="small">' + msg + '</p>' + DISCLAIMER; };
  const meter = (name, g, val) => '<div class="meter"><span>' + name + '</span><div class="bar-t"><div class="bar-f' + (Math.abs(g) > TAU ? ' over' : '') +
    '" style="width:' + (Math.min(Math.abs(g) / TAU, 1) * 100).toFixed(1) + '%"></div></div><b>' + val + '</b></div>';

  function render(r, title) {
    const s = simulate(r), w = +r.wind.toFixed(1) + ' m/s', hours = +(THETA * 10 / 60).toFixed(1);
    const scored = STATUS.filter((x) => NORMAL_STATUS.indexOf(x[0]) >= 0).map((x) => x[1].toLowerCase()).join(' or ');
    const powerLine = r.power <= 0.02 && s.expected >= 0.05
      ? 'the turbine is making almost no power (' + pct(r.power) + ') although at ' + w + ' a healthy turbine in this simulation makes about ' + pct(s.expected) + '.'
      : 'power output is ' + pct(r.power) + ', well ' + (r.power < s.expected ? 'below' : 'above') + ' the ' + pct(s.expected) + ' a healthy turbine in this simulation makes at ' + w + '.';
    const reactLine = 'reactive power is ' + pct(r.reactive) + ', far from its usual level of about ' + pct(REACTIVE.usual) + '.';
    const powerMain = Math.abs(s.gp) >= Math.abs(s.gq);
    let tag, why, next = '';
    if (s.kind === 'skipped') {
      tag = '<span class="tag ref">Not scored</span>';
      why = 'The turbine reports “' + statusName(r.status) + '”. The pipeline scores only readings with status ' + scored +
        ', so this one is skipped and the alarm counter stays where it was. Switch the status to Normal operation to see how the values themselves would score.';
    } else if (s.kind === 'flagged') {
      tag = '<span class="tag bad">Unusual reading</span>';
      why = 'This reading stands out because ' + (powerMain ? powerLine : reactLine) +
        (Math.min(Math.abs(s.gp), Math.abs(s.gq)) >= 2 ? (powerMain ? ' Reactive power is also unusual at ' + pct(r.reactive) + '.' : ' Power output is also off: ' + pct(r.power) + ' against about ' + pct(s.expected) + ' expected.') : '');
      next = 'One unusual reading is not an alarm. The real pipeline adds one to a counter for every flagged reading and takes one off for every normal one, and raises an alarm only when the counter reaches ' +
        THETA + ': about ' + hours + ' hours of unusual readings in a row, at one reading every 10 minutes. Even then, an alarm means the readings look unusual, not that a fault will follow.';
    } else {
      tag = '<span class="tag good">Looks normal</span>';
      why = 'Nothing here stands out. Power output is ' + pct(r.power) + ' where a healthy turbine in this simulation makes about ' + pct(s.expected) + ' at ' + w +
        ', and reactive power is ' + pct(r.reactive) + ' against a usual ' + pct(REACTIVE.usual) + '. Together the two gaps stay below the threshold.';
      next = 'In the real pipeline a normal reading takes one off the alarm counter (it never drops below zero), moving the turbine further from an alarm.';
    }
    $('t-result').classList.remove('stale');
    $('t-result').innerHTML = '<h3>' + esc(title) + '</h3>' +
      (s.kind === 'skipped' ? '' : '<div class="row2"><div class="big">' + s.score.toFixed(2) + '</div><span class="delta">simulated score<br>threshold ' + TAU.toFixed(2) + '</span></div>') +
      '<div>' + tag + '</div><p>' + why + '</p>' +
      (s.kind === 'skipped' ? '' : '<p class="small">How far each value is from usual, in percentage points of rated power. A full bar is enough to flag the reading on its own.</p>' +
        '<div class="meters">' + meter('Power', s.gp, pts(r.power - s.expected)) + meter('Reactive', s.gq, pts(r.reactive - REACTIVE.usual)) + '</div>') +
      (next ? '<p class="small">' + next + '</p>' : '') + DISCLAIMER;
    shown = true;
    // On a phone the result sits below the form, so bring it into view under the sticky header.
    const top = $('t-result').getBoundingClientRect().top;
    if (top > window.innerHeight - 80 || top < 0) {
      const head = document.querySelector('.site-head').offsetHeight;
      window.scrollTo({ top: window.scrollY + top - head - 12, behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
    }
  }

  function press(i) { [].forEach.call(document.querySelectorAll('#t-form .seg button'), (b) => b.setAttribute('aria-pressed', +b.dataset.i === i)); }
  function run(title) { const r = read(); if (r) render(r, title); else placeholder('Fix the highlighted values, then press <b>Test turbine</b>.'); }

  $('t-form').addEventListener('click', (e) => {
    const b = e.target.closest('.seg button'); if (!b) return;
    const p = PRESETS[+b.dataset.i];
    $('t-status').value = p.status;
    FIELDS.forEach((f) => { $('t-' + f.id).value = p[f.id]; });
    press(+b.dataset.i);
    run('Example: ' + p.name);
  });
  $('t-form').addEventListener('submit', (e) => { e.preventDefault(); press(-1); run('Your reading'); });
  $('t-form').addEventListener('input', (e) => {
    press(-1);
    const f = FIELDS.find((x) => 't-' + x.id === e.target.id);
    if (f && e.target.getAttribute('aria-invalid') === 'true') setError(f, '');
    if (shown && !$('t-result').classList.contains('stale')) {
      $('t-result').classList.add('stale');
      $('t-result').insertAdjacentHTML('afterbegin', '<p class="warn-line t-stale">The readings have changed. Press Test turbine to update this result.</p>');
    }
  });
  $('t-form').addEventListener('reset', () => { FIELDS.forEach((f) => setError(f, '')); press(-1); placeholder('Pick an example, or enter a reading and press <b>Test turbine</b>.'); });
  placeholder('Pick an example, or enter a reading and press <b>Test turbine</b>.');

  /* ---------- how it compares with the real pipeline ---------- */
  const share = +((1 - Q) * 100).toFixed(2);
  $('t-notes').innerHTML = [
    ['What this page does', 'It compares power output with the repository’s theoretical power curve (cut-in ' + CURVE.cutIn + ' m/s, rated ' + CURVE.rated + ' m/s, cut-out ' +
      CURVE.cutOut + ' m/s), and reactive power with its usual level of ' + pct(REACTIVE.usual) + ' ± ' + pct(REACTIVE.spread) + ' (the mean and spread of 368,614 normal-status readings summarised in notebooks/01_eda.ipynb). ' +
      'The power gap is divided by ' + Math.round(POWER_SPREAD * 100) + ' points, a made-up tolerance, and the reactive gap by ' + +(REACTIVE.spread * 100).toFixed(1) + ' points. The score is the length of the two scaled gaps together. ' +
      'The threshold, ' + TAU.toFixed(2) + ', is the score that only ' + share + '% of healthy readings would exceed under the simulation’s own assumption that healthy gaps follow a bell curve: the same ' + Q + '-quantile rule the real thresholds use.'],
    ['What the real models read', N_INPUTS + ' numbers per reading: ' + (N_INPUTS - N_ANGLES) + ' ten-minute averages, with each of the ' + N_ANGLES + ' angle channels turned into a sine and a cosine. ' +
      'Wind Farm C names only its wind-speed, power and reactive-power channels; the rest are anonymised as sensor_0, sensor_1 and so on, which is why the form asks for three values. ' +
      'Each turbine standardises every channel with the mean and spread of its own training readings, and a missing value becomes that mean.'],
    ['How the real models score it', 'An autoencoder trained only on that turbine’s healthy readings tries to rebuild all ' + N_INPUTS + ' numbers, and the size of the miss is the score. ' +
      'Each turbine’s threshold is the ' + Q + ' quantile of its scores on held-out healthy readings, and a counter turns flagged readings into alarms. <a href="#maths">CARE maths</a> shows both steps.'],
    ['Why there is no real result here', 'The trained weights are not stored in the repository and the CARE to Compare data are not redistributed, so this page cannot run the models, and the simulation has no accuracy figure. ' +
      'The scores in <a href="#present">Present</a> measure how well the real models flagged past, labelled events. A flagged reading means the readings look unusual, not that a fault will happen.']
  ].map((n) => '<div class="card note"><b>' + n[0] + '</b><span>' + n[1] + '</span></div>').join('');

  // Self-check: every example must still give the outcome its group promises.
  PRESETS.forEach((p) => console.assert(simulate({ status: p.status, wind: p.wind, power: p.power / 100, reactive: p.reactive / 100 }).kind === p.expect,
    'Test a turbine: example "' + p.name + '" no longer gives ' + p.expect));
})();
