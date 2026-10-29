/* The Test a turbine tab: replays a month of real Wind Farm C readings (turbine 35) and runs the CARE criticality
   counter on them, sounding an alarm if it reaches the threshold. Readings come from site/sim_data.js, built by
   site/build_sim_data.py. No trained model runs here: each reading's flag comes from a transparent stand-in score
   (see that script), and the counter follows src/windturbine/scoring/care.py. */
(function () {
  'use strict';
  const SIM = window.WT_SIM;
  if (!SIM) return;
  const $ = (id) => document.getElementById(id);
  const NORMAL_STATUS = [0, 2];         // normal_status_ids in src/windturbine/config.py, checked by tests/test_site_data.py
  const STATUS = ['Normal operation', 'Derated', 'Idling', 'Service mode', 'Down', 'Other'];
  const SPEEDS = [['Slow', 3], ['Normal', 12], ['Fast', 48]];
  const st = { snap: SIM.snapshots[0], i0: 0, i: 0, c: 0, peak: 0, alarmAt: -1, playing: false, raf: 0, speed: 12, theta: SIM.theta_default, muted: false };

  const fmtTime = (s, k) => { const d = new Date(new Date(s.start.replace(' ', 'T') + 'Z').getTime() + s.t[k] * 600000); return d.toISOString().slice(0, 16).replace('T', ' '); };
  const dateOf = (s, k) => fmtTime(s, k).slice(0, 10);

  /* ---------- controls ---------- */
  $('sim-snap').innerHTML = SIM.snapshots.map((s, k) => '<option value="' + k + '">' + s.label + '</option>').join('');
  $('sim-speed').innerHTML = SPEEDS.map((s) => '<button type="button" data-v="' + s[1] + '" aria-pressed="' + (s[1] === st.speed) + '">' + s[0] + '</button>').join('');
  $('sim-theta').value = st.theta;

  function pickSnap() {
    st.snap = SIM.snapshots[+$('sim-snap').value];
    const d = $('sim-date');
    d.min = dateOf(st.snap, 0); d.max = dateOf(st.snap, st.snap.t.length - 1); d.value = d.min;
    $('sim-note').textContent = st.snap.note + ' Event ' + st.snap.event + ', turbine ' + SIM.turbine + '.';
    reset();
  }
  function startIndex() {
    const want = $('sim-date').value, s = st.snap;
    for (let k = 0; k < s.t.length; k++) if (dateOf(s, k) >= want) return k;
    return 0;
  }
  function reset() {
    stop(); st.i0 = st.i = startIndex(); st.c = 0; st.peak = 0; st.alarmAt = -1;
    $('sim-dash').classList.remove('alarm', 'done'); draw(); panel();
  }

  /* ---------- the counter: same rule as care.py ---------- */
  function step() {
    const s = st.snap, k = st.i;
    if (NORMAL_STATUS.indexOf(s.status[k]) >= 0) st.c = s.flag[k] ? st.c + 1 : Math.max(st.c - 1, 0);
    st.peak = Math.max(st.peak, st.c);
    if (st.alarmAt < 0 && st.c >= st.theta) { st.alarmAt = k; ring(); }
    st.i++;
  }
  function frame() {
    for (let n = 0; n < st.speed && st.i < st.snap.t.length; n++) step();
    draw(); panel();
    if (st.i >= st.snap.t.length) { stop(); $('sim-dash').classList.add('done'); panel(); return; }
    st.raf = requestAnimationFrame(frame);
  }
  function play() { if (st.i >= st.snap.t.length) reset(); st.playing = true; $('sim-play').textContent = 'Pause'; st.raf = requestAnimationFrame(frame); }
  function stop() { st.playing = false; cancelAnimationFrame(st.raf); $('sim-play').textContent = st.i > st.i0 && st.i < st.snap.t.length ? 'Resume' : 'Play'; }

  /* ---------- alarm sound (starts only after the user pressed Play) ---------- */
  let ctx = null;
  function ring() {
    $('sim-dash').classList.add('alarm');
    if (st.muted) return;
    try {
      ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
      for (let b = 0; b < 3; b++) {
        const o = ctx.createOscillator(), g = ctx.createGain(), t = ctx.currentTime + b * 0.35;
        o.type = 'square'; o.frequency.value = 880; g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(0.15, t + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.25);
        o.connect(g).connect(ctx.destination); o.start(t); o.stop(t + 0.3);
      }
    } catch (e) { /* audio not available: the visual alarm still shows */ }
  }

  /* ---------- status panel ---------- */
  function panel() {
    const s = st.snap, k = Math.max(st.i - 1, st.i0), done = st.i >= s.t.length, alarm = st.alarmAt >= 0;
    const scored = s.status.slice(st.i0, st.i).filter((v) => NORMAL_STATUS.indexOf(v) >= 0).length;
    const flags = s.flag.slice(st.i0, st.i).filter((v, j) => v && NORMAL_STATUS.indexOf(s.status[st.i0 + j]) >= 0).length;
    $('sim-state').textContent = alarm ? 'ALARM' : done ? 'No alarm' : st.i > st.i0 ? 'Monitoring' : 'Ready';
    $('sim-when').textContent = st.i > st.i0 ? fmtTime(s, k) : 'Press Play';
    $('sim-c').textContent = st.c; $('sim-th').textContent = st.theta;
    $('sim-bar').style.width = Math.min(100, st.c / st.theta * 100) + '%';
    $('sim-reading').innerHTML = st.i > st.i0
      ? 'Status: <b>' + STATUS[s.status[k]] + '</b> · wind ' + s.wind[k] + ' m/s · power ' + s.power[k] + ' · ' + SIM.channels.sensor.replace('_avg', '') + ' ' + s.sensor[k] +
        (NORMAL_STATUS.indexOf(s.status[k]) < 0 ? ' · <i>not scored</i>' : s.flag[k] ? ' · <b class="bad">flagged</b>' : ' · normal')
      : '';
    $('sim-sum').textContent = st.i > st.i0
      ? scored + ' readings scored, ' + flags + ' flagged, highest counter ' + st.peak + '.' +
        (alarm ? ' Alarm at ' + fmtTime(s, st.alarmAt) + ', ' + Math.round((st.alarmAt - st.i0) / 6) + ' hours after monitoring started.'
          : done ? ' The counter never reached ' + st.theta + ', so no alarm was raised.' : '')
      : '';
  }

  /* ---------- charts ---------- */
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  function canvas(id) {
    const cv = $(id), r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    cv.width = Math.max(1, r.width * dpr); cv.height = Math.max(1, r.height * dpr);
    const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, r.width, r.height);
    return { g, w: r.width, h: r.height };
  }
  function draw() {
    const s = st.snap, n = s.t.length, L = 44, R = 10;
    const col = { fg: css('--fg'), mut: css('--muted'), bd: css('--border'), ac: css('--accent'), bad: css('--bad'), soft: css('--accent-soft'), warm: css('--warm') };
    const X = (k, w) => L + (k / (n - 1)) * (w - L - R);
    // sensor with its usual band
    let { g, w, h } = canvas('sim-cv1');
    const vals = s.sensor.concat(s.lo, s.hi).filter((v) => v !== null), lo = Math.min(...vals), hi = Math.max(...vals);
    const Y = (v) => 10 + (1 - (v - lo) / (hi - lo || 1)) * (h - 30);
    axes(g, w, h, L, lo, hi, Y, col, SIM.channels.sensor.replace('_avg', '') + ' and its usual range');
    g.fillStyle = col.soft; g.globalAlpha = 0.9;
    for (let k = 0; k < n; k++) if (s.lo[k] !== null) g.fillRect(X(k, w), Y(s.hi[k]), Math.max(1, X(1, w) - X(0, w) + .5), Y(s.lo[k]) - Y(s.hi[k]));
    g.globalAlpha = 1;
    for (let k = st.i0; k < st.i; k++) {
      if (s.sensor[k] === null) continue;
      const sc = NORMAL_STATUS.indexOf(s.status[k]) >= 0;
      g.fillStyle = !sc ? col.bd : s.flag[k] ? col.bad : col.fg;
      g.fillRect(X(k, w) - 1, Y(s.sensor[k]) - 1, 2, 2);
    }
    marks(g, w, h, X, col);
    // counter
    ({ g, w, h } = canvas('sim-cv2'));
    let c = 0, top = Math.max(st.theta * 1.3, 10);
    for (let k = st.i0; k < st.i; k++) { if (NORMAL_STATUS.indexOf(s.status[k]) >= 0) c = s.flag[k] ? c + 1 : Math.max(c - 1, 0); top = Math.max(top, c * 1.1); }
    const Yc = (v) => 10 + (1 - v / top) * (h - 30);
    axes(g, w, h, L, 0, top, Yc, col, 'Criticality counter');
    g.strokeStyle = col.bad; g.setLineDash([6, 5]); g.beginPath(); g.moveTo(L, Yc(st.theta)); g.lineTo(w - R, Yc(st.theta)); g.stroke(); g.setLineDash([]);
    g.fillStyle = col.bad; g.font = '600 11px Inter, system-ui, sans-serif'; g.fillText('threshold ' + st.theta, L + 6, Yc(st.theta) - 5);
    g.strokeStyle = col.ac; g.lineWidth = 2; g.beginPath(); c = 0;
    for (let k = st.i0; k < st.i; k++) {
      if (NORMAL_STATUS.indexOf(s.status[k]) >= 0) c = s.flag[k] ? c + 1 : Math.max(c - 1, 0);
      k === st.i0 ? g.moveTo(X(k, w), Yc(c)) : g.lineTo(X(k, w), Yc(c));
    }
    g.stroke(); g.lineWidth = 1;
    marks(g, w, h, X, col);
  }
  function axes(g, w, h, L, lo, hi, Y, col, title) {
    g.strokeStyle = col.bd; g.fillStyle = col.mut; g.font = '11px Inter, system-ui, sans-serif';
    for (let j = 0; j <= 3; j++) {
      const v = lo + (hi - lo) * j / 3, y = Y(v);
      g.beginPath(); g.moveTo(L, y); g.lineTo(w - 10, y); g.stroke(); g.fillText(v.toFixed(hi - lo < 5 ? 1 : 0), 4, y + 4);
    }
    const s = st.snap;
    [0, Math.floor(s.t.length / 2), s.t.length - 1].forEach((k, j) => {
      const x = L + (k / (s.t.length - 1)) * (w - L - 10);
      g.textAlign = j === 0 ? 'left' : j === 2 ? 'right' : 'center'; g.fillText(dateOf(s, k), x, h - 4);
    });
    g.textAlign = 'left'; g.fillStyle = col.fg; g.font = '600 12px Inter, system-ui, sans-serif'; g.fillText(title, L + 6, 22);
  }
  function marks(g, w, h, X, col) {
    const s = st.snap;
    const p = s.split.indexOf('p');
    if (p > 0) { g.strokeStyle = col.mut; g.setLineDash([2, 4]); g.beginPath(); g.moveTo(X(p, w), 8); g.lineTo(X(p, w), h - 18); g.stroke(); g.setLineDash([]); }
    if (st.i0 > 0) { g.fillStyle = col.bd; g.globalAlpha = .35; g.fillRect(X(0, w), 8, X(st.i0, w) - X(0, w), h - 26); g.globalAlpha = 1; }
    if (st.alarmAt >= 0) { g.strokeStyle = col.bad; g.lineWidth = 2; g.beginPath(); g.moveTo(X(st.alarmAt, w), 8); g.lineTo(X(st.alarmAt, w), h - 18); g.stroke(); g.lineWidth = 1; }
  }

  /* ---------- wiring ---------- */
  $('sim-snap').addEventListener('change', pickSnap);
  $('sim-date').addEventListener('change', reset);
  $('sim-play').addEventListener('click', () => (st.playing ? stop() : play()));
  $('sim-reset').addEventListener('click', reset);
  $('sim-mute').addEventListener('click', () => { st.muted = !st.muted; $('sim-mute').setAttribute('aria-pressed', st.muted); $('sim-mute').textContent = st.muted ? 'Sound off' : 'Sound on'; });
  $('sim-speed').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; st.speed = +b.dataset.v; [].forEach.call($('sim-speed').children, (c) => c.setAttribute('aria-pressed', c === b)); });
  $('sim-theta').addEventListener('input', (e) => { st.theta = +e.target.value; $('sim-thv').textContent = st.theta + ' readings (' + (st.theta / 6).toFixed(st.theta % 6 ? 1 : 0) + ' h)'; reset(); });
  window.addEventListener('resize', () => draw());
  $('sim-thv').textContent = st.theta + ' readings (' + st.theta / 6 + ' h)';
  window.WT_SIMULATOR = { redraw: draw };
  pickSnap();
})();
