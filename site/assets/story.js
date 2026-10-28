/* Animated, plain-language walkthrough of every experiment. Scenes are SVG drawn from window.WT_DATA; the motion is CSS.
   Exposes window.WT_STORY for the full-screen mode in app.js. */
(function () {
  'use strict';
  var D = window.WT_DATA, X = D.experiments, G = D.groups, M = D.meta, REF = M.reference;
  var reduce = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var MINUS = '−';

  function $(id) { return document.getElementById(id); }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
  function f3(v) { return v.toFixed(3); }
  function sgn(v) { return (v < 0 ? MINUS : '+') + Math.abs(v).toFixed(3); }
  function seeded(n) { var s = n; return function () { s = (s * 16807) % 2147483647; return s / 2147483647; }; }
  function txt(x, y, s, cls, anchor) {
    return '<text x="' + x + '" y="' + y + '"' + (cls ? ' class="' + cls + '"' : '') + (anchor ? ' text-anchor="' + anchor + '"' : '') + '>' + esc(s) + '</text>';
  }

  /* ---------- drawing parts ---------- */
  function turbine(x, y, s, cls) {
    var blade = '<path d="M0 0 L-2.2 -27 L2.2 -27 Z"/>';
    return '<g transform="translate(' + x + ',' + y + ') scale(' + s + ')" class="turb ' + (cls || '') + '">' +
      '<path class="tower" d="M-3.2 0 L3.2 0 L1.6 -46 L-1.6 -46 Z"/>' +
      '<g transform="translate(0,-46)"><g class="blades">' + blade + '<g transform="rotate(120)">' + blade + '</g><g transform="rotate(240)">' + blade + '</g></g>' +
      '<circle class="hub" r="3.4"/></g></g>';
  }
  function learner(x, y, s, kind, cls) {
    var dots = [[-13, -9], [-13, 9], [0, 0], [13, -9], [13, 9]], lines = '', i;
    for (i = 0; i < 2; i++) lines += '<path d="M-13 ' + (i ? 9 : -9) + ' L0 0 L13 -9 M0 0 L13 9"/>';
    var body = kind === 'split'
      ? '<rect class="part-shared" x="-27" y="-20" width="31" height="40" rx="10"/><rect class="part-own" x="5" y="-20" width="22" height="40" rx="10"/>'
      : '<rect class="lrn-box" x="-27" y="-20" width="54" height="40" rx="11"/>';
    return '<g transform="translate(' + x + ',' + y + ') scale(' + s + ')"><g class="lrn ' + (cls || '') + '">' + body +
      '<g class="lrn-lines">' + lines + '</g><g class="lrn-dots">' +
      dots.map(function (d) { return '<circle cx="' + d[0] + '" cy="' + d[1] + '" r="3.2"/>'; }).join('') + '</g></g></g>';
  }
  function ground() { return '<line class="ground" x1="20" y1="330" x2="620" y2="330"/>'; }
  function pkt(x, y, dx, dy, dir, delay, kind) {
    return '<g transform="translate(' + x + ',' + y + ')"><g class="pkt ' + dir + ' ' + (kind || '') + '" style="--dx:' + dx + 'px;--dy:' + dy + 'px;animation-delay:' + delay + 's">' +
      '<rect x="-6" y="-6" width="12" height="12" rx="3.5"/></g></g>';
  }
  function spark(x, y, w, seed, spikeAt) {
    var r = seeded(seed), pts = [], n = 14, i, py;
    for (i = 0; i < n; i++) {
      py = y - 6 - r() * 10;
      if (spikeAt !== undefined && i >= spikeAt) py -= (i - spikeAt + 1) * 4;
      pts.push((x + i * w / (n - 1)).toFixed(1) + ',' + py.toFixed(1));
    }
    return '<polyline class="spark' + (spikeAt !== undefined ? ' spiky' : '') + '" points="' + pts.join(' ') + '"/>';
  }
  var TX = [100, 240, 400, 540];

  /* ---------- scenes ---------- */
  function s1() {
    var xs = [70, 180, 290, 400, 510], h = ground();
    xs.forEach(function (x, i) {
      var alert = i === 2;
      h += turbine(x, 330, 1.7, (alert ? 'alert ' : '') + 'sp' + (i % 3));
      h += spark(x - 40, 362, 80, 11 + i, alert ? 9 : undefined);
    });
    h += '<g transform="translate(290,160)"><g class="pop"><circle r="15" class="warn-dot"/>' + txt(0, 7, '!', 'bang', 'middle') + '</g></g>';
    h += txt(320, 52, 'One wind farm, ' + M.n_turbines + ' turbines', 'head', 'middle');
    h += txt(320, 78, 'Each one streams sensor readings all day', 'note opt', 'middle');
    return h;
  }
  function s2() {
    var x0 = 210, x1 = 610, n = 40, r = seeded(7), exp = [], sen = [], i, x, e, dev, ys, thr = 34, band = [];
    for (i = 0; i <= n; i++) {
      x = x0 + (x1 - x0) * i / n; e = 215 - 24 * Math.sin((x - x0) / 46);
      dev = x > 450 ? -(x - 450) * 0.9 : 0;
      ys = e + (r() - .5) * 10 + dev;
      exp.push(x.toFixed(1) + ',' + e.toFixed(1)); sen.push(x.toFixed(1) + ',' + ys.toFixed(1));
      band.push([x, e]);
    }
    var up = band.map(function (b) { return b[0].toFixed(1) + ',' + (b[1] - thr).toFixed(1); });
    var dn = band.slice().reverse().map(function (b) { return b[0].toFixed(1) + ',' + (b[1] + thr).toFixed(1); });
    var ax = 490, ay = 215 - 24 * Math.sin((ax - x0) / 46) - 40 * 0.9 - 6;
    return turbine(88, 330, 1.5) + ground() +
      '<path class="link" d="M120 250 L196 230"/>' +
      '<polygon class="band" points="' + up.concat(dn).join(' ') + '"/>' +
      '<polyline class="ln expect draw" pathLength="1" points="' + exp.join(' ') + '"/>' +
      '<polyline class="ln sense draw late" pathLength="1" points="' + sen.join(' ') + '"/>' +
      '<g transform="translate(' + ax + ',' + (ay - 14).toFixed(0) + ')"><g class="pop late2"><path class="flag" d="M0 14 L0 -18 L26 -10 L0 -2"/></g></g>' +
      txt(ax + 32, (ay - 20).toFixed(0), 'Alarm', 'alarm') +
      '<g transform="translate(200,40)"><line class="ln sense" x1="0" y1="0" x2="26" y2="0"/>' + txt(34, 5, 'What the sensors say', 'note') + '</g>' +
      '<g transform="translate(200,64)"><line class="ln expect" x1="0" y1="0" x2="26" y2="0"/>' + txt(34, 5, 'What the model expects when healthy', 'note') + '</g>' +
      txt(88, 362, 'one turbine', 'note opt', 'middle');
  }
  function s3() {
    return '<line class="axis" x1="60" y1="140" x2="600" y2="140"/><line class="axis" x1="60" y1="310" x2="600" y2="310"/>' +
      txt(60, 52, 'A turbine with a fault', 'lane') + txt(60, 252, 'A healthy turbine', 'lane') +
      '<rect class="fault-zone" x="380" y="66" width="190" height="112" rx="10"/>' + txt(475, 88, 'fault period', 'note', 'middle') +
      '<g class="appear a1"><path class="flag good" d="M350 140 L350 100 L376 108 L350 116"/></g>' + txt(304, 168, 'early alarm', 'note', 'middle') +
      '<rect class="cover" x="396" y="150" width="130" height="12" rx="6"/>' + txt(461, 200, 'fault flagged', 'note', 'middle') +
      '<g class="appear a3"><path class="flag" d="M250 310 L250 272 L276 280 L250 288"/></g>' + txt(250, 340, 'false alarm', 'note', 'middle') +
      '<g class="quiet">' + [330, 380, 430, 480, 530].map(function (x) { return '<circle cx="' + x + '" cy="310" r="4"/>'; }).join('') + '</g>' +
      txt(430, 340, 'stays quiet', 'note', 'middle');
  }
  function s4() {
    var h = '', i, size = 18, gap = 6, x0 = 56, over = [3, 4, 11, 12, 19];
    h += txt(x0, 66, 'Rows the model learns from', 'lane') + txt(x0, 248, 'Rows we test it on (the fault events)', 'lane');
    for (i = 0; i < 24; i++) {
      var x = x0 + i * (size + gap), o = over.indexOf(i) >= 0;
      h += '<rect class="sq train' + (o ? ' dup' : '') + '" x="' + x + '" y="84" width="' + size + '" height="' + size + '" rx="4"/>';
      h += '<rect class="sq test' + (o ? ' dup' : '') + '" x="' + x + '" y="196" width="' + size + '" height="' + size + '" rx="4"/>';
      if (o) h += '<line class="dup-link" x1="' + (x + 9) + '" y1="104" x2="' + (x + 9) + '" y2="194"/>';
    }
    h += txt(320, 284, 'The glowing rows are on both sides', 'note warn', 'middle');
    h += '<g class="appear a4">' + txt(320, 334, 'Removed from training', 'head sm', 'middle') + '</g>';
    return h;
  }
  function s5() {
    var h = ground();
    TX.forEach(function (x, i) {
      h += '<ellipse class="bubble" cx="' + x + '" cy="236" rx="66" ry="122"/>' +
        turbine(x, 330, 1.4, 'sp' + (i % 3)) + learner(x, 186, 1, 'plain') +
        '<rect class="track" x="' + (x - 27) + '" y="212" width="54" height="7" rx="3.5"/>' +
        '<rect class="fill" style="animation-delay:' + (i * 0.6) + 's" x="' + (x - 27) + '" y="212" width="54" height="7" rx="3.5"/>';
    });
    return h + txt(320, 46, 'Every turbine learns on its own', 'head', 'middle') + txt(320, 72, 'its own readings stay inside its bubble', 'note opt', 'middle');
  }
  function s6() {
    var h = ground(), i, d;
    TX.forEach(function (x, i2) {
      h += '<path class="link" d="M' + x + ' 222 L320 156"/>' + turbine(x, 330, 1.4, 'sp' + (i2 % 3));
      for (i = 0; i < 3; i++) { d = i * 0.7 + i2 * 0.18; h += pkt(x, 222, 320 - x, -66, 'up2', d, 'data'); }
    });
    return h + learner(320, 108, 2.2, 'plain', 'merge slow') + txt(320, 178, 'one model for everyone', 'note opt', 'middle') +
      txt(320, 40, 'All the raw data is sent to one place', 'head', 'middle');
  }
  function cycle(split) {
    var h = ground(), off = split ? 8 : 0;
    TX.forEach(function (x, i) {
      h += '<path class="link" d="M' + x + ' 168 L320 114"/>' + turbine(x, 330, 1.35, 'sp' + (i % 3)) +
        '<circle class="ring" style="animation-delay:' + (i * 0.12) + 's" cx="' + x + '" cy="188" r="30"/>' +
        learner(x, 188, 0.95, split ? 'split' : 'plain') +
        pkt(x - off, 188, 320 - x + off, -108, 'up', i * 0.1, 'shared') +
        pkt(320, 80, x - 320, 108, 'down', i * 0.1, 'shared');
    });
    return h + learner(320, 80, 1.6, 'plain', 'merge') +
      txt(320, 134, split ? 'averages the shared part only' : 'the server averages the lessons', 'note opt', 'middle') +
      '<text id="rnd" class="note" x="30" y="40">Round 1</text>';
  }
  function s7() { return cycle(false); }
  function s8() {
    return cycle(true) +
      '<g transform="translate(150,360)"><rect class="part-shared" x="0" y="-12" width="16" height="16" rx="5"/>' + txt(24, 1, 'shared part', 'note') + '</g>' +
      '<g transform="translate(360,360)"><rect class="part-own" x="0" y="-12" width="16" height="16" rx="5"/>' + txt(24, 1, 'personal part', 'note') + '</g>';
  }
  var MU_P = [0, 0.15, 0.3, 0.5, 0.8, 0.97];
  function s9() {
    var h = ground();
    TX.forEach(function (x, i) {
      h += turbine(x, 330, 1.4, 'sp' + (i % 3)) +
        '<line class="tether" data-i="' + i + '" x1="' + x + '" y1="188" x2="320" y2="96"/>' +
        '<g class="mv" data-i="' + i + '" data-x="' + x + '"><g transform="translate(' + x + ',188)"><g class="wander" style="--amp:12px;animation-delay:-' + i * 0.9 + 's">' +
        learner(0, 0, 0.9, 'plain') + '</g></g></g>';
    });
    return h + learner(320, 92, 1.5, 'plain', 'merge slow') +
      txt(320, 40, 'A leash pulls each turbine toward the group', 'head sm', 'middle');
  }
  var SG_NOISE = [0, 0.45, 0.8, 1.2, 1.6];
  function s10() {
    var h = ground(), r = seeded(5), j;
    h += '<rect class="shield" x="30" y="142" width="580" height="38" rx="12"/>' + txt(320, 166, 'adds random noise', 'note', 'middle');
    TX.forEach(function (x, i) {
      var fuzz = '';
      for (j = 0; j < 7; j++) {
        fuzz += '<circle class="fuzz" r="3.6" style="--jx:' + ((r() - .5) * 46).toFixed(0) + 'px;--jy:' + ((r() - .5) * 46).toFixed(0) + 'px"/>';
      }
      h += '<path class="link" d="M' + x + ' 190 L320 112"/>' + turbine(x, 330, 1.3, 'sp' + (i % 3)) + learner(x, 212, 0.8, 'plain') +
        '<g transform="translate(' + x + ',212)"><g class="pkt up2 fz" style="--dx:' + (320 - x) + 'px;--dy:-106px;animation-delay:' + (i * 0.25) + 's">' + fuzz + '</g></g>' +
        pkt(x, 212, 320 - x, -106, 'up2', i * 0.25, 'shared');
    });
    return h + '<g class="srv-priv">' + learner(320, 84, 1.5, 'plain') + '</g>' + txt(320, 134, 'the shared model', 'note opt', 'middle') +
      txt(320, 40, 'Noise hides each turbine\u2019s private data', 'head sm', 'middle');
  }
  function s11() {
    var ids = [REF, 'central_sequential', 'F1', 'F2', bestMu(), 'F4_sigma1'], lo = 0.5, hi = 0.66, w = 330, x0 = 250, h = '', refX;
    var names = { central_sequential: 'Centralised', F1: 'FedAvg', F2: 'FedPer', F4_sigma1: 'With privacy noise' };
    names[REF] = 'Local training'; names[bestMu()] = 'FedProx, best μ';
    refX = x0 + (X[REF].metrics.care - lo) / (hi - lo) * w;
    ids.forEach(function (id, i) {
      var v = X[id].metrics.care, y = 64 + i * 42, bw = (v - lo) / (hi - lo) * w;
      h += txt(x0 - 12, y + 16, names[id], 'lane', 'end') +
        '<rect class="bar-bg" x="' + x0 + '" y="' + y + '" width="' + w + '" height="24" rx="6"/>' +
        '<rect class="bar-fg ' + (id === REF ? 'ref' : id === 'F4_sigma1' ? 'priv' : '') + '" style="animation-delay:' + (i * 0.25) + 's" x="' + x0 + '" y="' + y + '" width="' + bw.toFixed(1) + '" height="24" rx="6"/>' +
        txt(x0 + bw + 8, y + 17, f3(v), 'bar-val');
    });
    return h + '<line class="ref-line" x1="' + refX.toFixed(1) + '" y1="52" x2="' + refX.toFixed(1) + '" y2="318"/>' +
      txt(refX.toFixed(1), 340, 'local training', 'note', 'middle') + txt(x0, 366, 'Bars start at 0.50, not zero', 'note');
  }
  function bestMu() { return G.mu.slice().sort(function (a, b) { return X[b].metrics.care - X[a].metrics.care; })[0]; }

  /* ---------- narration ---------- */
  var refCare = X[REF].metrics.care;
  var maxFed = Math.max.apply(null, [].concat(G.mu, G.priv, ['F1', 'F2']).map(function (id) { return X[id].metrics.care; }));
  var noBeat = maxFed <= refCare;

  function result(id, extra) {
    var e = X[id], ci = e.ci;
    var delta = id === REF ? 'The bar every other approach has to beat.'
      : sgn(ci.diff) + ' vs local training · ' + (ci.excludes_zero ? 'differs from local' : 'not distinguishable from local');
    return '<div class="res"><div class="res-top"><span class="res-k">CARE</span><b class="res-n" data-to="' + e.metrics.care + '">' + f3(e.metrics.care) + '</b></div>' +
      '<div class="res-line">' + e.detected + ' of ' + M.n_anomaly + ' faults found · ' + e.false_alarms + ' of ' + M.n_normal + ' false alarms</div>' +
      '<div class="res-line muted">' + esc(delta) + '</div>' + (extra ? '<div class="res-line muted">' + extra + '</div>' : '') + '</div>';
  }
  function plain(t) { return '<div class="res"><div class="res-line">' + esc(t) + '</div></div>'; }
  function key(items) {
    return '<ul class="key">' + items.map(function (i) { return '<li><i class="' + i[0] + '"></i><span><b>' + esc(i[1]) + '</b> ' + esc(i[2]) + '</span></li>'; }).join('') + '</ul>';
  }

  var SCENES = [
    { kicker: 'The goal', title: 'Spotting a failing turbine early', svg: s1,
      text: ['A wind farm records readings from every turbine all day: temperatures, speeds, power.',
        'A model that has learned what “normal” looks like can raise an alarm when a turbine starts to drift, before it breaks.'],
      res: function () { return plain(M.n_turbines + ' turbines · ' + M.n_events + ' events to judge · ' + M.n_anomaly + ' with a real fault'); } },
    { kicker: 'The method', title: 'How a model notices trouble', svg: s2,
      text: ['The model learns to predict what the readings should look like when everything is healthy.',
        'When the real readings drift far away from that prediction, it raises an alarm.'],
      res: function () { return plain('The bigger the gap, the louder the alarm.'); } },
    { kicker: 'The scoring', title: 'How we keep score', svg: s3,
      text: ['Four things matter. They are combined into one number called CARE, and higher is better.'],
      extra: key([['k-a', 'Coverage', 'how much of the fault period was flagged.'], ['k-b', 'Earliness', 'how early the alarm came.'],
        ['k-c', 'Reliability', 'how trustworthy the alarms are, counting both misses and false alarms.'], ['k-d', 'Accuracy', 'how quiet the model stays on healthy turbines.']]),
      res: function () { return plain('One score, four ideas.'); } },
    { kicker: 'A fair test', title: 'Removing rows the model has already seen', svg: s4,
      text: ['In an earlier pooled experiment, about a quarter of the test rows (25.4%) had also been seen during training. That makes results look better than they really are.',
        'So every turbine here drops those rows from its training data first.'],
      res: function () { return plain('25.4% of test rows were also in training (40,216 of 158,528)'); } },
    { kicker: 'Approach 1 · Local training', title: 'Each turbine learns alone', svg: s5,
      text: ['Every turbine trains its own model on its own readings. Nothing is shared.', 'This is the bar that every other approach has to beat.'],
      res: function () { return result(REF); } },
    { kicker: 'Approach 2 · Centralised', title: 'Everything in one place', svg: s6,
      text: ['All the raw data is collected in one place and a single model learns from all of it.',
        'It can work well, but every turbine has to hand over its data.'],
      res: function () { return result('central_sequential', 'In the paper this comparator trained one turbine at a time and drifted, so treat it as a rough reference.'); } },
    { kicker: 'Approach 3 · FedAvg', title: 'Share the lessons, not the data', svg: s7, after: roundCounter,
      text: ['Each turbine trains briefly on its own data and sends what it learned to a server. The server averages the lessons and sends the result back. Then everyone repeats.',
        'Only what the models learned travels. The readings never leave the turbine.'],
      res: function () { return result('F1'); } },
    { kicker: 'Approach 4 · FedPer', title: 'Share some, keep some', svg: s8, after: roundCounter,
      text: ['Same idea, but each model has two parts. The shared part is averaged across turbines. A personal part stays on each turbine so it can fit its own quirks.'],
      res: function () { return result('F2'); } },
    { kicker: 'Approach 5 · FedProx', title: 'A leash toward the group', svg: s9, ctl: 'mu',
      text: ['FedProx adds a leash that pulls each turbine’s model back toward the group while it trains.', 'Drag the slider to change how tight the leash is.'],
      res: function () { return result(G.mu[story.mu]); } },
    { kicker: 'Approach 6 · Privacy', title: 'Hiding what each turbine sends', svg: s10, ctl: 'sg',
      text: ['Before sharing, each turbine blurs what it sends with random noise, so nobody can work out its private data.', 'Drag the slider to add more noise.'],
      res: function () { var id = G.priv[story.sg], e = X[id]; return result(id, e.epsilon !== null ? 'Privacy budget ε = ' + e.epsilon.toFixed(2) + ' (smaller means stronger privacy)' : 'Secure aggregation only, no noise'); } },
    { kicker: 'The result', title: 'Putting it together', svg: s11,
      text: noBeat
        ? ['No federated approach scored higher than the turbines training alone.', 'Federated learning came close without any turbine handing over raw data. Heavy privacy noise and a very tight leash cost the most.']
        : ['The best federated result was ' + f3(maxFed) + ' against ' + f3(refCare) + ' for local training.'],
      res: function () { return plain((M.n_events === 58 ? 'One seed, ' : '') + M.n_events + ' events, ' + M.n_turbines + ' turbines. Differences this small are not reliable on their own.'); } }
  ];

  /* ---------- player ---------- */
  var story = { i: 0, mu: 3, sg: 2, timers: [], raf: 0, auto: false, autoT: 0, token: 0, shown: 0 };
  var stage = $('stage');

  function clearTimers() { story.timers.forEach(clearInterval); story.timers = []; if (story.raf) cancelAnimationFrame(story.raf); story.raf = 0; }

  function roundCounter() {
    var n = 1, node = $('rnd');
    if (!node || reduce) return;
    story.timers.push(setInterval(function () { n += 1; node.textContent = 'Round ' + n; }, 6000));
  }

  /* The result card counts up from 0 when a scene opens and glides from the previous value when a slider moves. */
  function setResult(html, fresh) {
    var prev = fresh ? 0 : story.shown, node, to;
    if (story.raf) cancelAnimationFrame(story.raf);
    $('st-result').innerHTML = html;
    node = document.querySelector('#st-result .res-n');
    if (!node) return;
    to = parseFloat(node.dataset.to);
    if (reduce) { node.textContent = f3(to); story.shown = to; return; }
    node.textContent = f3(prev);
    var t0 = performance.now(), dur = fresh ? 900 : 380;
    function step(now) {
      var p = Math.min(1, (now - t0) / dur), v = prev + (to - prev) * (1 - Math.pow(1 - p, 3));
      node.textContent = f3(v); story.shown = v;
      if (p < 1) story.raf = requestAnimationFrame(step); else story.shown = to;
    }
    story.raf = requestAnimationFrame(step);
  }

  function muCaption(k) {
    var id = G.mu[k], mu = parseFloat(X[id].label.replace(/^[^=]*= /, ''));
    if (mu === 0) return 'No leash at all. Each turbine’s model is free to wander, and the group simply averages them.';
    if (mu <= 0.01) return 'A very loose leash. The models still wander and CARE is about the same as with no leash.';
    if (mu <= 0.1) return 'A gentle leash: the models stay near the group but keep some freedom. This scored best.';
    return 'A tight leash. Every turbine is forced to copy the group, so local detail is lost and the score drops.';
  }
  function sgCaption(k) {
    if (X[G.priv[k]].epsilon === null) return 'No noise. Turbines share clean updates, protected only by secure aggregation.';
    return 'More noise hides each turbine better, but the shared model gets blurrier and barely learns.';
  }

  function applyControl(kind, fresh) {
    var k, cap = $('st-control').querySelector('.cap');
    if (!cap) return;
    if (kind === 'mu') {
      k = story.mu;
      [].forEach.call(stage.querySelectorAll('.mv'), function (g) {
        var x = +g.dataset.x, p = MU_P[k];
        g.style.transform = 'translate(' + ((320 - x) * p).toFixed(1) + 'px,' + ((96 - 188) * p).toFixed(1) + 'px)';
        g.querySelector('.wander').style.setProperty('--amp', (12 * (1 - p)).toFixed(1) + 'px');
      });
      [].forEach.call(stage.querySelectorAll('.tether'), function (l) { l.style.strokeWidth = (1 + MU_P[k] * 6).toFixed(1); l.style.opacity = (0.35 + MU_P[k] * 0.6).toFixed(2); });
      cap.textContent = muCaption(k);
    } else {
      k = story.sg;
      stage.style.setProperty('--noise', SG_NOISE[k]);
      stage.style.setProperty('--blur', (SG_NOISE[k] * 2.4).toFixed(1) + 'px');
      cap.textContent = sgCaption(k);
    }
    setResult(SCENES[story.i].res(), fresh);
  }

  function buildControl(kind) {
    var node = $('st-control'), ids = kind === 'mu' ? G.mu : G.priv, cur = kind === 'mu' ? story.mu : story.sg;
    node.hidden = false;
    node.innerHTML = '<label for="st-slider">' + (kind === 'mu' ? 'Leash strength (μ)' : 'Noise level (σ)') + '</label>' +
      '<input type="range" id="st-slider" min="0" max="' + (ids.length - 1) + '" step="1" value="' + cur + '">' +
      '<div class="ticks">' + ids.map(function (id) { return '<span>' + esc(X[id].label.replace(/^[^=]*= /, '')) + '</span>'; }).join('') + '</div>' +
      '<p class="cap"></p>';
    $('st-slider').addEventListener('input', function (e) {
      if (kind === 'mu') story.mu = +e.target.value; else story.sg = +e.target.value;
      applyControl(kind, false);
    });
  }

  /* Caption, controls and result for scene i. The card keeps one height across scenes (see lockHeight). */
  function fillNarr(i) {
    var sc = SCENES[i], node = $('st-control');
    $('st-kicker').textContent = sc.kicker;
    $('st-title').textContent = sc.title;
    $('st-text').innerHTML = sc.text.map(function (t) { return '<p>' + esc(t) + '</p>'; }).join('') + (sc.extra || '');
    if (sc.ctl) buildControl(sc.ctl); else { node.hidden = true; node.innerHTML = ''; }
    $('st-result').innerHTML = sc.res();
  }

  function lockHeight() {
    var narr = document.querySelector('.narr'), max = 0, i;
    narr.style.minHeight = '';
    if (!narr.offsetWidth) return false;
    for (i = 0; i < SCENES.length; i++) { fillNarr(i); max = Math.max(max, narr.offsetHeight); }
    narr.style.minHeight = max + 'px';
    return true;
  }

  function settle() {
    var sc = SCENES[story.i];
    fillNarr(story.i);
    if (sc.ctl) applyControl(sc.ctl, false); else setResult(sc.res(), false);
  }

  function relayout() { if (lockHeight()) settle(); }

  function show(i) {
    var sc, my, n0;
    i = Math.max(0, Math.min(SCENES.length - 1, i));
    story.i = i; sc = SCENES[i];
    clearTimers();
    my = ++story.token;
    stage.classList.add('swap');
    fillNarr(i);
    n0 = document.querySelector('#st-result .res-n');
    if (n0 && !reduce) n0.textContent = f3(0);
    stage.style.removeProperty('--noise'); stage.style.removeProperty('--blur');
    [].forEach.call($('st-dots').children, function (b, k) { b.setAttribute('aria-current', k === i); });
    $('st-count').textContent = 'Step ' + (i + 1) + ' of ' + SCENES.length;
    $('st-prev').disabled = i === 0; $('st-next').disabled = i === SCENES.length - 1;
    setTimeout(function () {
      if (my !== story.token) return;   // a newer step has been requested; do not draw this one over it
      stage.innerHTML = sc.svg();
      stage.setAttribute('aria-label', sc.title);
      stage.classList.remove('swap');
      if (sc.ctl) applyControl(sc.ctl, true); else setResult(sc.res(), true);
      if (sc.after) sc.after();
    }, reduce ? 0 : 160);
  }

  function replay() { show(story.i); }
  function setAuto(on) {
    story.auto = on; clearTimeout(story.autoT);
    $('st-auto').setAttribute('aria-pressed', on);
    $('st-auto').textContent = on ? 'Pause' : 'Auto-play';
    if (on) tick();
  }
  function tick() {
    clearTimeout(story.autoT);
    story.autoT = setTimeout(function () {
      if (story.i >= SCENES.length - 1) { setAuto(false); return; }
      show(story.i + 1); tick();
    }, SCENES[story.i].ctl ? 14000 : 11000);
  }

  $('st-dots').innerHTML = SCENES.map(function (s, k) { return '<button type="button" aria-label="Step ' + (k + 1) + ': ' + esc(s.title) + '" data-i="' + k + '"></button>'; }).join('');
  $('st-dots').addEventListener('click', function (e) { var b = e.target.closest('button'); if (b) { show(+b.dataset.i); if (story.auto) tick(); } });
  $('st-prev').addEventListener('click', function () { show(story.i - 1); if (story.auto) tick(); });
  $('st-next').addEventListener('click', function () { show(story.i + 1); if (story.auto) tick(); });
  $('st-replay').addEventListener('click', replay);
  $('st-auto').addEventListener('click', function () { setAuto(!story.auto); });
  if (reduce) $('st-auto').hidden = true;

  /* Arrow keys step the story whenever it is on screen, without having to click it first. */
  document.addEventListener('keydown', function (e) {
    var t = e.target && e.target.tagName, r, view = $('view-present');
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    if (t === 'INPUT' || t === 'SELECT' || t === 'TEXTAREA') return;
    if (document.documentElement.classList.contains('presenting') || !view || view.hidden) return;
    r = $('story').getBoundingClientRect();
    if (r.bottom < window.innerHeight * 0.15 || r.top > window.innerHeight * 0.85) return;
    e.preventDefault();
    show(story.i + (e.key === 'ArrowRight' ? 1 : -1));
    if (story.auto) tick();
  });

  var resizeT = 0;
  window.addEventListener('resize', function () { clearTimeout(resizeT); resizeT = setTimeout(relayout, 160); });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(relayout);

  window.WT_STORY = { count: SCENES.length, show: show, replay: replay, relayout: relayout };
  lockHeight();
  show(0);
})();
