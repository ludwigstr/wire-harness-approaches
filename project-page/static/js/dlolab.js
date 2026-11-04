/* DLO-Lab page: fit progress, per-joint parameter explorer, gauges, parameter budget. Means only, never medians. */
(function () {
  'use strict';
  var F = window.DLOLAB_FIT, A = window.APPAGE;
  if (!F || !A) return;
  var el = A.el, text = A.text, esc = A.esc, hover = A.hover, svgIn = A.svgIn, MUTED = A.MUTED;
  var PHYS = A.COLORS.physics;
  var CABLES = ['Cable 1', 'Cable 2', 'Cable 3', 'Cable 4 (thin bundle)'];
  var CABLE_VARS = ['var(--cable-1)', 'var(--cable-2)', 'var(--cable-3)', 'var(--cable-4)'];

  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  var SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  function sci(v) {
    if (v === 0) return '0';
    if (Math.abs(v) >= 1e4 || Math.abs(v) < 1e-2) {
      var parts = v.toExponential(2).split('e'), e = parseInt(parts[1], 10); // toExponential avoids 10.00·10⁸
      return parts[0] + '·' + '10' + String(e).split('').map(function (c) { return SUP[c]; }).join('');
    }
    return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
  }

  // ---------------------------------------------------------------- KPIs
  var models = A.MODELS.slice().sort(function (a, b) { return a.score - b.score; });
  var me = models.filter(function (m) { return m.key === 'dlolab'; })[0];
  var rank = models.indexOf(me) + 1, best = models[0];
  var gain = F.base_f - F.best_f;
  document.getElementById('fit-kpis').innerHTML = [
    { v: F.best_f.toFixed(2), l: 'best mm/node so far (gen ' + F.best_gen + ')' },
    { v: F.base_f.toFixed(2), l: 'start, generation 0' },
    { v: '−' + (gain / F.base_f * 100).toFixed(0) + ' %', l: '−' + gain.toFixed(2) + ' mm/node since the start' },
    { v: String(F.n_params), l: 'fitted parameters' },
    { v: String(F.last_gen), l: 'generations (population ' + F.run.popsize + ')' },
    { v: '#' + rank + ' of ' + models.length, l: 'best: ' + best.name + ' (' + (me.score - best.score).toFixed(1) + ' lower)' }
  ].map(function (k) { return '<div class="stat"><span class="stat-value">' + k.v + '</span><span class="stat-label">' + esc(k.l) + '</span></div>'; }).join('');

  // ---------------------------------------------------------------- fit progress
  var pBox = document.getElementById('fit-progress');
  document.getElementById('fit-progress-badge').textContent = '−' + gain.toFixed(2) + ' mm/node in ' + F.last_gen + ' generations';
  document.getElementById('fit-progress-note').textContent =
    'The main machine stopped after generation ' + F.last_gen + ' (shared RAM limit, 4 Nov 2026). Another machine continues from the best state with a fresh CMA-ES step size; ' +
    'the result is validated in fresh processes. Candidates that crash the solver are scored 1000 and never become the best.';
  function drawProgress() {
    var h = F.hist, W = Math.max(300, pBox.clientWidth), H = 260, L = 44, R = W - 14, T = 12, B = H - 34;
    var vals = h.map(function (x) { return x.best; }).concat(h.map(function (x) { return x.best_ever; }));
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    var step = A.niceStep(hi - lo, 5), y0 = Math.floor(lo / step) * step, y1 = A.niceTop(hi, step);
    var g0 = h[0].gen, g1 = h[h.length - 1].gen;
    var sx = function (g) { return L + ((g - g0) / (g1 - g0)) * (R - L); };
    var sy = function (v) { return B - ((v - y0) / (y1 - y0)) * (B - T); };
    var svg = svgIn(pBox, W, H);
    var grid = el('g', { 'class': 'grid' }, svg), ticks = el('g', { 'class': 'tick' }, svg);
    for (var t = y0; t <= y1 + 1e-9; t += step) { el('line', { x1: L, x2: R, y1: sy(t), y2: sy(t) }, grid); text(ticks, L - 8, sy(t) + 4, +t.toFixed(2), { 'text-anchor': 'end' }); }
    var gs = Math.max(1, Math.round(A.niceStep(g1 - g0, 8)));
    for (var g = g0; g <= g1; g += gs) text(ticks, sx(g), B + 16, g, { 'text-anchor': 'middle' });
    el('line', { 'class': 'baseline', x1: L, x2: R, y1: B, y2: B }, svg);
    // best-so-far as a step line
    var d = '';
    h.forEach(function (x, i) {
      d += (i ? ' H' + sx(x.gen) + ' V' : 'M' + sx(x.gen) + ',') + sy(x.best_ever);
    });
    el('path', { d: d, fill: 'none', style: 'stroke: var(--ink); stroke-width: 2; stroke-linejoin: round' }, svg);
    var cross = el('line', { x1: 0, x2: 0, y1: T, y2: B, style: 'stroke: var(--ink); stroke-width: 1; opacity: 0', 'pointer-events': 'none' }, svg);
    h.forEach(function (x) {
      el('circle', { cx: sx(x.gen), cy: sy(x.best), r: x.gen === F.best_gen ? 6 : 4.5,
        style: 'fill:' + PHYS + '; stroke:' + (x.gen === F.best_gen ? 'var(--ink)' : 'var(--surface)') + '; stroke-width: 2', 'pointer-events': 'none' }, svg);
    });
    var bg = h.filter(function (x) { return x.gen === F.best_gen; })[0];
    text(svg, sx(bg.gen), sy(bg.best) - 12, 'best ' + bg.best.toFixed(2), { 'text-anchor': 'middle' },
      'fill: var(--ink); font-size: 11px; font-weight: 600; paint-order: stroke; stroke: var(--surface); stroke-width: 4px');
    text(svg, R, H - 4, 'CMA-ES generation', { 'text-anchor': 'end' }, MUTED);
    var hit = el('rect', { x: L, y: T, width: R - L, height: B - T, fill: 'transparent' }, svg);
    hit.addEventListener('mousemove', function (e) {
      var r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * (W / r.width);
      var gi = Math.round(g0 + (px - L) / (R - L) * (g1 - g0));
      var x = h.filter(function (q) { return q.gen === gi; })[0];
      if (!x) return;
      cross.setAttribute('x1', sx(x.gen)); cross.setAttribute('x2', sx(x.gen)); cross.style.opacity = 0.25;
      A.showTip('<div class="tt-title">Generation ' + x.gen + '</div>' +
        '<div class="tt-row"><span>Best candidate</span><b>' + x.best.toFixed(3) + ' mm/node</b></div>' +
        '<div class="tt-row"><span>Best so far</span><b>' + x.best_ever.toFixed(3) + '</b></div>', e.clientX, e.clientY);
    });
    hit.addEventListener('mouseleave', function () { cross.style.opacity = 0; A.hideTip(); });
  }

  // ---------------------------------------------------------------- per-joint explorer
  var PJ_ORDER = ['hinge', 'stretch', 'twist', 'kappa1', 'kappa2'];
  var PJ_TEXT = {
    hinge: 'Multiplier on the bending stiffness of each hinge (1× = the shared value from K and E). Hinge 1 is next to the hub.',
    stretch: 'Multiplier on the stretch stiffness of each of the 15 edges between the 16 nodes. Edge 1 starts at the hub.',
    twist: 'Multiplier on the twist stiffness of each hinge. Hinge 1 is next to the hub.',
    kappa1: 'Rest curvature of each hinge, first component, in 1/m: the bend the cable keeps when nothing pushes it (its plastic set).',
    kappa2: 'Rest curvature of each hinge, second component, in 1/m.'
  };
  var pj = { key: 'hinge', view: 'heat' };
  var tabs = document.getElementById('pj-tabs');
  tabs.innerHTML = PJ_ORDER.map(function (k) {
    return '<button type="button" role="tab" class="chip-btn' + (k === pj.key ? ' is-active' : '') + '" data-pj="' + k + '">' + esc(F.perJoint[k].label) + '</button>';
  }).join('');
  tabs.addEventListener('click', function (e) {
    var b = e.target.closest('[data-pj]'); if (!b) return;
    pj.key = b.getAttribute('data-pj');
    tabs.querySelectorAll('[data-pj]').forEach(function (x) { x.classList.toggle('is-active', x === b); x.setAttribute('aria-selected', x === b ? 'true' : 'false'); });
    drawPJ();
  });
  document.querySelectorAll('.seg-btn[data-view]').forEach(function (b) {
    b.addEventListener('click', function () {
      pj.view = b.getAttribute('data-view');
      document.querySelectorAll('.seg-btn[data-view]').forEach(function (x) { x.classList.toggle('is-on', x === b); });
      drawPJ();
    });
  });
  var pjBox = document.getElementById('pj-chart'), pjLegend = document.getElementById('pj-legend');

  function hexToRgb(h) { h = h.replace('#', ''); if (h.length === 3) h = h.split('').map(function (c) { return c + c; }).join(''); var n = parseInt(h, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
  function mix(a, b, t) { return [0, 1, 2].map(function (i) { return Math.round(a[i] + (b[i] - a[i]) * t); }); }
  function divColor(t) { // t in [-1, 1]
    var neg = hexToRgb(css('--div-neg')), mid = hexToRgb(css('--div-mid')), pos = hexToRgb(css('--div-pos'));
    var c = t < 0 ? mix(mid, neg, Math.min(1, -t)) : mix(mid, pos, Math.min(1, t));
    return c;
  }
  function lum(c) { return (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255; }
  function tOf(P, v) { return P.kind === 'scale' ? Math.log(v) / Math.log(P.hi) : v / P.hi; }
  function fmtP(P, v) { return P.kind === 'scale' ? (v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toPrecision(2)) + '×' : (v > 0 ? '+' : '') + v.toFixed(3); }
  function axisName(P) { return P.axis === 'edge' ? 'edge' : 'hinge'; }

  function drawPJ() {
    var P = F.perJoint[pj.key], V = P.values, n = V[0].length;
    document.getElementById('pj-sub').textContent = PJ_TEXT[pj.key] + ' Bounds ' + (P.kind === 'scale' ? P.lo + '–' + P.hi + '×.' : '±' + P.hi + ' 1/m.');
    if (pj.view === 'heat') drawHeat(P, V, n); else drawProfile(P, V, n);
    drawPJCards(P, V);
  }

  function drawHeat(P, V, n) {
    // colour range: symmetric around 1x (scales, in log) or 0 (curvature), stretched to the data so small values stay visible
    var flat = [].concat.apply([], V);
    var ext = Math.max.apply(null, flat.map(function (v) { return Math.abs(P.kind === 'scale' ? Math.log(v) : v); })) || 1;
    var tOfData = function (v) { return (P.kind === 'scale' ? Math.log(v) : v) / ext; };
    var lo = P.kind === 'scale' ? fmtP(P, Math.exp(-ext)) + ' (softer)' : '−' + ext.toFixed(2);
    var hi = P.kind === 'scale' ? fmtP(P, Math.exp(ext)) + ' (stiffer)' : '+' + ext.toFixed(2) + ' 1/m';
    var mid = P.kind === 'scale' ? '1× (unchanged)' : '0 (straight)';
    pjLegend.innerHTML = '<span class="grad-legend"><span>' + lo + '</span><span class="grad-bar"></span><span>' + hi + '</span></span><span class="muted">middle = ' + mid + '</span>';
    var W = Math.max(300, pjBox.clientWidth), narrow = W < 560;
    var L = narrow ? 64 : 150, R = W - 6, T = 4, rowH = narrow ? 34 : 40;
    var H = T + 4 * rowH + 28, cw = (R - L) / n;
    var svg = svgIn(pjBox, W, H);
    var ticks = el('g', { 'class': 'tick' }, svg);
    V.forEach(function (row, ci) {
      text(svg, L - 10, T + ci * rowH + rowH / 2 + 4, narrow ? 'C' + (ci + 1) : CABLES[ci], { 'class': 'row-label', 'text-anchor': 'end' });
      row.forEach(function (v, j) {
        var c = divColor(tOfData(v)), x = L + j * cw + 1, y = T + ci * rowH + 1, w = cw - 2, hh = rowH - 2;
        var g = el('g', { tabindex: 0 }, svg);
        el('rect', { x: x, y: y, width: w, height: hh, rx: 3, fill: 'rgb(' + c.join(',') + ')' }, g);
        var label = P.kind === 'scale' ? (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2)) : v.toFixed(2);
        if (w >= (narrow ? 30 : 34)) text(g, x + w / 2, y + hh / 2 + 4, label, { 'text-anchor': 'middle' },
          'font-size: 11px; font-variant-numeric: tabular-nums; pointer-events: none; fill:' + (lum(c) > 0.55 ? '#0b0b0b' : '#ffffff'));
        hover(g, '<div class="tt-title">' + CABLES[ci] + ' · ' + axisName(P) + ' ' + (j + 1) + '</div>' +
          '<div class="tt-row"><span>' + esc(P.label) + '</span><b>' + fmtP(P, v) + (P.kind === 'signed' ? ' 1/m' : '') + '</b></div>' +
          '<div class="tt-row"><span>Bounds</span><b>' + (P.kind === 'scale' ? P.lo + '–' + P.hi + '×' : '±' + P.hi) + '</b></div>');
      });
    });
    var every = n > 10 && narrow ? 3 : 1;
    for (var j = 0; j < n; j += every) text(ticks, L + j * cw + cw / 2, T + 4 * rowH + 16, j + 1, { 'text-anchor': 'middle' });
    text(svg, R, H - 2, axisName(P) + ' along the cable, hub → plug', { 'text-anchor': 'end' }, MUTED);
  }

  function drawProfile(P, V, n) {
    pjLegend.innerHTML = CABLES.map(function (c, i) { return '<span><span class="line-swatch" style="--sw:' + CABLE_VARS[i] + '"></span>' + c + '</span>'; }).join('') +
      '<span><span class="tick-key"></span>' + (P.kind === 'scale' ? '1× (unchanged)' : 'zero curvature') + '</span>';
    var W = Math.max(300, pjBox.clientWidth), H = 300, L = 48, R = W - 12, T = 10, B = H - 34;
    var all = [].concat.apply([], V);
    var f = P.kind === 'scale' ? function (v) { return Math.log(v) / Math.LN10; } : function (v) { return v; };
    var lo = Math.min.apply(null, all.map(f)), hi = Math.max.apply(null, all.map(f));
    if (P.kind === 'scale') { lo = Math.min(lo, 0); hi = Math.max(hi, 0); } else { var m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m; }
    var pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    var sx = function (j) { return L + (j / (n - 1)) * (R - L); };
    var sy = function (u) { return B - ((u - lo) / (hi - lo)) * (B - T); };
    var svg = svgIn(pjBox, W, H);
    var grid = el('g', { 'class': 'grid' }, svg), ticks = el('g', { 'class': 'tick' }, svg);
    var yt = P.kind === 'scale'
      ? [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20].filter(function (v) { var u = f(v); return u >= lo && u <= hi; })
      : (function () { var s = A.niceStep(hi - lo, 6), out = []; for (var v = Math.ceil(lo / s) * s; v <= hi + 1e-9; v += s) out.push(+v.toFixed(4)); return out; })();
    yt.forEach(function (v) { el('line', { x1: L, x2: R, y1: sy(f(v)), y2: sy(f(v)) }, grid); text(ticks, L - 8, sy(f(v)) + 4, P.kind === 'scale' ? v + '×' : v, { 'text-anchor': 'end' }); });
    for (var j = 0; j < n; j++) if (n <= 15 || j % 2 === 0) text(ticks, sx(j), B + 16, j + 1, { 'text-anchor': 'middle' });
    el('line', { x1: L, x2: R, y1: sy(f(P.kind === 'scale' ? 1 : 0)), y2: sy(f(P.kind === 'scale' ? 1 : 0)), style: 'stroke: var(--ink); stroke-width: 1.5; opacity: 0.5' }, svg);
    el('line', { 'class': 'baseline', x1: L, x2: R, y1: B, y2: B }, svg);
    V.forEach(function (row, ci) {
      var d = row.map(function (v, j) { return (j ? 'L' : 'M') + sx(j) + ',' + sy(f(v)); }).join(' ');
      el('path', { d: d, fill: 'none', style: 'stroke:' + CABLE_VARS[ci] + '; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round' }, svg);
      row.forEach(function (v, j) { el('circle', { cx: sx(j), cy: sy(f(v)), r: 3.5, style: 'fill:' + CABLE_VARS[ci] + '; stroke: var(--surface); stroke-width: 1.5', 'pointer-events': 'none' }, svg); });
    });
    var cross = el('line', { x1: 0, x2: 0, y1: T, y2: B, style: 'stroke: var(--ink); stroke-width: 1; opacity: 0', 'pointer-events': 'none' }, svg);
    var hit = el('rect', { x: L - 10, y: T, width: R - L + 20, height: B - T, fill: 'transparent' }, svg);
    hit.addEventListener('mousemove', function (e) {
      var r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * (W / r.width);
      var j = Math.max(0, Math.min(n - 1, Math.round((px - L) / (R - L) * (n - 1))));
      cross.setAttribute('x1', sx(j)); cross.setAttribute('x2', sx(j)); cross.style.opacity = 0.25;
      A.showTip('<div class="tt-title">' + axisName(P) + ' ' + (j + 1) + ' · ' + esc(P.label) + '</div>' + V.map(function (row, ci) {
        return '<div class="tt-row"><span><span class="dot" style="--sw:' + CABLE_VARS[ci] + '"></span>' + CABLES[ci] + '</span><b>' + fmtP(P, row[j]) + '</b></div>';
      }).join(''), e.clientX, e.clientY);
    });
    hit.addEventListener('mouseleave', function () { cross.style.opacity = 0; A.hideTip(); });
    text(svg, R, H - 4, axisName(P) + ' along the cable, hub → plug' + (P.kind === 'scale' ? ' · log scale' : ''), { 'text-anchor': 'end' }, MUTED);
  }

  function drawPJCards(P, V) {
    document.getElementById('pj-cards').innerHTML = V.map(function (row, ci) {
      var mn = Math.min.apply(null, row), mx = Math.max.apply(null, row);
      var avg = P.kind === 'scale'
        ? Math.exp(row.reduce(function (s, v) { return s + Math.log(v); }, 0) / row.length)
        : row.reduce(function (s, v) { return s + v; }, 0) / row.length;
      return '<div class="pj-card"><b><span class="dot" style="--sw:' + CABLE_VARS[ci] + '"></span>' + CABLES[ci] + '</b>' +
        '<span>' + (P.kind === 'scale' ? 'geometric mean' : 'mean') + ' <b>' + fmtP(P, avg) + '</b></span>' +
        '<span>lowest <b>' + fmtP(P, mn) + '</b> (' + axisName(P) + ' ' + (row.indexOf(mn) + 1) + ')</span>' +
        '<span>highest <b>' + fmtP(P, mx) + '</b> (' + axisName(P) + ' ' + (row.indexOf(mx) + 1) + ')</span></div>';
    }).join('');
  }

  // ---------------------------------------------------------------- gauges
  function gauge(box, rows) {
    box.innerHTML = rows.map(function (g, i) {
      return '<div class="gauge" tabindex="0" data-i="' + i + '">' +
        '<div class="gauge-head"><span class="gauge-name">' + g.dot + esc(g.label) + '</span><b class="gauge-val">' + g.valText + '</b></div>' +
        '<div class="gauge-track"><span class="gauge-warm" style="left:' + (g.pw * 100).toFixed(2) + '%"></span>' +
        '<span class="gauge-dot" style="left:' + (g.pv * 100).toFixed(2) + '%; background:' + g.color + '"></span></div>' +
        '<div class="gauge-ends"><span>' + g.loText + '</span><span>' + g.hiText + '</span></div></div>';
    }).join('');
    box.querySelectorAll('.gauge').forEach(function (node) {
      var g = rows[+node.getAttribute('data-i')];
      hover(node, '<div class="tt-title">' + esc(g.label) + '</div>' +
        '<div class="tt-row"><span>Fitted</span><b>' + g.valText + '</b></div>' +
        '<div class="tt-row"><span>Warm start</span><b>' + g.warmText + '</b></div>' +
        '<div class="tt-row"><span>Bounds</span><b>' + g.loText + ' – ' + g.hiText + '</b></div>' +
        '<div class="tt-row"><span>Change</span><b>' + g.change + '</b></div>');
    });
  }
  function pos(scale, v, lo, hi) {
    if (scale === 'log') return (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
    if (scale === 'sqrt') return Math.sqrt(Math.max(0, v)) / Math.sqrt(hi);
    return (v - lo) / (hi - lo);
  }
  function change(v, w) { return w ? '×' + (v / w).toFixed(2) + ' vs warm start' : 'from 0'; }
  gauge(document.getElementById('fit-globals'), F.globals.map(function (g) {
    return { label: g.label, dot: '', color: PHYS, valText: sci(g.value), warmText: sci(g.warm), loText: sci(g.lo), hiText: sci(g.hi),
      pv: pos(g.scale, g.value, g.lo, g.hi), pw: pos(g.scale, g.warm, g.lo, g.hi), change: change(g.value, g.warm) };
  }));
  var M = F.mass;
  gauge(document.getElementById('fit-mass'), M.values.map(function (v, i) {
    return { label: CABLES[i], dot: '<span class="dot" style="--sw:' + CABLE_VARS[i] + '"></span>', color: CABLE_VARS[i],
      valText: v.toFixed(4) + ' ' + M.unit, warmText: M.warm + ' ' + M.unit, loText: M.lo + ' ' + M.unit, hiText: M.hi + ' ' + M.unit,
      pv: pos('log', v, M.lo, M.hi), pw: pos('log', M.warm, M.lo, M.hi), change: change(v, M.warm) };
  }));

  // ---------------------------------------------------------------- simple horizontal bars
  function hbars(box, rows, opts) {
    var W = Math.max(260, box.clientWidth), L = opts.labelW || 170, R = W - 56, rowH = 34, barH = 18, T = 4;
    var H = T + rows.length * rowH + (opts.axis ? 26 : 6);
    var vmax = Math.max.apply(null, rows.map(function (r) { return r.v; }));
    var top = opts.axis ? A.niceTop(vmax, A.niceStep(vmax, 4)) : vmax;
    var sx = function (v) { return L + (v / top) * (R - L); };
    var svg = svgIn(box, W, H);
    if (opts.axis) {
      var grid = el('g', { 'class': 'grid' }, svg), ticks = el('g', { 'class': 'tick' }, svg), st = A.niceStep(vmax, 4);
      for (var t = 0; t <= top + 1e-9; t += st) { el('line', { x1: sx(t), x2: sx(t), y1: T, y2: T + rows.length * rowH }, grid); text(ticks, sx(t), T + rows.length * rowH + 16, t, { 'text-anchor': 'middle' }); }
    }
    el('line', { 'class': 'baseline', x1: L, x2: L, y1: T, y2: T + rows.length * rowH }, svg);
    rows.forEach(function (r, i) {
      var y = T + i * rowH + (rowH - barH) / 2;
      var g = el('g', { 'class': 'bar-row', tabindex: 0 }, svg);
      el('rect', { x: 0, y: T + i * rowH, width: W, height: rowH, fill: 'transparent' }, g);
      text(g, L - 10, y + barH / 2 + 4, r.label, { 'class': 'row-label', 'text-anchor': 'end' });
      el('path', { 'class': 'bar', d: A.hbarPath(L, sx(r.v), y, barH), style: 'fill:' + (r.color || PHYS) }, g);
      text(g, sx(r.v) + 7, y + barH / 2 + 4, r.vText, { 'class': 'value-label' });
      if (r.tip) hover(g, r.tip);
    });
  }
  var gBox = document.getElementById('fit-groups'), rBox = document.getElementById('fit-rows');
  function drawGroups() {
    hbars(gBox, F.groups.map(function (g) {
      return { label: g.label.replace(/ \(.*\)$/, ''), v: g.n, vText: String(g.n),
        tip: '<div class="tt-title">' + esc(g.label) + '</div><div class="tt-row"><span>Values</span><b>' + g.n + '</b></div><div class="tt-row"><span>Layout</span><b>' + esc(g.detail) + '</b></div><div class="tt-row"><span>Share</span><b>' + (g.n / F.n_params * 100).toFixed(0) + ' %</b></div>' };
    }), { labelW: gBox.clientWidth < 420 ? 120 : 150 });
  }
  var ROWS = [
    { label: 'Row 15 · uncalibrated', v: 33.9, note: 'K 1e4, E 1e5 as shipped' },
    { label: 'Row 16 · uniform K, E', v: 18.6, note: 'gradient fit, one clip' },
    { label: 'Row 17 · + per-joint', v: 17.0, note: '14 hinge scales, K/E frozen, one clip' },
    { label: 'Row 18 · + rest curvature', v: 16.56, note: 'full fit over all 14 clips' }
  ];
  function drawRows() {
    hbars(rBox, ROWS.map(function (r, i) {
      return { label: r.label, v: r.v, vText: String(r.v), color: i === ROWS.length - 1 ? PHYS : 'var(--axis)',
        tip: '<div class="tt-title">' + esc(r.label) + '</div><div class="tt-row"><span>Error</span><b>' + r.v + ' mm/node</b></div><div class="tt-row"><span>Fit</span><b>' + esc(r.note) + '</b></div>' };
    }), { labelW: rBox.clientWidth < 420 ? 150 : 190, axis: true });
  }

  // ---------------------------------------------------------------- spec sheet
  document.getElementById('fit-numbers').innerHTML = [
    { v: String(F.n_params), l: 'parameters' }, { v: String(F.run.popsize), l: 'candidates per generation' },
    { v: String(F.run.sims_per_gen), l: 'simulations per generation' }, { v: '~' + (F.run.sec_per_gen / 3600).toFixed(1) + ' h', l: 'per generation, ' + F.run.workers + ' workers' },
    { v: F.run.backend, l: 'backend (no GPU)' }, { v: '36', l: 'clips, 5,853 frames' }
  ].map(function (n) { return '<div class="num-tile"><b>' + esc(n.v) + '</b><span>' + esc(n.l) + '</span></div>'; }).join('');
  document.getElementById('fit-spec').innerHTML = [
    ['Model', 'Cosserat rods of the DLO-Lab fork of Genesis (1.0.0), one per cable, ends driven by the measured motion'],
    ['Patched solver', 'Per-element stiffness scales added to the rod solver (dlolab_fit_t36/extern/DLO-Lab-perjoint)'],
    ['Optimiser', 'CMA-ES, population 20, fixed per-coordinate step sizes, bounds per parameter group'],
    ['Objective', 'The table metric: mean node distance over all 36 clips, 16 nodes, every frame'],
    ['Start', 'The 14-clip optimum (16.56 mm/node there) copied onto every cable; μ 0.1, damping 40, mass 0.045 kg/m'],
    ['Status', 'Best ' + F.best_f.toFixed(3) + ' mm/node at generation ' + F.best_gen + '; ' + F.run.stopped],
    ['Source', F.source]
  ].map(function (r) { return '<div><dt>' + esc(r[0]) + '</dt><dd>' + esc(r[1]) + '</dd></div>'; }).join('');

  // ---------------------------------------------------------------- render + re-render on resize / theme change
  function all() { drawProgress(); drawPJ(); drawGroups(); drawRows(); }
  all();
  A.onResize(all);
  new MutationObserver(function () { drawPJ(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
})();
