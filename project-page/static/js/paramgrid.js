/* Approach pages: per-joint / per-spring parameter heatmap (same look as the DLO-Lab and PhysWorld sections).
   Data: window.PARAM_GRIDS[<data-approach>], built by tools/build_param_grids.py. Means only, never medians. */
(function () {
  'use strict';
  var A = window.APPAGE, ALL = window.PARAM_GRIDS;
  var box = document.getElementById('pg-chart');
  var key = document.body.getAttribute('data-approach');
  var G = ALL && ALL[key];
  if (!A || !box || !G || !G.q || !G.q.length) return;
  var el = A.el, text = A.text, esc = A.esc, hover = A.hover, svgIn = A.svgIn, MUTED = A.MUTED;
  var ROWS = G.rows || ['Cable 1', 'Cable 2', 'Cable 3', 'Cable 4'];
  var ROW_VARS = G.rows ? ['var(--muted)'] : ['var(--cable-1)', 'var(--cable-2)', 'var(--cable-3)', 'var(--cable-4)'];
  var AXIS = G.axis || 'joint';
  var cur = 0;

  function css(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
  function rgb(h) { h = h.replace('#', ''); if (h.length === 3) h = h.split('').map(function (c) { return c + c; }).join(''); var n = parseInt(h, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
  function mix(a, b, t) { return [0, 1, 2].map(function (i) { return Math.round(a[i] + (b[i] - a[i]) * t); }); }
  function lum(c) { return (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255; }
  function divColor(t) {
    var n = rgb(css('--div-neg')), m = rgb(css('--div-mid')), p = rgb(css('--div-pos'));
    return t < 0 ? mix(m, n, Math.min(1, -t)) : mix(m, p, Math.min(1, t));
  }
  function fmt(Q, v) {
    if (Q.kind === 'signed') return (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(1);
    var a = Math.abs(v);
    return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toPrecision(2);
  }
  function withUnit(Q, v) { return fmt(Q, v) + (Q.unit ? (Q.unit === '°' ? '°' : ' ' + Q.unit) : ''); }
  function gmean(a) { return Math.exp(a.reduce(function (s, v) { return s + Math.log(v); }, 0) / a.length); }
  function mean(a) { return a.reduce(function (s, v) { return s + v; }, 0) / a.length; }

  var tabs = document.getElementById('pg-tabs');
  tabs.innerHTML = G.q.map(function (Q, i) {
    return '<button type="button" role="tab" class="chip-btn' + (i === cur ? ' is-active' : '') + '" aria-selected="' + (i === cur) + '" data-i="' + i + '">' + esc(Q.label) + '</button>';
  }).join('');
  tabs.addEventListener('click', function (e) {
    var b = e.target.closest('[data-i]'); if (!b) return;
    cur = +b.getAttribute('data-i');
    tabs.querySelectorAll('[data-i]').forEach(function (x) { x.classList.toggle('is-active', x === b); x.setAttribute('aria-selected', x === b ? 'true' : 'false'); });
    draw();
  });
  var legend = document.getElementById('pg-legend'), sub = document.getElementById('pg-sub');

  function draw() {
    var Q = G.q[cur], V = Q.values, n = V[0].length, flat = [].concat.apply([], V);
    var signed = Q.kind === 'signed';
    var c, ext, tOf;
    if (signed) {
      c = 0; ext = Math.max.apply(null, flat.map(Math.abs)) || 1;
      tOf = function (v) { return v / ext; };
    } else {
      var logs = flat.map(Math.log);
      c = mean(logs); ext = Math.max.apply(null, logs.map(function (v) { return Math.abs(v - c); })) || 1;
      tOf = function (v) { return (Math.log(v) - c) / ext; };
    }
    sub.textContent = Q.what + (Q.unit && Q.unit !== '°' ? ' In ' + Q.unit + '.' : signed ? ' In degrees.' : '');
    legend.innerHTML = signed
      ? '<span class="grad-legend"><span>' + withUnit(Q, -ext) + '</span><span class="grad-bar"></span><span>' + withUnit(Q, ext) + '</span></span><span class="muted">middle = 0° (straight)</span>'
      : '<span class="grad-legend"><span>' + fmt(Q, Math.exp(c - ext)) + ' (lower)</span><span class="grad-bar"></span><span>' + withUnit(Q, Math.exp(c + ext)) + ' (higher)</span></span>' +
        '<span class="muted">middle = ' + withUnit(Q, Math.exp(c)) + ', the average of the values shown (log scale)</span>';

    var W = Math.max(300, box.clientWidth), narrow = W < 560, nr = V.length;
    var L = narrow ? (G.rows ? 60 : 34) : (G.rows ? 96 : 78), R = W - 4, T = 4, rowH = narrow ? 34 : 40;
    var H = T + nr * rowH + 30, cw = (R - L) / n;
    var svg = svgIn(box, W, H), ticks = el('g', { 'class': 'tick' }, svg);
    V.forEach(function (row, ri) {
      text(svg, L - 10, T + ri * rowH + rowH / 2 + 4, narrow && !G.rows ? 'C' + (ri + 1) : ROWS[ri], { 'class': 'row-label', 'text-anchor': 'end' });
      row.forEach(function (v, j) {
        var col = divColor(tOf(v)), x = L + j * cw + 1, y = T + ri * rowH + 1, w = cw - 2, hh = rowH - 2;
        var g = el('g', { tabindex: 0 }, svg);
        el('rect', { x: x, y: y, width: w, height: hh, rx: 3, fill: 'rgb(' + col.join(',') + ')' }, g);
        if (w >= 30) text(g, x + w / 2, y + hh / 2 + 4, fmt(Q, v), { 'text-anchor': 'middle' },
          'font-size: 10.5px; font-variant-numeric: tabular-nums; pointer-events: none; fill:' + (lum(col) > 0.55 ? '#0b0b0b' : '#ffffff'));
        hover(g, '<div class="tt-title">' + esc(ROWS[ri]) + ' · ' + AXIS + ' ' + (j + 1) + '</div>' +
          '<div class="tt-row"><span>' + esc(Q.label) + '</span><b>' + withUnit(Q, v) + '</b></div>');
      });
    });
    for (var j = 0; j < n; j += (narrow ? 3 : 1)) text(ticks, L + j * cw + cw / 2, T + nr * rowH + 16, j + 1, { 'text-anchor': 'middle' });
    text(svg, R, H - 2, AXIS + ' along the cable, hub → plug', { 'text-anchor': 'end' }, MUTED);

    document.getElementById('pg-cards').innerHTML = V.map(function (row, ri) {
      var mn = Math.min.apply(null, row), mx = Math.max.apply(null, row);
      var avg = signed ? mean(row) : gmean(row);
      return '<div class="pj-card"><b><span class="dot" style="--sw:' + ROW_VARS[ri % ROW_VARS.length] + '"></span>' + esc(ROWS[ri]) + '</b>' +
        '<span>' + (signed ? 'mean' : 'geometric mean') + ' <b>' + withUnit(Q, avg) + '</b></span>' +
        '<span>lowest <b>' + fmt(Q, mn) + '</b> (' + AXIS + ' ' + (row.indexOf(mn) + 1) + ')</span>' +
        '<span>highest <b>' + fmt(Q, mx) + '</b> (' + AXIS + ' ' + (row.indexOf(mx) + 1) + ')</span></div>';
    }).join('');
  }

  var note = document.getElementById('pg-note');
  if (note) note.innerHTML = '<i class="fas fa-circle-info"></i> ' + esc(G.note) + ' Source: <code>' + esc(G.source) + '</code>.';
  draw();
  A.onResize(draw);
})();
