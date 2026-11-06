/* PhysWorld page: per-node Phi explorer (stiffness E and density rho on the 16 nodes of each cable).
   Data: window.PHYSWORLD_PHI, built by tools/build_physworld_phi.py. Means only, never medians. */
(function () {
  'use strict';
  var P = window.PHYSWORLD_PHI, A = window.APPAGE;
  var box = document.getElementById('phi-chart');
  if (!P || !A || !box || !P.clips || !P.clips.length) return;
  var el = A.el, text = A.text, esc = A.esc, hover = A.hover, svgIn = A.svgIn, MUTED = A.MUTED;
  var CABLES = ['Cable 1', 'Cable 2', 'Cable 3', 'Cable 4'];
  var CABLE_VARS = ['var(--cable-1)', 'var(--cable-2)', 'var(--cable-3)', 'var(--cable-4)'];
  var NODES = 16, BOX = P.box || 1.5;

  var Q = {
    E: { label: 'Stiffness E', unit: 'MPa', scale: 1e-6,
         what: 'Young’s modulus of each node: how stiff the cable is at that point. Higher = stiffer.' },
    rho: { label: 'Density ρ', unit: 'kg/m³', scale: 1,
           what: 'Density of each node: how heavy the cable is at that point.' }
  };
  var st = { q: 'E', show: 'fit', view: 'heat', clip: -1 }; // clip -1 = all clips

  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function rgb(h) { h = h.replace('#', ''); if (h.length === 3) h = h.split('').map(function (c) { return c + c; }).join(''); var n = parseInt(h, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
  function mix(a, b, t) { return [0, 1, 2].map(function (i) { return Math.round(a[i] + (b[i] - a[i]) * t); }); }
  function lum(c) { return (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255; }
  function seqColor(t) { // 0..1, light to dark (one hue)
    var a = rgb(css('--seq-0')), b = rgb(css('--seq-1')), c = rgb(css('--seq-2'));
    t = Math.max(0, Math.min(1, t));
    return t < 0.5 ? mix(a, b, t * 2) : mix(b, c, (t - 0.5) * 2);
  }
  function divColor(t) { // -1..1 around "unchanged"
    var n = rgb(css('--div-neg')), m = rgb(css('--div-mid')), p = rgb(css('--div-pos'));
    return t < 0 ? mix(m, n, Math.min(1, -t)) : mix(m, p, Math.min(1, t));
  }
  function gmean(a) { return Math.exp(a.reduce(function (s, v) { return s + Math.log(v); }, 0) / a.length); }

  // values [cable][node] for the current quantity / clip; "all" = geometric mean over the clips
  function grid(which) {
    if (st.clip >= 0) return P.clips[st.clip][st.q][which];
    return [0, 1, 2, 3].map(function (c) {
      return Array.apply(null, Array(NODES)).map(function (_, j) {
        return gmean(P.clips.map(function (k) { return k[st.q][which][c][j]; }));
      });
    });
  }
  function shown() {
    var fit = grid('fit'), twin = grid('twin');
    if (st.show === 'fit') return { v: fit, fit: fit, twin: twin };
    return { v: fit.map(function (row, c) { return row.map(function (v, j) { return v / twin[c][j]; }); }), fit: fit, twin: twin };
  }
  // colour scale as on the DLO-Lab page: centred on the average of the values shown, stretched to them so differences stay visible
  function fitRange(V) {
    var all = [].concat.apply([], V);
    var logs = all.map(Math.log), c = logs.reduce(function (s, v) { return s + v; }, 0) / logs.length;
    var ext = Math.max.apply(null, logs.map(function (v) { return Math.abs(v - c); })) || 1;
    return { c: c, ext: ext };
  }
  function fmtVal(v) {
    var x = v * Q[st.q].scale;
    return x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x >= 1 ? x.toFixed(2) : x.toPrecision(2);
  }
  function fmtRatio(r) { return '×' + (r >= 10 ? r.toFixed(0) : r.toFixed(2)); }
  function cellLabel(v) { return st.show === 'fit' ? fmtVal(v) : '×' + (v >= 10 ? v.toFixed(0) : v.toFixed(1)); }

  // ---- controls
  var sel = document.getElementById('phi-clip');
  if (sel) sel.innerHTML = '<option value="-1">All ' + P.clips.length + ' clips (geometric mean)</option>' + P.clips.map(function (k, i) {
    return '<option value="' + i + '">Clip ' + (i + 1) + ' · ' + k.mean.toFixed(2) + ' mm/node</option>';
  }).join('');
  if (sel) sel.addEventListener('change', function () { st.clip = +sel.value; draw(); });
  var tabs = document.getElementById('phi-tabs');
  tabs.innerHTML = Object.keys(Q).map(function (k) {
    return '<button type="button" role="tab" class="chip-btn' + (k === st.q ? ' is-active' : '') + '" aria-selected="' + (k === st.q) + '" data-q="' + k + '">' + esc(Q[k].label) + '</button>';
  }).join('');
  tabs.addEventListener('click', function (e) {
    var b = e.target.closest('[data-q]'); if (!b) return;
    st.q = b.getAttribute('data-q');
    tabs.querySelectorAll('[data-q]').forEach(function (x) { x.classList.toggle('is-active', x === b); x.setAttribute('aria-selected', x === b ? 'true' : 'false'); });
    draw();
  });
  function seg(attr, key) {
    document.querySelectorAll('[' + attr + ']').forEach(function (b) {
      b.addEventListener('click', function () {
        st[key] = b.getAttribute(attr);
        document.querySelectorAll('[' + attr + ']').forEach(function (x) { x.classList.toggle('is-on', x === b); x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
        draw();
      });
    });
  }
  seg('data-pview', 'view');
  seg('data-pshow', 'show');

  var legend = document.getElementById('phi-legend'), sub = document.getElementById('phi-sub');

  function draw() {
    var D = shown(), q = Q[st.q];
    var who = st.clip >= 0 ? 'clip ' + (st.clip + 1) + ' (' + P.clips[st.clip].mean.toFixed(2) + ' mm/node after Φ, ' + P.clips[st.clip].before.toFixed(2) + ' before)'
                           : 'the geometric mean over the ' + P.clips.length + ' clips';
    sub.textContent = q.what + ' ' + (st.show === 'fit'
      ? 'Values after fine-tuning on the real clip, in ' + q.unit + ', for ' + who + '.'
      : 'Change from the twin’s value (fitted ÷ twin) for ' + who + '. The fit may move each value at most ×' + Math.exp(BOX).toFixed(1) + ' up or down.');
    if (st.view === 'heat') heat(D); else profile(D);
    cards(D);
  }

  function heat(D) {
    var q = Q[st.q], fr = fitRange(D.v);
    var chExt = Math.min(BOX, Math.max.apply(null, [].concat.apply([], D.v).map(function (v) { return Math.abs(Math.log(v)); })) || 0.1);
    var tOf = st.show === 'fit'
      ? function (v) { return (Math.log(v) - fr.c) / fr.ext; }
      : function (v) { return Math.log(v) / chExt; };
    if (st.show === 'fit') {
      var lowW = st.q === 'E' ? 'softer' : 'lighter', highW = st.q === 'E' ? 'stiffer' : 'heavier';
      legend.innerHTML = '<span class="grad-legend"><span>' + fmtVal(Math.exp(fr.c - fr.ext)) + ' (' + lowW + ')</span><span class="grad-bar"></span><span>' + fmtVal(Math.exp(fr.c + fr.ext)) + ' ' + q.unit + ' (' + highW + ')</span></span>' +
        '<span class="muted">middle = ' + fmtVal(Math.exp(fr.c)) + ' ' + q.unit + ', the average of the values shown (log scale)</span>';
    } else {
      legend.innerHTML = '<span class="grad-legend"><span>' + fmtRatio(Math.exp(-chExt)) + ' (lower)</span><span class="grad-bar"></span><span>' + fmtRatio(Math.exp(chExt)) + ' (higher)</span></span>' +
        '<span class="muted">middle = ×1 (as in the twin)</span>';
    }
    var W = Math.max(300, box.clientWidth), narrow = W < 560;
    var L = narrow ? 34 : 78, R = W - 4, T = 4, rowH = narrow ? 34 : 40;
    var H = T + 4 * rowH + 30, cw = (R - L) / NODES;
    var svg = svgIn(box, W, H);
    var ticks = el('g', { 'class': 'tick' }, svg);
    D.v.forEach(function (row, ci) {
      text(svg, L - 10, T + ci * rowH + rowH / 2 + 4, narrow ? 'C' + (ci + 1) : CABLES[ci], { 'class': 'row-label', 'text-anchor': 'end' });
      row.forEach(function (v, j) {
        var c = divColor(tOf(v));
        var x = L + j * cw + 1, y = T + ci * rowH + 1, w = cw - 2, hh = rowH - 2;
        var g = el('g', { tabindex: 0 }, svg);
        el('rect', { x: x, y: y, width: w, height: hh, rx: 3, fill: 'rgb(' + c.join(',') + ')' }, g);
        if (w >= 30) text(g, x + w / 2, y + hh / 2 + 4, cellLabel(v), { 'text-anchor': 'middle' },
          'font-size: 10.5px; font-variant-numeric: tabular-nums; pointer-events: none; fill:' + (lum(c) > 0.55 ? '#0b0b0b' : '#ffffff'));
        hover(g, '<div class="tt-title">' + CABLES[ci] + ' · node ' + (j + 1) + '</div>' +
          '<div class="tt-row"><span>Fitted</span><b>' + fmtVal(D.fit[ci][j]) + ' ' + q.unit + '</b></div>' +
          '<div class="tt-row"><span>Twin</span><b>' + fmtVal(D.twin[ci][j]) + ' ' + q.unit + '</b></div>' +
          '<div class="tt-row"><span>Change</span><b>' + fmtRatio(D.fit[ci][j] / D.twin[ci][j]) + '</b></div>');
      });
    });
    for (var j = 0; j < NODES; j += (narrow ? 3 : 1)) text(ticks, L + j * cw + cw / 2, T + 4 * rowH + 16, j + 1, { 'text-anchor': 'middle' });
    text(svg, R, H - 2, 'node along the cable, hub → plug', { 'text-anchor': 'end' }, MUTED);
  }

  function profile(D) {
    var q = Q[st.q], change = st.show !== 'fit';
    legend.innerHTML = CABLES.map(function (c, i) { return '<span><span class="line-swatch" style="--sw:' + CABLE_VARS[i] + '"></span>' + c + '</span>'; }).join('') +
      (change ? '<span><span class="tick-key"></span>×1 (as in the twin)</span>' : '<span><span class="line-swatch dashed" style="--sw: var(--muted)"></span>twin (start)</span>');
    var W = Math.max(300, box.clientWidth), H = 300, L = 54, R = W - 12, T = 10, B = H - 34;
    var series = change ? [D.v] : [D.fit, D.twin];
    var all = [].concat.apply([], [].concat.apply([], series)).map(Math.log);
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    if (change) { var m = Math.max(Math.abs(lo), Math.abs(hi), 0.1); lo = -m; hi = m; }
    var pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    var sx = function (j) { return L + (j / (NODES - 1)) * (R - L); };
    var sy = function (u) { return B - ((u - lo) / (hi - lo)) * (B - T); };
    var svg = svgIn(box, W, H);
    var gridG = el('g', { 'class': 'grid' }, svg), ticks = el('g', { 'class': 'tick' }, svg);
    // log-axis ticks at 1-2-5 steps
    var yt = [];
    for (var e = Math.floor(lo / Math.LN10) - 1; e <= Math.ceil(hi / Math.LN10) + 1; e++) [1, 2, 5].forEach(function (m) { var v = m * Math.pow(10, e), u = Math.log(v); if (u >= lo && u <= hi) yt.push(v); });
    if (yt.length > 7) yt = yt.filter(function (v) { return /^1/.test(v.toPrecision(1)); });
    yt.forEach(function (v) {
      el('line', { x1: L, x2: R, y1: sy(Math.log(v)), y2: sy(Math.log(v)) }, gridG);
      text(ticks, L - 8, sy(Math.log(v)) + 4, change ? '×' + v : fmtVal(v), { 'text-anchor': 'end' });
    });
    for (var j = 0; j < NODES; j++) if (W >= 520 || j % 3 === 0) text(ticks, sx(j), B + 16, j + 1, { 'text-anchor': 'middle' });
    if (change) el('line', { x1: L, x2: R, y1: sy(0), y2: sy(0), style: 'stroke: var(--ink); stroke-width: 1.5; opacity: 0.5' }, svg);
    el('line', { 'class': 'baseline', x1: L, x2: R, y1: B, y2: B }, svg);
    function line(V, ci, dashed) {
      var d = V[ci].map(function (v, j) { return (j ? 'L' : 'M') + sx(j) + ',' + sy(Math.log(v)); }).join(' ');
      el('path', { d: d, fill: 'none', style: 'stroke:' + CABLE_VARS[ci] + '; stroke-width:' + (dashed ? 1.5 : 2) + (dashed ? '; stroke-dasharray: 4 4; opacity: 0.7' : '') + '; stroke-linejoin: round; stroke-linecap: round' }, svg);
      if (!dashed) V[ci].forEach(function (v, j) { el('circle', { cx: sx(j), cy: sy(Math.log(v)), r: 3.5, style: 'fill:' + CABLE_VARS[ci] + '; stroke: var(--surface); stroke-width: 1.5', 'pointer-events': 'none' }, svg); });
    }
    [0, 1, 2, 3].forEach(function (ci) { if (!change) line(D.twin, ci, true); line(D.v, ci, false); });
    var cross = el('line', { x1: 0, x2: 0, y1: T, y2: B, style: 'stroke: var(--ink); stroke-width: 1; opacity: 0', 'pointer-events': 'none' }, svg);
    var hit = el('rect', { x: L - 10, y: T, width: R - L + 20, height: B - T, fill: 'transparent' }, svg);
    hit.addEventListener('mousemove', function (ev) {
      var r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * (W / r.width);
      var j = Math.max(0, Math.min(NODES - 1, Math.round((px - L) / (R - L) * (NODES - 1))));
      cross.setAttribute('x1', sx(j)); cross.setAttribute('x2', sx(j)); cross.style.opacity = 0.25;
      A.showTip('<div class="tt-title">Node ' + (j + 1) + ' · ' + esc(q.label) + '</div>' + [0, 1, 2, 3].map(function (ci) {
        return '<div class="tt-row"><span><span class="dot" style="--sw:' + CABLE_VARS[ci] + '"></span>' + CABLES[ci] + '</span><b>' +
          (change ? fmtRatio(D.v[ci][j]) : fmtVal(D.fit[ci][j]) + ' ' + q.unit) + '</b></div>';
      }).join(''), ev.clientX, ev.clientY);
    });
    hit.addEventListener('mouseleave', function () { cross.style.opacity = 0; A.hideTip(); });
    text(svg, R, H - 4, 'node along the cable, hub → plug · log scale' + (change ? '' : ', ' + q.unit), { 'text-anchor': 'end' }, MUTED);
  }

  function cards(D) {
    var q = Q[st.q];
    document.getElementById('phi-cards').innerHTML = D.fit.map(function (row, ci) {
      var mn = Math.min.apply(null, row), mx = Math.max.apply(null, row);
      var ch = gmean(row.map(function (v, j) { return v / D.twin[ci][j]; }));
      return '<div class="pj-card"><b><span class="dot" style="--sw:' + CABLE_VARS[ci] + '"></span>' + CABLES[ci] + '</b>' +
        '<span>geometric mean <b>' + fmtVal(gmean(row)) + ' ' + q.unit + '</b></span>' +
        '<span>lowest <b>' + fmtVal(mn) + '</b> (node ' + (row.indexOf(mn) + 1) + ')</span>' +
        '<span>highest <b>' + fmtVal(mx) + '</b> (node ' + (row.indexOf(mx) + 1) + ')</span>' +
        '<span>vs twin <b>' + fmtRatio(ch) + '</b> on average</span></div>';
    }).join('');
  }

  draw();
  A.onResize(draw);
})();
