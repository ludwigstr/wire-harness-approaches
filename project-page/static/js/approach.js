/* Approach pages: theme toggle, mobile nav, and either the data dashboard (#ap-dash)
   or the simple per-clip chart (#ap-perclip) for pages not converted yet. */
(function () {
  'use strict';

  var D = window.BENCH_DATA || {};
  var SVGNS = 'http://www.w3.org/2000/svg';
  var COLORS = { world: 'var(--series-world)', dynamics: 'var(--series-dynamics)', physics: 'var(--series-physics)' };
  // Page scores (same as the results chart on the main page).
  var MODELS = [
    { key: 'physworld', name: 'PhysWorld', paradigm: 'world', score: 4.5 },
    { key: 'newton', name: 'Newton', paradigm: 'physics', score: 8.1 },
    { key: 'mujoco', name: 'MuJoCo', paradigm: 'physics', score: 10.9 },
    { key: 'genesis', name: 'Genesis', paradigm: 'physics', score: 12.7 },
    { key: 'yang', name: 'Yang et al.', paradigm: 'dynamics', score: 13.3 },
    { key: 'caporali', name: 'Caporali et al.', paradigm: 'dynamics', score: 15.8 },
    { key: 'dlolab', name: 'DLO-Lab', paradigm: 'physics', score: 17.0 },
    { key: 'ropedreamer', name: 'RopeDreamer', paradigm: 'world', score: 18.3 },
    { key: 'phystwin', name: 'PhysTwin', paradigm: 'world', score: 22.0 },
    { key: 'adaptigraph', name: 'AdaptiGraph', paradigm: 'dynamics', score: 29.4 }
  ];
  var BY = {};
  MODELS.forEach(function (m) { BY[m.key] = m; });

  // ---------------------------------------------------------------- shared helpers
  function el(tag, attrs, parent) {
    var n = document.createElementNS(SVGNS, tag);
    for (var k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function f2(v) { return v == null ? '–' : Number(v).toFixed(2); }
  function niceStep(span, target) {
    var raw = span / (target || 6), p = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10)), m = raw / p;
    return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  }
  function niceTop(v, step) { return Math.ceil(v / step - 1e-9) * step; }
  function quantile(sorted, q) {
    var pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }
  function hbarPath(x0, x1, y, h) { // horizontal bar, 4px rounded data end, square at the baseline
    var r = Math.min(4, (x1 - x0) / 2, h / 2);
    return 'M' + x0 + ',' + y + ' H' + (x1 - r) + ' Q' + x1 + ',' + y + ' ' + x1 + ',' + (y + r) +
      ' V' + (y + h - r) + ' Q' + x1 + ',' + (y + h) + ' ' + (x1 - r) + ',' + (y + h) + ' H' + x0 + ' Z';
  }
  function vbarPath(x, w, y, base) { // vertical column, rounded top
    var r = Math.min(4, w / 2, (base - y) / 2);
    return 'M' + x + ',' + base + ' V' + (y + r) + ' Q' + x + ',' + y + ' ' + (x + r) + ',' + y +
      ' H' + (x + w - r) + ' Q' + (x + w) + ',' + y + ' ' + (x + w) + ',' + (y + r) + ' V' + base + ' Z';
  }
  function svgIn(box, W, H) {
    box.innerHTML = '';
    return el('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H }, box);
  }
  function text(parent, x, y, str, attrs, style) {
    var t = el('text', Object.assign({ x: x, y: y }, attrs || {}), parent);
    t.textContent = str;
    if (style) t.setAttribute('style', style);
    return t;
  }
  var MUTED = 'fill: var(--muted); font-size: 11px';

  // ---------------------------------------------------------------- theme + nav
  var themeBtn = document.getElementById('theme-toggle');
  function theme() { return document.documentElement.getAttribute('data-theme') || 'light'; }
  function syncIcon() { themeBtn.innerHTML = theme() === 'dark' ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>'; }
  themeBtn.addEventListener('click', function () {
    var next = theme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('dlo-theme-v2', next); } catch (e) {}
    syncIcon();
  });
  syncIcon();
  var burger = document.getElementById('nav-burger');
  var links = document.getElementById('nav-links');
  burger.addEventListener('click', function () {
    var open = links.classList.toggle('is-open');
    burger.setAttribute('aria-expanded', open ? 'true' : 'false');
  });

  // ---------------------------------------------------------------- tooltip
  var tip = document.getElementById('tooltip');
  function showTip(html, x, y) {
    tip.innerHTML = html; tip.hidden = false;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(window.innerWidth - w - 8, Math.max(8, x + 14)) + 'px';
    var top = y - h - 12; if (top < 8) top = y + 18;
    tip.style.top = top + 'px';
  }
  function hideTip() { tip.hidden = true; }
  function hover(node, html) {
    node.addEventListener('mousemove', function (e) { showTip(typeof html === 'function' ? html() : html, e.clientX, e.clientY); });
    node.addEventListener('mouseleave', hideTip);
    node.addEventListener('focus', function () { var r = node.getBoundingClientRect(); showTip(typeof html === 'function' ? html() : html, r.left + r.width / 2, r.top); });
    node.addEventListener('blur', hideTip);
  }
  document.addEventListener('scroll', hideTip, { passive: true });

  // ---------------------------------------------------------------- per-clip series
  var key = document.body.getAttribute('data-approach');
  var me = BY[key] || { name: key, paradigm: 'world' };
  var color = COLORS[me.paradigm];
  function seriesFor(k) {
    if (D.perClip && D.perClip[k]) return { key: k, name: BY[k].name, paradigm: BY[k].paradigm, values: D.perClip[k], interim: false };
    if (k === 'newton' && D.newtonGen0) return { key: k, name: 'Newton (gen 0)', paradigm: 'physics', values: D.newtonGen0.clips, interim: true };
    return null;
  }
  var mine = seriesFor(key);
  var others = MODELS.map(function (m) { return m.key; }).filter(function (k) { return k !== key; }).map(seriesFor).filter(Boolean);
  var clips = D.clips || [], N = clips.length;

  var resizers = [];
  var lastW = window.innerWidth, timer;
  window.addEventListener('resize', function () {
    clearTimeout(timer);
    timer = setTimeout(function () { if (window.innerWidth !== lastW) { lastW = window.innerWidth; resizers.forEach(function (f) { f(); }); } }, 100);
  });

  // Helpers for page-specific scripts (e.g. dlolab.js).
  window.APPAGE = {
    el: el, text: text, esc: esc, niceStep: niceStep, niceTop: niceTop, hbarPath: hbarPath, vbarPath: vbarPath,
    svgIn: svgIn, hover: hover, showTip: showTip, hideTip: hideTip, MUTED: MUTED, COLORS: COLORS, MODELS: MODELS,
    onResize: function (f) { resizers.push(f); }
  };

  if (document.getElementById('ap-dash')) dashboard();
  else legacy();

  // ================================================================= dashboard
  function dashboard() {
    var cfg = {};
    try { cfg = JSON.parse(document.getElementById('ap-data').textContent); } catch (e) {}
    var vals = mine ? clips.map(function (c) { return mine.values[c]; }) : [];
    var sorted = vals.slice().sort(function (a, b) { return a - b; });
    var mean = vals.length ? vals.reduce(function (s, v) { return s + v; }, 0) / N : null;

    // ---- KPI tiles
    var rankList = MODELS.slice().sort(function (a, b) { return a.score - b.score; });
    var rank = rankList.indexOf(me) + 1;
    var neighbour = rank === 1 ? rankList[1] : rankList[0];
    var kpis = [];
    if (mine) {
      var bi = vals.indexOf(sorted[0]), wi = vals.indexOf(sorted[N - 1]);
      kpis.push({ v: f2(mean), l: 'mean mm/node' });
      kpis.push({ v: f2(quantile(sorted, 0.5)), l: 'median' });
      kpis.push({ v: f2(sorted[0]), l: 'best clip (' + (bi + 1) + ')' });
      kpis.push({ v: f2(sorted[N - 1]), l: 'worst clip (' + (wi + 1) + ')' });
      kpis.push({ v: f2(quantile(sorted, 0.25)) + '–' + f2(quantile(sorted, 0.75)), l: 'middle 50 % of clips', small: true });
    } else {
      kpis.push({ v: me.score.toFixed(1), l: 'mm/node' });
    }
    kpis.push({ v: '#' + rank + ' of ' + MODELS.length, l: rank === 1
      ? 'next: ' + neighbour.name + ' (+' + (neighbour.score - me.score).toFixed(1) + ')'
      : 'best: ' + neighbour.name + ' (' + (me.score - neighbour.score).toFixed(1) + ' lower)' });
    document.getElementById('ap-kpis').innerHTML = kpis.map(function (k) {
      return '<div class="stat"><span class="stat-value' + (k.small ? ' kpi-small' : '') + '">' + k.v + '</span><span class="stat-label">' + esc(k.l) + '</span></div>';
    }).join('');

    // ---- per-clip chart with order toggle and comparison overlay
    var order = 'clip', compareKey = '';
    var cmpSel = document.getElementById('ap-compare');
    cmpSel.innerHTML = '<option value="">none</option>' + others.map(function (s) {
      return '<option value="' + s.key + '">' + esc(s.name) + '</option>';
    }).join('');
    cmpSel.addEventListener('change', function () { compareKey = cmpSel.value; drawClips(); });
    document.querySelectorAll('.seg-btn[data-order]').forEach(function (b) {
      b.addEventListener('click', function () {
        order = b.getAttribute('data-order');
        document.querySelectorAll('.seg-btn[data-order]').forEach(function (x) { x.classList.toggle('is-on', x === b); });
        drawClips();
      });
    });
    var clipBox = document.getElementById('ap-clips'), clipLegend = document.getElementById('ap-clips-legend');
    function drawClips() {
      if (!mine) { clipBox.innerHTML = '<p class="muted-note">No per-clip results for this model yet.</p>'; return; }
      var cmp = compareKey ? others.filter(function (s) { return s.key === compareKey; })[0] : null;
      clipLegend.innerHTML = '<span><span class="swatch" style="--sw:' + color + '"></span>' + esc(me.name) + '</span>' +
        (cmp ? '<span><span class="cmp-key"></span>' + esc(cmp.name) + '</span>' : '') +
        '<span><span class="tick-key"></span>' + esc(me.name) + ' mean</span>';
      var idx = clips.map(function (c, i) { return i; });
      if (order === 'sorted') idx.sort(function (a, b) { return vals[a] - vals[b]; });
      var W = Math.max(300, clipBox.clientWidth), H = 280, L = 40, R = W - 8, T = 12, B = H - 34;
      var vmax = sorted[N - 1];
      if (cmp) clips.forEach(function (c) { vmax = Math.max(vmax, cmp.values[c]); });
      var step = niceStep(vmax, 5), top = niceTop(vmax * 1.04, step);
      var sy = function (v) { return B - (v / top) * (B - T); };
      var band = (R - L) / N, bw = Math.min(24, band - 2);
      var svg = svgIn(clipBox, W, H);
      var grid = el('g', { 'class': 'grid' }, svg), ticks = el('g', { 'class': 'tick' }, svg);
      for (var t = 0; t <= top + 1e-9; t += step) {
        el('line', { x1: L, x2: R, y1: sy(t), y2: sy(t) }, grid);
        text(ticks, L - 8, sy(t) + 4, step < 1 ? t.toFixed(1) : Math.round(t), { 'text-anchor': 'end' });
      }
      el('line', { 'class': 'baseline', x1: L, x2: R, y1: B, y2: B }, svg);
      var cmpLayer = el('g', {}, svg);
      idx.forEach(function (ci, pos) {
        var c = clips[ci], v = vals[ci], x = L + pos * band + (band - bw) / 2;
        var g = el('g', { 'class': 'bar-row', tabindex: 0 }, svg);
        el('rect', { x: L + pos * band, y: T, width: band, height: B - T, fill: 'transparent' }, g);
        el('path', { 'class': 'bar', d: vbarPath(x, bw, sy(v), B), style: 'fill:' + color }, g);
        if (order === 'clip' ? ((ci + 1) % 5 === 0 || ci === 0) : (pos === 0 || (pos + 1) % 5 === 0)) {
          text(ticks, L + pos * band + band / 2, B + 16, order === 'clip' ? ci + 1 : pos + 1, { 'text-anchor': 'middle' });
        }
        if (cmp) {
          var cv = cmp.values[c];
          el('circle', { cx: L + pos * band + band / 2, cy: sy(cv), r: 4.5, style: 'fill: var(--ink-2); stroke: var(--surface); stroke-width: 2; pointer-events: none' }, cmpLayer);
        }
        hover(g, function () {
          var h = '<div class="tt-title">Clip ' + (ci + 1) + ' of ' + N + '</div>' +
            '<div class="tt-row"><span>' + esc(me.name) + '</span><b>' + f2(v) + ' mm/node</b></div>';
          if (cmp) {
            var d = v - cmp.values[c];
            h += '<div class="tt-row"><span>' + esc(cmp.name) + '</span><b>' + f2(cmp.values[c]) + '</b></div>' +
              '<div class="tt-row"><span>Difference</span><b>' + (d > 0 ? '+' : '') + d.toFixed(2) + '</b></div>';
          }
          return h + '<div class="tt-row"><span>Frames</span><b>' + D.frames[c] + '</b></div>';
        });
      });
      svg.appendChild(cmpLayer);
      el('line', { x1: L, x2: R, y1: sy(mean), y2: sy(mean), style: 'stroke: var(--ink); stroke-width: 1.5; opacity: 0.6; pointer-events: none' }, svg);
      text(svg, R, sy(mean) - 6, 'mean ' + mean.toFixed(2), { 'text-anchor': 'end' },
        'fill: var(--ink); font-size: 11px; font-weight: 600; paint-order: stroke; stroke: var(--surface); stroke-width: 4px; stroke-linejoin: round; pointer-events: none');
      text(svg, R, H - 4, order === 'clip' ? 'clip, in recording order' : 'clips, sorted from easiest to hardest', { 'text-anchor': 'end' }, MUTED);
    }

    // ---- head-to-head
    var h2hBox = document.getElementById('ap-h2h');
    function drawH2H() {
      if (!mine || !others.length) { h2hBox.innerHTML = '<p class="muted-note">Needs per-clip results.</p>'; return; }
      var rows = others.map(function (s) {
        var wins = 0, diff = 0;
        clips.forEach(function (c) { if (mine.values[c] < s.values[c]) wins++; diff += mine.values[c] - s.values[c]; });
        return { s: s, wins: wins, diff: diff / N };
      }).sort(function (a, b) { return b.wins - a.wins || a.diff - b.diff; });
      h2hBox.innerHTML = '<div class="h2h">' + rows.map(function (r) {
        var pct = (r.wins / N * 100).toFixed(1);
        return '<div class="h2h-row" tabindex="0" data-tip="' + esc(r.s.name) + '|' + r.wins + '|' + r.diff.toFixed(2) + '">' +
          '<span class="h2h-name">vs ' + esc(r.s.name) + '</span>' +
          '<span class="h2h-track"><span class="h2h-fill" style="width:' + pct + '%;background:' + color + '"></span></span>' +
          '<span class="h2h-val">' + r.wins + '/' + N + '</span>' +
          '<span class="h2h-diff">' + (r.diff > 0 ? '+' : '') + r.diff.toFixed(1) + ' mm</span></div>';
      }).join('') + '</div><p class="muted-note">Right column: average difference per clip (negative = ' + esc(me.name) + ' is lower).</p>';
      h2hBox.querySelectorAll('.h2h-row').forEach(function (row) {
        var p = row.getAttribute('data-tip').split('|');
        hover(row, '<div class="tt-title">' + esc(me.name) + ' vs ' + p[0] + '</div>' +
          '<div class="tt-row"><span>Lower error on</span><b>' + p[1] + ' of ' + N + ' clips</b></div>' +
          '<div class="tt-row"><span>Mean difference</span><b>' + (p[2] > 0 ? '+' : '') + p[2] + ' mm/node</b></div>');
      });
    }

    // ---- histogram
    var histBox = document.getElementById('ap-hist');
    function drawHist() {
      if (!mine) { histBox.innerHTML = ''; return; }
      var lo = sorted[0], hi = sorted[N - 1];
      var bw = niceStep(hi - lo || 1, 8), start = Math.floor(lo / bw) * bw;
      var nb = Math.max(1, Math.ceil((hi - start) / bw + 1e-9));
      var bins = []; for (var i = 0; i < nb; i++) bins.push([]);
      clips.forEach(function (c, ci) { var b = Math.min(nb - 1, Math.floor((vals[ci] - start) / bw + 1e-9)); bins[b].push(ci + 1); });
      var cmax = Math.max.apply(null, bins.map(function (b) { return b.length; }));
      var W = Math.max(260, histBox.clientWidth), H = 220, L = 30, R = W - 8, T = 10, B = H - 32;
      var ys = Math.max(1, Math.round(niceStep(cmax, 4))), topC = niceTop(cmax, ys);
      var sy = function (v) { return B - (v / topC) * (B - T); };
      var sx = function (v) { return L + ((v - start) / (nb * bw)) * (R - L); };
      var svg = svgIn(histBox, W, H);
      var grid = el('g', { 'class': 'grid' }, svg), ticks = el('g', { 'class': 'tick' }, svg);
      for (var t = 0; t <= topC; t += ys) { el('line', { x1: L, x2: R, y1: sy(t), y2: sy(t) }, grid); text(ticks, L - 6, sy(t) + 4, t, { 'text-anchor': 'end' }); }
      el('line', { 'class': 'baseline', x1: L, x2: R, y1: B, y2: B }, svg);
      var every = nb > 10 ? 2 : 1;
      for (var k = 0; k <= nb; k += every) text(ticks, sx(start + k * bw), B + 15, +(start + k * bw).toFixed(2), { 'text-anchor': 'middle' });
      bins.forEach(function (b, i) {
        if (!b.length) return;
        var x0 = sx(start + i * bw) + 1, x1 = sx(start + (i + 1) * bw) - 1;
        var g = el('g', { 'class': 'bar-row', tabindex: 0 }, svg);
        el('path', { 'class': 'bar', d: vbarPath(x0, x1 - x0, sy(b.length), B), style: 'fill:' + color }, g);
        hover(g, '<div class="tt-title">' + +(start + i * bw).toFixed(2) + '–' + +(start + (i + 1) * bw).toFixed(2) + ' mm/node</div>' +
          '<div class="tt-row"><span>Clips</span><b>' + b.length + '</b></div>' +
          '<div class="tt-row"><span>Clip numbers</span><b>' + b.join(', ') + '</b></div>');
      });
      var med = quantile(sorted, 0.5);
      el('line', { x1: sx(med), x2: sx(med), y1: T, y2: B, style: 'stroke: var(--ink); stroke-width: 1.5; opacity: 0.6' }, svg);
      text(svg, sx(med) + 5, T + 10, 'median ' + med.toFixed(2), {}, 'fill: var(--ink); font-size: 11px; font-weight: 600; paint-order: stroke; stroke: var(--surface); stroke-width: 4px');
      document.getElementById('ap-hist-sub').textContent = N + ' clips in ' + bw + ' mm/node bins';
    }

    // ---- variants
    var varBox = document.getElementById('ap-variants');
    function drawVariants() {
      var vs = cfg.variants || [];
      if (!vs.length) { varBox.closest('.dash-card').hidden = true; return; }
      var W = Math.max(300, varBox.clientWidth), narrow = W < 560;
      var L = narrow ? Math.min(170, W * 0.45) : 270, R = W - 60, rowH = 34, T = 4, barH = 18;
      var H = T + vs.length * rowH + 40;
      var vmax = Math.max.apply(null, vs.map(function (v) { return v.mean; }));
      var step = niceStep(vmax, 5), top = niceTop(vmax, step);
      var sx = function (v) { return L + (v / top) * (R - L); };
      var svg = svgIn(varBox, W, H);
      var grid = el('g', { 'class': 'grid' }, svg), ticks = el('g', { 'class': 'tick' }, svg);
      var B = T + vs.length * rowH;
      for (var t = 0; t <= top + 1e-9; t += step) { el('line', { x1: sx(t), x2: sx(t), y1: T, y2: B }, grid); text(ticks, sx(t), B + 16, Math.round(t * 10) / 10, { 'text-anchor': 'middle' }); }
      el('line', { 'class': 'baseline', x1: L, x2: L, y1: T, y2: B }, svg);
      vs.forEach(function (v, i) {
        var y = T + i * rowH + (rowH - barH) / 2;
        var g = el('g', { 'class': 'bar-row', tabindex: 0 }, svg);
        el('rect', { x: 0, y: T + i * rowH, width: W, height: rowH, fill: 'transparent' }, g);
        var lab = text(g, L - 10, y + barH / 2 + 4, '', { 'class': 'row-label', 'text-anchor': 'end' });
        var maxChars = narrow ? 22 : 40;
        lab.textContent = v.label.length > maxChars ? v.label.slice(0, maxChars - 1) + '…' : v.label;
        el('path', { 'class': 'bar', d: hbarPath(L, sx(v.mean), y, barH), style: 'fill:' + (v.current ? color : 'var(--axis)') }, g);
        if (v.median != null) el('line', { x1: sx(v.median), x2: sx(v.median), y1: y - 3, y2: y + barH + 3, style: 'stroke: var(--ink); stroke-width: 2' }, g);
        var vt = text(g, sx(v.mean) + 7, y + barH / 2 + 4, v.mean.toFixed(2), { 'class': 'value-label' });
        var cl = el('tspan', { dx: 5 }, vt); cl.textContent = v.clips + ' clips'; cl.setAttribute('style', MUTED + '; font-weight: 400');
        hover(g, '<div class="tt-title">' + esc(v.label) + '</div>' +
          '<div class="tt-row"><span>Mean</span><b>' + v.mean.toFixed(2) + ' mm/node</b></div>' +
          (v.median != null ? '<div class="tt-row"><span>Median</span><b>' + v.median.toFixed(2) + '</b></div>' : '') +
          '<div class="tt-row"><span>Clips</span><b>' + v.clips + '</b></div>');
      });
      text(svg, R, H - 2, 'mm/node', { 'text-anchor': 'end' }, MUTED);
    }

    // ---- error vs clip length
    var scBox = document.getElementById('ap-scatter');
    function drawScatter() {
      if (!mine) { scBox.closest('.dash-card').hidden = true; return; }
      var fr = clips.map(function (c) { return D.frames[c]; });
      var n = N, mx = fr.reduce(function (a, b) { return a + b; }, 0) / n, my = mean;
      var sxy = 0, sxx = 0, syy = 0;
      for (var i = 0; i < n; i++) { sxy += (fr[i] - mx) * (vals[i] - my); sxx += (fr[i] - mx) * (fr[i] - mx); syy += (vals[i] - my) * (vals[i] - my); }
      var r = sxy / Math.sqrt(sxx * syy), slope = sxy / sxx, icpt = my - slope * mx;
      var strength = Math.abs(r) < 0.2 ? 'no clear link'
        : (Math.abs(r) < 0.5 ? 'weak: ' : '') + (r > 0 ? 'higher error on longer clips' : 'lower error on longer clips');
      document.getElementById('ap-corr').textContent = 'r = ' + r.toFixed(2) + ' · ' + strength;
      var fmin = Math.min.apply(null, fr), fmax = Math.max.apply(null, fr);
      var W = Math.max(260, scBox.clientWidth), H = 220, L = 34, R = W - 10, T = 10, B = H - 32;
      var x0 = Math.floor((fmin - 5) / 10) * 10, x1 = Math.ceil((fmax + 5) / 10) * 10;
      var ys = niceStep(sorted[N - 1], 4), ytop = niceTop(sorted[N - 1] * 1.05, ys);
      var px = function (v) { return L + ((v - x0) / (x1 - x0)) * (R - L); };
      var py = function (v) { return B - (v / ytop) * (B - T); };
      var svg = svgIn(scBox, W, H);
      var grid = el('g', { 'class': 'grid' }, svg), ticks = el('g', { 'class': 'tick' }, svg);
      for (var t = 0; t <= ytop + 1e-9; t += ys) { el('line', { x1: L, x2: R, y1: py(t), y2: py(t) }, grid); text(ticks, L - 6, py(t) + 4, ys < 1 ? t.toFixed(1) : Math.round(t), { 'text-anchor': 'end' }); }
      var xs = niceStep(x1 - x0, 5);
      for (var u = Math.ceil(x0 / xs) * xs; u <= x1; u += xs) text(ticks, px(u), B + 15, u, { 'text-anchor': 'middle' });
      el('line', { 'class': 'baseline', x1: L, x2: R, y1: B, y2: B }, svg);
      el('line', { x1: px(x0), x2: px(x1), y1: py(icpt + slope * x0), y2: py(icpt + slope * x1), style: 'stroke: var(--ink); stroke-width: 1; opacity: 0.35' }, svg);
      clips.forEach(function (c, ci) {
        var g = el('g', { tabindex: 0 }, svg);
        el('circle', { cx: px(fr[ci]), cy: py(vals[ci]), r: 11, fill: 'transparent' }, g);
        el('circle', { cx: px(fr[ci]), cy: py(vals[ci]), r: 4.5, style: 'fill:' + color + '; stroke: var(--surface); stroke-width: 2' }, g);
        hover(g, '<div class="tt-title">Clip ' + (ci + 1) + ' of ' + N + '</div><div class="tt-row"><span>Frames</span><b>' + fr[ci] + '</b></div><div class="tt-row"><span>Error</span><b>' + f2(vals[ci]) + ' mm/node</b></div>');
      });
      text(svg, R, H - 2, 'frames in the clip (4 fps)', { 'text-anchor': 'end' }, MUTED);
    }

    // ---- compute
    var cBox = document.getElementById('ap-compute');
    function drawCompute() {
      var cs = cfg.compute || [];
      if (!cs.length) { cBox.closest('.dash-card').hidden = true; return; }
      var total = cs.reduce(function (s, c) { return s + c.hours; }, 0);
      document.getElementById('ap-compute-total').textContent = total.toFixed(1) + ' h total';
      document.getElementById('ap-compute-sub').textContent = cfg.computeNote || '';
      var W = Math.max(260, cBox.clientWidth), L = 110, R = W - 54, rowH = 34, barH = 18, T = 4;
      var H = T + cs.length * rowH + 8;
      var vmax = Math.max.apply(null, cs.map(function (c) { return c.hours; }));
      var sx = function (v) { return L + (v / vmax) * (R - L); };
      var svg = svgIn(cBox, W, H);
      el('line', { 'class': 'baseline', x1: L, x2: L, y1: T, y2: T + cs.length * rowH }, svg);
      cs.forEach(function (c, i) {
        var y = T + i * rowH + (rowH - barH) / 2;
        var g = el('g', { 'class': 'bar-row', tabindex: 0 }, svg);
        text(g, L - 10, y + barH / 2 + 4, c.label, { 'class': 'row-label', 'text-anchor': 'end' });
        el('path', { 'class': 'bar', d: hbarPath(L, sx(c.hours), y, barH), style: 'fill:' + color }, g);
        text(g, sx(c.hours) + 7, y + barH / 2 + 4, c.hours.toFixed(1) + ' h', { 'class': 'value-label' });
        hover(g, '<div class="tt-title">' + esc(c.label) + '</div><div class="tt-row"><span>Wall clock</span><b>' + c.hours.toFixed(1) + ' h</b></div><div class="tt-row"><span>Share</span><b>' + (c.hours / total * 100).toFixed(0) + ' %</b></div>');
      });
    }

    // ---- spec sheet
    document.getElementById('ap-numbers').innerHTML = (cfg.numbers || []).map(function (n) {
      return '<div class="num-tile"><b>' + esc(n.value) + '</b><span>' + esc(n.label) + '</span></div>';
    }).join('');
    document.getElementById('ap-spec').innerHTML = (cfg.spec || []).map(function (r) {
      return '<div><dt>' + esc(r[0]) + '</dt><dd>' + esc(r[1]) + '</dd></div>';
    }).join('');

    function all() { drawClips(); drawH2H(); drawHist(); drawVariants(); drawScatter(); drawCompute(); }
    all();
    resizers.push(function () { drawClips(); drawHist(); drawVariants(); drawScatter(); drawCompute(); });
  }

  // ================================================================= simple per-clip chart (pages not converted yet)
  function legacy() {
    var box = document.getElementById('ap-perclip');
    if (!mine || !box || !N) return;
    box.hidden = false;
    var vals = clips.map(function (c) { return mine.values[c]; });
    var sorted = vals.slice().sort(function (a, b) { return a - b; });
    var mean = vals.reduce(function (s, v) { return s + v; }, 0) / N;
    var median = quantile(sorted, 0.5);
    var best = vals.indexOf(sorted[0]), worst = vals.indexOf(sorted[N - 1]);
    document.getElementById('ap-perclip-stats').innerHTML =
      '<div class="stat"><span class="stat-value">' + mean.toFixed(2) + '</span><span class="stat-label">mean mm/node</span></div>' +
      '<div class="stat"><span class="stat-value">' + median.toFixed(2) + '</span><span class="stat-label">median</span></div>' +
      '<div class="stat"><span class="stat-value">' + sorted[0].toFixed(2) + '</span><span class="stat-label">best: clip ' + (best + 1) + '</span></div>' +
      '<div class="stat"><span class="stat-value">' + sorted[N - 1].toFixed(2) + '</span><span class="stat-label">worst: clip ' + (worst + 1) + '</span></div>' +
      (mine.interim ? '<p class="muted-note ap-note">Generation 0 of the running 36-clip fit (the warm start); the fit is still improving.</p>' : '');
    var chartBox = document.getElementById('ap-perclip-chart');
    function render() {
      var W = Math.max(300, chartBox.clientWidth), H = 260, L = 40, R = W - 8, T = 10, B = H - 34;
      var step = niceStep(sorted[N - 1], 5), top = niceTop(sorted[N - 1] * 1.05, step);
      var sy = function (v) { return B - (v / top) * (B - T); };
      var band = (R - L) / N, bw = Math.min(24, band - 2);
      var svg = svgIn(chartBox, W, H);
      var grid = el('g', { 'class': 'grid' }, svg), ticks = el('g', { 'class': 'tick' }, svg);
      for (var t = 0; t <= top + 1e-9; t += step) { el('line', { x1: L, x2: R, y1: sy(t), y2: sy(t) }, grid); text(ticks, L - 8, sy(t) + 4, Math.round(t * 10) / 10, { 'text-anchor': 'end' }); }
      el('line', { 'class': 'baseline', x1: L, x2: R, y1: B, y2: B }, svg);
      clips.forEach(function (c, i) {
        var v = vals[i], x = L + i * band + (band - bw) / 2;
        var g = el('g', { 'class': 'bar-row', tabindex: 0 }, svg);
        el('rect', { x: L + i * band, y: T, width: band, height: B - T, fill: 'transparent' }, g);
        el('path', { 'class': 'bar', d: vbarPath(x, bw, sy(v), B), style: 'fill:' + color }, g);
        if ((i + 1) % 5 === 0 || i === 0) text(ticks, L + i * band + band / 2, B + 16, i + 1, { 'text-anchor': 'middle' });
        hover(g, '<div class="tt-title">Clip ' + (i + 1) + ' of ' + N + '</div><div class="tt-row"><span>Error</span><b>' + v.toFixed(2) + ' mm/node</b></div><div class="tt-row"><span>Frames</span><b>' + D.frames[c] + '</b></div>');
      });
      el('line', { x1: L, x2: R, y1: sy(mean), y2: sy(mean), style: 'stroke: var(--ink); stroke-width: 1.5; opacity: 0.55' }, svg);
      text(svg, R, sy(mean) - 6, 'mean ' + mean.toFixed(2), { 'text-anchor': 'end' }, 'fill: var(--ink); font-size: 11px; font-weight: 600; paint-order: stroke; stroke: var(--surface); stroke-width: 4px; stroke-linejoin: round');
      text(svg, R, H - 4, 'clip, in recording order', { 'text-anchor': 'end' }, MUTED);
    }
    render();
    resizers.push(render);
  }
})();
