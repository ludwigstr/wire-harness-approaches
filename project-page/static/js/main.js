/* Project page interactions: theme, nav, pipeline stepper, approach tabs, charts, per-clip explorer. */
(function () {
  'use strict';

  var D = window.BENCH_DATA || { clips: [], frames: {}, perClip: {}, meta: {}, stills: {} };
  var FPS = 4;
  var SVGNS = 'http://www.w3.org/2000/svg';

  var PARADIGM = {
    world:    { label: 'World Model',    color: 'var(--series-world)' },
    dynamics: { label: 'Dynamics Model', color: 'var(--series-dynamics)' },
    physics:  { label: 'Physics Engine', color: 'var(--series-physics)' }
  };

  function med(key) { return D.meta[key] ? D.meta[key].median : null; }

  // Scores shown on the page. Physics-engine values are interim (fits running).
  var APPROACHES = [
    { key: 'physworld',   name: 'PhysWorld',       paradigm: 'world',    score: 4.5,  median: med('physworld'),   status: 'final',   clips: 36, scored: '16 nodes, frames 5…T−1' },
    { key: 'newton',      name: 'Newton',          paradigm: 'physics',  score: 8.1,  median: null,               status: 'running', gen: 12,   clips: 36, scored: '16 nodes, every frame' },
    { key: 'mujoco',      name: 'MuJoCo',          paradigm: 'physics',  score: 10.9, median: null,               status: 'running', gen: 20,   clips: 36, scored: '16 nodes, every frame' },
    { key: 'genesis',     name: 'Genesis',         paradigm: 'physics',  score: 12.7, median: null,               status: 'running', gen: 29,   clips: 36, scored: '16 nodes, every frame' },
    { key: 'yang',        name: 'Yang et al.',     paradigm: 'dynamics', score: 13.3, median: 11.68,              status: 'partial', clips: 14,   scored: '16 nodes, frames 5…T−1' },
    { key: 'caporali',    name: 'Caporali et al.', paradigm: 'dynamics', score: 15.8, median: med('caporali'),    status: 'final',   clips: 36, scored: '16 nodes, every frame incl. 0' },
    { key: 'dlolab',      name: 'DLO-Lab',         paradigm: 'physics',  score: 17.0, median: null,               status: 'running', gen: 19,   clips: 36, scored: '16 nodes, every frame' },
    { key: 'ropedreamer', name: 'RopeDreamer',     paradigm: 'world',    score: 18.3, median: 13.00,              status: 'partial', clips: 14,   scored: '16 nodes, frames 5…T−1' },
    { key: 'phystwin',    name: 'PhysTwin',        paradigm: 'world',    score: 22.0, median: med('phystwin'),    status: 'final',   clips: 36, scored: '14 inner nodes, every frame' },
    { key: 'adaptigraph', name: 'AdaptiGraph',     paradigm: 'dynamics', score: 29.4, median: med('adaptigraph'), status: 'final',   clips: 36, scored: '16 nodes, frames 5…T−1' }
  ];
  APPROACHES.forEach(function (a, i) { a.rank = i + 1; });
  var BY_KEY = {};
  APPROACHES.forEach(function (a) { BY_KEY[a.key] = a; });

  // The four approaches with finished 36-clip refits and videos, ordered by mean.
  var VIDEO_KEYS = ['physworld', 'caporali', 'phystwin', 'adaptigraph'].filter(function (k) { return D.perClip[k]; });

  function statusText(a) {
    if (a.status === 'final') return 'Final';
    if (a.status === 'running') return 'Running (gen ' + a.gen + ')';
    return '14-clip result';
  }
  function statusChip(a) {
    var icon = a.status === 'final' ? 'fa-check' : a.status === 'running' ? 'fa-rotate' : 'fa-circle-info';
    return '<span class="status-chip ' + a.status + '"><i class="fas ' + icon + '"></i>' + statusText(a) + '</span>';
  }
  function fmt(v, d) { return v == null ? '–' : Number(v).toFixed(d == null ? 1 : d); }
  function el(tag, attrs, parent) {
    var n = document.createElementNS(SVGNS, tag);
    for (var k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function clipLabel(c) {
    var m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/.exec(c);
    if (!m) return c;
    var mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][+m[2] - 1];
    return { date: (+m[3]) + ' ' + mon + ' ' + m[1], time: m[4] + ':' + m[5] + ':' + m[6] };
  }
  function niceMax(v, step) { return Math.ceil(v / step) * step; }
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------------------------------------------------------------- tooltip
  var tip = document.getElementById('tooltip');
  function showTip(html, x, y) {
    tip.innerHTML = html;
    tip.hidden = false;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = Math.min(window.innerWidth - w - 8, Math.max(8, x + 14));
    var top = y - h - 12;
    if (top < 8) top = y + 18;
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }
  function hideTip() { tip.hidden = true; }
  function tipAt(target, html) {
    var r = target.getBoundingClientRect();
    showTip(html, r.left + r.width / 2, r.top);
  }

  // ---------------------------------------------------------------- theme
  var themeBtn = document.getElementById('theme-toggle');
  function effectiveTheme() {
    // Light (white) by default; dark only when the viewer picks it with the toggle.
    return document.documentElement.getAttribute('data-theme') || 'light';
  }
  function syncThemeIcon() {
    themeBtn.innerHTML = effectiveTheme() === 'dark' ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
  }
  themeBtn.addEventListener('click', function () {
    var next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('dlo-theme-v2', next); } catch (e) {}
    syncThemeIcon();
  });
  syncThemeIcon();

  // ---------------------------------------------------------------- nav
  var burger = document.getElementById('nav-burger');
  var links = document.getElementById('nav-links');
  burger.addEventListener('click', function () {
    var open = links.classList.toggle('is-open');
    burger.setAttribute('aria-expanded', open ? 'true' : 'false');
  });
  links.addEventListener('click', function (e) {
    if (e.target.tagName === 'A') { links.classList.remove('is-open'); burger.setAttribute('aria-expanded', 'false'); }
  });
  if ('IntersectionObserver' in window) {
    var navMap = {};
    links.querySelectorAll('a').forEach(function (a) { navMap[a.getAttribute('href').slice(1)] = a; });
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting && navMap[en.target.id]) {
          links.querySelectorAll('a').forEach(function (a) { a.classList.remove('is-current'); });
          navMap[en.target.id].classList.add('is-current');
        }
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    Object.keys(navMap).forEach(function (id) { var s = document.getElementById(id); if (s) spy.observe(s); });
  }

  // ---------------------------------------------------------------- hero numbers
  if (D.clips.length) {
    var fr = D.clips.map(function (c) { return D.frames[c]; });
    var total = fr.reduce(function (s, v) { return s + v; }, 0);
    document.getElementById('stat-frames').textContent = total.toLocaleString('en-US');
    document.getElementById('frames-range').textContent = Math.min.apply(null, fr) + '–' + Math.max.apply(null, fr);
  }

  // ---------------------------------------------------------------- teaser
  var teaser = document.getElementById('teaser-video');
  if (teaser && !reduceMotion) {
    var p = teaser.play();
    if (p && p.catch) p.catch(function () {});
  } else if (teaser) {
    teaser.controls = true;
  }

  // ---------------------------------------------------------------- pipeline stepper
  var steps = document.querySelectorAll('.stepper .step');
  var stepPanels = document.querySelectorAll('.step-panel');
  steps.forEach(function (btn) {
    btn.addEventListener('click', function () {
      var i = btn.getAttribute('data-step');
      steps.forEach(function (b) {
        var on = b === btn;
        b.classList.toggle('is-active', on);
        b.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      stepPanels.forEach(function (p) {
        var on = p.getAttribute('data-step') === i;
        p.classList.toggle('is-active', on);
        p.hidden = !on;
      });
    });
  });

  // ---------------------------------------------------------------- category tabs + accordions
  function showCategory(cat) {
    document.querySelectorAll('.category-tab').forEach(function (tab) {
      var on = tab.getAttribute('data-cat') === cat;
      tab.classList.toggle('is-active', on);
      tab.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    document.querySelectorAll('.category-panel').forEach(function (panel) {
      var on = panel.id === 'panel-' + cat;
      panel.classList.toggle('is-active', on);
      panel.hidden = !on;
    });
  }
  document.querySelectorAll('.category-tab').forEach(function (tab) {
    tab.addEventListener('click', function () { showCategory(tab.getAttribute('data-cat')); });
  });

  function setOpen(detail, open) {
    detail.classList.toggle('is-open', open);
    detail.querySelector('.approach-toggle').setAttribute('aria-expanded', open ? 'true' : 'false');
  }
  document.querySelectorAll('.approach-detail').forEach(function (d) {
    d.querySelector('.approach-toggle').addEventListener('click', function () {
      setOpen(d, !d.classList.contains('is-open'));
    });
  });
  document.querySelectorAll('.expand-all').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var panel = btn.closest('.category-panel');
      var details = panel.querySelectorAll('.approach-detail');
      var anyClosed = Array.prototype.some.call(details, function (d) { return !d.classList.contains('is-open'); });
      details.forEach(function (d) { setOpen(d, anyClosed); });
      btn.textContent = anyClosed ? 'Collapse all' : 'Expand all';
    });
  });

  // Open an approach from anywhere (chart click, table click).
  function openApproach(key) {
    var a = BY_KEY[key];
    if (!a) return;
    showCategory(a.paradigm);
    var d = document.getElementById('detail-' + key);
    if (d) { setOpen(d, true); d.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' }); }
  }

  // ---------------------------------------------------------------- results: filter state
  var filter = { paradigm: 'all', interim: true };
  function visible(a) {
    if (filter.paradigm !== 'all' && a.paradigm !== filter.paradigm) return false;
    if (!filter.interim && a.status === 'running') return false;
    return true;
  }
  document.querySelectorAll('.filter-row .chip-btn').forEach(function (b) {
    b.addEventListener('click', function () {
      filter.paradigm = b.getAttribute('data-filter');
      document.querySelectorAll('.filter-row .chip-btn').forEach(function (x) { x.classList.toggle('is-active', x === b); });
      renderRank();
    });
  });
  document.getElementById('show-interim').addEventListener('change', function (e) {
    filter.interim = e.target.checked; renderRank();
  });

  function approachTip(a) {
    var p = PARADIGM[a.paradigm];
    return '<div class="tt-title"><span class="swatch" style="--sw:' + p.color + '"></span>' + esc(a.name) + '</div>' +
      '<div class="tt-row"><span>Error</span><b>' + fmt(a.score) + ' mm/node</b></div>' +
      '<div class="tt-row"><span>Paradigm</span><b>' + p.label + '</b></div>' +
      '<div class="tt-row"><span>Clips</span><b>' + a.clips + '</b></div>';
  }

  // ---------------------------------------------------------------- results: ranked bar chart
  var rankBox = document.getElementById('rank-chart');
  function renderRank() {
    var rows = APPROACHES.filter(visible);
    var W = Math.max(300, rankBox.clientWidth);
    var narrow = W < 520;
    var labelW = narrow ? 104 : 132, valueW = narrow ? 76 : 96;
    var rowH = 34, barH = 18, top = 6, axisH = 30;
    var H = top + rows.length * rowH + axisH;
    var x0 = labelW, x1 = W - valueW;
    var xMax = niceMax(Math.max.apply(null, APPROACHES.map(function (a) { return a.score; })), 5);
    var sx = function (v) { return x0 + (v / xMax) * (x1 - x0); };

    rankBox.innerHTML = '';
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H }, rankBox);
    var grid = el('g', { 'class': 'grid' }, svg);
    var ticks = el('g', { 'class': 'tick' }, svg);
    var plotBottom = top + rows.length * rowH;
    for (var t = 0; t <= xMax; t += 5) {
      el('line', { x1: sx(t), x2: sx(t), y1: top, y2: plotBottom }, grid);
      var tx = el('text', { x: sx(t), y: plotBottom + 16, 'text-anchor': 'middle' }, ticks);
      tx.textContent = t;
    }
    el('line', { 'class': 'baseline', x1: x0, x2: x0, y1: top, y2: plotBottom }, svg);
    var ax = el('text', { x: x1, y: plotBottom + 29, 'text-anchor': 'end' }, svg);
    ax.textContent = 'mm/node, lower is better';
    ax.setAttribute('style', 'fill: var(--muted); font-size: 11px');

    if (!rows.length) {
      var none = el('text', { x: (x0 + x1) / 2, y: top + 20, 'text-anchor': 'middle' }, svg);
      none.textContent = 'No approach matches the filter';
    }

    rows.forEach(function (a, i) {
      var y = top + i * rowH + (rowH - barH) / 2;
      var g = el('g', { 'class': 'bar-row', tabindex: 0, role: 'button', 'aria-label': a.name + ', ' + fmt(a.score) + ' mm/node, ' + statusText(a) }, svg);
      el('rect', { 'class': 'hit', x: 0, y: top + i * rowH, width: W, height: rowH, fill: 'transparent' }, g);
      var lab = el('text', { 'class': 'row-label', x: x0 - 10, y: y + barH / 2 + 4, 'text-anchor': 'end' }, g);
      lab.textContent = a.name;
      var bx = sx(a.score), r = 4, h = barH;
      var d = 'M' + x0 + ',' + y + ' H' + (bx - r) + ' Q' + bx + ',' + y + ' ' + bx + ',' + (y + r) +
              ' V' + (y + h - r) + ' Q' + bx + ',' + (y + h) + ' ' + (bx - r) + ',' + (y + h) + ' H' + x0 + ' Z';
      var bar = el('path', { 'class': 'bar', d: d }, g);
      bar.setAttribute('style', 'fill:' + PARADIGM[a.paradigm].color + (a.status === 'running' ? '; opacity: 0.42' : ''));
      var v = el('text', { 'class': 'value-label', x: bx + 7, y: y + barH / 2 + 4 }, g);
      v.textContent = fmt(a.score);
      if (a.status !== 'final') {
        var s = el('tspan', { dx: 5 }, v);
        s.textContent = a.status === 'running' ? 'interim' : '14 clips';
        s.setAttribute('style', 'fill: var(--muted); font-weight: 400; font-size: 11px');
      }
      g.addEventListener('mousemove', function (e) { showTip(approachTip(a), e.clientX, e.clientY); });
      g.addEventListener('mouseleave', hideTip);
      g.addEventListener('focus', function () { tipAt(bar, approachTip(a)); });
      g.addEventListener('blur', hideTip);
      g.addEventListener('click', function () { hideTip(); openApproach(a.key); });
      g.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); hideTip(); openApproach(a.key); } });
    });
  }

  // ---------------------------------------------------------------- per-clip explorer (clip x error, 2-D zoom)
  var stripBox = document.getElementById('strip-chart');
  var legendBox = document.getElementById('strip-legend');
  var N = D.clips.length;

  // Series with real per-clip numbers. Colour = paradigm (as everywhere on the page), shape = model.
  var SERIES = [];
  function addSeries(key, label, values) {
    if (values && BY_KEY[key]) SERIES.push({ key: key, label: label, paradigm: BY_KEY[key].paradigm, values: values });
  }
  addSeries('physworld', 'PhysWorld', D.perClip.physworld);
  addSeries('phystwin', 'PhysTwin', D.perClip.phystwin);
  addSeries('caporali', 'Caporali et al.', D.perClip.caporali);
  addSeries('adaptigraph', 'AdaptiGraph', D.perClip.adaptigraph);
  addSeries('newton', 'Newton (gen 0, interim)', D.newtonGen0 && D.newtonGen0.clips);
  var SHAPES = ['circle', 'square', 'triangle'];
  var shapeIx = {};
  SERIES.forEach(function (s) { var i = shapeIx[s.paradigm] || 0; s.shape = SHAPES[i % SHAPES.length]; shapeIx[s.paradigm] = i + 1; });
  var SERIES_BY_KEY = {};
  SERIES.forEach(function (s) { SERIES_BY_KEY[s.key] = s; });

  function marker(shape, parent) {
    if (shape === 'square') return el('rect', { x: -4, y: -4, width: 8, height: 8, rx: 1.5 }, parent);
    if (shape === 'triangle') return el('path', { d: 'M0,-5.4 L5,3.4 L-5,3.4 Z' }, parent);
    return el('circle', { r: 4.6 }, parent);
  }
  function markerIcon(s) {
    var svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('viewBox', '-7 -7 14 14'); svg.setAttribute('width', 14); svg.setAttribute('height', 14);
    svg.setAttribute('aria-hidden', 'true');
    var m = marker(s.shape, svg);
    m.setAttribute('style', 'fill:' + PARADIGM[s.paradigm].color);
    return svg.outerHTML;
  }

  // ---- method picker: "All methods" + one menu per paradigm
  var pick = { paradigm: 'all', key: null };
  var pickAll = document.getElementById('pc-all');
  var pickers = { world: document.getElementById('pc-world'), dynamics: document.getElementById('pc-dynamics'), physics: document.getElementById('pc-physics') };
  var MENU_TITLE = { world: 'World Models', dynamics: 'Dynamics Models', physics: 'Physics Engines' };
  var MENU_ALL = { world: 'All world models', dynamics: 'All dynamics models', physics: 'All physics engines' };
  Object.keys(pickers).forEach(function (par) {
    var sel = pickers[par];
    if (!sel) return;
    var html = '<option value="">' + MENU_TITLE[par] + '</option><option value="all">' + MENU_ALL[par] + '</option>';
    APPROACHES.filter(function (a) { return a.paradigm === par; }).forEach(function (a) {
      var s = SERIES_BY_KEY[a.key];
      html += s ? '<option value="' + a.key + '">' + esc(s.label) + '</option>'
                : '<option value="' + a.key + '" disabled>' + esc(a.name) + ' (no per-clip data yet)</option>';
    });
    sel.innerHTML = html;
    sel.addEventListener('change', function () {
      if (!sel.value) return;
      pick = { paradigm: par, key: sel.value === 'all' ? null : sel.value };
      syncPicker();
      resetZoom();
    });
  });
  if (pickAll) pickAll.addEventListener('click', function () { pick = { paradigm: 'all', key: null }; syncPicker(); resetZoom(); });
  function syncPicker() {
    if (pickAll) pickAll.classList.toggle('is-active', pick.paradigm === 'all');
    Object.keys(pickers).forEach(function (par) {
      var sel = pickers[par];
      if (!sel) return;
      var on = pick.paradigm === par;
      sel.parentNode.classList.toggle('is-active', on);
      sel.value = on ? (pick.key || 'all') : '';
    });
  }
  function activeSeries() {
    return SERIES.filter(function (s) {
      if (pick.paradigm === 'all') return true;
      return s.paradigm === pick.paradigm && (!pick.key || s.key === pick.key);
    });
  }

  // ---- zoom state: x = clip index (1..N), y = mm/node
  var state = { clip: D.clips.indexOf('20261021_164755') >= 0 ? '20261021_164755' : D.clips[0], hover: null };
  var mode = 'box';
  function fullDomain() {
    var vmax = 0;
    activeSeries().forEach(function (s) { D.clips.forEach(function (c) { if (s.values[c] > vmax) vmax = s.values[c]; }); });
    vmax = vmax || 10;
    return { x0: 0.5, x1: N + 0.5, y0: 0, y1: niceMax(vmax * 1.04, vmax > 40 ? 10 : vmax > 15 ? 5 : 2) };
  }
  var FULL = fullDomain();
  var zoom = { x0: FULL.x0, x1: FULL.x1, y0: FULL.y0, y1: FULL.y1 };
  var MIN_X = 2, MIN_Y = 0.5;
  function clampAxis(a0, a1, f0, f1, minSpan) {
    var span = a1 - a0;
    if (span < minSpan) { var c = (a0 + a1) / 2; a0 = c - minSpan / 2; a1 = c + minSpan / 2; span = minSpan; }
    if (span >= f1 - f0) return [f0, f1];
    if (a0 < f0) { a1 += f0 - a0; a0 = f0; }
    if (a1 > f1) { a0 -= a1 - f1; a1 = f1; }
    return [a0, a1];
  }
  function setZoom(x0, x1, y0, y1) {
    var x = clampAxis(x0, x1, FULL.x0, FULL.x1, MIN_X), y = clampAxis(y0, y1, FULL.y0, FULL.y1, MIN_Y);
    zoom = { x0: x[0], x1: x[1], y0: y[0], y1: y[1] };
    hideTip();
    renderStrip();
  }
  function resetZoom() { FULL = fullDomain(); setZoom(FULL.x0, FULL.x1, FULL.y0, FULL.y1); }
  function isZoomed() { return zoom.x0 > FULL.x0 + 1e-9 || zoom.x1 < FULL.x1 - 1e-9 || zoom.y0 > FULL.y0 + 1e-9 || zoom.y1 < FULL.y1 - 1e-9; }
  function zoomBy(f, cx, cy) {
    if (cx == null) cx = (zoom.x0 + zoom.x1) / 2;
    if (cy == null) cy = (zoom.y0 + zoom.y1) / 2;
    setZoom(cx - (cx - zoom.x0) * f, cx + (zoom.x1 - cx) * f, cy - (cy - zoom.y0) * f, cy + (zoom.y1 - cy) * f);
  }
  function niceStep(span, target) {
    var raw = span / (target || 6), p = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10)), m = raw / p;
    return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  }

  var zb = {
    box: document.getElementById('mode-box'), pan: document.getElementById('mode-pan'),
    inn: document.getElementById('zoom-in'), out: document.getElementById('zoom-out'),
    reset: document.getElementById('zoom-reset'), range: document.getElementById('zoom-range')
  };
  function setMode(m) {
    mode = m;
    zb.box.classList.toggle('is-on', m === 'box'); zb.box.setAttribute('aria-pressed', m === 'box' ? 'true' : 'false');
    zb.pan.classList.toggle('is-on', m === 'pan'); zb.pan.setAttribute('aria-pressed', m === 'pan' ? 'true' : 'false');
    stripBox.classList.toggle('pan-mode', m === 'pan');
  }
  if (zb.box) {
    zb.box.addEventListener('click', function () { setMode('box'); });
    zb.pan.addEventListener('click', function () { setMode('pan'); });
    zb.inn.addEventListener('click', function () { zoomBy(1 / 1.6); });
    zb.out.addEventListener('click', function () { zoomBy(1.6); });
    zb.reset.addEventListener('click', resetZoom);
  }
  function syncZoomControls() {
    if (!zb.inn) return;
    var z = isZoomed();
    zb.out.disabled = !z;
    zb.reset.disabled = !z;
    zb.inn.disabled = (zoom.x1 - zoom.x0) <= MIN_X + 1e-9 && (zoom.y1 - zoom.y0) <= MIN_Y + 1e-9;
    var c0 = Math.max(1, Math.ceil(zoom.x0)), c1 = Math.min(N, Math.floor(zoom.x1));
    var d = zoom.y1 - zoom.y0 < 10 ? 1 : 0;
    zb.range.textContent = 'Clips ' + c0 + '–' + c1 + ' · ' + zoom.y0.toFixed(d) + '–' + zoom.y1.toFixed(d) + ' mm/node';
  }

  function renderLegend() {
    if (!legendBox) return;
    legendBox.innerHTML = activeSeries().map(function (s) {
      return '<span>' + markerIcon(s) + esc(s.label) + '</span>';
    }).join('');
  }

  function clipTip(c) {
    var L = clipLabel(c);
    var html = '<div class="tt-title">Clip ' + (D.clips.indexOf(c) + 1) + ' &middot; ' + L.date + ', ' + L.time + '</div>' +
      '<div class="tt-row"><span>Frames</span><b>' + D.frames[c] + '</b></div>';
    activeSeries().forEach(function (s) {
      html += '<div class="tt-row"><span><span class="dot" style="--sw:' + PARADIGM[s.paradigm].color + '"></span>' + esc(s.label) + '</span><b>' + fmt(s.values[c], 2) + '</b></div>';
    });
    return html + '<div class="tt-row"><span></span><span>Click to pin</span></div>';
  }

  // Ctrl + wheel / pinch zooms both axes at the cursor.
  stripBox.addEventListener('wheel', function (e) {
    if (!(e.ctrlKey || e.metaKey) || !stripBox._geo) return;
    var g = stripBox._geo, p = g.toPt(e);
    if (p.x < g.L || p.x > g.R || p.y < g.T || p.y > g.B) return;
    e.preventDefault();
    zoomBy(e.deltaY < 0 ? 1 / 1.25 : 1.25, g.invX(p.x), g.invY(p.y));
  }, { passive: false });

  var pts = [];
  function renderStrip() {
    if (!SERIES.length) { stripBox.innerHTML = '<p class="muted-note">No per-clip data found.</p>'; return; }
    renderLegend();
    var W = Math.max(300, stripBox.clientWidth);
    var narrow = W < 520;
    var L = narrow ? 40 : 48, R = W - 14, T = 10, H = narrow ? 300 : 380, B = H - 44;
    var z = zoom;
    var sx = function (i) { return L + ((i - z.x0) / (z.x1 - z.x0)) * (R - L); };
    var sy = function (v) { return B - ((v - z.y0) / (z.y1 - z.y0)) * (B - T); };
    var invX = function (px) { return z.x0 + ((px - L) / (R - L)) * (z.x1 - z.x0); };
    var invY = function (py) { return z.y0 + ((B - py) / (B - T)) * (z.y1 - z.y0); };

    stripBox.innerHTML = '';
    pts = [];
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H }, stripBox);
    var toPt = function (e) { var r = svg.getBoundingClientRect(); return { x: (e.clientX - r.left) * (W / r.width), y: (e.clientY - r.top) * (H / r.height) }; };
    var defs = el('defs', {}, svg);
    var cp = el('clipPath', { id: 'pc-clip' }, defs);
    el('rect', { x: L, y: T - 6, width: R - L, height: B - T + 12 }, cp);

    // y grid + labels
    var grid = el('g', { 'class': 'grid' }, svg), ticks = el('g', { 'class': 'tick' }, svg);
    var ys = niceStep(z.y1 - z.y0, narrow ? 4 : 6);
    for (var v = Math.ceil(z.y0 / ys - 1e-9) * ys; v <= z.y1 + 1e-9; v += ys) {
      el('line', { x1: L, x2: R, y1: sy(v), y2: sy(v) }, grid);
      var ty = el('text', { x: L - 8, y: sy(v) + 4, 'text-anchor': 'end' }, ticks);
      ty.textContent = ys < 1 ? v.toFixed(1) : Math.round(v);
    }
    // x ticks: clip numbers
    var xs = Math.max(1, Math.round(niceStep(z.x1 - z.x0, narrow ? 5 : 9)));
    for (var i = Math.max(1, Math.ceil(z.x0 / xs) * xs); i <= Math.min(N, z.x1); i += xs) {
      el('line', { x1: sx(i), x2: sx(i), y1: B, y2: B + 4, style: 'stroke: var(--axis)' }, svg);
      var tx = el('text', { x: sx(i), y: B + 17, 'text-anchor': 'middle' }, ticks);
      tx.textContent = i;
    }
    el('line', { 'class': 'baseline', x1: L, x2: R, y1: B, y2: B }, svg);
    var xl = el('text', { x: R, y: H - 6, 'text-anchor': 'end' }, svg);
    xl.textContent = 'clip, in recording order';
    xl.setAttribute('style', 'fill: var(--muted); font-size: 11px');
    var yl = el('text', { x: 4, y: T + 2, 'text-anchor': 'start', transform: 'rotate(-90 4 ' + (T + 2) + ')' }, svg);
    yl.setAttribute('style', 'fill: var(--muted); font-size: 11px');
    yl.setAttribute('x', -(B - 4)); yl.setAttribute('y', 12); yl.setAttribute('transform', 'rotate(-90)');
    yl.setAttribute('text-anchor', 'start');
    yl.textContent = 'mm/node';

    var plot = el('g', { 'clip-path': 'url(#pc-clip)' }, svg);
    var guide = el('g', {}, plot);
    var series = activeSeries();
    series.forEach(function (s) {
      D.clips.forEach(function (c, ci) {
        var val = s.values[c];
        if (val == null) return;
        var idx = ci + 1;
        if (idx < z.x0 - 0.5 || idx > z.x1 + 0.5 || val < z.y0 - (z.y1 - z.y0) * 0.05 || val > z.y1 + (z.y1 - z.y0) * 0.05) return;
        var x = sx(idx), y = sy(val);
        var m = marker(s.shape, plot);
        pts.push({ clip: c, s: s, m: m, x: x, y: y });
      });
    });

    // interaction layer on top: nearest-point hover, box zoom / pan, click to pin
    var area = el('rect', { 'class': 'zoom-area', x: L, y: T, width: R - L, height: B - T, fill: 'transparent' }, svg);
    var drag = null;
    function nearest(p) {
      var best = null, bd = 22 * 22;
      pts.forEach(function (q) { var dx = q.x - p.x, dy = q.y - p.y, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = q; } });
      return best;
    }
    area.addEventListener('pointerdown', function (e) {
      if (e.button !== 0) return;
      var p = toPt(e);
      drag = { x: p.x, y: p.y, moved: false, z: { x0: z.x0, x1: z.x1, y0: z.y0, y1: z.y1 } };
      if (mode === 'box') drag.rect = el('rect', { 'class': 'brush', x: p.x, y: p.y, width: 0, height: 0 }, svg);
      area.setPointerCapture(e.pointerId);
    });
    area.addEventListener('pointermove', function (e) {
      var p = toPt(e);
      if (drag) {
        if (Math.abs(p.x - drag.x) > 4 || Math.abs(p.y - drag.y) > 4) { drag.moved = true; hideTip(); }
        if (mode === 'box' && drag.rect) {
          var cx = Math.max(L, Math.min(R, p.x)), cy = Math.max(T, Math.min(B, p.y));
          drag.rect.setAttribute('x', Math.min(cx, drag.x)); drag.rect.setAttribute('width', Math.abs(cx - drag.x));
          drag.rect.setAttribute('y', Math.min(cy, drag.y)); drag.rect.setAttribute('height', Math.abs(cy - drag.y));
        } else if (mode === 'pan' && drag.moved) {
          var dx = (p.x - drag.x) / (R - L) * (drag.z.x1 - drag.z.x0), dy = (p.y - drag.y) / (B - T) * (drag.z.y1 - drag.z.y0);
          var d0 = drag;
          if (!d0.raf) d0.raf = requestAnimationFrame(function () {
            d0.raf = 0;
            setZoom(d0.z.x0 - dx, d0.z.x1 - dx, d0.z.y0 + dy, d0.z.y1 + dy);
          });
        }
        return;
      }
      var q = nearest(p);
      var c = q ? q.clip : null;
      if (c !== state.hover) { state.hover = c; paintStrip(); }
      if (c) showTip(clipTip(c), e.clientX, e.clientY); else hideTip();
    });
    area.addEventListener('pointerup', function (e) {
      if (!drag) return;
      var d = drag; drag = null;
      var p = toPt(e);
      if (d.rect && d.rect.parentNode) d.rect.parentNode.removeChild(d.rect);
      if (!d.moved) { var q = nearest(p); if (q) selectClip(q.clip); return; }
      if (mode === 'box') {
        var ax = Math.max(L, Math.min(R, p.x)), ay = Math.max(T, Math.min(B, p.y));
        var w = Math.abs(ax - d.x), h = Math.abs(ay - d.y);
        if (w < 6 && h < 6) return;
        var nx0 = w >= 6 ? invX(Math.min(ax, d.x)) : z.x0, nx1 = w >= 6 ? invX(Math.max(ax, d.x)) : z.x1;
        var ny0 = h >= 6 ? invY(Math.max(ay, d.y)) : z.y0, ny1 = h >= 6 ? invY(Math.min(ay, d.y)) : z.y1;
        setZoom(nx0, nx1, ny0, ny1);
      }
    });
    area.addEventListener('pointercancel', function () { if (drag && drag.rect && drag.rect.parentNode) drag.rect.parentNode.removeChild(drag.rect); drag = null; });
    area.addEventListener('pointerleave', function () { if (!drag && state.hover) { state.hover = null; paintStrip(); hideTip(); } });
    area.addEventListener('dblclick', resetZoom);

    stripBox._guide = guide;
    stripBox._geo = { L: L, R: R, T: T, B: B, toPt: toPt, invX: invX, invY: invY, sx: sx };
    syncZoomControls();
    paintStrip();
  }

  function paintStrip() {
    var focus = state.hover || state.clip;
    var g = stripBox._geo, guide = stripBox._guide;
    if (guide) {
      guide.innerHTML = '';
      var fi = D.clips.indexOf(focus) + 1;
      if (g && fi > 0) el('line', { 'class': 'clip-guide', x1: g.sx(fi), x2: g.sx(fi), y1: g.T, y2: g.B }, guide);
    }
    pts.forEach(function (q) {
      var on = q.clip === focus;
      q.m.setAttribute('transform', 'translate(' + q.x + ' ' + q.y + ')' + (on ? ' scale(1.45)' : ''));
      q.m.setAttribute('style', 'fill:' + PARADIGM[q.s.paradigm].color + '; stroke:' + (on ? 'var(--ink)' : 'var(--surface)') +
        '; stroke-width: 1.6; vector-effect: non-scaling-stroke' + (state.hover && !on ? '; opacity: 0.35' : ''));
    });
    // keep the focused clip's markers on top
    pts.forEach(function (q) { if (q.clip === focus && q.m.parentNode) q.m.parentNode.appendChild(q.m); });
  }

  // ---------------------------------------------------------------- clip card
  var clipCard = document.getElementById('clip-card');
  function rankOf(s, clip) {
    var vals = D.clips.map(function (c) { return s.values[c]; }).sort(function (a, b) { return a - b; });
    return vals.indexOf(s.values[clip]) + 1;
  }
  function renderClipCard() {
    var c = state.clip;
    if (!c) { clipCard.hidden = true; return; }
    var L = clipLabel(c);
    var series = activeSeries();
    var vmax = Math.max.apply(null, series.map(function (s) { return s.values[c]; }).concat([10]));
    var bars = series.map(function (s) {
      var v = s.values[c], col = PARADIGM[s.paradigm].color;
      return '<span class="cb-name">' + markerIcon(s) + esc(s.label) + '</span>' +
        '<span class="cb-track"><span class="cb-fill" style="display:block;width:' + (v / vmax * 100).toFixed(1) + '%;background:' + col + '"></span></span>' +
        '<span class="cb-val">' + fmt(v, 2) + '<span class="cb-rank">#' + rankOf(s, c) + '/' + N + '</span></span>';
    }).join('');
    clipCard.innerHTML =
      '<div><h4>Clip ' + (D.clips.indexOf(c) + 1) + ' &middot; ' + L.date + ', ' + L.time + '</h4>' +
      '<p class="clip-meta"><code>' + c + '</code><br>' + D.frames[c] + ' frames &middot; ' + (D.frames[c] / FPS).toFixed(1) + ' s at 4 fps</p></div>' +
      '<div><div class="clip-bars">' + bars + '</div>' +
      '<p class="muted-note">mm/node on this clip; #rank is the clip\'s rank among the 36 for that method (1 = easiest).</p></div>';
  }

  function selectClip(c) {
    if (!c) return;
    state.clip = c;
    paintStrip();
    renderClipCard();
  }
  var _render = renderStrip;
  renderStrip = function () { _render(); renderClipCard(); };
  syncPicker();

  // ---------------------------------------------------------------- BibTeX copy
  var copyBtn = document.getElementById('copy-bib');
  copyBtn.addEventListener('click', function () {
    var text = document.getElementById('bibtex-code').textContent;
    function done(ok) {
      copyBtn.querySelector('span').textContent = ok ? 'Copied' : 'Copy failed';
      copyBtn.querySelector('i').className = ok ? 'fas fa-check' : 'fas fa-xmark';
      setTimeout(function () { copyBtn.querySelector('span').textContent = 'Copy'; copyBtn.querySelector('i').className = 'fas fa-copy'; }, 1800);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(false); });
    } else {
      var ta = document.createElement('textarea');
      ta.value = text; document.body.appendChild(ta); ta.select();
      try { done(document.execCommand('copy')); } catch (e) { done(false); }
      document.body.removeChild(ta);
    }
  });

  // ---------------------------------------------------------------- initial render + resize
  renderRank();
  renderStrip();
  renderClipCard();

  // Deep links: #detail-<key> opens that approach.
  function fromHash() {
    var m = /^#detail-([a-z]+)$/.exec(location.hash);
    if (m && BY_KEY[m[1]]) openApproach(m[1]);
  }
  window.addEventListener('hashchange', fromHash);
  fromHash();

  var resizeTimer, lastW = rankBox.clientWidth;
  function rerender() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      var w = rankBox.clientWidth;
      if (w === lastW) return; // height changes from our own re-render must not loop
      lastW = w;
      renderRank(); renderStrip();
    }, 80);
  }
  if ('ResizeObserver' in window) {
    new ResizeObserver(rerender).observe(rankBox);
  } else {
    window.addEventListener('resize', rerender);
  }
  document.addEventListener('scroll', hideTip, { passive: true });
})();
