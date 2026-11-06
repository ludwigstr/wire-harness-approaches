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

  // Results come only from the finished 36-clip refits in the repo (clips36_final_v1/prediction_videos).
  // An approach without them keeps its name on the charts and nothing else.
  function mean(key) { return D.meta[key] ? D.meta[key].mean : null; }
  var APPROACHES = [
    { key: 'physworld',   name: 'PhysWorld',       paradigm: 'world' },
    { key: 'ropedreamer', name: 'RopeDreamer',     paradigm: 'world' },
    { key: 'phystwin',    name: 'PhysTwin',        paradigm: 'world' },
    { key: 'yang',        name: 'Yang et al.',     paradigm: 'dynamics' },
    { key: 'caporali',    name: 'Caporali et al.', paradigm: 'dynamics' },
    { key: 'adaptigraph', name: 'AdaptiGraph',     paradigm: 'dynamics' },
    { key: 'newton',      name: 'Newton',          paradigm: 'physics' },
    { key: 'mujoco',      name: 'MuJoCo',          paradigm: 'physics' },
    { key: 'genesis',     name: 'Genesis',         paradigm: 'physics' },
    { key: 'dlolab',      name: 'DLO-Lab',         paradigm: 'physics' }
  ];
  APPROACHES.forEach(function (a) {
    a.score = mean(a.key);
    a.clips = D.perClip[a.key] ? Object.keys(D.perClip[a.key]).length : 0;
  });
  // with results first (best first), then the ones without, in menu order
  var RANKED = APPROACHES.filter(function (a) { return a.score != null; }).sort(function (x, y) { return x.score - y.score; })
    .concat(APPROACHES.filter(function (a) { return a.score == null; }));
  var BY_KEY = {};
  APPROACHES.forEach(function (a) { BY_KEY[a.key] = a; });

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
  // The left sidebar menu (burger, scroll spy) lives in ux.js, shared by every page.

  // ---------------------------------------------------------------- hero numbers
  if (D.clips.length) {
    var fr = D.clips.map(function (c) { return D.frames[c]; });
    var total = fr.reduce(function (s, v) { return s + v; }, 0);
    var sf = document.getElementById('stat-frames'), frr = document.getElementById('frames-range');
    if (sf) sf.textContent = total.toLocaleString('en-US');
    if (frr) frr.textContent = Math.min.apply(null, fr) + '–' + Math.max.apply(null, fr);
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
  // Every approach has its own page.
  function openApproach(key) {
    if (BY_KEY[key]) location.href = 'approaches/' + key + '.html';
  }

  // ---------------------------------------------------------------- results: filter state
  var filter = { paradigm: 'all' };
  function visible(a) {
    return filter.paradigm === 'all' || a.paradigm === filter.paradigm;
  }
  var rankChips = document.querySelectorAll('#rank-filters .chip-btn');
  rankChips.forEach(function (b) {
    b.addEventListener('click', function () {
      filter.paradigm = b.getAttribute('data-filter');
      rankChips.forEach(function (x) { x.classList.toggle('is-active', x === b); });
      renderRank();
    });
  });

  function approachTip(a) {
    var p = PARADIGM[a.paradigm];
    return '<div class="tt-title"><span class="swatch" style="--sw:' + p.color + '"></span>' + esc(a.name) + '</div>' +
      '<div class="tt-row"><span>Error</span><b>' + fmt(a.score, 2) + ' mm/node</b></div>' +
      '<div class="tt-row"><span>Paradigm</span><b>' + p.label + '</b></div>' +
      '<div class="tt-row"><span>Clips</span><b>' + a.clips + '</b></div>';
  }

  // ---------------------------------------------------------------- results: ranked bar chart
  var rankBox = document.getElementById('rank-chart');
  function renderRank() {
    var rows = RANKED.filter(visible);
    var W = Math.max(300, rankBox.clientWidth);
    var narrow = W < 520;
    var labelW = narrow ? 104 : 132, valueW = narrow ? 76 : 96;
    var rowH = 34, barH = 18, top = 6, axisH = 30;
    var H = top + rows.length * rowH + axisH;
    var x0 = labelW, x1 = W - valueW;
    var xMax = niceMax(Math.max.apply(null, RANKED.filter(function (a) { return a.score != null; }).map(function (a) { return a.score; }).concat([5])), 5);
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
      if (a.score == null) { // no result in the repo yet: the name only
        var nl = el('text', { 'class': 'row-label pending', x: x0 - 10, y: y + barH / 2 + 4, 'text-anchor': 'end' }, svg);
        nl.textContent = a.name;
        return;
      }
      var g = el('g', { 'class': 'bar-row', tabindex: 0, role: 'button', 'aria-label': a.name + ', ' + fmt(a.score) + ' mm/node' }, svg);
      el('rect', { 'class': 'hit', x: 0, y: top + i * rowH, width: W, height: rowH, fill: 'transparent' }, g);
      var lab = el('text', { 'class': 'row-label', x: x0 - 10, y: y + barH / 2 + 4, 'text-anchor': 'end' }, g);
      lab.textContent = a.name;
      var bx = sx(a.score), r = 4, h = barH;
      var d = 'M' + x0 + ',' + y + ' H' + (bx - r) + ' Q' + bx + ',' + y + ' ' + bx + ',' + (y + r) +
              ' V' + (y + h - r) + ' Q' + bx + ',' + (y + h) + ' ' + (bx - r) + ',' + (y + h) + ' H' + x0 + ' Z';
      var bar = el('path', { 'class': 'bar', d: d }, g);
      bar.setAttribute('style', 'fill:' + PARADIGM[a.paradigm].color);
      var v = el('text', { 'class': 'value-label', x: bx + 7, y: y + barH / 2 + 4 }, g);
      v.textContent = fmt(a.score, 2);
      g.addEventListener('mousemove', function (e) { showTip(approachTip(a), e.clientX, e.clientY); });
      g.addEventListener('mouseleave', hideTip);
      g.addEventListener('focus', function () { tipAt(bar, approachTip(a)); });
      g.addEventListener('blur', hideTip);
      g.addEventListener('click', function () { hideTip(); openApproach(a.key); });
      g.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); hideTip(); openApproach(a.key); } });
    });
  }

  // ---------------------------------------------------------------- per-clip explorer (one row per method, 2-D zoom)
  var stripBox = document.getElementById('strip-chart');
  var legendBox = document.getElementById('strip-legend');
  var N = D.clips.length;

  // Series with real per-clip numbers. Colour = paradigm (as everywhere on the page); the row names the method.
  var SERIES = [];
  function addSeries(key, label, values) {
    if (values && BY_KEY[key]) SERIES.push({ key: key, label: label, paradigm: BY_KEY[key].paradigm, values: values });
  }
  // methods with per-clip results, grouped by paradigm; the others follow as name-only rows
  APPROACHES.forEach(function (a) { addSeries(a.key, a.name, D.perClip[a.key]); });
  APPROACHES.forEach(function (a) { if (!D.perClip[a.key]) SERIES.push({ key: a.key, label: a.name, paradigm: a.paradigm, values: {}, empty: true }); });
  var SERIES_BY_KEY = {};
  SERIES.forEach(function (s) { SERIES_BY_KEY[s.key] = s; });

  // ---- method picker: "All methods" + one multi-select menu per paradigm.
  // Any number of models can be ticked, from one family or from several.
  var AVAILABLE = SERIES.filter(function (s) { return !s.empty; }).map(function (s) { return s.key; });
  var pick = { all: true, keys: {} };   // all = every method (with the name-only rows); else the ticked keys
  var pickAll = document.getElementById('pc-all');
  var pcStatus = document.getElementById('pc-status');
  var menus = Array.prototype.slice.call(document.querySelectorAll('.pc-filters .ms'));
  function isOn(k) { return pick.all ? AVAILABLE.indexOf(k) >= 0 : !!pick.keys[k]; }
  function chosen() { return AVAILABLE.filter(isOn); }
  function setKeys(keys) {
    pick = { all: false, keys: {} };
    keys.forEach(function (k) { pick.keys[k] = true; });
    if (keys.length === AVAILABLE.length) pick = { all: true, keys: {} };
    syncPicker(); resetZoom();
  }
  function closeMenus(except) {
    menus.forEach(function (m) {
      if (m === except) return;
      m.classList.remove('is-open');
      m.querySelector('.ms-btn').setAttribute('aria-expanded', 'false');
      m.querySelector('.ms-panel').hidden = true;
    });
  }
  menus.forEach(function (m) {
    var par = m.getAttribute('data-par'), btn = m.querySelector('.ms-btn'), panel = m.querySelector('.ms-panel');
    var models = APPROACHES.filter(function (a) { return a.paradigm === par; });
    panel.innerHTML = '<div class="ms-tools"><button type="button" data-ms="all">Select all</button><button type="button" data-ms="none">None</button></div>' +
      models.map(function (a) {
        var ok = AVAILABLE.indexOf(a.key) >= 0;
        return '<label class="ms-item' + (ok ? '' : ' is-off') + '"><input type="checkbox" value="' + a.key + '"' + (ok ? '' : ' disabled') + '>' +
          '<span class="dot" style="--sw:' + PARADIGM[par].color + '"></span><span>' + esc(a.name) + '</span></label>';
      }).join('');
    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      var open = !m.classList.contains('is-open');
      closeMenus(m);
      m.classList.toggle('is-open', open);
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      panel.hidden = !open;
    });
    panel.addEventListener('click', function (e) { e.stopPropagation(); });
    panel.addEventListener('change', function (e) {
      if (e.target.type !== 'checkbox') return;
      var keys = chosen().filter(function (k) { return k !== e.target.value; });
      if (e.target.checked) keys.push(e.target.value);
      setKeys(keys);
    });
    panel.querySelectorAll('[data-ms]').forEach(function (t) {
      t.addEventListener('click', function () {
        var mine = models.map(function (a) { return a.key; }).filter(function (k) { return AVAILABLE.indexOf(k) >= 0; });
        var rest = chosen().filter(function (k) { return mine.indexOf(k) < 0; });
        setKeys(t.getAttribute('data-ms') === 'all' ? rest.concat(mine) : rest);
      });
    });
  });
  document.addEventListener('click', function () { closeMenus(); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeMenus(); });
  if (pickAll) pickAll.addEventListener('click', function () { pick = { all: true, keys: {} }; syncPicker(); resetZoom(); });
  function syncPicker() {
    if (pickAll) pickAll.classList.toggle('is-active', pick.all);
    menus.forEach(function (m) {
      var par = m.getAttribute('data-par');
      var mine = AVAILABLE.filter(function (k) { return SERIES_BY_KEY[k].paradigm === par; });
      var n = mine.filter(isOn).length;
      m.querySelectorAll('input[type=checkbox]').forEach(function (cb) { cb.checked = isOn(cb.value); });
      m.classList.toggle('has-pick', !pick.all && n > 0);
      m.querySelector('.ms-count').textContent = pick.all ? '' : n ? n + '/' + mine.length : '';
    });
    var c = chosen().length;
    if (pcStatus) pcStatus.textContent = pick.all ? 'All methods' : c === 0 ? 'No model selected' : 'Comparing ' + c + ' model' + (c === 1 ? '' : 's');
  }
  function withData(list) { return list.filter(function (s) { return !s.empty; }); }
  function activeSeries() {
    return SERIES.filter(function (s) { return pick.all ? true : !!pick.keys[s.key]; });
  }

  // ---- zoom state: x = mm/node, y = row position (row 1 at the top; a row spans ±0.5)
  var state = { clip: D.clips.indexOf('20261021_164755') >= 0 ? '20261021_164755' : D.clips[0], hover: null };
  var mode = 'box';
  function fullDomain() {
    var vmax = 0, rows = activeSeries();
    rows.forEach(function (s) { D.clips.forEach(function (c) { if (s.values[c] > vmax) vmax = s.values[c]; }); });
    vmax = vmax || 10;
    return { x0: 0, x1: niceMax(vmax * 1.02, vmax > 40 ? 10 : vmax > 15 ? 5 : 2), y0: 0.5, y1: rows.length + 0.5 };
  }
  var FULL = fullDomain();
  var zoom = { x0: FULL.x0, x1: FULL.x1, y0: FULL.y0, y1: FULL.y1 };
  var MIN_X = 1, MIN_Y = 0.25;
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
    var d = zoom.x1 - zoom.x0 < 10 ? 1 : 0;
    var vz = (FULL.y1 - FULL.y0) / (zoom.y1 - zoom.y0);
    zb.range.textContent = 'Showing ' + zoom.x0.toFixed(d) + '–' + zoom.x1.toFixed(d) + ' mm/node' +
      (vz > 1.01 ? ' · vertical zoom ×' + vz.toFixed(1) : '');
  }

  function renderLegend() {
    if (!legendBox) return;
    var seen = {};
    withData(activeSeries()).forEach(function (s) { seen[s.paradigm] = true; });
    legendBox.innerHTML = ['world', 'dynamics', 'physics'].filter(function (p) { return seen[p]; }).map(function (p) {
      return '<span><span class="swatch round" style="--sw:' + PARADIGM[p].color + '"></span>' + PARADIGM[p].label + '</span>';
    }).join('');
  }

  // Tooltip: clip number, frames and every shown method's error (no date or time).
  function clipTip(c) {
    var html = '<div class="tt-title">Clip ' + (D.clips.indexOf(c) + 1) + ' of ' + N + '</div>' +
      '<div class="tt-row"><span>Frames</span><b>' + D.frames[c] + '</b></div>';
    withData(activeSeries()).forEach(function (s) {
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
    if (!activeSeries().length) {
      stripBox.innerHTML = '<p class="pc-empty"><i class="fas fa-hand-pointer"></i>Pick at least one model from the menus above.</p>';
      if (legendBox) legendBox.innerHTML = '';
      stripBox._geo = null; pts = []; return;
    }
    renderLegend();
    var rows = activeSeries();
    var W = Math.max(300, stripBox.clientWidth);
    var narrow = W < 520;
    var R = W - 16, T = 8;
    var plotH = Math.max(rows.length * 58, 150);
    var B = T + plotH, H = B + 34;
    var z = zoom;
    // label column wide enough for the longest row name actually shown
    var probe = el('svg', { width: 0, height: 0, style: 'position:absolute' }, stripBox);
    var pt = el('text', { 'class': 'row-label' }, probe), longest = 0;
    rows.forEach(function (s) { pt.textContent = s.label; longest = Math.max(longest, pt.getComputedTextLength()); });
    stripBox.removeChild(probe);
    var L = Math.min(Math.round(W * 0.4), Math.max(narrow ? 80 : 100, Math.ceil(longest) + 22));
    var sx = function (v) { return L + ((v - z.x0) / (z.x1 - z.x0)) * (R - L); };
    var sy = function (r) { return T + ((r - z.y0) / (z.y1 - z.y0)) * (B - T); };
    var invX = function (px) { return z.x0 + ((px - L) / (R - L)) * (z.x1 - z.x0); };
    var invY = function (py) { return z.y0 + ((py - T) / (B - T)) * (z.y1 - z.y0); };

    stripBox.innerHTML = '';
    pts = [];
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H }, stripBox);
    var toPt = function (e) { var r = svg.getBoundingClientRect(); return { x: (e.clientX - r.left) * (W / r.width), y: (e.clientY - r.top) * (H / r.height) }; };
    var defs = el('defs', {}, svg);
    var cp = el('clipPath', { id: 'pc-clip' }, defs);
    el('rect', { x: L - 2, y: T, width: R - L + 4, height: B - T }, cp);
    var cpl = el('clipPath', { id: 'pc-clip-labels' }, defs);
    el('rect', { x: 0, y: T - 4, width: L, height: B - T + 8 }, cpl);

    // x grid + ticks
    var grid = el('g', { 'class': 'grid' }, svg), ticks = el('g', { 'class': 'tick' }, svg);
    var xs = niceStep(z.x1 - z.x0, narrow ? 4 : 7);
    for (var v = Math.ceil(z.x0 / xs - 1e-9) * xs; v <= z.x1 + 1e-9; v += xs) {
      el('line', { x1: sx(v), x2: sx(v), y1: T, y2: B }, grid);
      var tx = el('text', { x: sx(v), y: B + 16, 'text-anchor': 'middle' }, ticks);
      tx.textContent = xs < 1 ? v.toFixed(1) : Math.round(v);
    }
    el('line', { 'class': 'baseline', x1: L, x2: L, y1: T, y2: B }, svg);
    el('line', { 'class': 'baseline', x1: L, x2: R, y1: B, y2: B }, svg);
    var xl = el('text', { x: R, y: H - 4, 'text-anchor': 'end' }, svg);
    xl.textContent = 'mm/node per clip';
    xl.setAttribute('style', 'fill: var(--muted); font-size: 11px');

    // rows: centre line + label (labels stay put on the left, clipped to the plot height)
    var labels = el('g', { 'clip-path': 'url(#pc-clip-labels)' }, svg);
    var plot = el('g', { 'clip-path': 'url(#pc-clip)' }, svg);
    var linkLayer = el('g', {}, plot);
    // band half-height in row units; zooming in vertically spreads the dots apart
    var JIT = 0.3;
    rows.forEach(function (s, ri) {
      var r = ri + 1;
      var cy = sy(r);
      el('line', { x1: L, x2: R, y1: cy, y2: cy, style: 'stroke: var(--line); pointer-events: none' }, plot);
      if (r >= z.y0 - 0.5 && r <= z.y1 + 0.5) {
        var lab = el('text', { 'class': 'row-label' + (s.empty ? ' pending' : ''), x: L - 10, y: cy + 4, 'text-anchor': 'end' }, labels);
        lab.textContent = s.label;
      }
      D.clips.forEach(function (c, ci) {
        var val = s.values[c];
        if (val == null) return;
        var j = ((((ci * 37) % 11) - 5) / 5) * JIT; // deterministic spread in -0.3..0.3 rows
        var yr = r + j;
        if (val < z.x0 - (z.x1 - z.x0) * 0.02 || val > z.x1 + (z.x1 - z.x0) * 0.02 || yr < z.y0 - 0.1 || yr > z.y1 + 0.1) return;
        var dot = el('circle', { 'class': 'clip-dot', cx: sx(val), cy: sy(yr), r: 4.5 }, plot);
        pts.push({ clip: c, s: s, m: dot, x: sx(val), y: sy(yr) });
      });
    });

    // interaction layer on top: nearest-point hover, box zoom / pan, click to pin
    var area = el('rect', { 'class': 'zoom-area', x: L, y: T, width: R - L, height: B - T, fill: 'transparent' }, svg);
    var drag = null;
    function nearest(p) {
      var best = null, bd = 20 * 20;
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
            setZoom(d0.z.x0 - dx, d0.z.x1 - dx, d0.z.y0 - dy, d0.z.y1 - dy);
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
        var ny0 = h >= 6 ? invY(Math.min(ay, d.y)) : z.y0, ny1 = h >= 6 ? invY(Math.max(ay, d.y)) : z.y1;
        setZoom(nx0, nx1, ny0, ny1);
      }
    });
    area.addEventListener('pointercancel', function () { if (drag && drag.rect && drag.rect.parentNode) drag.rect.parentNode.removeChild(drag.rect); drag = null; });
    area.addEventListener('pointerleave', function () { if (!drag && state.hover) { state.hover = null; paintStrip(); hideTip(); } });
    area.addEventListener('dblclick', resetZoom);

    stripBox._link = linkLayer;
    stripBox._geo = { L: L, R: R, T: T, B: B, toPt: toPt, invX: invX, invY: invY };
    syncZoomControls();
    paintStrip();
  }

  function paintStrip() {
    var focus = state.hover || state.clip;
    var link = stripBox._link;
    if (link) link.innerHTML = '';
    var line = [];
    pts.forEach(function (q) {
      var on = q.clip === focus;
      var isSel = q.clip === state.clip;
      q.m.setAttribute('r', on ? 7 : 4.5);
      q.m.setAttribute('style', 'fill:' + PARADIGM[q.s.paradigm].color + '; stroke:' + (on || isSel ? 'var(--ink)' : 'var(--surface)') +
        '; stroke-width: 2' + (state.hover && !on ? '; opacity: 0.3' : ''));
      if (on) line.push(q);
    });
    // connect the focused clip across rows, top to bottom, and keep its dots on top
    line.sort(function (a, b) { return a.y - b.y; });
    if (link && line.length > 1) el('polyline', { 'class': 'clip-link', points: line.map(function (q) { return q.x + ',' + q.y; }).join(' '), fill: 'none' }, link);
    line.forEach(function (q) { if (q.m.parentNode) q.m.parentNode.appendChild(q.m); });
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
    var series = withData(activeSeries());
    var vmax = Math.max.apply(null, series.map(function (s) { return s.values[c]; }).concat([10]));
    var bars = series.map(function (s) {
      var v = s.values[c], col = PARADIGM[s.paradigm].color;
      return '<span class="cb-name"><span class="dot" style="--sw:' + col + '"></span>' + esc(s.label) + '</span>' +
        '<span class="cb-track"><span class="cb-fill" style="display:block;width:' + (v / vmax * 100).toFixed(1) + '%;background:' + col + '"></span></span>' +
        '<span class="cb-val">' + fmt(v, 2) + '<span class="cb-rank">#' + rankOf(s, c) + '/' + N + '</span></span>';
    }).join('');
    clipCard.innerHTML =
      '<div><h4>Clip ' + (D.clips.indexOf(c) + 1) + ' of ' + N + '</h4>' +
      '<p class="clip-meta">' + D.frames[c] + ' frames &middot; ' + (D.frames[c] / FPS).toFixed(1) + ' s at 4 fps</p></div>' +
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

  // Old deep links (#detail-<key>) forward to the approach's own page.
  function fromHash() {
    var m = /^#detail-([a-z]+)$/.exec(location.hash);
    if (m && BY_KEY[m[1]]) location.replace('approaches/' + m[1] + '.html');
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
