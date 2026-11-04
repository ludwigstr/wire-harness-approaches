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
      renderRank(); renderTable();
    });
  });
  document.getElementById('show-interim').addEventListener('change', function (e) {
    filter.interim = e.target.checked; renderRank(); renderTable();
  });

  function approachTip(a) {
    var p = PARADIGM[a.paradigm];
    return '<div class="tt-title"><span class="swatch" style="--sw:' + p.color + '"></span>' + esc(a.name) + '</div>' +
      '<div class="tt-row"><span>Mean</span><b>' + fmt(a.score) + ' mm/node</b></div>' +
      '<div class="tt-row"><span>Median</span><b>' + fmt(a.median, 2) + '</b></div>' +
      '<div class="tt-row"><span>Paradigm</span><b>' + p.label + '</b></div>' +
      '<div class="tt-row"><span>Status</span><b>' + statusText(a) + '</b></div>' +
      '<div class="tt-row"><span>Clips</span><b>' + a.clips + '</b></div>' +
      '<div class="tt-row"><span>Scored</span><b>' + esc(a.scored) + '</b></div>' +
      '<div class="tt-row"><span></span><span>Click for details</span></div>';
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

  // ---------------------------------------------------------------- results: sortable table
  var tbody = document.querySelector('#results-table tbody');
  var sortState = { key: 'score', dir: 1 };
  function renderTable() {
    var rows = APPROACHES.slice().sort(function (p, q) {
      var a = p[sortState.key], b = q[sortState.key];
      if (sortState.key === 'paradigm') { a = PARADIGM[a].label; b = PARADIGM[b].label; }
      if (a == null) return 1;
      if (b == null) return -1;
      if (typeof a === 'string') return a.localeCompare(b) * sortState.dir;
      return (a - b) * sortState.dir;
    });
    tbody.innerHTML = rows.map(function (a) {
      var p = PARADIGM[a.paradigm];
      return '<tr class="' + (a.rank === 1 ? 'best-row ' : '') + (visible(a) ? '' : 'is-hidden') + '" data-key="' + a.key + '">' +
        '<td class="num">' + a.rank + '</td>' +
        '<td><a href="#detail-' + a.key + '" class="row-link" data-key="' + a.key + '">' + esc(a.name) + '</a></td>' +
        '<td><span class="dot" style="--sw:' + p.color + '"></span>' + p.label + '</td>' +
        '<td class="num">' + fmt(a.score) + '</td>' +
        '<td class="num">' + fmt(a.median, 2) + '</td>' +
        '<td>' + statusChip(a) + '</td></tr>';
    }).join('');
  }
  document.querySelectorAll('#results-table th[data-sort]').forEach(function (th) {
    th.tabIndex = 0;
    function go() {
      var k = th.getAttribute('data-sort');
      sortState.dir = sortState.key === k ? -sortState.dir : 1;
      sortState.key = k;
      document.querySelectorAll('#results-table th[data-sort]').forEach(function (x) { x.removeAttribute('aria-sort'); });
      th.setAttribute('aria-sort', sortState.dir === 1 ? 'ascending' : 'descending');
      renderTable();
    }
    th.addEventListener('click', go);
    th.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
  });
  tbody.addEventListener('click', function (e) {
    var a = e.target.closest('.row-link');
    if (a) { e.preventDefault(); openApproach(a.getAttribute('data-key')); }
  });

  // ---------------------------------------------------------------- per-clip strip chart
  var stripBox = document.getElementById('strip-chart');
  var state = { clip: D.clips.length ? '20261021_164755' : null, hover: null };
  if (state.clip && D.clips.indexOf(state.clip) < 0) state.clip = D.clips[0];

  function rankOf(key, clip) {
    var vals = D.clips.map(function (c) { return D.perClip[key][c]; }).sort(function (a, b) { return a - b; });
    return vals.indexOf(D.perClip[key][clip]) + 1;
  }
  function clipTip(c) {
    var L = clipLabel(c);
    var html = '<div class="tt-title">' + L.date + ', ' + L.time + '</div>' +
      '<div class="tt-row"><span>Frames</span><b>' + D.frames[c] + '</b></div>';
    VIDEO_KEYS.forEach(function (k) {
      html += '<div class="tt-row"><span><span class="dot" style="--sw:' + PARADIGM[BY_KEY[k].paradigm].color + '"></span>' + BY_KEY[k].name + '</span><b>' + fmt(D.perClip[k][c], 2) + '</b></div>';
    });
    return html + '<div class="tt-row"><span></span><span>Click to pin</span></div>';
  }

  var stripDots = [];
  function renderStrip() {
    if (!VIDEO_KEYS.length) { stripBox.innerHTML = '<p class="muted-note">No per-clip data found.</p>'; return; }
    var W = Math.max(300, stripBox.clientWidth);
    var narrow = W < 520;
    var labelW = narrow ? 92 : 124, rightPad = 16;
    var rowH = 54, top = 6, axisH = 30;
    var H = top + VIDEO_KEYS.length * rowH + axisH;
    var x0 = labelW, x1 = W - rightPad;
    var all = [];
    VIDEO_KEYS.forEach(function (k) { D.clips.forEach(function (c) { all.push(D.perClip[k][c]); }); });
    var vmax = Math.max.apply(null, all);
    var step = vmax > 40 ? 10 : 5;
    var xMax = niceMax(vmax, step);
    var sx = function (v) { return x0 + (v / xMax) * (x1 - x0); };

    stripBox.innerHTML = '';
    stripDots = [];
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H }, stripBox);
    var grid = el('g', { 'class': 'grid' }, svg);
    var ticks = el('g', { 'class': 'tick' }, svg);
    var plotBottom = top + VIDEO_KEYS.length * rowH;
    for (var t = 0; t <= xMax; t += step) {
      el('line', { x1: sx(t), x2: sx(t), y1: top, y2: plotBottom }, grid);
      var tx = el('text', { x: sx(t), y: plotBottom + 16, 'text-anchor': 'middle' }, ticks);
      tx.textContent = t;
    }
    el('line', { 'class': 'baseline', x1: x0, x2: x0, y1: top, y2: plotBottom }, svg);
    var ax = el('text', { x: x1, y: plotBottom + 29, 'text-anchor': 'end' }, svg);
    ax.textContent = 'mm/node per clip';
    ax.setAttribute('style', 'fill: var(--muted); font-size: 11px');

    var linkLayer = el('g', {}, svg);
    VIDEO_KEYS.forEach(function (k, ri) {
      var cy = top + ri * rowH + rowH / 2;
      var a = BY_KEY[k];
      el('line', { x1: x0, x2: x1, y1: cy, y2: cy, style: 'stroke: var(--line)' }, svg);
      var lab = el('text', { 'class': 'row-label', x: x0 - 10, y: cy - 2, 'text-anchor': 'end' }, svg);
      lab.textContent = a.name;
      var sub = el('text', { x: x0 - 10, y: cy + 13, 'text-anchor': 'end' }, svg);
      sub.textContent = 'median ' + fmt(D.meta[k].median, 1);
      sub.setAttribute('style', 'fill: var(--muted); font-size: 11px');
      var mx = sx(D.meta[k].median);
      el('line', { 'class': 'median-tick', x1: mx, x2: mx, y1: cy - 14, y2: cy + 14 }, svg);

      D.clips.forEach(function (c, ci) {
        var v = D.perClip[k][c];
        var jitter = (((ci * 37) % 11) - 5) * 1.6; // deterministic spread so dots don't sit on one line
        var cx = sx(v), y = cy + jitter;
        var g = el('g', { tabindex: -1 }, svg);
        var dot = el('circle', { 'class': 'clip-dot', cx: cx, cy: y, r: 4.5, 'data-clip': c }, g);
        dot.setAttribute('style', 'fill:' + PARADIGM[a.paradigm].color + '; stroke: var(--surface); stroke-width: 2');
        var hit = el('circle', { cx: cx, cy: y, r: 11, fill: 'transparent', 'data-clip': c, style: 'cursor:pointer' }, g);
        stripDots.push({ clip: c, key: k, dot: dot, x: cx, y: y, color: PARADIGM[a.paradigm].color });
        hit.addEventListener('mouseenter', function (e) { setHover(c); showTip(clipTip(c), e.clientX, e.clientY); });
        hit.addEventListener('mousemove', function (e) { showTip(clipTip(c), e.clientX, e.clientY); });
        hit.addEventListener('mouseleave', function () { setHover(null); hideTip(); });
        hit.addEventListener('click', function () { hideTip(); selectClip(c); });
      });
    });
    stripBox._linkLayer = linkLayer;
    paintStrip();
  }
  function paintStrip() {
    var focus = state.hover || state.clip;
    var layer = stripBox._linkLayer;
    if (layer) layer.innerHTML = '';
    var pts = [];
    stripDots.forEach(function (s) {
      var isF = s.clip === focus;
      var isSel = s.clip === state.clip;
      s.dot.classList.toggle('is-dim', !!state.hover && !isF);
      s.dot.setAttribute('r', isF ? 7 : 4.5);
      s.dot.setAttribute('style', 'fill:' + s.color + '; stroke:' + (isF || isSel ? 'var(--ink)' : 'var(--surface)') + '; stroke-width: 2');
      if (isF) {
        pts.push(s.x + ',' + s.y);
        // Raise the selected clip's dots; never re-parent under the pointer (would flicker the hover).
        if (!state.hover) s.dot.parentNode.parentNode.appendChild(s.dot.parentNode);
      }
    });
    if (layer && pts.length > 1) el('polyline', { 'class': 'clip-link', points: pts.join(' '), fill: 'none' }, layer);
  }
  function setHover(c) { state.hover = c; paintStrip(); }

  // ---------------------------------------------------------------- clip card
  var clipCard = document.getElementById('clip-card');
  function renderClipCard() {
    var c = state.clip;
    if (!c) { clipCard.hidden = true; return; }
    var L = clipLabel(c);
    var vals = VIDEO_KEYS.map(function (k) { return D.perClip[k][c]; });
    var vmax = Math.max.apply(null, vals.concat([10]));
    var bars = VIDEO_KEYS.map(function (k) {
      var a = BY_KEY[k], v = D.perClip[k][c];
      return '<span class="cb-name"><span class="dot" style="--sw:' + PARADIGM[a.paradigm].color + '"></span>' + a.name + '</span>' +
        '<span class="cb-track"><span class="cb-fill" style="display:block;width:' + (v / vmax * 100).toFixed(1) + '%;background:' + PARADIGM[a.paradigm].color + '"></span></span>' +
        '<span class="cb-val">' + fmt(v, 2) + '<span class="cb-rank">#' + rankOf(k, c) + '/' + D.clips.length + '</span></span>';
    }).join('');
    clipCard.innerHTML =
      '<div><h4>' + L.date + ', ' + L.time + '</h4>' +
      '<p class="clip-meta"><code>' + c + '</code><br>' + D.frames[c] + ' frames &middot; ' + (D.frames[c] / FPS).toFixed(1) + ' s at 4 fps</p>' + '</div>' +
      '<div><div class="clip-bars">' + bars + '</div>' +
      '<p class="muted-note">mm/node on this clip; #rank is the clip\'s rank among the 36 for that approach (1 = easiest).</p></div>';
  }

  function selectClip(c) {
    if (!c) return;
    state.clip = c;
    paintStrip();
    renderClipCard();
  }

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
  renderTable();
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
