/* Viewing experience for every page: reading progress, nav shadow, back-to-top, scroll reveal,
   count-up numbers, and a sticky jump bar on dashboard pages. Content is never changed. */
(function () {
  'use strict';
  var root = document.documentElement;
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var nav = document.getElementById('topnav');

  // ---- left sidebar menu (docked on wide screens, a drawer behind the burger on small ones)
  var here = document.body.getAttribute('data-approach') || '';
  var home = here ? '../index.html' : '';
  var apBase = here ? '' : 'approaches/';
  var MENU = [
    { label: 'Overview', href: '#top', items: [['Abstract', '#overview'], ['Pipeline Overview', '#pipeline'], ['Benchmark Setup', '#setup']] },
    { label: 'Approaches', href: '#approaches', cats: [
      { cat: 'world', label: 'World Models', color: 'var(--series-world)',
        models: [['physworld', 'PhysWorld'], ['ropedreamer', 'RopeDreamer'], ['phystwin', 'PhysTwin']] },
      { cat: 'dynamics', label: 'Dynamics Models', color: 'var(--series-dynamics)',
        models: [['yang', 'Yang et al.'], ['caporali', 'Caporali et al.'], ['adaptigraph', 'AdaptiGraph']] },
      { cat: 'physics', label: 'Physics Engines', color: 'var(--series-physics)',
        models: [['newton', 'Newton'], ['mujoco', 'MuJoCo'], ['genesis', 'Genesis'], ['dlolab', 'DLO-Lab']] }
    ] },
    { label: 'Results', href: '#results', items: [['Mean Error per Approach', '#ranking'], ['Per-clip Explorer', '#per-clip']] }
  ];
  function escHtml(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
  var side = document.createElement('aside');
  side.className = 'side-nav';
  side.id = 'side-nav';
  side.setAttribute('aria-label', 'Contents');
  side.tabIndex = -1;
  var h = '<div class="sn-scroll"><p class="sn-kicker">Contents</p>';
  MENU.forEach(function (g, gi) {
    h += '<div class="sn-group"><a class="sn-head" href="' + home + g.href + '"><span class="sn-num">0' + (gi + 1) + '</span>' + escHtml(g.label) + '</a><ul class="sn-list">';
    (g.items || []).forEach(function (it) {
      h += '<li><a class="sn-link" href="' + home + it[1] + '">' + escHtml(it[0]) + '</a></li>';
    });
    (g.cats || []).forEach(function (c) {
      var mine = c.models.some(function (m) { return m[0] === here; });
      h += '<li class="sn-cat is-open" style="--cat: ' + c.color + '"><div class="sn-cat-row">' +
        '<a class="sn-link sn-cat-link' + (mine ? ' is-parent' : '') + '" href="' + (here ? home + '#cat-' + c.cat : '#approaches') + '" data-cat="' + c.cat + '">' +
        '<span class="swatch" style="--sw: ' + c.color + '"></span>' + escHtml(c.label) + '</a>' +
        '<button class="sn-caret" type="button" aria-expanded="true" aria-label="Show or hide ' + escHtml(c.label) + '"><i class="fas fa-chevron-down"></i></button></div><ul class="sn-sub">';
      c.models.forEach(function (m) {
        var cur = m[0] === here;
        h += '<li><a class="sn-link sn-model' + (cur ? ' is-current' : '') + '" href="' + apBase + m[0] + '.html"' + (cur ? ' aria-current="page"' : '') + '>' + escHtml(m[1]) + '</a></li>';
      });
      h += '</ul></li>';
    });
    h += '</ul></div>';
  });
  h += '<div class="sn-foot"><a class="sn-link" href="' + home + '#BibTeX"><i class="fas fa-quote-right"></i>BibTeX</a>' +
    '<a class="sn-link" href="https://github.com/ludwigstr/wire-harness-approaches"><i class="fab fa-github"></i>Code on GitHub</a></div></div>';
  side.innerHTML = h;
  var backdrop = document.createElement('div');
  backdrop.className = 'side-backdrop';
  backdrop.setAttribute('aria-hidden', 'true');
  document.body.insertBefore(backdrop, document.body.firstChild);
  document.body.insertBefore(side, document.body.firstChild);
  root.classList.add('has-side');
  if (here) side.querySelector('.sn-group:nth-child(3) .sn-head').classList.add('is-active');

  var burger = document.getElementById('nav-burger');
  var dock = window.matchMedia('(min-width: 1200px)');
  function setDrawer(open) {
    root.classList.toggle('side-open', open);
    if (burger) {
      burger.setAttribute('aria-expanded', open ? 'true' : 'false');
      burger.setAttribute('aria-controls', 'side-nav');
      var bi = burger.querySelector('i');
      if (bi) bi.className = open ? 'fas fa-xmark' : 'fas fa-bars';
    }
    if (open) side.focus({ preventScroll: true }); // Tab then moves into the menu links
  }
  if (burger) { burger.setAttribute('aria-controls', 'side-nav'); burger.setAttribute('aria-label', 'Contents menu'); }
  if (burger) burger.addEventListener('click', function () { setDrawer(!root.classList.contains('side-open')); });
  backdrop.addEventListener('click', function () { setDrawer(false); });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && root.classList.contains('side-open')) { setDrawer(false); if (burger) burger.focus(); }
  });
  (dock.addEventListener ? dock.addEventListener.bind(dock, 'change') : dock.addListener.bind(dock))(function () { setDrawer(false); });

  side.addEventListener('click', function (e) {
    var caret = e.target.closest('.sn-caret');
    if (caret) {
      var li = caret.closest('.sn-cat'), open = !li.classList.contains('is-open');
      li.classList.toggle('is-open', open);
      caret.setAttribute('aria-expanded', open ? 'true' : 'false');
      return;
    }
    var a = e.target.closest('a');
    if (!a) return;
    // On the main page a paradigm entry also switches the Approaches tabs to that paradigm.
    var cat = a.getAttribute('data-cat');
    if (cat && !here) { var tab = document.getElementById('tab-' + cat); if (tab) tab.click(); }
    if (!dock.matches) setDrawer(false);
  });

  // Coming from an approach page with #cat-<paradigm>: open that tab and land on the Approaches section.
  var cm = !here && /^#cat-(world|dynamics|physics)$/.exec(location.hash);
  if (cm) {
    var tab0 = document.getElementById('tab-' + cm[1]), sec = document.getElementById('approaches');
    if (tab0) tab0.click();
    if (history.replaceState) history.replaceState(null, '', '#approaches');
    if (sec) requestAnimationFrame(function () { sec.scrollIntoView(); });
  }

  // Scroll spy on the main page: highlight the entry for the part being read.
  // The Approaches section is one entry; inside it the paradigm of the open tab is highlighted.
  var spyLinks = [];
  if (!here) {
    side.querySelectorAll('.sn-list > li > .sn-link, .sn-head[href="#approaches"], .sn-foot .sn-link').forEach(function (a) {
      var el = document.getElementById((a.getAttribute('href') || '').replace(/^#/, ''));
      if (el) spyLinks.push({ a: a, el: el, head: a.closest('.sn-group') && a.closest('.sn-group').querySelector('.sn-head') });
    });
  }
  var catLinks = side.querySelectorAll('.sn-cat-link');
  var lastSpy = null;
  function spy() {
    if (!spyLinks.length) return;
    var line = window.innerHeight * 0.32, cur = null;
    spyLinks.forEach(function (l) { if (l.el.getBoundingClientRect().top <= line) cur = l; });
    if ((root.scrollTop || window.scrollY) + window.innerHeight >= root.scrollHeight - 4) cur = spyLinks[spyLinks.length - 1];
    var inAp = !!cur && cur.el.id === 'approaches';
    var openTab = document.querySelector('.category-tab.is-active');
    var openCat = inAp && openTab ? openTab.getAttribute('data-cat') : '';
    catLinks.forEach(function (a) { a.classList.toggle('is-current', a.getAttribute('data-cat') === openCat); });
    if (cur === lastSpy) return;
    lastSpy = cur;
    spyLinks.forEach(function (l) { l.a.classList.toggle('is-current', l === cur && !l.a.classList.contains('sn-head')); });
    side.querySelectorAll('.sn-head').forEach(function (hd) { hd.classList.toggle('is-active', !!cur && cur.head === hd); });
    if (cur && dock.matches) {
      var sc = side.querySelector('.sn-scroll'), r = cur.a.getBoundingClientRect(), sr = sc.getBoundingClientRect();
      if (r.top < sr.top + 40 || r.bottom > sr.bottom - 40) sc.scrollTop += r.top - sr.top - sr.height / 3;
    }
  }

  // ---- reading progress + nav shadow + back to top
  var bar = document.createElement('div');
  bar.className = 'read-progress';
  bar.setAttribute('aria-hidden', 'true');
  document.body.appendChild(bar);
  var toTop = document.createElement('button');
  toTop.type = 'button';
  toTop.className = 'to-top';
  toTop.setAttribute('aria-label', 'Back to top');
  toTop.title = 'Back to top';
  toTop.innerHTML = '<i class="fas fa-arrow-up"></i>';
  toTop.addEventListener('click', function () { window.scrollTo({ top: 0, behavior: reduce ? 'auto' : 'smooth' }); });
  document.body.appendChild(toTop);
  var ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () {
      ticking = false;
      var y = window.scrollY || root.scrollTop, max = root.scrollHeight - root.clientHeight;
      bar.style.transform = 'scaleX(' + (max > 0 ? Math.min(1, y / max) : 0) + ')';
      if (nav) nav.classList.toggle('is-scrolled', y > 8);
      toTop.classList.toggle('is-visible', y > 900);
      spy();
    });
  }
  document.querySelectorAll('.category-tab').forEach(function (t) { t.addEventListener('click', function () { setTimeout(spy, 0); }); });
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  onScroll();

  // ---- scroll reveal (skipped when the visitor prefers reduced motion)
  if (!reduce && 'IntersectionObserver' in window) {
    root.classList.add('js-reveal');
    var sel = ['.section-title', '.sub-title', '.prose', '.callout', '.teaser-figure', '.fig-image', '.teaser-sub', '.stepper', '.step-panels',
      '.setup-video', '.setup-text', '.category-tabs', '.approach-link', '.filter-row', '.chart-card', '.clip-card', '.bibtex',
      '.dash-card', '.method-details', '.ap-pager', '.stat-row .stat', '.kpi-grid .stat'].join(',');
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add('is-in'); io.unobserve(en.target); }
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.06 });
    document.querySelectorAll(sel).forEach(function (n) {
      // small stagger for siblings in a row or grid
      var sibs = n.parentElement ? Array.prototype.filter.call(n.parentElement.children, function (c) { return c.matches(sel); }) : [];
      var i = sibs.indexOf(n);
      if (sibs.length > 1 && i > 0) n.style.transitionDelay = Math.min(i * 60, 300) + 'ms';
      n.classList.add('reveal');
      io.observe(n);
    });
  }

  // ---- videos marked data-inview-play run (muted) only while on screen
  var vids = document.querySelectorAll('video[data-inview-play]');
  if (vids.length && !reduce && 'IntersectionObserver' in window) {
    var vo = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        var v = en.target;
        if (en.isIntersecting) { var pr = v.play(); if (pr && pr.catch) pr.catch(function () {}); } else if (!v.paused) v.pause();
      });
    }, { threshold: 0.35 });
    vids.forEach(function (v) { vo.observe(v); });
  }

  // ---- count-up for plain headline numbers (e.g. 36, 5,853, 4.51)
  function countUp(node) {
    var raw = node.textContent.trim();
    if (!/^[\d,]*\.?\d+$/.test(raw)) return;
    var target = parseFloat(raw.replace(/,/g, '')), dec = (raw.split('.')[1] || '').length, comma = raw.indexOf(',') >= 0;
    var t0 = null, dur = 900;
    function fmt(v) { var s = v.toFixed(dec); return comma ? Number(s).toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec }) : s; }
    function step(ts) {
      if (t0 === null) t0 = ts;
      var p = Math.min(1, (ts - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      node.textContent = fmt(target * e);
      if (p < 1) requestAnimationFrame(step); else node.textContent = raw;
    }
    requestAnimationFrame(step);
  }
  if (!reduce && 'IntersectionObserver' in window) {
    var co = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { countUp(en.target); co.unobserve(en.target); } });
    }, { threshold: 0.6 });
    document.querySelectorAll('.stat-row .stat-value, .kpi-grid .stat-value').forEach(function (n) { co.observe(n); });
  }

  // ---- sticky jump bar on dashboard pages (built from the card headings)
  var dash = document.querySelector('.ap-dash');
  if (dash) {
    var heads = dash.querySelectorAll('.dash-card .dash-head h3');
    if (heads.length > 2) {
      var jump = document.createElement('nav');
      jump.className = 'jump-nav';
      jump.setAttribute('aria-label', 'On this page');
      var inner = document.createElement('div');
      inner.className = 'wrap';
      jump.appendChild(inner);
      var links = [];
      heads.forEach(function (h, i) {
        var card = h.closest('.dash-card');
        if (!card.id) card.id = 'sec-' + (i + 1) + '-' + h.textContent.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 30);
        var a = document.createElement('a');
        a.href = '#' + card.id;
        a.textContent = h.textContent;
        inner.appendChild(a);
        links.push({ a: a, card: card });
      });
      var hero = document.querySelector('.ap-hero');
      if (hero && hero.parentNode) hero.parentNode.insertBefore(jump, hero.nextSibling);
      root.classList.add('has-jump');
      if ('IntersectionObserver' in window) {
        var jo = new IntersectionObserver(function (entries) {
          entries.forEach(function (en) {
            if (!en.isIntersecting) return;
            links.forEach(function (l) {
              var on = l.card === en.target;
              l.a.classList.toggle('is-current', on);
              if (on && l.a.scrollIntoView && inner.scrollWidth > inner.clientWidth) {
                inner.scrollTo({ left: l.a.offsetLeft - 16, behavior: reduce ? 'auto' : 'smooth' });
              }
            });
          });
        }, { rootMargin: '-40% 0px -55% 0px' });
        links.forEach(function (l) { jo.observe(l.card); });
      }
    }
  }
})();
