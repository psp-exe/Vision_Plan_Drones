/* ============================================================
   script.js — GitHub Pages thesis site interactivity
   ============================================================ */

// ── Navbar scroll behaviour ──────────────────────────────────
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 40);
});

// ── Mobile hamburger ─────────────────────────────────────────
const hamburger = document.getElementById('hamburger');
const navLinks  = document.querySelector('.nav-links');
hamburger.addEventListener('click', () => {
  navLinks.classList.toggle('open');
});
document.querySelectorAll('.nav-links a').forEach(a => {
  a.addEventListener('click', () => navLinks.classList.remove('open'));
});

// ── Hero particle field ──────────────────────────────────────
(function spawnParticles() {
  const container = document.getElementById('particles');
  if (!container) return;
  const count = 60;
  for (let i = 0; i < count; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    p.style.cssText = `
      left:${Math.random() * 100}%;
      top:${Math.random() * 100}%;
      width:${Math.random() * 3 + 1}px;
      height:${Math.random() * 3 + 1}px;
      animation-delay:${Math.random() * 4}s;
      animation-duration:${Math.random() * 3 + 3}s;
      opacity:${Math.random() * 0.6};
    `;
    // Randomly tint blue or purple
    p.style.background = Math.random() > 0.5 ? '#6395ff' : '#a78bfa';
    container.appendChild(p);
  }
})();

// ── Smooth anchor scroll with offset ─────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) {
      e.preventDefault();
      const offset = 80;
      const top = target.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({ top, behavior: 'smooth' });
    }
  });
});

// ── Intersection Observer: reveal on scroll ──────────────────
const revealObserver = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        revealObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12 }
);

document.querySelectorAll(
  '.problem-card, .arch-layer, .vlm-card, .future-card, ' +
  '.sim-spec-card, .ft-step, .stat-card, .safety-item, ' +
  '.results-block, .finetuning-section'
).forEach(el => {
  el.classList.add('reveal');
  revealObserver.observe(el);
});

// ── Animated counter (stats band) ────────────────────────────
function animateCounter(el, target, decimals = 0, duration = 1600) {
  const start = performance.now();
  const update = now => {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // ease-out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = eased * target;
    el.textContent = current.toFixed(decimals);
    if (progress < 1) requestAnimationFrame(update);
    else el.textContent = target.toFixed(decimals);
  };
  requestAnimationFrame(update);
}

const statsObserver = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const numberEl = entry.target.querySelector('.stat-number');
      if (!numberEl || numberEl.dataset.animated) return;
      numberEl.dataset.animated = '1';
      const target = parseFloat(numberEl.dataset.target);
      const decimals = target % 1 !== 0 ? 2 : 0;
      animateCounter(numberEl, target, decimals);
    });
  },
  { threshold: 0.5 }
);
document.querySelectorAll('.stat-card').forEach(c => statsObserver.observe(c));

// ── Animate bar chart when visible ───────────────────────────
const barObserver = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.querySelectorAll('.bar').forEach((bar, i) => {
          setTimeout(() => bar.classList.add('animate'), i * 120);
        });
        barObserver.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.3 }
);
document.querySelectorAll('.bar-chart').forEach(c => barObserver.observe(c));

// ── Active nav link highlight ─────────────────────────────────
const sections = document.querySelectorAll('section[id]');
const navAnchors = document.querySelectorAll('.nav-links a[href^="#"]');

const activeSectionObserver = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navAnchors.forEach(a => a.classList.remove('active-nav'));
        const match = document.querySelector(`.nav-links a[href="#${entry.target.id}"]`);
        if (match) match.classList.add('active-nav');
      }
    });
  },
  { rootMargin: '-40% 0px -55% 0px' }
);
sections.forEach(s => activeSectionObserver.observe(s));

// ── Copy BibTeX to clipboard ──────────────────────────────────
function copyBibtex() {
  const text = document.getElementById('bibtex-block').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copy-bibtex');
    const originalHTML = btn.innerHTML;
    btn.classList.add('copied');
    btn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="20 6 9 17 4 12"/>
      </svg>
      Copied!`;
    setTimeout(() => {
      btn.classList.remove('copied');
      btn.innerHTML = originalHTML;
    }, 2500);
  });
}

// Expose to window for onclick attribute
window.copyBibtex = copyBibtex;

// ── Staggered card entrance delays ───────────────────────────
function applyStagger(selector, delay = 80) {
  document.querySelectorAll(selector).forEach((el, i) => {
    el.style.transitionDelay = `${i * delay}ms`;
  });
}
applyStagger('.problem-card');
applyStagger('.future-card');
applyStagger('.vlm-card', 100);
applyStagger('.stat-card', 60);
