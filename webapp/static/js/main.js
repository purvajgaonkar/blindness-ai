/* ═══════════════════════════════════════════════════════════════════
   main.js — RetinalAI Frontend Logic
   Vanilla JS: Particles · IntersectionObserver · Upload · Predict · UI
   ═══════════════════════════════════════════════════════════════════ */

'use strict';

// ─────────────────────────────────────────────────────────────────
// ① PARTICLES BACKGROUND
// ─────────────────────────────────────────────────────────────────
(function initParticles() {
  const canvas = document.getElementById('particles-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const particles = [];
  const NUM_PARTICLES = 80;

  function resizeCanvas() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas, { passive: true });

  class Particle {
    constructor() { this.reset(true); }
    reset(init = false) {
      this.x    = Math.random() * canvas.width;
      this.y    = init ? Math.random() * canvas.height : canvas.height + 10;
      this.size = Math.random() * 2 + 0.5;
      this.speedX = (Math.random() - 0.5) * 0.3;
      this.speedY = -(Math.random() * 0.5 + 0.2);
      this.opacity = Math.random() * 0.5 + 0.1;
      this.color = Math.random() > 0.5 ? '0,212,255' : '124,58,237';
    }
    update() {
      this.x += this.speedX;
      this.y += this.speedY;
      if (this.y < -10) this.reset();
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${this.color},${this.opacity})`;
      ctx.fill();
    }
  }

  for (let i = 0; i < NUM_PARTICLES; i++) particles.push(new Particle());

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw connecting lines between close particles
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 80) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(0,212,255,${0.05 * (1 - dist / 80)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
      particles[i].update();
      particles[i].draw();
    }
    requestAnimationFrame(animate);
  }
  animate();
})();


// ─────────────────────────────────────────────────────────────────
// ② NAVBAR — appear on scroll
// ─────────────────────────────────────────────────────────────────
(function initNavbar() {
  const navbar   = document.getElementById('navbar');
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-link[data-section]');

  if (!navbar) return;

  let lastScrollY = 0;

  window.addEventListener('scroll', () => {
    const scrollY = window.scrollY;

    // Show/hide navbar
    if (scrollY > 80) navbar.classList.add('visible');
    else              navbar.classList.remove('visible');

    // Active link highlighting
    sections.forEach(section => {
      const top    = section.offsetTop - 120;
      const bottom = top + section.offsetHeight;
      if (scrollY >= top && scrollY < bottom) {
        const id = section.id;
        navLinks.forEach(link => {
          link.classList.toggle('active', link.dataset.section === id);
        });
      }
    });

    lastScrollY = scrollY;
  }, { passive: true });

  // Smooth scroll for nav links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      const target = document.querySelector(anchor.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });
})();


// ─────────────────────────────────────────────────────────────────
// ③ INTERSECTION OBSERVER — scroll reveal animations
// ─────────────────────────────────────────────────────────────────
(function initReveal() {
  const revealEls = document.querySelectorAll('.reveal-up, .step-card');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15 });

  revealEls.forEach(el => observer.observe(el));
})();


// ─────────────────────────────────────────────────────────────────
// ④ STAT COUNTER ANIMATION — runs once on intersection
// ─────────────────────────────────────────────────────────────────
(function initCounters() {
  const counterEls = document.querySelectorAll('.stat-number[data-target]');
  if (!counterEls.length) return;

  function animateCount(el) {
    const target  = parseFloat(el.dataset.target);
    const prefix  = el.dataset.prefix || '';
    const suffix  = el.dataset.suffix || '';
    const isFloat = target % 1 !== 0;
    const duration = 2000;
    const start    = performance.now();

    function step(now) {
      const t    = Math.min((now - start) / duration, 1);
      const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; // easeInOut
      const val  = target * ease;
      el.textContent = prefix + (isFloat ? val.toFixed(1) : Math.round(val)) + suffix;
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  const obs = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCount(entry.target);
        obs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });

  counterEls.forEach(el => obs.observe(el));
})();


// ─────────────────────────────────────────────────────────────────
// ⑤ PIPELINE CAROUSEL — drag to scroll
// ─────────────────────────────────────────────────────────────────
(function initCarousel() {
  const carousel = document.getElementById('pipeline-carousel');
  if (!carousel) return;

  let isDown = false, startX, scrollLeft;

  carousel.addEventListener('mousedown', e => {
    isDown = true;
    startX = e.pageX - carousel.offsetLeft;
    scrollLeft = carousel.scrollLeft;
    carousel.style.cursor = 'grabbing';
  });
  carousel.addEventListener('mouseleave', () => { isDown = false; carousel.style.cursor = 'grab'; });
  carousel.addEventListener('mouseup',    () => { isDown = false; carousel.style.cursor = 'grab'; });
  carousel.addEventListener('mousemove', e => {
    if (!isDown) return;
    e.preventDefault();
    carousel.scrollLeft = scrollLeft - (e.pageX - carousel.offsetLeft - startX);
  });
})();


// ─────────────────────────────────────────────────────────────────
// ⑥ FILE UPLOAD & DRAG AND DROP
// ─────────────────────────────────────────────────────────────────
const state = {
  file: null,
  results: null,
};

(function initUpload() {
  const uploadArea     = document.getElementById('upload-area');
  const fileInput      = document.getElementById('file-input');
  const uploadContent  = document.getElementById('upload-content');
  const retinalViewer  = document.getElementById('retinal-viewer');
  const previewImg     = document.getElementById('preview-img');
  const analyzeBtn     = document.getElementById('analyze-btn');
  const changeImageBtn = document.getElementById('change-image-btn');

  if (!uploadArea) return;

  // Click to browse
  uploadArea.addEventListener('click', (e) => {
    if (e.target === changeImageBtn || changeImageBtn?.contains(e.target)) return;
    if (!state.file) fileInput.click();
  });

  uploadArea.addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && !state.file) fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  // Drag & drop
  uploadArea.addEventListener('dragover',  (e) => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
  uploadArea.addEventListener('dragleave', ()  => { uploadArea.classList.remove('drag-over'); });
  uploadArea.addEventListener('drop',      (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // Change image button
  if (changeImageBtn) {
    changeImageBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      resetUpload();
    });
  }

  function handleFile(file) {
    if (!file.type.startsWith('image/')) {
      showToast('Please upload an image file (PNG, JPG)', 'error');
      return;
    }
    state.file = file;
    const reader = new FileReader();
    reader.onload = (ev) => {
      previewImg.src = ev.target.result;
      uploadContent.style.display = 'none';
      retinalViewer.style.display  = 'flex';
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function resetUpload() {
    state.file = null;
    fileInput.value = '';
    uploadContent.style.display = 'flex';
    retinalViewer.style.display  = 'none';
    analyzeBtn.disabled = true;
    // Hide results
    const results = document.getElementById('results');
    if (results) results.style.display = 'none';
  }

  // Analyze button
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', () => {
      if (!state.file) return;

      // Ripple effect
      const ripple = document.getElementById('btn-ripple');
      if (ripple) {
        ripple.classList.remove('active');
        void ripple.offsetWidth; // reflow
        ripple.classList.add('active');
      }

      runAnalysis();
    });
  }
})();


// ─────────────────────────────────────────────────────────────────
// ⑦ ANALYSIS — send to Flask backend
// ─────────────────────────────────────────────────────────────────
const LOADING_MESSAGES = [
  'Preprocessing image...',
  'Detecting blood vessels...',
  'Analyzing lesion patterns...',
  'Running EfficientNet-B4 classifier...',
  'Generating Grad-CAM heatmap...',
  'Compiling clinical report...',
];

function runAnalysis() {
  const loadingPanel = document.getElementById('loading-panel');
  const analyzeBtn   = document.getElementById('analyze-btn');
  const resultsSec   = document.getElementById('results');

  if (!state.file) return;

  // Show loading
  loadingPanel.style.display = 'flex';
  analyzeBtn.disabled = true;
  if (resultsSec) resultsSec.style.display = 'none';

  // Animate progress bar through loading messages
  let progress = 0;
  let msgIdx   = 0;
  const msgEl  = document.getElementById('loading-message');
  const barEl  = document.getElementById('loading-bar');
  const pctEl  = document.getElementById('loading-percent');

  msgEl.textContent = LOADING_MESSAGES[0];

  const progressInterval = setInterval(() => {
    progress += Math.random() * 12 + 4;
    if (progress > 90) progress = 90;

    msgIdx = Math.min(Math.floor(progress / (90 / LOADING_MESSAGES.length)),
                      LOADING_MESSAGES.length - 1);
    msgEl.textContent = LOADING_MESSAGES[msgIdx];

    barEl.style.width  = `${progress}%`;
    pctEl.textContent = `${Math.round(progress)}%`;
  }, 600);

  const formData = new FormData();
  formData.append('image', state.file);

  fetch('/predict', { method: 'POST', body: formData })
    .then(res => {
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      return res.json();
    })
    .then(data => {
      clearInterval(progressInterval);

      // Fill bar to 100%
      barEl.style.width  = '100%';
      pctEl.textContent = '100%';
      msgEl.textContent = '✓ Analysis complete!';

      setTimeout(() => {
        loadingPanel.style.display = 'none';
        analyzeBtn.disabled = false;
        state.results = data;
        renderResults(data);
      }, 600);
    })
    .catch(err => {
      clearInterval(progressInterval);
      loadingPanel.style.display = 'none';
      analyzeBtn.disabled = false;
      console.error('[RetinalAI] Fetch error:', err);
      showToast(`Analysis failed: ${err.message}`, 'error');
    });
}


// ─────────────────────────────────────────────────────────────────
// ⑧ RENDER RESULTS
// ─────────────────────────────────────────────────────────────────
const CLASS_NAMES  = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR'];
const GRADE_COLORS = ['#22c55e', '#84cc16', '#f59e0b', '#ef4444', '#991b1b'];

function renderResults(data) {
  const resultsSec = document.getElementById('results');
  if (!resultsSec) return;

  // Demo warning banner
  const demoBanner = document.getElementById('demo-warning-banner');
  if (data.demo_mode && data.demo_warning) {
    demoBanner.style.display = 'flex';
    document.getElementById('demo-warning-text').textContent = data.demo_warning;
  } else {
    demoBanner.style.display = 'none';
  }

  const grade = data.predicted_class ?? 0;
  const color = GRADE_COLORS[grade] || '#00d4ff';

  // ── Diagnosis card ─────────────────────────────────────────────
  const diagName = document.getElementById('diagnosis-name');
  diagName.textContent = data.prediction || 'Unknown';
  diagName.style.color = color;

  // Confidence ring
  const conf     = (data.confidence || 0) * 100;
  const ringFill = document.getElementById('confidence-ring-fill');
  const circum   = 2 * Math.PI * 52; // r=52
  setTimeout(() => {
    ringFill.style.stroke            = color;
    ringFill.style.strokeDashoffset  = circum * (1 - conf / 100);
  }, 200);

  animateCountUp(document.getElementById('confidence-pct'), 0, conf, '%', 1, 1200);

  // Risk badge
  const riskBadge = document.getElementById('risk-badge');
  riskBadge.textContent  = data.risk_level || '—';
  riskBadge.style.color  = color;
  riskBadge.style.borderColor = color + '80';
  riskBadge.style.background  = color + '15';

  // ── Probability bars ────────────────────────────────────────────
  const probsContainer = document.getElementById('prob-bars');
  probsContainer.innerHTML = '';
  const probs = data.probabilities || {};
  const probValues = CLASS_NAMES.map(n => (probs[n] || 0) * 100);

  CLASS_NAMES.forEach((name, i) => {
    const pct = probValues[i];
    const div = document.createElement('div');
    div.className = 'prob-item';
    div.innerHTML = `
      <div class="prob-header">
        <span class="prob-name">${name}</span>
        <span class="prob-pct" id="prob-pct-${i}">0%</span>
      </div>
      <div class="prob-bar-bg">
        <div class="prob-bar prob-bar-${i}" id="prob-bar-${i}" style="width:0%"></div>
      </div>`;
    probsContainer.appendChild(div);

    setTimeout(() => {
      document.getElementById(`prob-bar-${i}`).style.width = `${pct}%`;
      animateCountUp(document.getElementById(`prob-pct-${i}`), 0, pct, '%', 1, 1000);
    }, 300 + i * 80);
  });

  // ── Blindness risk gauge ─────────────────────────────────────────
  const riskPct  = data.blindness_risk_percent || 0;
  const gaugeFill   = document.getElementById('gauge-fill');
  const gaugeNeedle = document.getElementById('gauge-needle');
  const gaugeVal    = document.getElementById('gauge-value');

  const arcLen = 267; // stroke-dasharray (matching path arc length)
  setTimeout(() => {
    gaugeFill.style.strokeDashoffset = arcLen * (1 - riskPct / 100);
    // Needle: -90deg = 0%, 0deg = 50%, +90deg = 100%
    const angle = -90 + (riskPct / 100) * 180;
    gaugeNeedle.style.transform = `rotate(${angle}deg)`;
  }, 300);
  animateCountUp(gaugeVal, 0, riskPct, '%', 0, 1500);

  // ── Images ──────────────────────────────────────────────────────
  setImg('img-original',     data.original_b64);
  setImg('img-preprocessed', data.preprocessed_b64);
  setImg('img-vessels',      data.vessel_mask_b64);
  setImg('img-lesions',      data.lesion_overlay_b64);
  setImg('img-gradcam',      data.gradcam_b64);

  // Update pipeline preview images
  setPipeImg('pipe-img-original',  data.original_b64);
  setPipeImg('pipe-img-vessels',   data.vessel_mask_b64);
  setPipeImg('pipe-img-lesions',   data.lesion_overlay_b64);
  setPipeImg('pipe-img-gradcam',   data.gradcam_b64);
  setPipeImg('pipe-img-bgraham',   data.preprocessed_b64);
  setPipeImg('pipe-img-clahe',     data.preprocessed_b64);
  setPipeImg('pipe-img-denoise',   data.preprocessed_b64);

  // ── Clinical findings ────────────────────────────────────────────
  const findingsList = document.getElementById('findings-list');
  findingsList.innerHTML = '';
  (data.clinical_findings || []).forEach((finding, i) => {
    const li = document.createElement('li');
    li.className = 'finding-item';
    li.innerHTML = `<i class="fa-solid fa-circle-dot"></i><span>${finding}</span>`;
    findingsList.appendChild(li);
    setTimeout(() => li.classList.add('in'), 200 + i * 100);
  });

  // ── Recommendations ──────────────────────────────────────────────
  const recsList   = document.getElementById('recs-list');
  const urgBadge   = document.getElementById('urgency-badge');
  recsList.innerHTML = '';

  const urgency = (data.urgency || 'Routine').toLowerCase();
  urgBadge.textContent  = data.urgency || 'ROUTINE';
  urgBadge.className    = `urgency-badge ${urgency}`;

  (data.recommendations || []).forEach(rec => {
    const li = document.createElement('li');
    li.className = 'rec-item';
    li.innerHTML = `<i class="fa-solid fa-circle-check"></i><span>${rec}</span>`;
    recsList.appendChild(li);
  });

  // ── Processing steps ─────────────────────────────────────────────
  const procList = document.getElementById('processing-list');
  procList.innerHTML = '';
  (data.processing_steps || []).forEach(step => {
    const li = document.createElement('li');
    li.className = 'processing-step';
    li.innerHTML = `<i class="fa-solid fa-check-circle" style="color:var(--emerald)"></i>${step}`;
    procList.appendChild(li);
  });

  // ── Feature analysis ─────────────────────────────────────────────
  const vd    = data.vessel_density || 0;
  const vdBar = document.getElementById('vessel-density-bar');
  const vdVal = document.getElementById('vessel-density-val');
  setTimeout(() => { if (vdBar) vdBar.style.width = `${Math.min(vd * 3, 100)}%`; }, 400);
  if (vdVal) animateCountUp(vdVal, 0, vd, '%', 1, 1200);

  animateCountUp(document.getElementById('exudate-count'),     0, data.exudate_count || 0,      '', 0, 1000);
  animateCountUp(document.getElementById('hemorrhage-count'),  0, data.hemorrhage_count || 0,   '', 0, 1000);
  animateCountUp(document.getElementById('microaneurysm-count'), 0, data.microaneurysm_count || 0, '', 0, 1000);

  // ── Show results section ─────────────────────────────────────────
  resultsSec.style.display = 'block';
  resultsSec.style.animation = 'fadeUp 0.6s ease';
  setTimeout(() => {
    resultsSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 200);

  // Scroll to make pipeline section show updates
  const pipelSec = document.getElementById('pipeline');
  if (pipelSec) {
    const heading = pipelSec.querySelector('.section-header');
    if (heading) {
      heading.querySelector('.section-tag').textContent = 'Live Analysis Results';
    }
  }
}

function setImg(elId, b64) {
  const el = document.getElementById(elId);
  if (el && b64) el.src = `data:image/png;base64,${b64}`;
}

function setPipeImg(wrapId, b64) {
  const wrap = document.getElementById(wrapId);
  if (!wrap || !b64) return;
  const placeholder = wrap.querySelector('.pipe-img-placeholder');
  if (placeholder) placeholder.remove();
  let img = wrap.querySelector('img');
  if (!img) {
    img = document.createElement('img');
    img.alt = '';
    img.style.width  = '100%';
    img.style.height = '100%';
    img.style.objectFit = 'cover';
    wrap.appendChild(img);
  }
  img.src = `data:image/png;base64,${b64}`;
}


// ─────────────────────────────────────────────────────────────────
// ⑨ IMAGE TABS
// ─────────────────────────────────────────────────────────────────
(function initTabs() {
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.tab-btn');
    if (!btn) return;

    const tabId   = btn.dataset.tab;
    const tabWrap = btn.closest('.image-tabs-card');
    if (!tabWrap) return;

    tabWrap.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.remove('active');
      b.setAttribute('aria-selected', 'false');
    });
    btn.classList.add('active');
    btn.setAttribute('aria-selected', 'true');

    tabWrap.querySelectorAll('.tab-panel').forEach(p => {
      p.classList.toggle('active', p.id === `tab-${tabId}`);
    });
  });
})();


// ─────────────────────────────────────────────────────────────────
// ⑩ FEATURE ACCORDION
// ─────────────────────────────────────────────────────────────────
(function initAccordion() {
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.accordion-header');
    if (!btn) return;
    const body = document.getElementById(btn.getAttribute('aria-controls'));
    if (!body) return;
    const expanded = btn.getAttribute('aria-expanded') === 'true';
    btn.setAttribute('aria-expanded', String(!expanded));
    body.style.display = expanded ? 'none' : 'block';
  });
})();


// ─────────────────────────────────────────────────────────────────
// ⑪ UTILITY FUNCTIONS
// ─────────────────────────────────────────────────────────────────

/**
 * Animate a number counting up in an element.
 * @param {HTMLElement} el
 * @param {number} from
 * @param {number} to
 * @param {string} suffix
 * @param {number} decimals
 * @param {number} duration  ms
 */
function animateCountUp(el, from, to, suffix = '', decimals = 0, duration = 1000) {
  if (!el) return;
  const start = performance.now();
  function step(now) {
    const t    = Math.min((now - start) / duration, 1);
    const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    const val  = from + (to - from) * ease;
    el.textContent = val.toFixed(decimals) + suffix;
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/**
 * Simple toast notification.
 */
function showToast(msg, type = 'info') {
  let toast = document.getElementById('retinal-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'retinal-toast';
    Object.assign(toast.style, {
      position: 'fixed', bottom: '24px', right: '24px', zIndex: '9999',
      padding: '14px 20px', borderRadius: '12px', fontSize: '0.85rem',
      fontFamily: 'Inter, sans-serif', maxWidth: '360px',
      backdropFilter: 'blur(20px)', border: '1px solid',
      transition: 'opacity 0.3s ease, transform 0.3s ease',
      boxShadow: '0 8px 30px rgba(0,0,0,0.4)',
    });
    document.body.appendChild(toast);
  }

  const styles = {
    error: { background: 'rgba(239,68,68,0.15)', border: 'rgba(239,68,68,0.4)', color: '#fca5a5' },
    info:  { background: 'rgba(0,212,255,0.1)',  border: 'rgba(0,212,255,0.3)', color: '#e2e8f0' },
  };
  const s = styles[type] || styles.info;
  toast.style.background   = s.background;
  toast.style.borderColor  = s.border;
  toast.style.color        = s.color;
  toast.textContent        = msg;
  toast.style.opacity      = '1';
  toast.style.transform    = 'translateY(0)';
  toast.style.display      = 'block';

  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    toast.style.opacity   = '0';
    toast.style.transform = 'translateY(10px)';
  }, 4000);
}


// ─────────────────────────────────────────────────────────────────
// ⑫ HEALTH CHECK on load
// ─────────────────────────────────────────────────────────────────
(function checkHealth() {
  fetch('/health')
    .then(r => r.json())
    .then(data => {
      if (!data.model_loaded) {
        console.info('[RetinalAI] Model not trained. Running in demo mode.');
        // Subtle indicator in footer (non-intrusive)
        const footer = document.querySelector('.footer-tagline');
        if (footer) {
          footer.textContent += ' · Demo Mode (model not trained)';
          footer.style.color = '#f59e0b';
        }
      } else {
        console.info('[RetinalAI] Model loaded. Ready for inference.');
      }
    })
    .catch(() => {
      // Backend not running — expected during static preview
    });
})();
