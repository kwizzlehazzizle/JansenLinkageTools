/**
 * Main application logic: UI wiring, animation loop, exports, shareable URLs.
 */

(function () {
  'use strict';

  // ── Default lengths ────────────────────────────────────────
  const DEFAULT_LENGTHS = {
    a: 38.0, b: 41.5, c: 39.3, d: 40.1,
    e: 55.8, f: 39.4, g: 36.7, h: 65.7,
    i: 49.0, j: 50.0, k: 61.9, l: 7.8, m: 15.0
  };

  // ── State ──────────────────────────────────────────────────
  let lengths = { ...DEFAULT_LENGTHS };
  let currentAngle = 0;
  let isPlaying = false;
  let animSpeed = 1.0;
  let showFootPath = true;
  let prevGuess = null;       // continuation state
  let currentJoints = null;   // current frame joints
  let footPath = [];          // accumulated foot path
  let animFrameId = null;
  let lastTimestamp = null;
  let highlightIntensities = {};       // barName → { current: 0..1, target: 0..1 }
  let highlightLastTime = null;        // last time we updated highlight intensities
  let currentHoveredBar = null;        // which bar input is currently under the mouse
  let integerScale = 1;                // current integer scale factor for margin compensation
  let scoreTimer = null;               // debounce timer for score computation

  // ── DOM refs ───────────────────────────────────────────────
  const canvas = document.getElementById('linkage-canvas');
  const sliderAngle = document.getElementById('slider-angle');
  const sliderSpeed = document.getElementById('slider-speed');
  const angleDisplay = document.getElementById('angle-display');
  const speedDisplay = document.getElementById('speed-display');
  const btnPlay = document.getElementById('btn-play');
  const btnReset = document.getElementById('btn-reset');
  const btnExportPng = document.getElementById('btn-export-png');
  const btnShare = document.getElementById('btn-share');
  const chkFootpath = document.getElementById('chk-footpath');
  const solverWarning = document.getElementById('solver-warning');
  const inputScale = document.getElementById('input-scale');
  const btnApplyScale = document.getElementById('btn-apply-scale');
  const integerWarning = document.getElementById('integer-warning');
  const scoreShape = document.getElementById('score-shape');
  const scoreFlatness = document.getElementById('score-flatness');

  // Input elements for each dimension
  const inputIds = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'];
  const inputs = {};
  inputIds.forEach(id => {
    inputs[id] = document.getElementById('input-' + id);
  });

  // Bar IDs that correspond to drawn bars on the canvas (a and l are pivot distances, not bars)
  const BAR_HIGHLIGHT_IDS = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'm'];
  // All highlightable IDs including pivot distances
  const ALL_HIGHLIGHT_IDS = ['a', 'l', ...BAR_HIGHLIGHT_IDS];

  // ── Flash helpers (sidebar + button, matches canvas crossfade) ──
  function flashSidebarBars() {
    // Flash the input-group rows for each bar length + pivot distances (a, l)
    ALL_HIGHLIGHT_IDS.forEach(id => {
      const row = inputs[id].closest('.input-group');
      if (row) {
        row.classList.remove('flash-highlight');
        void row.offsetWidth; // force reflow to restart animation
        row.classList.add('flash-highlight');
        row.addEventListener('animationend', () => row.classList.remove('flash-highlight'), { once: true });
      }
    });
    // Flash the slider value spans in the animation panel
    [angleDisplay, speedDisplay].forEach(el => {
      el.classList.remove('flash-value');
      void el.offsetWidth;
      el.classList.add('flash-value');
      el.addEventListener('animationend', () => el.classList.remove('flash-value'), { once: true });
    });
  }

  function flashButton(btnEl) {
    btnEl.classList.remove('flash-button');
    void btnEl.offsetWidth;
    btnEl.classList.add('flash-button');
    btnEl.addEventListener('animationend', () => btnEl.classList.remove('flash-button'), { once: true });
  }

  // ── Gait quality scoring ────────────────────────────────────

  /**
   * Schedule a debounced score computation (non-blocking via setTimeout).
   * @param {boolean} immediate - If true, compute immediately without debounce.
   */
  function scheduleScoreComputation(immediate = false) {
    if (scoreTimer) {
      clearTimeout(scoreTimer);
      scoreTimer = null;
    }
    if (!immediate) {
      scoreTimer = setTimeout(computeScores, 150);
      return;
    }
    setTimeout(computeScores, 0);
  }

  /**
   * Compute shape and flatness scores and update the UI.
   * Runs in a setTimeout callback so it doesn't block the current frame.
   */
  function computeScores() {
    scoreTimer = null;
    const result = Scorer.evaluate(lengths);

    if (!result.converged) {
      scoreShape.textContent  = 'Error';
      scoreShape.className    = 'score-value bad';
      scoreFlatness.textContent = 'Error';
      scoreFlatness.className   = 'score-value bad';
      return;
    }

    const shape  = 100 - result.shapeScore;
    const flat   = 100 - result.flatnessScore;

    // Format
    scoreShape.textContent  = formatScore(shape);
    scoreShape.className    = 'score-value ' + scoreColorClass(shape, 95, 90, 90);
    scoreFlatness.textContent = formatScore(flat);
    scoreFlatness.className    = 'score-value ' + scoreColorClass(flat, 90, 85, 85);
  }

  /**
   * Format a numeric score for display (100 = perfect).
   */
  function formatScore(value) {
    if (!isFinite(value) || value === Infinity) return '—';
    if (value <= 0) return '—';
    return value.toFixed(1) + '%';
  }

  /**
   * Return a CSS class based on how good the score is (higher = better).
   * @param {number} score - percentage (higher is better, 100 = perfect)
   * @param {number} greenThreshold - score at or above this is green
   * @param {number} orangeThreshold - score at or above this is orange (below is red)
   * @param {number} _unused - not used, kept for backwards compat
   */
  function scoreColorClass(score, greenThreshold, orangeThreshold, _unused) {
    if (score >= greenThreshold) return 'good';
    if (score >= orangeThreshold) return 'orange';
    return 'bad';
  }

  // ── Initialize ─────────────────────────────────────────────
  function init() {
    // Initialize renderer
    Renderer.init(canvas);

    // Parse URL params for shared configs
    const params = new URLSearchParams(window.location.search);
    let hasParams = false;
    inputIds.forEach(id => {
      const val = params.get(id);
      if (val !== null) {
        lengths[id] = parseFloat(val);
        hasParams = true;
      }
    });
    const angleParam = params.get('angle');
    if (angleParam !== null) {
      currentAngle = parseFloat(angleParam);
      hasParams = true;
    }

    // Populate inputs from current lengths
    updateInputsFromLengths();

    // Solve initial frame
    solveAndRender();

    // Set up event listeners
    setupEventListeners();

    // Handle resize
    window.addEventListener('resize', onResize);
    onResize();

    // Compute initial scores
    scheduleScoreComputation(true);
  }

  // ── Event listeners ────────────────────────────────────────
  function setupEventListeners() {
    // Dimension inputs: live-update on change
    inputIds.forEach(id => {
      inputs[id].addEventListener('input', onDimensionChange);
      inputs[id].addEventListener('blur', onDimensionChange);
      // Hover highlight for bars + pivot distances (a and l)
      if (BAR_HIGHLIGHT_IDS.includes(id) || id === 'a' || id === 'l') {
        inputs[id].addEventListener('mouseenter', () => onBarHover(id));
        inputs[id].addEventListener('mouseleave', onBarHoverOut);
      }
    });

    // Angle slider
    sliderAngle.addEventListener('input', () => {
      currentAngle = parseFloat(sliderAngle.value);
      angleDisplay.textContent = currentAngle.toFixed(1) + '°';
      // Reset foot path when scrubbing
      footPath = [];
      prevGuess = null;
      solveAndRender();
    });

    // Speed slider
    sliderSpeed.addEventListener('input', () => {
      animSpeed = parseFloat(sliderSpeed.value);
      speedDisplay.textContent = animSpeed.toFixed(1) + '×';
    });

    // Play/Pause
    btnPlay.addEventListener('click', togglePlay);

    // Reset
    btnReset.addEventListener('click', resetToDefaults);

    // Foot path toggle
    chkFootpath.addEventListener('change', () => {
      showFootPath = chkFootpath.checked;
      if (!isPlaying) solveAndRender();
    });

    // Export buttons
    btnExportPng.addEventListener('click', exportPng);
    btnShare.addEventListener('click', shareUrl);

    // Integer approximation
    btnApplyScale.addEventListener('click', applyIntegerApproximation);
  }

  // ── Dimension change handler ───────────────────────────────
  function onDimensionChange() {
    inputIds.forEach(id => {
      const val = parseFloat(inputs[id].value);
      if (!isNaN(val) && val > 0) {
        lengths[id] = val;
      }
    });
    // Reset continuation state on dimension change
    prevGuess = null;
    footPath = [];
    solveAndRender();
    scheduleScoreComputation();
  }

  // ── Bar highlight on hover ─────────────────────────────────
  let highlightAnimId = null;  // rAF handle for highlight animation when paused

  function onBarHover(barName) {
    // If another bar was hovered, schedule it to fade out after reaching full (or immediately if already full)
    if (currentHoveredBar && currentHoveredBar !== barName && highlightIntensities[currentHoveredBar]) {
      const prev = highlightIntensities[currentHoveredBar];
      if (prev.reachedFull) {
        prev.target = 0;
      } else {
        prev.fadeOutAfterFull = true;
      }
    }

    // Set new bar's target to 1 (fade in)
    if (!highlightIntensities[barName]) {
      highlightIntensities[barName] = { current: 0, target: 0, reachedFull: false, fadeOutAfterFull: false };
    }
    highlightIntensities[barName].target = 1;
    highlightIntensities[barName].reachedFull = false;
    highlightIntensities[barName].fadeOutAfterFull = false;
    currentHoveredBar = barName;
    highlightLastTime = performance.now();

    // Start animating the fade-in even when paused
    if (!isPlaying && currentJoints) {
      if (highlightAnimId) cancelAnimationFrame(highlightAnimId);
      animateHighlight();
    }
  }

  function onBarHoverOut() {
    // Schedule the bar to fade out after reaching full (or immediately if already full)
    if (currentHoveredBar && highlightIntensities[currentHoveredBar]) {
      const state = highlightIntensities[currentHoveredBar];
      if (state.reachedFull) {
        state.target = 0;
      } else {
        state.fadeOutAfterFull = true;
      }
    }
    currentHoveredBar = null;
    highlightLastTime = performance.now();
    if (!isPlaying) {
      if (highlightAnimId) cancelAnimationFrame(highlightAnimId);
      fadeHighlight();
    }
  }

  /**
   * Update all highlight intensities toward their targets.
   * Returns true if any bar is still transitioning.
   */
  function updateHighlightIntensity() {
    const now = performance.now();
    const dt = Math.min((now - (highlightLastTime || now)) / 1000, 0.1); // seconds, cap at 100ms
    highlightLastTime = now;

    let anyChanging = false;
    for (const barName in highlightIntensities) {
      const state = highlightIntensities[barName];
      if (state.current < state.target) {
        // Fading in over 0.5s (always completes — fade-in is irreversible)
        state.current += dt / 0.5;
        if (state.current >= 1) {
          state.current = 1;
          state.reachedFull = true;
          // If the mouse left before reaching full, start fading out now
          if (state.fadeOutAfterFull) {
            state.target = 0;
          }
        }
        anyChanging = true;
      } else if (state.current > state.target && state.reachedFull) {
        // Fading out over 2.5s (only after reaching full highlight)
        state.current -= dt / 2.5;
        if (state.current <= 0) {
          state.current = 0;
          delete highlightIntensities[barName];
        }
        anyChanging = true;
      }
    }
    return anyChanging;
  }

  /**
   * Check if any bar still has an active highlight (or is transitioning).
   */
  function hasActiveHighlights() {
    for (const barName in highlightIntensities) {
      const state = highlightIntensities[barName];
      if (Math.abs(state.current - state.target) > 0.001) return true;
      // Also keep animating if fadeOutAfterFull is pending (will trigger once reachedFull)
      if (state.fadeOutAfterFull && state.current >= 0.001) return true;
    }
    return false;
  }

  /**
   * Animation loop for highlight when paused (fade-in).
   */
  function animateHighlight() {
    if (!currentJoints) return;
    updateHighlightIntensity();
    Renderer.draw(currentJoints, footPath, lengths, currentAngle, showFootPath, false, highlightIntensities);

    // Keep animating while any bar is still transitioning
    if (hasActiveHighlights()) {
      highlightAnimId = requestAnimationFrame(animateHighlight);
    }
  }

  /**
   * Continue fading out highlights when paused.
   */
  function fadeHighlight() {
    if (!currentJoints) return;
    updateHighlightIntensity();
    Renderer.draw(currentJoints, footPath, lengths, currentAngle, showFootPath, false, highlightIntensities);

    if (hasActiveHighlights()) {
      highlightAnimId = requestAnimationFrame(fadeHighlight);
    } else {
      highlightAnimId = null;
      // Clean up fully-faded entries without redrawing — avoids a visible jump
      for (const barName in highlightIntensities) {
        if (highlightIntensities[barName].current <= 0.001) {
          delete highlightIntensities[barName];
        }
      }
    }
  }

  // ── Solve and render current frame ─────────────────────────
  function solveAndRender() {
    const result = Solver.solveLinkage(currentAngle, lengths, prevGuess);
    currentJoints = result.joints;

    if (result.converged) {
      solverWarning.style.display = 'none';
      // Update continuation state
      prevGuess = result.joints.slice(3).flat();
      // Accumulate foot path
      footPath.push(result.joints[7]);
    } else {
      solverWarning.style.display = 'block';
    }

    updateHighlightIntensity();
    Renderer.draw(currentJoints, footPath, lengths, currentAngle, showFootPath, false, highlightIntensities);
  }

  // ── Animation loop ─────────────────────────────────────────
  function togglePlay() {
    isPlaying = !isPlaying;
    if (isPlaying) {
      btnPlay.textContent = '⏸ Pause';
      lastTimestamp = performance.now();
      // Reset foot path for clean cycle
      footPath = [];
      prevGuess = null;
      animate();
    } else {
      btnPlay.textContent = '▶ Play';
      if (animFrameId) cancelAnimationFrame(animFrameId);
    }
  }

  function animate() {
    if (!isPlaying) return;

    const now = performance.now();
    const dt = (now - lastTimestamp) / 1000; // seconds
    lastTimestamp = now;

    // Advance angle: 360° per 5 seconds at 1× speed
    currentAngle += 360 * animSpeed * dt / 5;
    if (currentAngle >= 360) {
      currentAngle -= 360;
      // Keep foot path across cycle boundaries
    }

    // Update slider
    sliderAngle.value = currentAngle;
    angleDisplay.textContent = currentAngle.toFixed(1) + '°';

    solveAndRender();
    animFrameId = requestAnimationFrame(animate);
  }

  // ── Resize handler ─────────────────────────────────────────
  function onResize() {
    // Pre-solve a few frames to compute bounds
    const tempResult = Solver.solveLinkage(0, lengths, null);
    if (tempResult.converged) {
      // Solve a few angles to get bounds
      const sampleAngles = [0, 90, 180, 270, 360, 450, 540, 630];
      const allJoints = sampleAngles.map(a => {
        const r = Solver.solveLinkage(a, lengths, null);
        return r.joints;
      });
      // Scale margin proportionally with integerScale so the linkage stays visually the same size
      Renderer.computeBounds(allJoints, 15 * integerScale);
    }
    Renderer.resize();
    if (currentJoints) {
      Renderer.draw(currentJoints, footPath, lengths, currentAngle, showFootPath, false, highlightIntensities);
    }
  }

  // ── Reset to defaults ──────────────────────────────────────
  function resetToDefaults() {
    lengths = { ...DEFAULT_LENGTHS };
    currentAngle = 0;
    prevGuess = null;
    footPath = [];
    integerScale = 1;

    // Reset input step back to 0.1
    inputIds.forEach(id => {
      inputs[id].step = '0.1';
    });

    updateInputsFromLengths();
    sliderAngle.value = 0;
    angleDisplay.textContent = '0.0°';

    // Recompute view bounds for the reset lengths
    const tempResult = Solver.solveLinkage(0, lengths, null);
    if (tempResult.converged) {
      const sampleAngles = [0, 90, 180, 270, 360, 450, 540, 630];
      const allJoints = sampleAngles.map(a => {
        const r = Solver.solveLinkage(a, lengths, null);
        return r.joints;
      });
      Renderer.computeBounds(allJoints, 15 * integerScale);
    }
    Renderer.resize();

    // Highlight all bars + pivot distances with auto fade-out
    ALL_HIGHLIGHT_IDS.forEach(barName => {
      if (!highlightIntensities[barName]) {
        highlightIntensities[barName] = { current: 0, target: 0, reachedFull: false, fadeOutAfterFull: false };
      }
      highlightIntensities[barName].target = 1;
      highlightIntensities[barName].reachedFull = false;
      highlightIntensities[barName].fadeOutAfterFull = true;
    });
    highlightLastTime = performance.now();

    solveAndRender();

    // Start animating the fade-in (and subsequent fade-out) even when paused
    if (!isPlaying && currentJoints) {
      if (highlightAnimId) cancelAnimationFrame(highlightAnimId);
      animateHighlight();
    }

    // Flash sidebar bars + button
    flashSidebarBars();
    flashButton(btnReset);

    // Compute scores for the reset defaults
    scheduleScoreComputation(true);

    // Clear URL params
    window.history.replaceState(null, '', window.location.pathname);
  }

  function updateInputsFromLengths() {
    inputIds.forEach(id => {
      inputs[id].value = lengths[id].toFixed(1);
    });
  }

  // ── Export: PNG ────────────────────────────────────────────
  function exportPng() {
    // Temporarily render with the table visible for the exported image
    Renderer.draw(currentJoints, footPath, lengths, currentAngle, showFootPath, true);
    const dataUrl = canvas.toDataURL('image/png');
    // Restore the normal view (table hidden)
    Renderer.draw(currentJoints, footPath, lengths, currentAngle, showFootPath, false);

    const link = document.createElement('a');
    link.download = 'jansen_linkage.png';
    link.href = dataUrl;
    link.click();
  }

  // ── Integer Approximation ──────────────────────────────────
  function applyIntegerApproximation() {
    const scale = parseFloat(inputScale.value);
    if (isNaN(scale) || scale <= 0) {
      return;
    }

    // Reset to defaults first, then scale and round
    lengths = { ...DEFAULT_LENGTHS };
    currentAngle = 0;
    prevGuess = null;
    footPath = [];
    sliderAngle.value = 0;
    angleDisplay.textContent = '0.0°';

    // Track the integer scale factor for margin compensation
    integerScale = scale;

    let changed = false;
    inputIds.forEach(id => {
      const scaled = Math.round(lengths[id] * scale);
      if (scaled !== lengths[id]) {
        changed = true;
        lengths[id] = scaled;
      }
    });

    // Show warning
    integerWarning.style.display = 'block';

    // Switch input step to integer increments
    inputIds.forEach(id => {
      inputs[id].step = '1';
    });

    // Update inputs to reflect new integer values
    updateInputsFromLengths();

    // Recompute view bounds for the new scaled lengths
    const tempResult = Solver.solveLinkage(0, lengths, null);
    if (tempResult.converged) {
      const sampleAngles = [0, 90, 180, 270, 360, 450, 540, 630];
      const allJoints = sampleAngles.map(a => {
        const r = Solver.solveLinkage(a, lengths, null);
        return r.joints;
      });
      Renderer.computeBounds(allJoints, 15 * integerScale);
    }
    Renderer.resize();

    // Highlight all bars + pivot distances with auto fade-out
    ALL_HIGHLIGHT_IDS.forEach(barName => {
      if (!highlightIntensities[barName]) {
        highlightIntensities[barName] = { current: 0, target: 0, reachedFull: false, fadeOutAfterFull: false };
      }
      highlightIntensities[barName].target = 1;
      highlightIntensities[barName].reachedFull = false;
      highlightIntensities[barName].fadeOutAfterFull = true;
    });
    highlightLastTime = performance.now();

    solveAndRender();

    // Start animating the fade-in (and subsequent fade-out) even when paused
    if (!isPlaying && currentJoints) {
      if (highlightAnimId) cancelAnimationFrame(highlightAnimId);
      animateHighlight();
    }

    // Flash sidebar bars + button
    flashSidebarBars();
    flashButton(btnApplyScale);

    // Compute scores for the integer approximation
    scheduleScoreComputation(true);
  }

  // ── Shareable URL ──────────────────────────────────────────
  function shareUrl() {
    const params = new URLSearchParams();
    inputIds.forEach(id => {
      params.set(id, lengths[id].toFixed(1));
    });
    params.set('angle', currentAngle.toFixed(1));

    const url = window.location.origin + window.location.pathname + '?' + params.toString();

    navigator.clipboard.writeText(url).then(() => {
      exportStatus.textContent = '✅ Link copied to clipboard!';
      setTimeout(() => { exportStatus.textContent = ''; }, 3000);
    }).catch(() => {
      // Fallback
      prompt('Copy this URL:', url);
    });
  }

  // ── Boot ───────────────────────────────────────────────────
  window.addEventListener('DOMContentLoaded', init);
})();
