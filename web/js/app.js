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

  // Input elements for each dimension
  const inputIds = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'];
  const inputs = {};
  inputIds.forEach(id => {
    inputs[id] = document.getElementById('input-' + id);
  });

  // Bar IDs that correspond to drawn bars on the canvas (a and l are pivot distances, not bars)
  const BAR_HIGHLIGHT_IDS = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'm'];

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
  }

  // ── Event listeners ────────────────────────────────────────
  function setupEventListeners() {
    // Dimension inputs: live-update on change
    inputIds.forEach(id => {
      inputs[id].addEventListener('input', onDimensionChange);
      inputs[id].addEventListener('blur', onDimensionChange);
      // Hover highlight for bars that have corresponding bars on the canvas
      if (BAR_HIGHLIGHT_IDS.includes(id)) {
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
  }

  // ── Bar highlight on hover ─────────────────────────────────
  let highlightAnimId = null;  // rAF handle for highlight animation when paused

  function onBarHover(barName) {
    // If another bar was hovered, start fading it out
    if (currentHoveredBar && currentHoveredBar !== barName && highlightIntensities[currentHoveredBar]) {
      highlightIntensities[currentHoveredBar].target = 0;
    }

    // Set new bar's target to 1 (fade in)
    if (!highlightIntensities[barName]) {
      highlightIntensities[barName] = { current: 0, target: 0 };
    }
    highlightIntensities[barName].target = 1;
    currentHoveredBar = barName;
    highlightLastTime = performance.now();

    // Start animating the fade-in even when paused
    if (!isPlaying && currentJoints) {
      if (highlightAnimId) cancelAnimationFrame(highlightAnimId);
      animateHighlight();
    }
  }

  function onBarHoverOut() {
    // Set the previously hovered bar's target to 0 (fade out)
    if (currentHoveredBar && highlightIntensities[currentHoveredBar]) {
      highlightIntensities[currentHoveredBar].target = 0;
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
        // Fading in over 0.25s
        state.current += dt / 0.25;
        if (state.current >= 1) state.current = 1;
        anyChanging = true;
      } else if (state.current > state.target) {
        // Fading out over 1.0s
        state.current -= dt / 1.0;
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
      if (Math.abs(state.current - state.target) > 0.005) return true;
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
      highlightAnimId = null;
      highlightIntensities = {};
      Renderer.draw(currentJoints, footPath, lengths, currentAngle, showFootPath, false, {});
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

    // Advance angle: 360° per 10 seconds at 1× speed
    currentAngle += 360 * animSpeed * dt / 10;
    if (currentAngle >= 720) {
      currentAngle -= 720;
      footPath = []; // Reset foot path after 2 revolutions
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
      Renderer.computeBounds(allJoints, 15);
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
    updateInputsFromLengths();
    sliderAngle.value = 0;
    angleDisplay.textContent = '0.0°';
    solveAndRender();
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
