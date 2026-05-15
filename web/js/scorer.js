/**
 * Gait Quality Scorer
 *
 * Computes two metrics for a linkage configuration:
 *   — Shape Match: how close the foot path matches the original Jansen ideal
 *   — Flatness:    how flat the ground-contact phase of the stride is
 *
 * Both scores are percentages (lower = better).
 * The reference foot path is precomputed once from the default bar lengths.
 */

const Scorer = (() => {
  /* ── Precomputed reference foot path ───────────────────── */

  const DEFAULT_LENGTHS = {
    a: 38.0, b: 41.5, c: 39.3, d: 40.1,
    e: 55.8, f: 39.4, g: 36.7, h: 65.7,
    i: 49.0, j: 50.0, k: 61.9, l: 7.8, m: 15.0
  };

  /** @type {number[][]|null} — [[x,y], ...] for 360 angles */
  let referencePath = null;

  function ensureReference() {
    if (!referencePath) {
      const result = Solver.solveAllFrames(DEFAULT_LENGTHS, 360);
      referencePath = result.footPath;
    }
    return referencePath;
  }

  /* ── Compute foot path for arbitrary lengths ───────────── */

  /**
   * Solve all angles and return the foot path and convergence status.
   * @param {object} lengths
   * @param {number} [numFrames=360]
   * @returns {{footPath: number[][], allConverged: boolean}}
   */
  function computeFootPath(lengths, numFrames = 360) {
    const result = Solver.solveAllFrames(lengths, numFrames);
    return { footPath: result.footPath, allConverged: result.allConverged };
  }

  /* ── Shape score ───────────────────────────────────────── */
  // Port of compare_paths_simple from optimize_integer.py

  /**
   * Compare a test foot path against the reference Jansen path.
   * Returns RMS error as a percentage of the reference path size.
   * @param {number[][]} refPath
   * @param {number[][]} testPath
   * @returns {number} percentage (lower = better, 0 = identical)
   */
  function computeShapeScore(refPath, testPath) {
    const n = refPath.length;
    if (n === 0 || testPath.length === 0) return Infinity;

    // Centroids
    let refCx = 0, refCy = 0, tstCx = 0, tstCy = 0;
    for (let i = 0; i < n; i++) {
      refCx += refPath[i][0];  refCy += refPath[i][1];
      tstCx += testPath[i][0]; tstCy += testPath[i][1];
    }
    refCx /= n; refCy /= n;
    tstCx /= n; tstCy /= n;

    // Centered bounding boxes (for scale normalization)
    let refXmin = Infinity, refXmax = -Infinity;
    let refYmin = Infinity, refYmax = -Infinity;
    let tstXmin = Infinity, tstXmax = -Infinity;
    let tstYmin = Infinity, tstYmax = -Infinity;

    for (let i = 0; i < n; i++) {
      const rx = refPath[i][0] - refCx;
      const ry = refPath[i][1] - refCy;
      const tx = testPath[i][0] - tstCx;
      const ty = testPath[i][1] - tstCy;

      if (rx < refXmin) refXmin = rx;
      if (rx > refXmax) refXmax = rx;
      if (ry < refYmin) refYmin = ry;
      if (ry > refYmax) refYmax = ry;
      if (tx < tstXmin) tstXmin = tx;
      if (tx > tstXmax) tstXmax = tx;
      if (ty < tstYmin) tstYmin = ty;
      if (ty > tstYmax) tstYmax = ty;
    }

    const refSize = Math.sqrt((refXmax - refXmin) ** 2 + (refYmax - refYmin) ** 2);
    const tstSize = Math.sqrt((tstXmax - tstXmin) ** 2 + (tstYmax - tstYmin) ** 2);

    if (tstSize < 1e-10) return Infinity;
    const scale = refSize / tstSize;

    // Scaled RMS error
    let ss = 0;
    for (let i = 0; i < n; i++) {
      const dx = (refPath[i][0] - refCx) - (testPath[i][0] - tstCx) * scale;
      const dy = (refPath[i][1] - refCy) - (testPath[i][1] - tstCy) * scale;
      ss += dx * dx + dy * dy;
    }
    const rms = Math.sqrt(ss / n);

    if (refSize < 1e-10) return Infinity;
    return (rms / refSize) * 100;
  }

  /* ── Flatness score ────────────────────────────────────── */
  // Port of compute_flatness from optimize_integer.py

  /**
   * Compute flatness of the ground-contact phase.
   * Ground contact = frames where Y ≤ min_Y + 0.2 × Y_range
   * Flatness = std(Y) during ground contact, normalized by Y_range as a percentage.
   * Returns Infinity if the foot path is degenerate or has no meaningful ground phase.
   * @param {number[][]} footPath
   * @returns {number} percentage (lower = flatter, 0 = perfectly flat)
   */
  function computeFlatness(footPath) {
    const n = footPath.length;
    if (n === 0) return Infinity;

    // Check for NaN in the path
    for (let i = 0; i < n; i++) {
      if (isNaN(footPath[i][0]) || isNaN(footPath[i][1])) return Infinity;
    }

    // Y min and range
    let yMin = footPath[0][1];
    let yMax = footPath[0][1];
    for (let i = 1; i < n; i++) {
      const y = footPath[i][1];
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
    const yRange = yMax - yMin;
    // Degenerate: no meaningful vertical movement → can't assess flatness
    if (yRange < 1e-3) return Infinity;

    // Ground threshold
    const threshold = yMin + 0.2 * yRange;

    // Collect ground Y values
    let groundSum = 0;
    let groundCount = 0;
    for (let i = 0; i < n; i++) {
      if (footPath[i][1] <= threshold) {
        groundSum += footPath[i][1];
        groundCount++;
      }
    }
    // Require a meaningful ground-contact phase (at least 5% of the cycle)
    if (groundCount < Math.max(2, n * 0.05)) return Infinity;

    const groundMean = groundSum / groundCount;

    // Standard deviation
    let ss = 0;
    for (let i = 0; i < n; i++) {
      if (footPath[i][1] <= threshold) {
        const diff = footPath[i][1] - groundMean;
        ss += diff * diff;
      }
    }
    const std = Math.sqrt(ss / groundCount);

    // Normalize by Y range → percentage
    return (std / yRange) * 100;
  }

  /* ── Combined evaluation ───────────────────────────────── */

  /**
   * Compute both scores for a given set of bar lengths.
   * @param {object} lengths
   * @param {number} [numFrames=360]
   * @returns {{shapeScore: number, flatnessScore: number, converged: boolean}}
   */
  function evaluate(lengths, numFrames = 360) {
    const result = computeFootPath(lengths, numFrames);
    const footPath = result.footPath;

    // Solver didn't converge for at least one frame
    if (!result.allConverged) {
      return { shapeScore: Infinity, flatnessScore: Infinity, converged: false };
    }

    // Check for NaN in the path (safety net)
    for (let i = 0; i < footPath.length; i++) {
      if (isNaN(footPath[i][0]) || isNaN(footPath[i][1])) {
        return { shapeScore: Infinity, flatnessScore: Infinity, converged: false };
      }
    }

    const ref = ensureReference();
    return {
      shapeScore: computeShapeScore(ref, footPath),
      flatnessScore: computeFlatness(footPath),
      converged: true
    };
  }

  return { ensureReference, computeFootPath, computeShapeScore, computeFlatness, evaluate };
})();
