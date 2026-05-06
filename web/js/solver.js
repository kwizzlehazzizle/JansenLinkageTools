/**
 * Jansen Linkage Solver (ported from Python/SciPy)
 *
 * Solves 10 distance constraints for 5 unknown joint positions (J3–J7)
 * using a Levenberg–Marquardt optimizer with numerical Jacobian.
 *
 * Joint topology (same as Python reference):
 *   J0: Fixed pivot 1 (origin) — bars b, c, d
 *   J1: Fixed pivot 2 (a, l) — crank m
 *   J2: Crank tip — bars m, j, k
 *   J3: Top vertex — bars e, b, j
 *   J4: Top-left vertex — bars e, d, f
 *   J5: Left-middle vertex — bars f, g, h
 *   J6: Center-bottom vertex — bars c, g, i, k
 *   J7: Foot — bars h, i
 */

const Solver = (() => {
  /* ── Levenberg-Marquardt optimizer ───────────────────────── */

  /**
   * Solve min ||F(x)||² using Levenberg-Marquardt.
   * @param {function} F - Returns array of residuals given x
   * @param {number[]} x0 - Initial guess
   * @param {object} opts - Tolerance options
   * @returns {{x: number[], converged: boolean, cost: number}}
   */
  function levenbergMarquardt(F, x0, opts = {}) {
    const ftol = opts.ftol ?? 1e-12;
    const xtol = opts.xtol ?? 1e-12;
    const gtol = opts.gtol ?? 1e-12;
    const maxIter = opts.maxIter ?? 200;

    let x = Array.from(x0);
    let n = x.length;
    let lambda = 0.001;
    let scale = 1e-3; // finite-difference step scale

    let residuals = F(x);
    let cost = residuals.reduce((s, r) => s + r * r, 0);

    for (let iter = 0; iter < maxIter; iter++) {
      // Build Jacobian numerically (finite differences)
      let m = residuals.length;
      let J = new Float64Array(m * n); // row-major: J[i*n + j]
      for (let j = 0; j < n; j++) {
        let h = scale * (Math.abs(x[j]) + 1);
        x[j] += h;
        let fp = F(x);
        x[j] -= 2 * h;
        let fm = F(x);
        x[j] += h; // restore
        for (let i = 0; i < m; i++) {
          J[i * n + j] = (fp[i] - fm[i]) / (2 * h);
        }
      }

      // Jᵀ·J  (n×n, symmetric — store full for simplicity)
      let JtJ = new Float64Array(n * n);
      for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
          let s = 0;
          for (let k = 0; k < m; k++) {
            s += J[k * n + i] * J[k * n + j];
          }
          JtJ[i * n + j] = s;
          JtJ[j * n + i] = s;
        }
      }

      // Jᵀ·r  (n-vector)
      let Jtr = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let k = 0; k < m; k++) {
          s += J[k * n + i] * residuals[k];
        }
        Jtr[i] = s;
      }

      // Check gradient norm
      let gradNorm = Math.sqrt(Jtr.reduce((s, v) => s + v * v, 0));
      if (gradNorm < gtol) {
        return { x, converged: true, cost };
      }

      // Solve (JᵀJ + λ·diag(JᵀJ)) · dx = -Jᵀr
      let A = new Float64Array(n * n);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          A[i * n + j] = JtJ[i * n + j];
        }
        A[i * n + i] += lambda * (JtJ[i * n + i] + 1e-10);
      }
      let b = Jtr.map(v => -v);

      let dx = solveLinearSystem(A, b, n);
      if (!dx) {
        // Fallback: increase lambda and retry
        lambda *= 10;
        continue;
      }

      // Trial step
      let xp = x.map((v, i) => v + dx[i]);
      let rp = F(xp);
      let costp = rp.reduce((s, r) => s + r * r, 0);

      if (costp < cost) {
        let rho = (cost - costp) / (gradNorm * lambda);
        x = xp;
        residuals = rp;
        cost = costp;

        // Check convergence
        let maxDx = 0;
        for (let i = 0; i < n; i++) {
          let d = Math.abs(dx[i]) / (Math.abs(x[i]) + 1);
          if (d > maxDx) maxDx = d;
        }
        if (maxDx < xtol) return { x, converged: true, cost };
        if (cost < ftol) return { x, converged: true, cost };

        lambda = Math.max(lambda * Math.max(1 / 3, 1 - rho), 1e-10);
      } else {
        lambda *= 10;
      }
    }

    return { x, converged: cost < 1e-4, cost };
  }

  /**
   * Solve Ax = b using Gaussian elimination with partial pivoting.
   * @param {Float64Array} A - n×n matrix (row-major)
   * @param {number[]} b - RHS vector
   * @param {number} n - dimension
   * @returns {number[]|null} solution or null if singular
   */
  function solveLinearSystem(A, b, n) {
    // Augmented matrix [A|b]
    let M = new Float64Array(n * (n + 1));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) M[i * (n + 1) + j] = A[i * n + j];
      M[i * (n + 1) + n] = b[i];
    }

    for (let col = 0; col < n; col++) {
      // Partial pivoting
      let maxVal = Math.abs(M[col * (n + 1) + col]);
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        let v = Math.abs(M[row * (n + 1) + col]);
        if (v > maxVal) { maxVal = v; maxRow = row; }
      }
      if (maxVal < 1e-14) return null; // singular

      if (maxRow !== col) {
        for (let j = 0; j <= n; j++) {
          let tmp = M[col * (n + 1) + j];
          M[col * (n + 1) + j] = M[maxRow * (n + 1) + j];
          M[maxRow * (n + 1) + j] = tmp;
        }
      }

      // Eliminate below
      for (let row = col + 1; row < n; row++) {
        let factor = M[row * (n + 1) + col] / M[col * (n + 1) + col];
        for (let j = col; j <= n; j++) {
          M[row * (n + 1) + j] -= factor * M[col * (n + 1) + j];
        }
      }
    }

    // Back substitution
    let x = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = M[i * (n + 1) + n];
      for (let j = i + 1; j < n; j++) {
        s -= M[i * (n + 1) + j] * x[j];
      }
      x[i] = s / M[i * (n + 1) + i];
    }
    return x;
  }

  /* ── Linkage solver ──────────────────────────────────────── */

  /**
   * Solve the Jansen linkage for a given crank angle.
   *
   * @param {number} thetaDeg - Crank angle in degrees
   * @param {object} lengths - Bar lengths {a, b, c, ..., m}
   * @param {number[]|null} prevGuess - Previous solution for continuation
   * @returns {{joints: number[][], converged: boolean, cost: number}}
   */
  function solveLinkage(thetaDeg, lengths, prevGuess = null) {
    const theta = thetaDeg * Math.PI / 180;

    // Fixed pivots
    const J0 = [0, 0];
    const J1 = [lengths.a, lengths.l];

    // Crank tip (known from angle)
    const J2 = [
      J1[0] + lengths.m * Math.cos(theta),
      J1[1] + lengths.m * Math.sin(theta)
    ];

    // Constraint function: 10 distance equations
    const constraints = (x) => {
      const J3 = [x[0], x[1]];
      const J4 = [x[2], x[3]];
      const J5 = [x[4], x[5]];
      const J6 = [x[6], x[7]];
      const J7 = [x[8], x[9]];

      return [
        dist(J3, J0) - lengths.b,  // b: J0↔J3
        dist(J3, J2) - lengths.j,  // j: J2↔J3
        dist(J4, J3) - lengths.e,  // e: J3↔J4
        dist(J4, J0) - lengths.d,  // d: J0↔J4
        dist(J5, J4) - lengths.f,  // f: J4↔J5
        dist(J6, J0) - lengths.c,  // c: J0↔J6
        dist(J6, J2) - lengths.k,  // k: J2↔J6
        dist(J6, J5) - lengths.g,  // g: J5↔J6
        dist(J7, J5) - lengths.h,  // h: J5↔J7
        dist(J7, J6) - lengths.i,  // i: J6↔J7
      ];
    };

    // Initial guess (continuation or default)
    let x0;
    if (prevGuess) {
      x0 = Array.from(prevGuess);
    } else {
      x0 = [
        lengths.b * 0.3,  lengths.b * 0.9,   // J3
        -lengths.d * 0.7, lengths.d * 0.7,   // J4
        -lengths.f * 0.8, 0.0,               // J5
        lengths.c * 0.5, -lengths.c * 0.8,   // J6
        -lengths.h * 0.3, -lengths.h * 0.9,  // J7
      ];
    }

    const result = levenbergMarquardt(constraints, x0, {
      ftol: 1e-12, xtol: 1e-12, gtol: 1e-12, maxIter: 200
    });

    const J3 = [result.x[0], result.x[1]];
    const J4 = [result.x[2], result.x[3]];
    const J5 = [result.x[4], result.x[5]];
    const J6 = [result.x[6], result.x[7]];
    const J7 = [result.x[8], result.x[9]];

    return {
      joints: [J0, J1, J2, J3, J4, J5, J6, J7],
      converged: result.converged,
      cost: result.cost
    };
  }

  function dist(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    return Math.sqrt(dx * dx + dy * dy);
  }

  /* ── Pre-solve all frames (for GIF export) ───────────────── */

  /**
   * Pre-compute all linkage positions for 0°–720°.
   * @param {object} lengths - Bar lengths
   * @param {number} numFrames - Number of frames (default 720)
   * @returns {{frames: {joints: number[][], converged: boolean}[], footPath: number[][]}}
   */
  function solveAllFrames(lengths, numFrames = 720) {
    const frames = [];
    const footPath = [];
    let prevGuess = null;

    for (let i = 0; i < numFrames; i++) {
      const angle = (i / numFrames) * 720;
      const result = solveLinkage(angle, lengths, prevGuess);
      frames.push(result);
      footPath.push(result.joints[7]); // J7 is the foot
      prevGuess = result.joints.slice(3).flat();
    }

    return { frames, footPath };
  }

  return { solveLinkage, solveAllFrames };
})();
