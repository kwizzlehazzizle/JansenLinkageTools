/**
 * Canvas renderer for the Jansen Linkage visualization.
 * Draws bars, joints, crank circle, foot path, and a lengths table.
 */

const Renderer = (() => {
  // Bar colors (from Python reference)
  const BAR_COLORS = {
    b: '#ff6b6b', c: '#e8838b', d: '#f39c12', e: '#f1c40f',
    f: '#e74c3c', g: '#e67e22', h: '#9b59b6', i: '#3498db',
    j: '#1abc9c', k: '#2980b9', m: '#1a5276'
  };

  // Bar connections: [joint_a, joint_b, linewidth, bar_name]
  const BARS = [
    [0, 3, 4, 'b'], [0, 4, 4, 'd'], [3, 4, 4, 'e'],
    [0, 6, 4, 'c'], [4, 5, 4, 'f'], [5, 6, 4, 'g'],
    [5, 7, 4, 'h'], [6, 7, 4, 'i'], [2, 6, 4, 'k'],
    [2, 3, 4, 'j'], [1, 2, 5, 'm']
  ];

  // Background colors
  const BG_COLOR = '#1a1a2e';
  const PANEL_COLOR = '#16213e';
  const BORDER_COLOR = '#0f3460';
  const TEXT_COLOR = '#ecf0f1';
  const FOOT_COLOR = '#00ff88';
  const CRANK_CIRCLE_COLOR = '#2ecc71';
  const JOINT_COLOR = '#f1c40f';
  const JOINT_EDGE = '#e67e22';
  const FIXED_JOINT_COLOR = '#e74c3c';
  const FIXED_JOINT_EDGE = '#c0392b';

  let canvas, ctx;
  let viewXMin, viewXMax, viewYMin, viewYMax;
  let scale, offsetX, offsetY;
  let canvasWidth, canvasHeight;

  /**
   * Initialize the renderer.
   * @param {HTMLCanvasElement} canvasEl - The canvas element
   */
  function init(canvasEl) {
    canvas = canvasEl;
    ctx = canvas.getContext('2d', { willReadFrequently: true });
  }

  /**
   * Compute view bounds from all joint positions.
   * @param {number[][][]} allJoints - Array of frames, each frame is [8][2]
   * @param {number} margin - Margin in world units
   */
  function computeBounds(allJoints, margin = 15) {
    let xMin = Infinity, xMax = -Infinity;
    let yMin = Infinity, yMax = -Infinity;
    for (const frame of allJoints) {
      for (const [x, y] of frame) {
        if (x < xMin) xMin = x;
        if (x > xMax) xMax = x;
        if (y < yMin) yMin = y;
        if (y > yMax) yMax = y;
      }
    }
    viewXMin = xMin - margin;
    viewXMax = xMax + margin;
    viewYMin = yMin - margin;
    viewYMax = yMax + margin;
  }

  /**
   * Update the canvas size and compute the world-to-screen transform.
   */
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvasWidth = rect.width;
    canvasHeight = rect.height;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const worldW = viewXMax - viewXMin;
    const worldH = viewYMax - viewYMin;
    scale = Math.min(canvasWidth / worldW, canvasHeight / worldH);

    const cx = (viewXMin + viewXMax) / 2;
    const cy = (viewYMin + viewYMax) / 2;
    offsetX = canvasWidth / 2 - cx * scale;
    offsetY = canvasHeight / 2 + cy * scale; // flip Y (canvas Y is down)
  }

  /**
   * Transform world coordinates to canvas coordinates.
   */
  function toScreen(x, y) {
    return [x * scale + offsetX, -y * scale + offsetY];
  }

  /**
   * Draw a single frame of the linkage.
   * @param {number[][]} joints - 8 joint positions [x, y]
   * @param {number[][]} footPath - Accumulated foot path points
   * @param {object} lengths - Current bar lengths
   * @param {number} angle - Current crank angle in degrees
   * @param {boolean} showFootPath - Whether to show the foot path trace
   */
  function draw(joints, footPath, lengths, angle, showFootPath = true) {
    // Clear
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Draw lengths table (left side)
    drawTable(lengths);

    // Draw title
    ctx.save();
    ctx.fillStyle = TEXT_COLOR;
    ctx.font = 'bold 20px "Segoe UI", system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('Jansen Linkage — Strandbeest Walking Mechanism', canvasWidth / 2, 12);
    ctx.restore();

    // Draw crank circle (dashed)
    const [cx, cy] = toScreen(joints[1][0], joints[1][1]);
    const crankR = lengths.m * scale;
    ctx.save();
    ctx.strokeStyle = CRANK_CIRCLE_COLOR;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.4;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.arc(cx, cy, crankR, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    // Draw foot path trace
    if (showFootPath && footPath.length > 1) {
      ctx.save();
      ctx.strokeStyle = FOOT_COLOR;
      ctx.lineWidth = 3;
      ctx.globalAlpha = 0.7;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      const [fx0, fy0] = toScreen(footPath[0][0], footPath[0][1]);
      ctx.moveTo(fx0, fy0);
      for (let i = 1; i < footPath.length; i++) {
        const [fx, fy] = toScreen(footPath[i][0], footPath[i][1]);
        ctx.lineTo(fx, fy);
      }
      ctx.stroke();
      ctx.restore();
    }

    // Draw bars
    for (const [j1, j2, lw, name] of BARS) {
      const [x1, y1] = toScreen(joints[j1][0], joints[j1][1]);
      const [x2, y2] = toScreen(joints[j2][0], joints[j2][1]);
      ctx.save();
      ctx.strokeStyle = BAR_COLORS[name];
      ctx.lineWidth = lw;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
      ctx.restore();
    }

    // Draw joints
    for (let i = 0; i < 8; i++) {
      const [jx, jy] = toScreen(joints[i][0], joints[i][1]);
      const isFixed = (i === 0 || i === 1);
      const radius = isFixed ? 7 : 4;
      ctx.save();
      ctx.fillStyle = isFixed ? FIXED_JOINT_COLOR : JOINT_COLOR;
      ctx.strokeStyle = isFixed ? FIXED_JOINT_EDGE : JOINT_EDGE;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(jx, jy, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // Draw angle label
    ctx.save();
    ctx.fillStyle = '#f39c12';
    ctx.font = 'bold 15px "Segoe UI", system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText(`Crank Angle: ${angle.toFixed(1)}°`, canvasWidth - 15, 12);
    ctx.restore();
  }

  /**
   * Draw the bar lengths table on the left side.
   * @param {object} lengths - Bar lengths
   */
  function drawTable(lengths) {
    const tableX = 12;
    const tableY = 50;
    const tableW = 140;
    const rowH = 22;

    // Background panel
    ctx.save();
    ctx.fillStyle = PANEL_COLOR;
    ctx.strokeStyle = BORDER_COLOR;
    ctx.lineWidth = 1;
    roundRect(ctx, tableX, tableY, tableW, 280, 8);
    ctx.fill();
    ctx.stroke();

    // Title
    ctx.fillStyle = '#f39c12';
    ctx.font = 'bold 16px "Segoe UI", system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('Bar Lengths', tableX + tableW / 2, tableY + 8);

    // Separator
    ctx.strokeStyle = BORDER_COLOR;
    ctx.beginPath();
    ctx.moveTo(tableX + 10, tableY + 32);
    ctx.lineTo(tableX + tableW - 10, tableY + 32);
    ctx.stroke();

    // Rows
    const barNames = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'm'];
    ctx.font = 'bold 13px "Consolas", "Courier New", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    for (let i = 0; i < barNames.length; i++) {
      const name = barNames[i];
      const y = tableY + 44 + i * rowH;
      const color = BAR_COLORS[name];

      // Bar name (left)
      ctx.fillStyle = color;
      ctx.fillText(name, tableX + 15, y);

      // Length value (right)
      ctx.textAlign = 'right';
      ctx.fillText(lengths[name].toFixed(1), tableX + tableW - 15, y);
      ctx.textAlign = 'left';
    }

    ctx.restore();
  }

  /**
   * Draw a rounded rectangle path.
   */
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  return { init, computeBounds, resize, draw, toScreen };
})();
