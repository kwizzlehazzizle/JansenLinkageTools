# Start a local HTTP server for the Jansen Linkage web app
# Web Workers (required for GIF export) don't work with file:// protocol

Write-Host "Starting local server..." -ForegroundColor Green
Write-Host "Open http://localhost:8000 in your browser" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Use Python's built-in HTTP server (most users have Python installed)
python -m http.server 8000
