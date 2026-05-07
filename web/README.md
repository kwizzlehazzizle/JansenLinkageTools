# Jansen Linkage Simulator

Interactive web simulator for the 11-bar Jansen linkage (Strandbeest walking mechanism).

## Running the App

**You must serve the files over HTTP** — opening `index.html` directly with `file://` will not work because GIF export requires Web Workers, which browsers block on the `file://` protocol.

### Quick Start (PowerShell)

```powershell
cd web
.\start-server.ps1
```

Then open **http://localhost:8000** in your browser.

### Alternative Methods

**Python:**
```powershell
cd web
python -m http.server 8000
```

**Node.js:**
```powershell
cd web
npx serve
```

**Any other HTTP server** that serves static files from the `web/` directory.

## Features

- **Adjustable dimensions** — change all 13 bar lengths and pivot distances
- **Real-time animation** — play/pause with adjustable speed
- **Foot path trace** — toggle the walking foot trajectory
- **PNG export** — save the current frame
- **GIF export** — generate a 720-frame animation (~30-60 seconds)
- **Shareable URLs** — encode all dimensions in the URL for sharing

## License

MIT — see [LICENSE](LICENSE)

Third-party libraries: see [THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md)
