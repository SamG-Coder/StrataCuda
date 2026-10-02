@echo off
cd /d "%~dp0"
if not exist generated\strata.wasm (
  echo Run npm install and npm run build first. See README.md for Emscripten setup.
  pause
  exit /b 1
)
start "" http://127.0.0.1:8094
node scripts\serve.mjs
