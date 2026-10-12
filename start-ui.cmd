@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONPATH=%~dp0src;%PYTHONPATH%
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m mini_harness.ui
) else (
  where uv >nul 2>nul
  if errorlevel 1 (
    echo Please install uv first, then run this launcher again.
  ) else (
    uv run python -m mini_harness.ui
  )
)
pause
