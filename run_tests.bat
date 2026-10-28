@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m pytest -q tests
) else (
  python -m pytest -q tests
)
pause
