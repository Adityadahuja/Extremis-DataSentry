@echo off
cd /d "%~dp0"
echo ========================================================
echo   Running Extremis DataSentry Test Suite
echo ========================================================
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m pytest -v
) else if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" -m pytest -v
) else (
    python -m pytest -v
)
pause
