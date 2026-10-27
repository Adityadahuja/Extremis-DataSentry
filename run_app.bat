@echo off
cd /d "%~dp0"
echo ========================================================
echo   Launching Extremis DataSentry Web Dashboard
echo ========================================================
if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run app.py --browser.gatherUsageStats=false
) else if exist "..\.venv\Scripts\streamlit.exe" (
    "..\.venv\Scripts\streamlit.exe" run app.py --browser.gatherUsageStats=false
) else (
    streamlit run app.py --browser.gatherUsageStats=false
)
pause
