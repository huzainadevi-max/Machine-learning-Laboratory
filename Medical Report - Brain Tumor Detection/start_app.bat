@echo off
title NeuroScan AI - Brain Tumor Detection Platform
echo ======================================================================
echo   NEUROSCAN AI : BRAIN TUMOR DETECTION ^& ML DIAGNOSTIC STUDIO
echo   Opening Chrome / Default Browser on http://127.0.0.1:8000
echo ======================================================================

:: Launch browser in background after short delay
start "" http://127.0.0.1:8000

:: Run the FastAPI Web Server
python app.py
if errorlevel 1 (
    echo.
    echo [!] Falling back to absolute Python interpreter path...
    "%LOCALAPPDATA%\Python\bin\python.exe" app.py
)
pause
