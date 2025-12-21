@echo off
echo =========================================================
echo   RESTARTING BACKEND WITH FRESH CODE (NO CACHE)
echo =========================================================
echo.

cd /d "D:\Trading Analyzer"

echo [1/4] Killing ALL Python processes (nuclear option)...
taskkill /F /IM python.exe >nul 2>&1
if errorlevel 1 (
    echo   No Python processes found
) else (
    echo   ✓ All Python processes killed
)
echo   Waiting for processes to terminate...
timeout /t 3 /nobreak >nul

echo.
echo [2/4] Clearing Python cache...
if exist backend\__pycache__ (
    echo   Removing backend\__pycache__...
    rmdir /s /q backend\__pycache__
    echo   ✓ Cache cleared
) else (
    echo   No cache found
)

del /q backend\*.pyc 2>nul

echo.
echo [3/4] Verifying port 8002 is free...
netstat -ano | findstr ":8002" | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    echo   ✓ Port 8002 is free
) else (
    echo   ⚠ WARNING: Port 8002 still in use! Trying to kill again...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8002" ^| findstr "LISTENING" 2^>nul') do (
        taskkill /F /PID %%a >nul 2>&1
    )
    timeout /t 2 /nobreak >nul
)

echo.
echo [4/4] Starting backend with latest code...
echo   Using: python -m uvicorn backend.main:app --host 0.0.0.0 --port 8002 --reload
echo.
start "Trading Analyzer - Backend (Fresh)" cmd /k "cd /d %~dp0 && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8002 --reload"

echo.
echo Backend starting on http://localhost:8002
echo Wait 5 seconds for it to initialize...
echo.
echo IMPORTANT: Watch the backend terminal window for:
echo   - INFO: Application startup complete.
echo   - Then test: http://localhost:8002/api/indicators/ACTUSDT/4h
echo   - You should see [S/R] debug logs immediately!
echo.
pause

