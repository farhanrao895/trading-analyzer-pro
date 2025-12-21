@echo off
echo =========================================================
echo   KILLING ALL BACKEND PROCESSES AND CLEARING CACHE
echo =========================================================
echo.

cd /d "D:\Trading Analyzer"

echo [1/4] Killing ALL Python processes...
taskkill /F /IM python.exe >nul 2>&1
if errorlevel 1 (
    echo   No Python processes found
) else (
    echo   ✓ All Python processes killed
)

echo.
echo [2/4] Waiting for processes to terminate...
timeout /t 2 /nobreak >nul

echo [3/4] Clearing Python cache...
if exist backend\__pycache__ (
    echo   Removing backend\__pycache__...
    rmdir /s /q backend\__pycache__
    echo   ✓ Cache cleared
) else (
    echo   No cache found
)

del /q backend\*.pyc 2>nul

echo.
echo [4/4] Verifying port 8002 is free...
netstat -ano | findstr ":8002" | findstr "LISTENING"
if errorlevel 1 (
    echo   ✓ Port 8002 is free
) else (
    echo   ⚠ WARNING: Port 8002 still in use!
    echo   You may need to restart your computer to clear all processes
)

echo.
echo =========================================================
echo   CLEANUP COMPLETE
echo =========================================================
echo.
echo Now run: restart-backend-fresh.bat
echo.
pause

