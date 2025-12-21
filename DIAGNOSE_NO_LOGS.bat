@echo off
echo =========================================================
echo   DIAGNOSING: Why No Backend Logs Appear
echo =========================================================
echo.

cd /d "D:\Trading Analyzer"

echo [1/5] Checking if backend is running on port 8002...
netstat -ano | findstr ":8002" | findstr "LISTENING"
if errorlevel 1 (
    echo   ❌ ERROR: No process listening on port 8002
    echo   Backend is NOT running!
    goto :end
) else (
    echo   ✓ Backend is running on port 8002
)

echo.
echo [2/5] Checking Python processes...
tasklist | findstr "python.exe"
if errorlevel 1 (
    echo   ❌ No Python processes found
) else (
    echo   ✓ Python processes found
)

echo.
echo [3/5] Checking if main.py has new code...
findstr /C:"v2.0_fixed_support_detection" backend\main.py >nul
if errorlevel 1 (
    echo   ❌ ERROR: Version marker NOT found in main.py
    echo   The new code is NOT in the file!
) else (
    echo   ✓ Version marker found - new code is in file
)

echo.
echo [4/5] Checking for Python cache...
if exist backend\__pycache__ (
    echo   ⚠ WARNING: __pycache__ directory exists
    echo   This might be using old cached code!
    dir backend\__pycache__ /b
) else (
    echo   ✓ No __pycache__ found
)

echo.
echo [5/5] Testing backend directly...
echo   Calling: http://localhost:8002/api/indicators/ACTUSDT/4h
echo   WATCH THE BACKEND TERMINAL WINDOW NOW!
echo.
timeout /t 2 >nul
curl -s http://localhost:8002/api/indicators/ACTUSDT/4h >nul 2>&1

echo.
echo =========================================================
echo   DIAGNOSIS COMPLETE
echo =========================================================
echo.
echo IMPORTANT: Did you see logs in the BACKEND terminal?
echo.
echo If NO logs appeared:
echo   1. Find the backend terminal window (titled "Backend")
echo   2. Make sure it's visible (not minimized)
echo   3. Check if it shows "INFO: Application startup complete."
echo   4. The backend terminal is DIFFERENT from frontend terminal
echo.
echo If backend terminal shows startup but no API logs:
echo   - The endpoint might not be getting called
echo   - Or Python print() is being suppressed
echo   - Or there's a different backend instance running
echo.
pause

:end

