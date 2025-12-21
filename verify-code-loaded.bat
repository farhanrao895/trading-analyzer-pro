@echo off
echo =========================================================
echo   VERIFYING NEW CODE IS LOADED
echo =========================================================
echo.

cd /d "D:\Trading Analyzer"

echo [*] Checking if new code exists in backend/main.py...
echo.

findstr /C:"recency_weight: float = 0.7" backend\main.py >nul
if errorlevel 1 (
    echo   ❌ ERROR: recency_weight parameter NOT FOUND
    echo   The new code is NOT in the file!
) else (
    echo   ✓ Found: recency_weight parameter
)

findstr /C:"Max support distance" backend\main.py >nul
if errorlevel 1 (
    echo   ❌ ERROR: Max support distance filter NOT FOUND
) else (
    echo   ✓ Found: Max support distance filter
)

findstr /C:"FILTERED OUT" backend\main.py >nul
if errorlevel 1 (
    echo   ❌ ERROR: FILTERED OUT debug code NOT FOUND
) else (
    echo   ✓ Found: FILTERED OUT debug code
)

findstr /C:"find_support_resistance() CALLED" backend\main.py >nul
if errorlevel 1 (
    echo   ❌ ERROR: Debug logging NOT FOUND
) else (
    echo   ✓ Found: Debug logging statements
)

echo.
echo =========================================================
echo   VERIFICATION COMPLETE
echo =========================================================
echo.
echo If all checks passed (✓), the code is in the file.
echo If backend still shows old behavior, it's a cache issue.
echo.
echo Next steps:
echo   1. Run: clear-python-cache.bat
echo   2. Run: restart-backend-fresh.bat
echo   3. Test: http://localhost:8002/api/indicators/ACTUSDT/4h
echo   4. Check backend terminal for [S/R] logs
echo.
pause

