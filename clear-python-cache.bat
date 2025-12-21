@echo off
echo =========================================================
echo   CLEARING PYTHON CACHE FILES
echo =========================================================
echo.

cd /d "D:\Trading Analyzer"

echo [*] Removing __pycache__ directories...
if exist backend\__pycache__ (
    echo   Removing backend\__pycache__...
    rmdir /s /q backend\__pycache__
    echo   ✓ Removed
) else (
    echo   No backend\__pycache__ found
)

if exist __pycache__ (
    echo   Removing root __pycache__...
    rmdir /s /q __pycache__
    echo   ✓ Removed
) else (
    echo   No root __pycache__ found
)

echo.
echo [*] Removing .pyc files...
del /q backend\*.pyc 2>nul
if errorlevel 1 (
    echo   No .pyc files found
) else (
    echo   ✓ Removed .pyc files
)

echo.
echo [*] Cache cleared!
echo.
echo Now restart the backend to load fresh code.
echo.
pause

