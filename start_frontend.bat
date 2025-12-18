@echo off
title Trading Analyzer Pro - Frontend Server
color 0B

echo.
echo =========================================================
echo   TRADING ANALYZER PRO - FRONTEND SERVER
echo =========================================================
echo.
echo   Features:
echo   - Modern React/Next.js UI
echo   - Live Price Ticker (Auto-refresh)
echo   - Symbol & Timeframe Selection
echo   - Drag & Drop Chart Upload
echo   - Annotated Chart Display
echo   - Technical Indicators Dashboard
echo   - Confluence Score Visualization
echo   - Trade Setup with Entry/SL/TPs
echo   - Support/Resistance Display
echo   - Market Data Panel
echo.
echo =========================================================
echo.

cd /d "D:\Trading Analyzer"

echo [*] Checking Node.js environment...
node --version
if errorlevel 1 (
    echo [ERROR] Node.js not found. Please install Node.js 18+
    pause
    exit /b 1
)

echo.
echo [*] Checking dependencies...
if not exist "node_modules" (
    echo [*] Installing dependencies...
    npm install
)

echo.
echo [*] Starting Next.js development server...
echo [*] Frontend will be available at http://localhost:3000
echo.
echo Press Ctrl+C to stop the server
echo.

npm run dev

pause
