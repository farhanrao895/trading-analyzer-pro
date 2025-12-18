@echo off
title Trading Analyzer Pro - Backend Server
color 0A

echo.
echo =========================================================
echo   TRADING ANALYZER PRO - BACKEND SERVER
echo =========================================================
echo.
echo   Features:
echo   - FastAPI REST API (Port 8002)
echo   - Binance Market Data Integration
echo   - Technical Indicator Engine (RSI, EMA, MACD, BB, ATR)
echo   - Support/Resistance Detection
echo   - Fibonacci Retracement Levels
echo   - Order Book Depth Analysis
echo   - Gemini AI Chart Analysis
echo   - Automatic Price-to-Pixel Mapping
echo   - Chart Annotation Drawing
echo.
echo =========================================================
echo.

cd /d "D:\Trading Analyzer"

echo [*] Checking Python environment...
python --version
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

echo.
echo [*] Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [*] Installing dependencies...
    python -m pip install -r requirements.txt
)

echo.
echo [*] Starting FastAPI server on http://localhost:8002
echo [*] API Documentation: http://localhost:8002/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8002 --reload

pause
