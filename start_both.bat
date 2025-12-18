@echo off
title Trading Analyzer Pro - Launcher
color 0E

echo.
echo =========================================================
echo   TRADING ANALYZER PRO - SYSTEM LAUNCHER
echo =========================================================
echo.
echo   Starting both servers...
echo.
echo   Backend:  http://localhost:8002 (API + AI Analysis)
echo   Frontend: http://localhost:3000 (User Interface)
echo.
echo   API Docs: http://localhost:8002/docs
echo.
echo =========================================================
echo.

cd /d "D:\Trading Analyzer"

echo [1/2] Starting Backend Server...
start "Trading Analyzer - Backend" cmd /k "cd /d D:\Trading Analyzer && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8002 --reload"

echo [*] Waiting for backend to initialize...
timeout /t 3 /nobreak >nul

echo [2/2] Starting Frontend Server...
start "Trading Analyzer - Frontend" cmd /k "cd /d D:\Trading Analyzer && npm run dev"

echo.
echo [*] Both servers starting...
echo [*] Opening browser in 5 seconds...
timeout /t 5 /nobreak >nul

start http://localhost:3000

echo.
echo =========================================================
echo   SERVERS RUNNING
echo =========================================================
echo.
echo   Both servers are now running in separate windows.
echo   You can close this launcher window.
echo.
echo   To stop: Close the backend and frontend terminal windows
echo            or press Ctrl+C in each.
echo.
echo =========================================================
echo.
pause
