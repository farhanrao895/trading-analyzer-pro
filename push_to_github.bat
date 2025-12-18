@echo off
title Push to GitHub
color 0A

echo.
echo =========================================================
echo   PUSH TO GITHUB
echo =========================================================
echo.

set GITHUB_URL=https://github.com/farhanrao895/trading-analyzer-pro.git

echo [*] Using GitHub URL: %GITHUB_URL%
echo.

echo [*] Checking if remote exists...
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo [*] Adding remote origin...
    git remote add origin %GITHUB_URL%
) else (
    echo [*] Remote already exists, updating...
    git remote set-url origin %GITHUB_URL%
)

echo [*] Setting branch to main...
git branch -M main

echo [*] Pushing to GitHub...
echo.
echo NOTE: If asked for credentials:
echo - Username: farhanrao895
echo - Password: Use a Personal Access Token (not your password)
echo   Create one at: https://github.com/settings/tokens
echo   Select: repo (all repo permissions)
echo.

git push -u origin main

if errorlevel 1 (
    echo.
    echo =========================================================
    echo   PUSH FAILED - Common Solutions:
    echo =========================================================
    echo.
    echo 1. Repository not created yet?
    echo    Go to: https://github.com/new
    echo    Name: trading-analyzer-pro
    echo    Click: Create repository
    echo.
    echo 2. Authentication failed?
    echo    Create Personal Access Token:
    echo    https://github.com/settings/tokens
    echo    - Click "Generate new token (classic)"
    echo    - Check "repo" permission
    echo    - Copy token and use as password
    echo.
    echo 3. Try again after fixing above issues
    echo.
    pause
    exit /b 1
)

echo.
echo =========================================================
echo   SUCCESS! Code pushed to GitHub!
echo =========================================================
echo.
echo   Repository: https://github.com/farhanrao895/trading-analyzer-pro
echo.
echo   Next steps:
echo   1. Deploy backend to Railway (railway.app)
echo   2. Deploy frontend to Vercel (vercel.com)
echo.
echo   See COMPLETE_DEPLOYMENT_GUIDE.md for details
echo.
pause
